"""Stage 1 scored against the two targets instead of one overall precision / recall (lab test, caches, no GPU).

  1. Recall on knives a person can see clearly (label_stage1_visibility.py: c). Target 90-95 %.
  2. Precision on boxes at a fast-moving hand: a fast swing may be missed, but a box there must be a knife.

Also reported: recall on blurred (b) and hidden (h) knives, and knives whose visibility label is still unsure (?).
Hand speed: every wrist (pose conf >= 0.3) is matched to the nearest wrist in the labelled frames 10 before and 10 after
(same take); speed = mean shift / torso length per 10 frames (1/3 s). A box belongs to the nearest wrist within 0.5 torso;
a box with no wrist that near is "away from a hand". Boxes after NMS 0.5. Labels: lab_stage1_gt test + the by-eye fixes
(results/stage1_kd/cand_review/test_label_fix.json).

  .venv/Scripts/python.exe results/samples/poc/eval_stage1_targets.py [FAST=0.5]
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("LAB_FIX", os.path.join(REPO, "results", "stage1_kd", "cand_review", "test_label_fix.json"))
sys.argv = sys.argv[:1]
sys.path.insert(0, HERE)
import sweep_stage1_rules as S

VIS = json.load(open(os.path.join(REPO, "results", "stage1_kd", "vis_review", "visibility.json")))
FAST = float(os.environ.get("FAST", 0.5))
MODELS = {"shipped": "shipped", "A labgt": "labgtL", "C blur/dark": "labgtaugL", "C1 blur/dark s1": "labgtaugS1L",
          "L half blur": "labgtaugltL", "B KD from E": "kdgtNL", "H KD from F": "kdgtaugFL", "E teacher": "teachergtL",
          "F teacher blur/dark": "teachergtaug"}
CONFS = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50)


def nms(bs, thr=0.5):
    out = []
    for b in sorted(bs, key=lambda v: -v[4]):
        if all(S.calculate_iou(b[:4], o[:4]) < thr for o in out):
            out.append(b)
    return out


COVER = os.environ.get("MATCH") == "cover"  # also a hit: the box holds >= half of the knife label (hand + knife boxes)


def overlap(b, g):
    if not COVER:
        return S.calculate_iou(b, g)
    iw, ih = max(0, min(b[2], g[2]) - max(b[0], g[0])), max(0, min(b[3], g[3]) - max(b[1], g[1]))
    return max(S.calculate_iou(b, g), S.IOU_THRESH if iw * ih >= 0.5 * (g[2] - g[0]) * (g[3] - g[1]) else 0.0)


def match(rec, d, t):
    """Greedy match as evaluate_matches, after NMS: (hit per gt, [(box, is_tp)])."""
    bs = nms([tuple(b["xyxy"]) + (b["conf"],) for b in rec["det"][d] if b["conf"] >= t])
    got, out = [False] * len(rec["gt"]), []
    for b in bs:
        cand = [(overlap(b[:4], g), i) for i, g in enumerate(rec["gt"]) if not got[i]]
        v, i = max(cand, default=(0.0, -1))
        tp = v >= S.IOU_THRESH
        if tp:
            got[i] = True
        out.append((b, tp))
    return got, out


def wrists(rec):
    return [(np.array(w["xy"]), p["torso"] or 0.15 * rec["h"]) for p in rec["pose"] for w in p["wrists"] if w["conf"] >= 0.3]


recs = S.load_cache("lab")
BOXFIX = os.path.join(REPO, "results", "stage1_kd", "vis_review", "miss_check", "box_fix.json")
BOXFIX = json.load(open(BOXFIX)) if os.path.exists(BOXFIX) else {}  # boxes the human redrew in review_misses.bat
dropped = {"d": 0, "x": 0}
for r in recs:
    for j, b in BOXFIX.get(os.path.basename(r["path"]), {}).items():
        r["gt"][int(j)] = b  # d = second label on an already labelled knife, x = no knife (check_stage1_misses.py): not knives to find
    v = VIS.get(os.path.basename(r["path"]), {})
    keep = [j for j in range(len(r["gt"])) if v.get(str(j)) not in dropped]
    for j in range(len(r["gt"])):
        if v.get(str(j)) in dropped:
            dropped[v[str(j)]] += 1
    r["vis"] = [{"m": "w"}.get(v.get(str(j)), v.get(str(j))) or "?" for j in keep]  # m = hidden and box misses: scored with w
    r["gt"], r["gi"] = [r["gt"][j] for j in keep], keep  # gi: index in the original label list
frame ={(r["category"], int(os.path.splitext(r["path"])[0].rsplit("_f", 1)[1])): r for r in recs}
for (take, f), r in frame.items():
    ws = wrists(r)
    speed = []
    for xy, torso in ws:
        sh = [min(np.linalg.norm(xy - xy2) for xy2, _ in wrists(n)) / torso
              for n in (frame.get((take, f - 10)), frame.get((take, f + 10))) if n and wrists(n)]
        speed.append(float(np.mean(sh)) if sh else None)
    r["wr"] = [(xy, torso, s) for (xy, torso), s in zip(ws, speed)]


def hand_speed(rec, box):
    """Speed of the nearest wrist within 0.5 torso of the box centre; 'away' if none, None if the speed is unknown."""
    c = np.array([(box[0] + box[2]) / 2 * rec["w"], (box[1] + box[3]) / 2 * rec["h"]])
    near = [(np.linalg.norm(c - xy) / torso, s) for xy, torso, s in rec["wr"] if np.linalg.norm(c - xy) <= 0.5 * torso]
    return min(near, key=lambda v: v[0])[1] if near else "away"


vis = [v for r in recs for v in r["vis"]]
n = len(vis)
groups = {g: [i for i, v in enumerate(vis) if v == g] for g in ("c", "b", "h", "?", "w")}
print(f"Lab test, {n} knives (with the by-eye fixes; dropped {dropped['d']} duplicate and {dropped['x']} no-knife labels): "
      f"clear {len(groups['c'])}, blurred {len(groups['b'])}, hidden {len(groups['h'])}, unsure {len(groups['?'])}, "
      f"box misses the knife {len(groups['w'])}. Fast hand = speed >= {FAST} torso / 10 frames."
      f"{' MATCH=cover: a box holding >= half the knife label is a hit.' if COVER else ''}\n")

# how well does wrist speed separate the blurred knives from the clear ones?
sp = [hand_speed(r, g) for r in recs for g in r["gt"]]
print("| knife label | knives | hand speed known | median speed | share fast |\n|---|---|---|---|---|")
for g, name in (("c", "clear"), ("b", "blurred"), ("h", "hidden"), ("?", "unsure")):
    v = [sp[i] for i in groups[g] if isinstance(sp[i], float)]
    print(f"| {name} | {len(groups[g])} | {len(v)} | {np.median(v):.2f} | {np.mean(np.array(v) >= FAST) * 100:.0f} % |")

rows = {}
for name, d in MODELS.items():
    for t in CONFS:
        hit, boxes = [], {"fast": [0, 0], "slow": [0, 0], "away": [0, 0], "unknown": [0, 0]}
        for r in recs:
            got, bs = match(r, d, t)
            hit += got
            for b, tp in bs:
                s = hand_speed(r, b[:4])
                k = "away" if s == "away" else "unknown" if s is None else "fast" if s >= FAST else "slow"
                boxes[k][0 if tp else 1] += 1
        rows[(name, t)] = (hit, boxes)


def rec_(hit, g):
    return np.mean([hit[i] for i in groups[g]]) * 100 if groups[g] else float("nan")


def prec(c):
    return f"{c[0] / max(sum(c), 1) * 100:.1f} ({c[1]})"


print("\nRecall per knife group, and precision per box group: P (false boxes). Target: clear recall 90-95 %, fast-hand precision high.\n")
print("| model | conf | **clear R** | blurred R | hidden R | unsure R | all R | all P | **fast-hand P** | slow-hand P | away-from-hand P |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (name, t), (hit, bx) in rows.items():
    tot = [sum(v[0] for v in bx.values()), sum(v[1] for v in bx.values())]
    print(f"| {name} | {t:.2f} | **{rec_(hit, 'c'):.1f}** | {rec_(hit, 'b'):.1f} | {rec_(hit, 'h'):.1f} | {rec_(hit, '?'):.1f} | "
          f"{np.mean(hit) * 100:.1f} | {prec(tot)} | **{prec(bx['fast'])}** | {prec(bx['slow'])} | {prec(bx['away'])} |")

# the lowest conf where each model reaches 90 % on clear knives, and what the fast-hand precision is there
print("\nPer model: highest listed conf that still gives clear-knife recall >= 90 %, and the boxes there.\n")
print("| model | conf | clear R | blurred R | fast-hand P | all P |\n|---|---|---|---|---|---|")
for name in MODELS:
    ok = [(t, rows[(name, t)]) for t in CONFS if rec_(rows[(name, t)][0], "c") >= 90]
    if not ok:
        best = max(CONFS, key=lambda t: rec_(rows[(name, t)][0], "c"))
        print(f"| {name} | never (best {rec_(rows[(name, best)][0], 'c'):.1f} at {best:.2f}) | | | | |")
        continue
    t, (hit, bx) = max(ok, key=lambda v: v[0])
    tot = [sum(v[0] for v in bx.values()), sum(v[1] for v in bx.values())]
    print(f"| {name} | {t:.2f} | {rec_(hit, 'c'):.1f} | {rec_(hit, 'b'):.1f} | {prec(bx['fast'])} | {prec(tot)} |")
