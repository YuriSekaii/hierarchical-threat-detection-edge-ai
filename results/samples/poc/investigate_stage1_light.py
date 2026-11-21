"""Why does the half-length-blur student L lose lab knives and produce more false boxes? Lab test cache of
sweep_stage1_rules.py, no GPU.

A = labgtL, C = labgtaugL, C1 = labgtaugS1L (seed 1), L = labgtaugltL (half-length blur). Every model is
scored at its own working point (NMS 0.5, best recall with P >= 0.90) and at fixed conf 0.20 / 0.40. Then:
  - confidence on real knives (all / slow), and how L's lost knives split by L's own best conf,
  - false boxes after NMS at conf 0.25 on lab and gtv2: near a wrist, high conf, repeated across frames,
  - contact sheets of the knives C finds and L misses, and of L's highest-conf false boxes.

  .venv/Scripts/python.exe results/samples/poc/investigate_stage1_light.py
"""
import os, sys, json, random
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_stage1_rules as S

MODELS = {"A": "labgtL", "C": "labgtaugL", "C1": "labgtaugS1L", "L": "labgtaugltL"}
TS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
OUT = os.path.join("results", "stage1_kd", "light_invest")


def nms(bs, thr=0.5):
    out = []
    for b in sorted(bs, key=lambda v: -v[4]):
        if all(S.calculate_iou(b[:4], o[:4]) < thr for o in out):
            out.append(b)
    return out


def hits(rec, d, t):
    """Per test knife of rec: True if a kept box (NMS, conf >= t) is matched to it (greedy, as evaluate_matches)."""
    bs = nms([tuple(b["xyxy"]) + (b["conf"],) for b in rec["det"][d] if b["conf"] >= t])
    got, fp = [False] * len(rec["gt"]), 0
    for b in bs:
        cand = [(S.calculate_iou(b[:4], g), i) for i, g in enumerate(rec["gt"]) if not got[i]]
        v, i = max(cand, default=(0.0, -1))
        if v >= S.IOU_THRESH:
            got[i] = True
        else:
            fp += 1
    return got, fp


def false_boxes(rec, d, t):
    """Boxes kept by NMS with conf >= t that match no test knife (same greedy matching as hits)."""
    bs = nms([tuple(b["xyxy"]) + (b["conf"],) for b in rec["det"][d] if b["conf"] >= t])
    got, out = [False] * len(rec["gt"]), []
    for b in bs:
        cand = [(S.calculate_iou(b[:4], g), i) for i, g in enumerate(rec["gt"]) if not got[i]]
        v, i = max(cand, default=(0.0, -1))
        if v >= S.IOU_THRESH:
            got[i] = True
        else:
            out.append(b)
    return out


def match_conf(rec, d, g):
    """Best conf of a box with IoU >= IOU_THRESH to the knife (0 if none)."""
    return max((b["conf"] for b in rec["det"][d] if S.calculate_iou(b["xyxy"], g) >= S.IOU_THRESH), default=0.0)


def best_iou_box(rec, d, g):
    """L's best box on the knife by IoU, any conf: (xyxy, conf), or (None, 0)."""
    boxes = rec["det"][d]
    if not boxes:
        return None, 0.0
    b = max(boxes, key=lambda b: S.calculate_iou(b["xyxy"], g))
    return b["xyxy"], b["conf"]


def frame_no(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.rsplit("_f", 1)[-1]
    if tail.isdigit():
        return int(tail)
    digits = "".join(ch for ch in stem[::-1] if ch.isdigit())[::-1]
    return int(digits) if digits else None


def near_wrist(rec, box):
    W, H = rec["w"], rec["h"]
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    for person in rec["pose"]:
        torso = person["torso"] if person["torso"] is not None else 0.15 * H
        for wr in person["wrists"]:
            if wr["conf"] < 0.3:
                continue
            dist = ((wr["xy"][0] - cx) ** 2 + (wr["xy"][1] - cy) ** 2) ** 0.5
            if dist <= 0.5 * torso:
                return True
    return False


def make_tile(img, box, draws):
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    cw, ch = max((box[2] - box[0]) * W * 3, 96), max((box[3] - box[1]) * H * 3, 96)
    ax1, ay1 = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
    ax2, ay2 = int(min(W, cx + cw / 2)), int(min(H, cy + ch / 2))
    if ax2 - ax1 < 2 or ay2 - ay1 < 2:
        return None
    crop = cv2.resize(img[ay1:ay2, ax1:ax2], (160, 160))
    sx, sy = 160 / (ax2 - ax1), 160 / (ay2 - ay1)

    def proj(b):
        return (int((b[0] * W - ax1) * sx), int((b[1] * H - ay1) * sy),
                int((b[2] * W - ax1) * sx), int((b[3] * H - ay1) * sy))

    for b, color, text in draws:
        x1, y1, x2, y2 = proj(b)
        cv2.rectangle(crop, (x1, y1), (x2, y2), color, 1)
        if text:
            cv2.putText(crop, text, (max(1, x1), max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    return crop


def write_sheet(tiles, name):
    cols, cell = 8, 160
    rows = max(1, (len(tiles) + cols - 1) // cols)
    sheet = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = t
    path = os.path.join(OUT, name)
    cv2.imwrite(path, sheet)
    print(f"wrote {path} ({len(tiles)} tiles)")


rec0 = {"gt": [[0, 0, .1, .1], [.5, .5, .6, .6]], "det": {"d": [{"xyxy": [0, 0, .1, .1], "conf": .9}, {"xyxy": [0, 0, .1, .105], "conf": .8}]}}
assert hits(rec0, "d", 0.5) == ([True, False], 0)

os.makedirs(OUT, exist_ok=True)
recs = S.load_cache("lab")

point = {}
for name, d in MODELS.items():
    best = None
    for t in TS:
        h = [hits(r, d, t) for r in recs]
        tp = sum(sum(g) for g, _ in h)
        fp = sum(f for _, f in h)
        p = tp / max(tp + fp, 1)
        if p >= 0.90 and (best is None or tp > best[1]):
            best = (t, tp, p, fp, [x for g, _ in h for x in g])
    point[name] = best or (1.01, 0, 0.0, 0, [False] * sum(len(r["gt"]) for r in recs))

feat = []
confs = {m: [] for m in MODELS}
for r in recs:
    img = None
    for g in r["gt"]:
        img = cv2.imread(r["path"]) if img is None else img
        h, w = img.shape[:2]
        c = cv2.cvtColor(img[int(g[1] * h):int(g[3] * h), int(g[0] * w):int(g[2] * w)], cv2.COLOR_BGR2GRAY)
        feat.append(dict(rec=r, take=r["category"], f=frame_no(r["path"]), g=g,
                         side=((g[2] - g[0]) * 640 * (g[3] - g[1]) * 360) ** 0.5, mean=float(c.mean()) if c.size else 0.0,
                         path=r["path"]))
        for m, d in MODELS.items():
            confs[m].append(match_conf(r, d, g))
by = {}
for x in feat:
    by.setdefault((x["take"], x["f"]), []).append(x)
for x in feat:
    sh = []
    for df in (-10, 10):
        for y in by.get((x["take"], x["f"] + df), []):
            a, b = x["g"], y["g"]
            sh.append((((a[0] + a[2] - b[0] - b[2]) / 2 * 640) ** 2 + ((a[1] + a[3] - b[1] - b[3]) / 2 * 360) ** 2) ** 0.5 / max(x["side"], 1))
    x["speed"] = min(sh) if sh else None

n = len(feat)
print(f"Lab test, {n} knives. Working point: NMS 0.5, best recall with P >= 0.90\n")

print("## Confidence on real knives (best box with IoU >= %.2f, before NMS)\n" % S.IOU_THRESH)
print("| model | median all | median slow (< 0.5) | >= 0.40 | >= 0.20 | < 0.05 |\n|---|---|---|---|---|---|")
for m in MODELS:
    v = np.array(confs[m])
    slow = np.array([c for c, x in zip(confs[m], feat) if x["speed"] is not None and x["speed"] < 0.5])
    med = f"{np.median(v):.3f}" if len(v) else "n/a"
    meds = f"{np.median(slow):.3f}" if len(slow) else "n/a"
    print(f"| {m} | {med} | {meds} | {(v >= 0.40).mean() * 100:.1f} % | {(v >= 0.20).mean() * 100:.1f} % | {(v < 0.05).mean() * 100:.1f} % |")

print("\n| model | conf | P | R | false boxes |\n|---|---|---|---|---|")
for m, (t, tp, p, fp, _) in point.items():
    print(f"| {m} | {t:.2f} | {p * 100:.1f} | {tp / n * 100:.1f} | {fp} |")

print("\n| model | conf | P | R | false boxes |\n|---|---|---|---|---|")
for t in (0.20, 0.40):
    for m, d in MODELS.items():
        h = [hits(r, d, t) for r in recs]
        tp = sum(sum(g) for g, _ in h)
        fp = sum(f for _, f in h)
        print(f"| {m} | {t:.2f} | {tp / max(tp + fp, 1) * 100:.1f} | {tp / n * 100:.1f} | {fp} |")

fd = lambda ids: (sum((feat[i]["speed"] or 0) >= 2 for i in ids), sum(feat[i]["mean"] < 100 for i in ids))
top5 = lambda ids: ", ".join(f"{k} ({v})" for k, v in Counter(feat[i]["take"] for i in ids).most_common(5)) or "-"

for a in ("C", "A"):
    lt = point["L"][0]
    idx = [i for i in range(n) if point[a][4][i] and not point["L"][4][i]]
    print(f"\n## Knives {a} finds and L misses at their own working points ({len(idx)} knives; L conf = {lt:.2f})\n")
    print("| L's best box conf on the knife | knives | fast (>= 2) | dark (< 100) | top takes |\n|---|---|---|---|---|")
    lbest = {i: confs["L"][i] for i in idx}  # best L conf at IoU >= 0.2, as in the confidence table
    bins = [("< 0.05 (not seen)", lambda c: c < 0.05),
            (f"0.05 to {lt:.2f} (under threshold)", lambda c: 0.05 <= c < lt),
            (f">= {lt:.2f} (box elsewhere / NMS)", lambda c: c >= lt)]
    for lab, f in bins:
        ids = [i for i in idx if f(lbest[i])]
        fast, dark = fd(ids)
        print(f"| {lab} | {len(ids)} | {fast} | {dark} | {top5(ids)} |")
    fast, dark = fd(idx)
    print(f"| total | {len(idx)} | {fast} | {dark} | {top5(idx)} |")

print("\n## False boxes after NMS at conf 0.25\n")
sets = {"lab": recs, "gtv2": S.load_cache("gtv2")}
fbs_all = {}
for sname, rs in sets.items():
    print(f"### {sname}\n")
    print("| model | false boxes | near wrist (<= 0.5 torso) | median conf | conf >= 0.40 | repeat (same take, +/-1..15 frames, IoU >= 0.3) |\n|---|---|---|---|---|---|")
    fbs = {}
    for m, d in MODELS.items():
        cur = []
        for r in rs:
            for b in false_boxes(r, d, 0.25):
                cur.append(dict(rec=r, box=b, conf=b[4], take=r["category"], frame=frame_no(r["path"])))
        fbs[m] = cur
        groups = defaultdict(list)
        for i, fb in enumerate(cur):
            groups[fb["take"]].append(i)
        rep = 0
        for group in groups.values():
            for i in group:
                if cur[i]["frame"] is None:
                    continue
                if any(j != i and cur[j]["frame"] is not None and 1 <= abs(cur[i]["frame"] - cur[j]["frame"]) <= 15
                       and S.calculate_iou(cur[i]["box"][:4], cur[j]["box"][:4]) >= 0.3 for j in group):
                    rep += 1
        cs = np.array([fb["conf"] for fb in cur])
        print(f"| {m} | {len(cur)} | {sum(near_wrist(fb['rec'], fb['box']) for fb in cur)} | "
              f"{np.median(cs) if len(cs) else 0:.3f} | {int((cs >= 0.40).sum()) if len(cs) else 0} | {rep} |")
    for m in ("L", "C"):
        print(f"\ntop takes {m} ({MODELS[m]}): " + ", ".join(
            f"{k} ({v})" for k, v in Counter(fb["take"] for fb in fbs[m]).most_common(5)))
    fbs_all[sname] = fbs
    print()

lost = [i for i in range(n) if point["C"][4][i] and not point["L"][4][i]]
if len(lost) > 48:
    random.Random(0).shuffle(lost)
    lost = lost[:48]
imgs, tiles = {}, []
for i in lost:
    x = feat[i]
    if x["path"] not in imgs:
        imgs[x["path"]] = cv2.imread(x["path"])
    img = imgs[x["path"]]
    if img is None:
        continue
    lbox, lconf = best_iou_box(x["rec"], MODELS["L"], x["g"])
    draws = [(x["g"], (0, 255, 0), "")]
    if lbox is not None:
        draws.append((lbox, (0, 0, 255), f"L {lconf:.2f}"))
    tile = make_tile(img, x["g"], draws)
    if tile is None:
        continue
    cv2.putText(tile, f"C {confs['C'][i]:.2f}", (2, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
write_sheet(tiles, "lost_C_to_L.jpg")

ff = [fb for fb in fbs_all["lab"]["L"] if fb["conf"] >= 0.40]
if len(ff) > 48:
    random.Random(0).shuffle(ff)
    ff = ff[:48]
imgs, tiles = {}, []
for fb in ff:
    path = fb["rec"]["path"]
    if path not in imgs:
        imgs[path] = cv2.imread(path)
    img = imgs[path]
    if img is None:
        continue
    tile = make_tile(img, fb["box"][:4], [(fb["box"][:4], (0, 0, 255), f"{fb['conf']:.2f}")])
    if tile is None:
        continue
    cv2.putText(tile, fb["take"][-22:], (2, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
write_sheet(tiles, "false_L.jpg")

json.dump([dict(path=feat[i]["path"], gt=feat[i]["g"], speed=feat[i]["speed"], mean=feat[i]["mean"],
                conf={m: confs[m][i] for m in MODELS}, hit={m: point[m][4][i] for m in MODELS})
           for i in range(n)], open(os.path.join(OUT, "knives.json"), "w"), indent=0)
print(f"wrote {os.path.join(OUT, 'knives.json')} ({n} knives)")

# Are the lab false boxes really false? For each false box (NMS, conf 0.25): the best IoU with a test knife in the same
# frame (0 < IoU < 0.2 = knife found, box too far off to match), else whether a labelled knife of a frame 1..20 away (test frames are every 10th)
# in the same take overlaps it (IoU >= 0.1 = probably a knife whose label is missing in this frame), else nothing labelled near.
labelled = {(r["category"], frame_no(r["path"])): r["gt"] for r in recs}
print("\n## Lab false boxes at conf 0.25 (NMS): what is under them\n")
print("| model | false boxes | overlaps a labelled knife, IoU < 0.2 | labelled knife there 1-20 frames away | nothing labelled near |\n|---|---|---|---|---|")
for m in MODELS:
    k = [0, 0, 0]
    for fb in fbs_all["lab"][m]:
        r, b = fb["rec"], fb["box"][:4]
        if max((S.calculate_iou(b, g) for g in r["gt"]), default=0.0) > 0:
            k[0] += 1
        elif any(S.calculate_iou(b, g) >= 0.1 for df in range(-20, 21) if df
                 for g in labelled.get((fb["take"], (fb["frame"] or 0) + df), [])):
            k[1] += 1
        else:
            k[2] += 1
    print(f"| {m} | {sum(k)} | " + " | ".join(str(v) for v in k) + " |")
