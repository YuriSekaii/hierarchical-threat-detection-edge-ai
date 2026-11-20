"""Why does the blur/dark augmentation help the teacher but not the distilled student? Lab test cache of
sweep_stage1_rules.py, no GPU.

Every model is scored at its own working point: NMS 0.5 (YOLO26 has none, distilled students put two boxes on a knife),
then the lowest conf with precision >= 0.90 on the lab test. At that point every test knife is hit or missed. Then:
  - recall by knife speed and crop brightness (features as analyze_stage1_misses.py),
  - churn between pairs (without -> with augmentation): knives gained, knives lost.

  .venv/Scripts/python.exe results/samples/poc/investigate_stage1_aug.py
"""
import os, sys

import cv2
import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_stage1_rules as S

MODELS = {"A labgtL": "labgtL", "C labgtaugL": "labgtaugL", "E teachergtL": "teachergtL", "F teachergtaug": "teachergtaug",
          "B kdgtNL": "kdgtNL", "H kdgtaugFL": "kdgtaugFL",
          "C1 labgtaugS1L": "labgtaugS1L", "L labgtaugltL": "labgtaugltL"}  # C1 = C with seed 1 (noise), L = half-length blur
PAIRS = (("A labgtL", "C labgtaugL"), ("E teachergtL", "F teachergtaug"), ("B kdgtNL", "H kdgtaugFL"),
         ("F teachergtaug", "H kdgtaugFL"), ("E teachergtL", "B kdgtNL"),
         ("C labgtaugL", "C1 labgtaugS1L"), ("A labgtL", "C1 labgtaugS1L"), ("A labgtL", "L labgtaugltL"), ("C labgtaugL", "L labgtaugltL"))
TS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


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


rec0 = {"gt": [[0, 0, .1, .1], [.5, .5, .6, .6]], "det": {"d": [{"xyxy": [0, 0, .1, .1], "conf": .9}, {"xyxy": [0, 0, .1, .105], "conf": .8}]}}
assert hits(rec0, "d", 0.5) == ([True, False], 0)

recs = [r for r in S.load_cache("lab")]
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
    point[name] = best
n = len(point["A labgtL"][4])
print(f"Lab test, {n} knives. Working point: NMS 0.5, best recall with P >= 0.90\n")
print("| model | conf | P | R | false boxes |\n|---|---|---|---|---|")
for name, (t, tp, p, fp, _) in point.items():
    print(f"| {name} | {t:.2f} | {p * 100:.1f} | {tp / n * 100:.1f} | {fp} |")

# features per knife, same order as the hit lists
feat = []
for r in recs:
    img = None
    for g in r["gt"]:
        img = cv2.imread(r["path"]) if img is None else img
        h, w = img.shape[:2]
        c = cv2.cvtColor(img[int(g[1] * h):int(g[3] * h), int(g[0] * w):int(g[2] * w)], cv2.COLOR_BGR2GRAY)
        feat.append(dict(take=r["category"], f=int(os.path.splitext(r["path"])[0].rsplit("_f", 1)[1]), g=g,
                         side=((g[2] - g[0]) * 640 * (g[3] - g[1]) * 360) ** 0.5, mean=float(c.mean()) if c.size else 0.0,
                         path=r["path"]))
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

groups = [("slow (< 0.5)", lambda x: x["speed"] is not None and x["speed"] < 0.5),
          ("medium (0.5-2)", lambda x: x["speed"] is not None and 0.5 <= x["speed"] < 2),
          ("fast (2-4)", lambda x: x["speed"] is not None and 2 <= x["speed"] < 4),
          ("very fast (>= 4)", lambda x: x["speed"] is not None and x["speed"] >= 4),
          ("dark (grey < 100)", lambda x: x["mean"] < 100), ("mid (100-140)", lambda x: 100 <= x["mean"] < 140),
          ("bright (>= 140)", lambda x: x["mean"] >= 140), ("fast and dark", lambda x: (x["speed"] or 0) >= 2 and x["mean"] < 120)]
print("\n| group | knives | " + " | ".join(MODELS) + " |\n|---|---|" + "---|" * len(MODELS))
for lab, f in groups:
    idx = [i for i, x in enumerate(feat) if f(x)]
    print(f"| {lab} | {len(idx)} | " + " | ".join(f"{np.mean([point[m][4][i] for i in idx]) * 100:.1f} %" for m in MODELS) + " |")

print("\n| without -> with | gained | lost | net | gained: fast / dark | lost: fast / dark |\n|---|---|---|---|---|---|")
for a, b in PAIRS:
    ha, hb = point[a][4], point[b][4]
    gain = [i for i in range(n) if hb[i] and not ha[i]]
    lost = [i for i in range(n) if ha[i] and not hb[i]]
    fd = lambda ids: f"{sum((feat[i]['speed'] or 0) >= 2 for i in ids)} / {sum(feat[i]['mean'] < 100 for i in ids)}"
    print(f"| {a} -> {b} | {len(gain)} | {len(lost)} | {len(gain) - len(lost):+d} | {fd(gain)} | {fd(lost)} |")

# knives no model finds, at any working point above
never = [i for i in range(n) if not any(point[m][4][i] for m in MODELS)]
print(f"\nknives missed by every model: {len(never)} ({sum((feat[i]['speed'] or 0) >= 2 for i in never)} fast, "
      f"{sum(feat[i]['mean'] < 100 for i in never)} dark)")
OUT = os.path.join("results", "stage1_kd", "aug_invest")
os.makedirs(OUT, exist_ok=True)
import json
json.dump([dict(path=feat[i]["path"], g=feat[i]["g"], speed=feat[i]["speed"], mean=feat[i]["mean"]) for i in never],
          open(os.path.join(OUT, "never_found.json"), "w"), indent=0)
json.dump({m: dict(conf=v[0], hits=v[4]) for m, v in point.items()}, open(os.path.join(OUT, "points.json"), "w"))

# GPU=1: what do the teachers score on the blurred/darkened training copies, against the same frame unchanged?
# In distillation the student is pulled toward these scores on every copy, while the label says 1.0.
if os.environ.get("GPU"):
    import random
    from ultralytics import YOLO
    AUG = os.path.join("data", "stage1_negatives", "lab_gt_aug")
    train = [l.strip() for l in open(os.path.join("data", "stage1_negatives", "lab_gt", "train.txt")) if l.strip()]
    orig = {os.path.splitext(os.path.basename(p))[0]: p for p in train}
    copies = sorted(os.listdir(os.path.join(AUG, "images", "aug")))
    random.Random(0).shuffle(copies)
    copies = copies[:300]
    W = {"E teachergtL": "weights/stage1_repro846_teachergt_p250_last.pt", "F teachergtaug": "weights/stage1_repro846_teachergtaug_p250.pt",
         "G teachergtaugL": "weights/stage1_repro846_teachergtaug_p250_last.pt"}

    def knife_conf(model, img, boxes):
        res = model(img, imgsz=640, conf=0.001, device=os.environ.get("DEV", "0"), verbose=False)[0]
        h, w = img.shape[:2]
        out = []
        for g in boxes:
            c = [cf for b, cf, k in zip(res.boxes.xyxyn.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())
                 if res.names[int(k)] == "knife" and S.calculate_iou(b, g) >= S.IOU_THRESH]
            out.append(max(c, default=0.0))
        return out

    rows = {m: ([], []) for m in W}
    for m, p in W.items():
        model = YOLO(p)
        for name in copies:
            stem = os.path.splitext(name)[0][:-4]
            lbl = [l.split() for l in open(os.path.join(AUG, "labels", "aug", stem + "_aug.txt")) if l.startswith("1 ")]
            boxes = [[float(a) - float(c) / 2, float(b) - float(d) / 2, float(a) + float(c) / 2, float(b) + float(d) / 2] for _, a, b, c, d in lbl]
            rows[m][0].extend(knife_conf(model, cv2.imread(orig[stem]), boxes))
            rows[m][1].extend(knife_conf(model, cv2.imread(os.path.join(AUG, "images", "aug", name)), boxes))
    print(f"\nTeacher knife score on {len(copies)} training copies ({len(rows['E teachergtL'][0])} knives): unchanged frame -> blurred/dark copy")
    print("| teacher | trained on the copies | median, unchanged | median, copy | copy < 0.25 | copy < 0.5 |\n|---|---|---|---|---|---|")
    for m in W:
        a, b = np.array(rows[m][0]), np.array(rows[m][1])
        print(f"| {m} | {'no' if m.startswith('E') else 'yes'} | {np.median(a):.2f} | {np.median(b):.2f} | {(b < 0.25).mean() * 100:.0f} % | {(b < 0.5).mean() * 100:.0f} % |")
