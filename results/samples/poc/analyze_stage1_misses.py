"""Why does a Stage 1 detector miss lab knives? Uses the sweep_stage1_rules.py lab cache (no GPU).

Every test knife box gets one cause:
  hit        a box at conf >= T overlaps it at IoU >= 0.2
  bad_box    a box at conf >= T touches it (IoU > 0) but IoU < 0.2: found, box drawn wrong
  low_score  a box at IoU >= 0.2 exists, but its conf is below T: seen, not sure enough
  unseen     no box at any cached conf with IoU >= 0.2 (and none at conf >= T touching it)
and features: size at the 384x640 model input, take, blur and contrast of the crop, a wrist near it, whether the
teacher finds it. Writes a summary and contact sheets of the misses to results/stage1_kd/misses/.

  DET=kdgtNL T=0.40 .venv/Scripts/python.exe results/samples/poc/analyze_stage1_misses.py
"""
import collections, json, os, sys

import cv2
import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_stage1_rules as S

DET = os.environ.get("DET", "kdgtNL")
T = float(os.environ.get("T", "0.40"))
REF = os.environ.get("REF", "teachergtL")  # does the teacher find it at its own working point (0.25)?
OUT = os.path.join(S.REPO if hasattr(S, "REPO") else ".", "results", "stage1_kd", "misses", DET)
os.makedirs(OUT, exist_ok=True)


def iou(a, b):
    return S.calculate_iou(a, b)


def classify(rec, g, det, t):
    boxes = rec["det"][det]
    best = max([(iou(b["xyxy"], g), b["conf"]) for b in boxes], default=(0.0, 0.0))
    hi = [iou(b["xyxy"], g) for b in boxes if b["conf"] >= t]
    if any(v >= S.IOU_THRESH for v in hi):
        return "hit", best
    if any(v > 0 for v in hi):
        return "bad_box", best
    seen = [b["conf"] for b in boxes if iou(b["xyxy"], g) >= S.IOU_THRESH]
    if seen:
        return "low_score", (S.IOU_THRESH, max(seen))
    return "unseen", best


def wrist_near(rec, g, k=0.5):
    return S.gate_passes(rec, g, DET, {"kind": "wrist@k", "k": k})


def crop_stats(img, g):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(g[0] * w), int(g[1] * h), int(g[2] * w), int(g[3] * h)
    c = cv2.cvtColor(img[max(y1, 0):y2, max(x1, 0):x2], cv2.COLOR_BGR2GRAY)
    if c.size == 0:
        return 0.0, 0.0, 0.0
    return float(cv2.Laplacian(c, cv2.CV_64F).var()), float(c.std()), float(c.mean())


def self_check():
    rec = {"det": {"d": [{"xyxy": [0.1, 0.1, 0.2, 0.2], "conf": 0.9}, {"xyxy": [0.5, 0.5, 0.6, 0.6], "conf": 0.1}]}}
    assert classify(rec, [0.1, 0.1, 0.2, 0.2], "d", 0.4)[0] == "hit"
    assert classify(rec, [0.5, 0.5, 0.6, 0.6], "d", 0.4)[0] == "low_score"
    assert classify(rec, [0.8, 0.8, 0.9, 0.9], "d", 0.4)[0] == "unseen"
    assert classify(rec, [0.18, 0.18, 0.4, 0.4], "d", 0.4)[0] == "bad_box"


self_check()
recs = [r for r in S.load_cache("lab") if r["gt"]]
rows = []
for r in recs:
    img = None
    for g in r["gt"]:
        cause, (biou, bconf) = classify(r, g, DET, T)
        ref = classify(r, g, REF, 0.25)[0] == "hit"
        img = cv2.imread(r["path"]) if img is None else img
        lap, std, mean = crop_stats(img, g)
        rows.append(dict(path=r["path"], take=r["category"], g=g, cause=cause, best_iou=biou, best_conf=bconf, ref_hit=ref,
                         wrist=wrist_near(r, g), persons=len(r["pose"]),
                         side=((g[2] - g[0]) * 640 * (g[3] - g[1]) * 360) ** 0.5,  # sqrt area at the 384x640 input (360 rows of image)
                         aspect=max((g[2] - g[0]) * 16, (g[3] - g[1]) * 9) / max(min((g[2] - g[0]) * 16, (g[3] - g[1]) * 9), 1e-6),
                         lap=lap, std=std, mean=mean))

# knife speed: centre shift to the knife in the nearest labelled frame of the same take (frames are 10 apart), in box sides
by_frame = {(x["take"], int(os.path.splitext(x["path"])[0].rsplit("_f", 1)[1])): x for x in rows}
for (take, f), x in by_frame.items():
    sh = []
    for df in (-10, 10):
        y = by_frame.get((take, f + df))
        if y:
            a, b = x["g"], y["g"]
            d = ((((a[0] + a[2]) - (b[0] + b[2])) / 2 * 640) ** 2 + (((a[1] + a[3]) - (b[1] + b[3])) / 2 * 360) ** 2) ** 0.5
            sh.append(d / max(x["side"], 1))
    x["speed"] = min(sh) if sh else None

n = len(rows)
cnt = collections.Counter(x["cause"] for x in rows)
lines = [f"# {DET} at conf {T:.2f}, lab test: {n} boxes, recall {cnt['hit'] / n * 100:.1f} %", "",
         "| cause | boxes | share of misses | teacher finds it (0.25) | wrist within 0.5 torso |", "|---|---|---|---|---|"]
miss = [x for x in rows if x["cause"] != "hit"]
for c in ("bad_box", "low_score", "unseen"):
    xs = [x for x in rows if x["cause"] == c]
    if xs:
        lines.append(f"| {c} | {len(xs)} | {len(xs) / len(miss) * 100:.0f} % | {sum(x['ref_hit'] for x in xs) / len(xs) * 100:.0f} % "
                     f"| {sum(x['wrist'] for x in xs) / len(xs) * 100:.0f} % |")
lines.append(f"| hit | {cnt['hit']} | | {sum(x['ref_hit'] for x in rows if x['cause'] == 'hit') / cnt['hit'] * 100:.0f} % "
             f"| {sum(x['wrist'] for x in rows if x['cause'] == 'hit') / cnt['hit'] * 100:.0f} % |")
low = [x["best_conf"] for x in rows if x["cause"] == "low_score"]
if low:
    lines += ["", "low_score conf: " + ", ".join(f"{a:.2f}-{b:.2f}: {sum(a <= v < b for v in low)}"
                                             for a, b in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, T)))]


def table(title, key, bins):
    out = ["", f"| {title} | boxes | recall | misses |", "|---|---|---|---|"]
    for lab, f in bins:
        xs = [x for x in rows if x.get(key) is not None and f(x[key])]
        if xs:
            h = sum(x["cause"] == "hit" for x in xs)
            out.append(f"| {lab} | {len(xs)} | {h / len(xs) * 100:.1f} % | {len(xs) - h} |")
    return out


lines += table("box side at model input (px)", "side", [(f"{a}-{b}", lambda v, a=a, b=b: a <= v < b)
                                                         for a, b in ((0, 12), (12, 16), (16, 20), (20, 28), (28, 40), (40, 999))])
lines += table("knife shift to the frame 10 later or earlier (box sides)", "speed", [(f"{a}-{b}", lambda v, a=a, b=b: a <= v < b)
                                                                                  for a, b in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 999))])
lines += table("box long/short side", "aspect", [(f"{a}-{b}", lambda v, a=a, b=b: a <= v < b) for a, b in ((1, 1.5), (1.5, 2.5), (2.5, 4), (4, 99))])
lines += table("crop sharpness (Laplacian var)", "lap", [(f"{a}-{b}", lambda v, a=a, b=b: a <= v < b)
                                                                       for a, b in ((0, 20), (20, 50), (50, 100), (100, 300), (300, 99999))])
lines += table("crop brightness (grey mean)", "mean", [(f"{a}-{b}", lambda v, a=a, b=b: a <= v < b)
                                                      for a, b in ((0, 60), (60, 100), (100, 140), (140, 180), (180, 256))])
lines += table("crop contrast (grey std)", "std", [(f"{a}-{b}", lambda v, a=a, b=b: a <= v < b)
                                                                 for a, b in ((0, 10), (10, 20), (20, 35), (35, 999))])
lines += table("wrist within 0.5 torso", "wrist", [("yes", bool), ("no", lambda v: not v)])
lines += ["", "| take | boxes | recall | misses | unseen | low_score | bad_box |", "|---|---|---|---|---|---|---|"]
for take in sorted({x["take"] for x in rows}):
    xs = [x for x in rows if x["take"] == take]
    c = collections.Counter(x["cause"] for x in xs)
    lines.append(f"| {take} | {len(xs)} | {c['hit'] / len(xs) * 100:.1f} % | {len(xs) - c['hit']} | {c['unseen']} | {c['low_score']} | {c['bad_box']} |")
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(miss, open(os.path.join(OUT, "misses.json"), "w"), indent=0)
print("\n".join(lines))

# contact sheets: each miss as a crop 4x the box, box in green, best detection in the cache in red, conf printed
for c in ("unseen", "low_score", "bad_box"):
    xs = [x for x in miss if x["cause"] == c]
    tiles = []
    for x in xs:
        img = cv2.imread(x["path"])
        h, w = img.shape[:2]
        g = x["g"]
        cx, cy = (g[0] + g[2]) / 2 * w, (g[1] + g[3]) / 2 * h
        half = max((g[2] - g[0]) * w, (g[3] - g[1]) * h, 120) * 2
        x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x2, y2 = int(min(cx + half, w)), int(min(cy + half, h))
        im = img.copy()
        cv2.rectangle(im, (int(g[0] * w), int(g[1] * h)), (int(g[2] * w), int(g[3] * h)), (0, 255, 0), 2)
        rec_boxes = [b for r in recs if r["path"] == x["path"] for b in r["det"][DET]]
        for b in rec_boxes:
            if iou(b["xyxy"], g) > 0:
                bb = b["xyxy"]
                cv2.rectangle(im, (int(bb[0] * w), int(bb[1] * h)), (int(bb[2] * w), int(bb[3] * h)), (0, 0, 255), 2)
        t = cv2.resize(im[y1:y2, x1:x2], (256, 256))
        name = os.path.basename(x["path"]).replace("2026-", "")[:28]
        cv2.putText(t, f"{x['best_conf']:.2f} {'T' if x['ref_hit'] else '-'}{'W' if x['wrist'] else '-'}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(t, name, (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(t)
    for s in range(0, len(tiles), 48):
        chunk = tiles[s:s + 48] + [np.zeros((256, 256, 3), np.uint8)] * (-len(tiles[s:s + 48]) % 8)
        sheet = np.vstack([np.hstack(chunk[i:i + 8]) for i in range(0, len(chunk), 8)])
        cv2.imwrite(os.path.join(OUT, f"{c}_{s // 48:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
