"""Does more resolution find the lab knives Stage 1 misses? Second pass of the same detector on crops around each wrist
the pose model found (no knife label used to place a crop), boxes merged with the full-frame pass, scored on the lab test
labels as compare_stage1_cached.py. ZOOM=2 crops 1280x720 of the 2560x1440 frame (2x the full-frame resolution), ZOOM=4
crops 640x360 (native pixels).

  DET=kdgtNL WEIGHTS=weights/stage1_repro846_kdgt_p250_newteacher_last.pt DEV=0 \
    .venv/Scripts/python.exe results/samples/poc/zoom_stage1_wrists.py
"""
import json, os, sys

import cv2

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_stage1_rules as S
from ultralytics import YOLO

DET = os.environ.get("DET", "kdgtNL")
WEIGHTS = os.environ.get("WEIGHTS", "weights/stage1_repro846_kdgt_p250_newteacher_last.pt")
DEV = os.environ.get("DEV", "0")
OUT = os.path.join("results", "stage1_kd", "misses", DET)
os.makedirs(OUT, exist_ok=True)
model = YOLO(WEIGHTS, task="detect")


def merge(a, b, thr=0.5):
    """Union of two box lists (xyxy normalised + conf); a box overlapping a kept one at IoU >= thr is dropped."""
    out = []
    for x in sorted(a + b, key=lambda v: -v[4]):
        if all(S.calculate_iou(x[:4], y[:4]) < thr for y in out):
            out.append(x)
    return out


assert len(merge([[0, 0, 0.1, 0.1, 0.9]], [[0, 0, 0.1, 0.11, 0.5], [0.5, 0.5, 0.6, 0.6, 0.3]])) == 2

recs = S.load_cache("lab")
zoomed = {z: [] for z in (2, 4)}
for i, r in enumerate(recs):
    img = cv2.imread(r["path"])
    h, w = img.shape[:2]
    wrists = [wr["xy"] for p in r["pose"] for wr in p["wrists"] if wr["conf"] >= 0.3]
    for z in zoomed:
        cw, ch = w // z, h // z
        boxes = []
        for (wx, wy) in wrists:
            x1 = int(min(max(wx - cw / 2, 0), w - cw))
            y1 = int(min(max(wy - ch / 2, 0), h - ch))
            res = model(img[y1:y1 + ch, x1:x1 + cw], conf=0.01, imgsz=S.IMGSZ, half=True, device=DEV, verbose=False)[0]
            for b, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                if res.names[int(k)] != "knife":
                    continue
                boxes.append([(b[0] + x1) / w, (b[1] + y1) / h, (b[2] + x1) / w, (b[3] + y1) / h, c])
        zoomed[z].append(boxes)
    if i % 500 == 0:
        print(i, "/", len(recs), flush=True)

full = [[b["xyxy"] + [b["conf"]] for b in r["det"][DET]] for r in recs]
variants = {"full frame": full}
for z in zoomed:
    variants[f"full + wrist crops {z}x"] = [merge(f, zb) for f, zb in zip(full, zoomed[z])]
    variants[f"wrist crops {z}x only"] = [merge([], zb) for zb in zoomed[z]]


def score(boxes_per_rec, t, gate):
    tp = fp = fn = 0
    for r, bs in zip(recs, boxes_per_rec):
        keep = [tuple(b) for b in bs if b[4] >= t and (not gate or S.gate_passes(r, b[:4], DET, {"kind": "wrist@k", "k": 0.5}))]
        a, b, c = S.evaluate_matches(keep, r["gt"])
        tp, fp, fn = tp + a, fp + b, fn + c
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), fp


TS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
lines = [f"# {DET}: wrist-crop second pass, lab test ({len(recs)} frames, {sum(len(r['gt']) for r in recs)} boxes)", "",
         "| variant | 0.40: P / R | false boxes | 0.40 + gate: P / R | best R at P >= 0.90 (conf) | best R at P >= 0.90, gate (conf) |",
         "|---|---|---|---|---|---|"]
for name, bs in variants.items():
    p, rr, fp = score(bs, 0.40, False)
    pg, rg, _ = score(bs, 0.40, True)
    best = []
    for g in (False, True):
        ok = [(score(bs, t, g), t) for t in TS]
        ok = [(s, t) for s, t in ok if s[0] >= 0.90]
        best.append(f"{max(ok, key=lambda v: v[0][1])[0][1] * 100:.1f} % ({max(ok, key=lambda v: v[0][1])[1]:.2f})" if ok else "not reached")
    lines.append(f"| {name} | {p * 100:.1f} / {rr * 100:.1f} | {fp} | {pg * 100:.1f} / {rg * 100:.1f} | {best[0]} | {best[1]} |")

# which of the full-frame misses at 0.40 does each zoom find at 0.40?
miss = json.load(open(os.path.join(OUT, "misses.json")))
idx = {r["path"]: i for i, r in enumerate(recs)}
lines += ["", "| full-frame miss cause | boxes | found by wrist crops 2x at 0.40 | 4x at 0.40 |", "|---|---|---|---|"]
for c in ("unseen", "low_score", "bad_box"):
    xs = [x for x in miss if x["cause"] == c]
    found = {z: sum(any(S.calculate_iou(b[:4], x["g"]) >= S.IOU_THRESH and b[4] >= 0.40 for b in zoomed[z][idx[x["path"]]]) for x in xs)
             for z in zoomed}
    lines.append(f"| {c} | {len(xs)} | {found[2]} | {found[4]} |")
open(os.path.join(OUT, "zoom_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
