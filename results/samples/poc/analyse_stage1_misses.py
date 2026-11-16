"""What a Stage 1 detector finds and misses on the ground-truth set, box by box.

  DEV=0 IMGSZ=384,640 python analyse_stage1_misses.py <REPO> run name=weights [name=weights ...]
      -> results/stage1_kd/misses/<name>_<h>x<w>.json (one row per ground-truth box and per unmatched detection) and a table
  python analyse_stage1_misses.py <REPO> sheet <json> miss|fp <conf> <out.jpg>
      -> picture of the missed ground-truth boxes (or the detections with no ground-truth box) at that confidence
  DEV=0 python analyse_stage1_misses.py <REPO> trainneg <weights> [<weights> ...]
      -> data/stage1_negatives/orig_clean: the original training set without the box-less images in which any of the
         given detectors sees a knife at 0.05 or more (a knife with no label is trained as "no knife")

A ground-truth box counts as found at confidence c when any knife detection with score >= c overlaps it at IoU >= 0.20.
bench_stage1.py also counts a second detection on the same box as a false detection; here it is counted apart ("dup").
"""
import glob, json, os, sys

import cv2
import numpy as np

REPO, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
GT = os.environ.get("GT_DIR") or os.path.join(REPO, "data", "Ground_Truth_Dataset")  # GT_DIR=data/Ground_Truth_Dataset_v2: relabelled set
OUT = os.path.join(REPO, "results", "stage1_kd", "misses")
IMGSZ = tuple(int(x) for x in os.environ.get("IMGSZ", "384,640").split(","))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def run(name, weights):
    from ultralytics import YOLO
    model, rows = YOLO(weights, task="detect"), []
    for cat in ("With_Coat", "Without_Coat", "OOD"):
        for path in sorted(glob.glob(os.path.join(GT, cat, "images", "*"))):
            img = cv2.imread(path)
            h, w = img.shape[:2]
            scale = min(IMGSZ[0] / h, IMGSZ[1] / w)  # letterbox shrink: box size in pixels as the network sees it
            lab = os.path.join(GT, cat, "labels", os.path.splitext(os.path.basename(path))[0] + ".txt")
            gts = [[float(v) for v in l.split()[1:5]] for l in open(lab) if len(l.split()) >= 5] if os.path.exists(lab) else []
            gts = [[x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2] for x, y, bw, bh in gts]
            res = model(img, conf=0.01, imgsz=IMGSZ, half=True, device=int(os.environ.get("DEV", 0)), verbose=False)[0]
            preds = [(b.xyxyn[0].tolist(), float(b.conf[0])) for b in res.boxes if model.names[int(b.cls[0])] != "phone"]
            for g in gts:
                hit = [c for p, c in preds if iou(p, g) >= 0.2]
                rows.append(dict(kind="gt", img=path, cat=cat, box=g, conf=max(hit, default=0.0), n=len(hit),
                                 px=[(g[2] - g[0]) * w * scale, (g[3] - g[1]) * h * scale]))
            for p, c in preds:
                if not any(iou(p, g) >= 0.2 for g in gts):
                    rows.append(dict(kind="fp", img=path, cat=cat, box=p, conf=c))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, f"{name}_{IMGSZ[0]}x{IMGSZ[1]}.json"), "w"))
    table(name, rows)


def table(name, rows):
    gt = [r for r in rows if r["kind"] == "gt"]
    fp = [r for r in rows if r["kind"] == "fp"]
    long_side = np.array([max(r["px"]) for r in gt])
    conf = np.array([r["conf"] for r in gt])
    print(f"\n{name} at {IMGSZ[0]}x{IMGSZ[1]}: {len(gt)} ground-truth boxes, longer side median {np.median(long_side):.0f} px "
          f"(10th-90th percentile {np.percentile(long_side, 10):.0f}-{np.percentile(long_side, 90):.0f})")
    print("| Group | Boxes | Found at 0.45 | at 0.25 | at 0.15 | at 0.01 (ceiling) |\n|---|---|---|---|---|---|")
    groups = [("all", np.ones(len(gt), bool))]
    groups += [(c, np.array([r["cat"] == c for r in gt])) for c in ("With_Coat", "Without_Coat")]
    groups += [(f"longer side {a}-{b} px", (long_side >= a) & (long_side < b)) for a, b in ((0, 24), (24, 40), (40, 64), (64, 9999))]
    for label, m in groups:
        if m.sum():
            print(f"| {label} | {m.sum()} | " + " | ".join(f"{100 * (conf[m] >= t).mean():.1f} %" for t in (0.45, 0.25, 0.15, 0.01)) + " |")
    print("| Detections with no ground-truth box | | " + " | ".join(
        f"{sum(r['conf'] >= t for r in fp)} ({sum(r['conf'] >= t and r['cat'] == 'OOD' for r in fp)} OOD)" for t in (0.45, 0.25, 0.15, 0.01)) + " |")


def sheet(js, what, conf, out, cell=192, cols=10):
    rows = json.load(open(js))
    pick = [r for r in rows if (r["kind"] == "gt" and r["conf"] < conf) if what == "miss"] if what == "miss" else \
        [r for r in rows if r["kind"] == "fp" and r["conf"] >= conf and r["cat"] != "OOD"]
    tiles = []
    for r in pick[:cols * 8]:
        img = cv2.imread(r["img"])
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (np.array(r["box"]) * [w, h, w, h]).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        s = int(max(x2 - x1, y2 - y1) * 2) + 40  # context around the box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        crop = img[max(0, cy - s):cy + s, max(0, cx - s):cx + s]
        tile = cv2.resize(crop, (cell, cell))
        cv2.putText(tile, f"{r['conf']:.2f} {r['cat'][:7]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(tile)
    tiles += [np.zeros((cell, cell, 3), np.uint8)] * (-len(tiles) % cols)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
    print(len(pick), "boxes,", min(len(pick), cols * 8), "drawn ->", out)


def trainneg(weights, src="D:/Backup/Weapon_Detection_Split", thr=0.05):
    from ultralytics import YOLO
    ims = sorted(f.replace("\\", "/") for f in glob.glob(src + "/images/train/*"))
    lab = lambda f: f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    score = {f: 0.0 for f in ims if not (os.path.exists(lab(f)) and open(lab(f)).read().strip())}
    for w in weights:
        m = YOLO(w, task="detect")
        for f in score:
            r = m(cv2.imread(f), conf=0.02, imgsz=IMGSZ, half=True, device=int(os.environ.get("DEV", 0)), verbose=False)[0]
            score[f] = max([score[f]] + [float(b.conf[0]) for b in r.boxes if m.names[int(b.cls[0])] != "phone"])
    out = os.path.join(REPO, "data", "stage1_negatives", "orig_clean")
    os.makedirs(out, exist_ok=True)
    keep = [f for f in ims if score.get(f, 0.0) < thr]
    open(os.path.join(out, "train.txt"), "w").write("".join(f + "\n" for f in keep))
    yaml = ["path: " + src, "train: " + out.replace("\\", "/") + "/train.txt", "val: images/val", "names:", "  0: knife"]
    open(os.path.join(out, "data.yaml"), "w").write("".join(l + "\n" for l in yaml))
    json.dump(score, open(os.path.join(out, "boxless_scores.json"), "w"))
    print(len(ims), "images,", len(score), "without a box,", len(ims) - len(keep), "dropped,", len(keep), "kept ->", out)


if __name__ == "__main__":
    if MODE == "trainneg":
        trainneg(sys.argv[3:])
    elif MODE == "run":
        for spec in sys.argv[3:]:
            run(*spec.split("=", 1))
    else:
        sheet(sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6])
