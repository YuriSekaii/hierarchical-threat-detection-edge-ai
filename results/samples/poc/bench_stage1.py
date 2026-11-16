"""
results/samples/poc/bench_stage1.py

Benchmark Stage 1 weapon-detection "systems": different models / combinations get
the same three measurements (ground-truth P/R/F1 + OOD FP, lab phone frames,
lab knife frames/windows).

A system is a function(img_bgr) -> list of (x1, y1, x2, y2, conf) knife boxes
with *normalised* coordinates.  Every detector runs with imgsz=(384, 640),
half=True, device=int(os.environ.get("DEV", 0)).

Usage:
    DEV=1 .venv/Scripts/python.exe results/samples/poc/bench_stage1.py "<spec>" [<spec> ...]

Specs:
    det,<weights>,<conf>
        One YOLO detector; keep boxes with conf >= conf.  If the model has a
        class named "phone", drop phone-class boxes and also drop any knife box
        whose IoU with a phone box (phone conf >= 0.25) is >= 0.3.
    confirm,<weightsA>,<confA>,<weightsB>,<confB>
        Keep a box of detector A (conf >= confA) only when detector B has a box
        with conf >= confB and IoU >= 0.2 with it.
    coco,<weightsA>,<confA>,<cocoWeights>,<phoneConf>
        Keep a box of A unless the COCO detector (class "cell phone") has a cell
        phone box with conf >= phoneConf and IoU >= 0.2 with it.
    crop,<weightsA>,<confA>,<clsWeights>,<pKnife>
        Keep a box of A only when an ultralytics classification model (classes
        include "knife") gives probability >= pKnife for "knife" on the box crop
        enlarged by 50 % on each side (clipped to the image).
        Classifier weights do not exist yet; this path is assert-guarded.

Models are cached by path so a weights file is loaded once.

Measurements per system:
  * Ground truth: reuses the image list, label reading and TP/FP/FN/TN counting
    logic of scripts/validate_stage1_ground_truth.py (IoU 0.20).  Reports
    overall precision, recall, F1 and the OOD false-positive count.
  * Lab takes in data/lab_recordings/2026-11-05/train_attack
    (<num>_*.avi, frames cropped with scripts.record_takes.crop_4_3, every 3rd
    frame): phone takes 008, 006, 007, 009; knife takes 002, 005, 010, 014, 018,
    022.  Per take: share of sampled frames with >= 1 kept box, and share of
    10-sample sliding windows with >= 3 such frames.  Each video is decoded
    once per run and all systems receive the same frames.

Output: one markdown table (one row per system) printed with flushed stdout, and
the rows appended as JSON lines to results/stage1_phoneneg/bench.jsonl (spec
plus every raw number, including per-take values).
"""

import os
import sys
import glob
import json
import importlib.util
from collections import defaultdict

import cv2
import numpy as np

IMGSZ = tuple(int(x) for x in os.environ.get("IMGSZ", "384,640").split(","))  # test images are 16:10: 416,640 keeps the x0.25 shrink
IOU_THRESH = 0.20
DEVICE = int(os.environ.get("DEV", 0))

HERE = os.path.abspath(os.path.dirname(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # results/samples/poc -> repo root
SCRIPTS = os.path.join(WORKTREE, "scripts")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_data(*relparts):
    """Walk up from HERE looking for data/<relparts> (the data tree lives in the
    parent repo checkout, not necessarily inside this worktree)."""
    cur = HERE
    for _ in range(12):
        cand = os.path.join(cur, "data", *relparts)
        if os.path.isdir(cand):
            return cand
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return os.path.join(HERE, *([".."] * (2 + len(relparts))), "data", *relparts)


GT_DIR = os.environ.get("GT_DIR") or _find_data("Ground_Truth_Dataset")  # GT_DIR=<abs path of data/Ground_Truth_Dataset_v2>: relabelled set
TAKES_DIR = _find_data("lab_recordings", "2026-11-05", "train_attack")

# Reuse helpers from scripts/validate_stage1_ground_truth.py (importable without
# side effects: main() is __main__-guarded).
_vgt = _load_module("validate_stage1_ground_truth",
                    os.path.join(SCRIPTS, "validate_stage1_ground_truth.py"))
calculate_iou = _vgt.calculate_iou
load_ground_truth = _vgt.load_ground_truth
xywhn2xyxy = _vgt.xywhn2xyxy
calculate_metrics = _vgt.calculate_metrics
get_category_from_path = _vgt.get_category_from_path

_rt = _load_module("record_takes", os.path.join(SCRIPTS, "record_takes.py"))
crop_4_3 = _rt.crop_4_3

from ultralytics import YOLO  # noqa: E402  (needs the venv; system python lacks it)

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache = {}


def load_model(weights):
    if weights not in _model_cache:
        _model_cache[weights] = YOLO(weights, task="detect")
    return _model_cache[weights]


def class_name(model, cls):
    names = model.names
    if isinstance(names, dict):
        return names.get(int(cls), str(int(cls)))
    return names[int(cls)]


def _has_class(model, name):
    n = len(model.names)
    return any(class_name(model, c) == name for c in range(n))


def predict_boxes(model, img):
    res = model(img, conf=0.05, imgsz=IMGSZ, half=True, device=DEVICE, verbose=False)[0]
    out = []
    for b in res.boxes:
        x, y, w, h = b.xywhn[0].tolist()
        out.append({
            "xyxy": xywhn2xyxy(x, y, w, h),
            "conf": float(b.conf[0]),
            "cls": int(b.cls[0]),
            "name": class_name(model, int(b.cls[0])),
        })
    return out


def to_tuples(boxes):
    return [(b["xyxy"][0], b["xyxy"][1], b["xyxy"][2], b["xyxy"][3], b["conf"])
            for b in boxes]


# ---------------------------------------------------------------------------
# System factories
# ---------------------------------------------------------------------------
def make_det(weights, conf):
    model = load_model(weights)
    has_phone = _has_class(model, "phone")

    def system(img):
        boxes = predict_boxes(model, img)
        if not has_phone:
            kept = [b for b in boxes if b["conf"] >= conf]
        else:
            phones = [b for b in boxes if b["name"] == "phone" and b["conf"] >= 0.25]
            kept = [b for b in boxes if b["conf"] >= conf and b["name"] != "phone"]
            kept = [b for b in kept
                    if not any(calculate_iou(b["xyxy"], p["xyxy"]) >= 0.3 for p in phones)]
        return to_tuples(kept)

    return system


def make_confirm(w_a, c_a, w_b, c_b):
    m_a = load_model(w_a)
    m_b = load_model(w_b)

    def system(img):
        a = [b for b in predict_boxes(m_a, img) if b["conf"] >= c_a]
        b = [bb for bb in predict_boxes(m_b, img) if bb["conf"] >= c_b]
        kept = [a0 for a0 in a
                if any(calculate_iou(a0["xyxy"], bb["xyxy"]) >= 0.2 for bb in b)]
        return to_tuples(kept)

    return system


def make_coco(w_a, c_a, w_coco, phone_conf):
    m_a = load_model(w_a)
    m_c = load_model(w_coco)

    def system(img):
        a = [b for b in predict_boxes(m_a, img) if b["conf"] >= c_a]
        c = predict_boxes(m_c, img)
        phones = [bb for bb in c if bb["name"] == "cell phone" and bb["conf"] >= phone_conf]
        kept = [a0 for a0 in a
                if not any(calculate_iou(a0["xyxy"], p["xyxy"]) >= 0.2 for p in phones)]
        return to_tuples(kept)

    return system


def make_crop(w_a, c_a, w_cls, p_knife):
    # Classifier weights do not exist yet; keep the path simple and assert-guarded.
    assert os.path.isfile(w_cls), f"classifier weights not found: {w_cls}"
    m_a = load_model(w_a)
    m_cls = YOLO(w_cls, task="classify")
    knife_idx = None
    names = m_cls.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for c, name in items:
        if name == "knife":
            knife_idx = c
            break
    assert knife_idx is not None, "classifier has no 'knife' class"

    def system(img):
        a = [b for b in predict_boxes(m_a, img) if b["conf"] >= c_a]
        kept = []
        h, w = img.shape[:2]
        for b in a:
            x1, y1, x2, y2 = b["xyxy"][:4]
            bw = x2 - x1
            bh = y2 - y1
            ex1 = max(0.0, x1 - 0.5 * bw)
            ey1 = max(0.0, y1 - 0.5 * bh)
            ex2 = min(1.0, x2 + 0.5 * bw)
            ey2 = min(1.0, y2 + 0.5 * bh)
            cx1 = int(ex1 * w)
            cy1 = int(ey1 * h)
            cx2 = int(ex2 * w)
            cy2 = int(ey2 * h)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            res = m_cls(crop, verbose=False)[0]
            probs = res.probs
            if probs is None:
                continue
            p = float(probs.data[int(knife_idx)])
            if p >= p_knife:
                kept.append((x1, y1, x2, y2, b["conf"]))
        return kept

    return system


def make_roi(w_a, conf, w_pose, mode):
    """roi,<weights>,<conf>,<poseWeights>,only|plus : wrist-ROI zoom (WeDePE idea, test-time only, detector not retrained on
    crops). Pose gives elbow and wrist; a square of side 1.8 torso lengths centred a little beyond the wrist is cut from the
    frame and the detector runs on it enlarged. 'plus' also keeps the full-frame boxes.
    shortcut: one detector pass per hand instead of one packed grid, so this measures accuracy, not speed."""
    assert mode in ("only", "plus"), mode
    m_a, m_p = load_model(w_a), YOLO(w_pose)

    def system(img):
        h, w = img.shape[:2]
        out = [b["xyxy"] + (b["conf"],) for b in predict_boxes(m_a, img) if b["conf"] >= conf and b["name"] != "phone"] if mode == "plus" else []
        kp = m_p(img, imgsz=IMGSZ, half=True, device=DEVICE, verbose=False)[0].keypoints
        if kp is not None and kp.conf is not None:
            for xy, c in zip(kp.xy.cpu().numpy(), kp.conf.cpu().numpy()):
                if min(c[5], c[6], c[11], c[12]) < 0.3:
                    continue
                torso = np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2)
                for e, wr in ((7, 9), (8, 10)):
                    if c[wr] < 0.3 or torso < 8:
                        continue
                    ctr = xy[wr] + (0.4 * (xy[wr] - xy[e]) if c[e] >= 0.3 else 0)
                    x1, y1 = int(max(0, ctr[0] - 0.9 * torso)), int(max(0, ctr[1] - 0.9 * torso))
                    x2, y2 = int(min(w, ctr[0] + 0.9 * torso)), int(min(h, ctr[1] + 0.9 * torso))
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    for b in predict_boxes(m_a, img[y1:y2, x1:x2]):
                        if b["conf"] >= conf and b["name"] != "phone":
                            bx1, by1, bx2, by2 = b["xyxy"]
                            out.append(((x1 + bx1 * (x2 - x1)) / w, (y1 + by1 * (y2 - y1)) / h,
                                        (x1 + bx2 * (x2 - x1)) / w, (y1 + by2 * (y2 - y1)) / h, b["conf"]))
        keep = []  # greedy NMS: the two hand crops and the full frame can report the same knife
        for b in sorted(out, key=lambda b: -b[4]):
            if all(calculate_iou(b, k) < 0.3 for k in keep):
                keep.append(b)
        return keep

    return system


def make_roicrop(w_a, conf, w_pose, mode, w_cls, p_knife):
    """roicrop,<weights>,<conf>,<poseWeights>,only|plus,<clsWeights>,<pKnife> : wrist-ROI zoom, then the crop classifier."""
    base = make_roi(w_a, conf, w_pose, mode)
    m_cls = YOLO(w_cls, task="classify")
    knife_idx = [c for c, n in m_cls.names.items() if n == "knife"][0]

    def system(img):
        h, w = img.shape[:2]
        kept = []
        for x1, y1, x2, y2, c in base(img):
            bw, bh = x2 - x1, y2 - y1
            crop = img[int(max(0.0, y1 - 0.5 * bh) * h):int(min(1.0, y2 + 0.5 * bh) * h),
                       int(max(0.0, x1 - 0.5 * bw) * w):int(min(1.0, x2 + 0.5 * bw) * w)]
            if crop.size and float(m_cls(crop, verbose=False)[0].probs.data[knife_idx]) >= p_knife:
                kept.append((x1, y1, x2, y2, c))
        return kept

    return system


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------
def parse_spec(spec):
    p = spec.split(",")  # comma, not colon: Windows paths contain a colon
    kind = p[0]
    if kind == "det":
        return ("det", p[1], float(p[2]))
    if kind == "confirm":
        return ("confirm", p[1], float(p[2]), p[3], float(p[4]))
    if kind == "coco":
        return ("coco", p[1], float(p[2]), p[3], float(p[4]))
    if kind == "crop":
        return ("crop", p[1], float(p[2]), p[3], float(p[4]))
    if kind == "roicrop":
        return ("roicrop", p[1], float(p[2]), p[3], p[4], p[5], float(p[6]))
    if kind == "roi":
        return ("roi", p[1], float(p[2]), p[3], p[4])
    raise ValueError(f"unknown system spec: {spec}")


def build_system(spec):
    kind = parse_spec(spec)
    if kind[0] == "det":
        return spec, make_det(kind[1], kind[2])
    if kind[0] == "confirm":
        return spec, make_confirm(kind[1], kind[2], kind[3], kind[4])
    if kind[0] == "coco":
        return spec, make_coco(kind[1], kind[2], kind[3], kind[4])
    if kind[0] == "crop":
        return spec, make_crop(kind[1], kind[2], kind[3], kind[4])
    if kind[0] == "roicrop":
        return spec, make_roicrop(*kind[1:])
    if kind[0] == "roi":
        return spec, make_roi(kind[1], kind[2], kind[3], kind[4])
    raise ValueError(kind[0])


# ---------------------------------------------------------------------------
# Ground truth measurement (faithful copy of validate_stage1_ground_truth logic)
# ---------------------------------------------------------------------------
def evaluate_gt(system, image_files):
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total": 0})
    overall = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total": 0}
    for img_path in image_files:
        category = get_category_from_path(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        grandparent_dir = os.path.dirname(os.path.dirname(img_path))
        gt_txt_path = os.path.join(grandparent_dir, "labels", filename + ".txt")
        gt_boxes = load_ground_truth(gt_txt_path)
        gt_xyxy = [xywhn2xyxy(*g[1:]) for g in gt_boxes]

        img = cv2.imread(img_path)
        pred_boxes = system(img)

        matched_gt = set()
        matched_pred = set()
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_g = -1
            for g_idx, g_box in enumerate(gt_xyxy):
                if g_idx in matched_gt:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g = g_idx
            if best_iou >= IOU_THRESH:
                stats[category]["TP"] += 1
                overall["TP"] += 1
                matched_gt.add(best_g)
                matched_pred.add(p_idx)
            else:
                stats[category]["FP"] += 1
                overall["FP"] += 1

        fn = len(gt_boxes) - len(matched_gt)
        stats[category]["FN"] += fn
        overall["FN"] += fn
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            stats[category]["TN"] += 1
            overall["TN"] += 1
        stats[category]["total"] += 1
        overall["total"] += 1

    return stats, overall


def gt_measurement(system, image_files):
    stats, overall = evaluate_gt(system, image_files)
    p, r, f1, _spec, _acc = calculate_metrics(overall["TP"], overall["FP"],
                                              overall["FN"], overall["TN"])
    return {
        "precision": p * 100.0,
        "recall": r * 100.0,
        "f1": f1 * 100.0,
        "ood_fp": stats["OOD"]["FP"],
    }


# ---------------------------------------------------------------------------
# Lab takes measurement
# ---------------------------------------------------------------------------
PHONE_TAKES = ["008", "006", "007", "009"]
KNIFE_TAKES = ["002", "005", "010", "014", "018", "022"]


def list_take_videos(takes_dir):
    videos = {}
    for path in sorted(glob.glob(os.path.join(takes_dir, "*_*.avi"))):
        num = os.path.basename(path).split("_", 1)[0]
        videos[num] = path
    return videos


def eval_take(video_path, systems):
    """Decode once; every system sees the same sampled frames (every 3rd,
    crop_4_3 applied). Returns {system_name: [bool per sampled frame]}."""
    cap = cv2.VideoCapture(video_path)
    flags = {name: [] for name, _ in systems}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 3 == 0:
            img = crop_4_3(frame)
            for name, fn in systems:
                flags[name].append(len(fn(img)) > 0)
        idx += 1
    cap.release()
    return flags


def share(frames_flags):
    n = len(frames_flags)
    if n == 0:
        return 0.0
    return sum(1 for f in frames_flags if f) / n


def windows_share(frames_flags, win=10, min_hits=3):
    n = len(frames_flags)
    if n < win:
        return 0.0
    total = n - win + 1
    pos = 0
    for i in range(total):
        if sum(1 for f in frames_flags[i:i + win] if f) >= min_hits:
            pos += 1
    return pos / total


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def lab_measurement(systems, takes_dir):
    videos = list_take_videos(takes_dir)
    takes = {}
    all_takes = PHONE_TAKES + KNIFE_TAKES
    for num in all_takes:
        path = videos.get(num)
        if path is None:
            takes[num] = {"frames": {name: 0.0 for name, _ in systems},
                          "windows": {name: 0.0 for name, _ in systems}}
            continue
        flags = eval_take(path, systems)
        takes[num] = {
            "frames": {name: share(flags[name]) for name, _ in systems},
            "windows": {name: windows_share(flags[name]) for name, _ in systems},
        }
    return takes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: bench_stage1.py <spec> [<spec> ...]", file=sys.stderr)
        return 2

    specs = sys.argv[1:]
    systems = [build_system(s) for s in specs]

    image_files = sorted(glob.glob(os.path.join(GT_DIR, "**", "*.jpg"), recursive=True))
    print(f"Loaded {len(image_files)} Ground Truth images from {GT_DIR}", flush=True)
    print(f"Lab takes dir: {TAKES_DIR}", flush=True)

    # Lab videos are decoded once and shared by all systems.
    lab = lab_measurement(systems, TAKES_DIR)

    rows = []
    for spec, system in systems:
        gt = gt_measurement(system, image_files)
        tk = lab
        p008 = tk["008"]["frames"][spec]
        pmean = mean(tk[t]["frames"][spec] for t in ["006", "007", "009"])
        kmin = min(tk[t]["frames"][spec] for t in KNIFE_TAKES)
        kmean = mean(tk[t]["frames"][spec] for t in KNIFE_TAKES)
        pwmax = max(tk[t]["windows"][spec] for t in PHONE_TAKES)
        kwmin = min(tk[t]["windows"][spec] for t in KNIFE_TAKES)

        row = {
            "spec": spec,
            "gt_precision": round(gt["precision"], 4),
            "gt_recall": round(gt["recall"], 4),
            "gt_f1": round(gt["f1"], 4),
            "gt_ood_fp": gt["ood_fp"],
            "phone_frames_008": round(p008, 4),
            "phone_frames_mean_006_007_009": round(pmean, 4),
            "knife_frames_min": round(kmin, 4),
            "knife_frames_mean": round(kmean, 4),
            "phone_windows_max": round(pwmax, 4),
            "knife_windows_min": round(kwmin, 4),
            "take_frames": {t: round(tk[t]["frames"][spec], 4) for t in PHONE_TAKES + KNIFE_TAKES},
            "take_windows": {t: round(tk[t]["windows"][spec], 4) for t in PHONE_TAKES + KNIFE_TAKES},
        }
        rows.append(row)

    # Markdown table
    header = ("| spec | GT precision | GT recall | GT F1 | OOD FP | "
              "phone frames 008 | phone frames 006/007/009 | "
              "knife frames min | knife frames mean | "
              "phone windows max | knife windows min |")
    sep = "|" + "|".join([" ---" ] * (len(header.split("|")) - 1)) + "|"
    print()
    print(header, flush=True)
    print(sep, flush=True)
    for row in rows:
        line = ("| {spec} | {gt_precision:.2f}% | {gt_recall:.2f}% | {gt_f1:.2f}% | {gt_ood_fp} | "
                "{p008:.1%} | {pmean:.1%} | {kmin:.1%} | {kmean:.1%} | "
                "{pwmax:.1%} | {kwmin:.1%} |").format(
            spec=row["spec"],
            gt_precision=row["gt_precision"],
            gt_recall=row["gt_recall"],
            gt_f1=row["gt_f1"],
            gt_ood_fp=row["gt_ood_fp"],
            p008=row["phone_frames_008"],
            pmean=row["phone_frames_mean_006_007_009"],
            kmin=row["knife_frames_min"],
            kmean=row["knife_frames_mean"],
            pwmax=row["phone_windows_max"],
            kwmin=row["knife_windows_min"],
        )
        print(line, flush=True)
    print(flush=True)

    # JSON lines
    out_dir = os.path.join(WORKTREE, "results", "stage1_phoneneg")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench.jsonl")
    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Appended {len(rows)} row(s) to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
