"""
results/samples/poc/sweep_stage1_rules.py

Cache detector detections once, then sweep two-threshold rules offline to keep
precision while raising recall.

Modes:
    cache <set>        Run four detectors (+ classifier + pose) on every image of
                       a set and cache per-image detections to
                       results/stage1_kd/rules/cache_<set>.json.
                       (needs the venv: ultralytics, GPU via DEV)
    sweep <set>...     Offline sweep of single/hysteresis rules over cached
                       detections. No GPU, no ultralytics import.

Usage:
    DEV=0 .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . cache gtv2
    LIMIT=40 DEV=0 .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . cache gtv2
    .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . sweep gtv2
    .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . sweep gtv2 lab \
        --apply-to lab --min-precision 0.90

Env:
    LIMIT=n            cache mode: only the first n images
    DEV=n              cache mode: cuda device (default 0)
    GT_DIR             override Ground_Truth_Dataset_v2 dir
    LAB_DIR            override lab_stage1_gt dir
"""

import os
import sys
import re
import json
import glob
import argparse
import importlib.util
from collections import defaultdict

import cv2
import numpy as np

IMGSZ = (384, 640)
IOU_THRESH = 0.20
DEVICE = int(os.environ.get("DEV", "0"))

HERE = os.path.abspath(os.path.dirname(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # results/samples/poc -> repo root
SCRIPTS = os.path.join(WORKTREE, "scripts")
RULES_DIR = os.path.join(WORKTREE, "results", "stage1_kd", "rules")

DETECTORS = {
    "shipped": "weights/yolo_weapon_distilled.pt",
    "teacher": "weights/stage1_repro846_teacher.pt",
    "student": "weights/stage1_repro846_labv2_p50.pt",
    "kd": "weights/stage1_repro846_kd.pt",
}
CLASSIFIER = "weights/stage1_cropcls2.pt"
POSE = "weights/yolo26s-pose.pt"

DROP_CLASSES = {"phone", "hand_grip_knife"}

SINGLE_THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
HI_VALUES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1.01]  # 1.01: never kept on its own score, only through the gate
LO_VALUES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45]
TEACHER_C = [0.05, 0.1, 0.15, 0.25, 0.35, 0.45]
CLS_P = [0.3, 0.5, 0.7, 0.9, 0.95]
WRIST_K = [0.25, 0.5, 0.75, 1.0]
TARGET_PRECISIONS = [0.85, 0.88, 0.90, 0.92, 0.95, 0.96]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vgt = _load_module("validate_stage1_ground_truth",
                    os.path.join(SCRIPTS, "validate_stage1_ground_truth.py"))
calculate_iou = _vgt.calculate_iou
load_ground_truth = _vgt.load_ground_truth
xywhn2xyxy = _vgt.xywhn2xyxy
get_category_from_path = _vgt.get_category_from_path
calculate_metrics = _vgt.calculate_metrics


def _find_data(*relparts):
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


GT_DIR = os.environ.get("GT_DIR") or _find_data("Ground_Truth_Dataset_v2")
LAB_DIR = os.environ.get("LAB_DIR") or _find_data("lab_stage1_gt")


# ---------------------------------------------------------------------------
# Matcher (shared by sweep)
# ---------------------------------------------------------------------------

def matcher_self_check():
    gts = [(0.1, 0.1, 0.3, 0.3), (0.6, 0.6, 0.8, 0.8)]
    # 1 TP (matches gt0), 1 FP (matches nothing), gt1 left unmatched -> 1 FN
    preds = [(0.12, 0.12, 0.32, 0.32, 0.9), (0.9, 0.9, 0.99, 0.99, 0.9)]
    tp, fp, fn = evaluate_matches(preds, gts)
    assert (tp, fp, fn) == (1, 1, 1), (tp, fp, fn)


def evaluate_matches(pred_boxes, gt_boxes, iou_thresh=IOU_THRESH):
    """Greedy by confidence. pred_boxes: (x1,y1,x2,y2,conf). Returns (tp,fp,fn)."""
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_boxes[i][4])
    matched_gt = set()
    tp = 0
    fp = 0
    for i in order:
        box = pred_boxes[i][:4]
        best_iou = 0.0
        best_g = -1
        for g, gbox in enumerate(gt_boxes):
            if g in matched_gt:
                continue
            iou = calculate_iou(box, gbox)
            if iou > best_iou:
                best_iou = iou
                best_g = g
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_g)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Detection caching
# ---------------------------------------------------------------------------

def resolve_weights(rel):
    cand = os.path.join(WORKTREE, rel)
    if os.path.isfile(cand):
        return cand
    return rel


class Detector:
    def __init__(self, name, rel):
        self.name = name
        self.path = resolve_weights(rel)
        self.model = None
        self.names = None
        self.has_drop = False

    def load(self):
        if self.model is None:
            from ultralytics import YOLO  # venv only
            self.model = YOLO(self.path, task="detect")
            self.names = self.model.names
            n = len(self.names)
            self.has_drop = any(self._cname(c) in DROP_CLASSES for c in range(n))
        return self.model

    def _cname(self, c):
        if isinstance(self.names, dict):
            return self.names.get(int(c), str(int(c)))
        return self.names[int(c)]

    def predict(self, img):
        self.load()
        res = self.model(img, conf=0.01, imgsz=IMGSZ, half=True, device=DEVICE, verbose=False)[0]
        out = []
        for b in res.boxes:
            x, y, w, h = b.xywhn[0].tolist()
            cls = int(b.cls[0])
            name = self._cname(cls)
            if self.has_drop and name in DROP_CLASSES:
                continue
            out.append({
                "xyxy": list(xywhn2xyxy(x, y, w, h)),
                "conf": float(b.conf[0]),
                "cls": cls,
                "name": name,
            })
        out.sort(key=lambda d: -d["conf"])
        return out


class Classifier:
    def __init__(self, rel):
        self.path = resolve_weights(rel)
        self.model = None
        self.knife_idx = None

    def load(self):
        if self.model is None:
            from ultralytics import YOLO  # venv only
            self.model = YOLO(self.path, task="classify")
            names = self.model.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            for c, nm in items:
                if nm == "knife":
                    self.knife_idx = int(c)
                    break
        return self.model

    def prob(self, img, box):
        self.load()
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        ex1 = max(0.0, x1 - 0.5 * bw)
        ey1 = max(0.0, y1 - 0.5 * bh)
        ex2 = min(1.0, x2 + 0.5 * bw)
        ey2 = min(1.0, y2 + 0.5 * bh)
        h, w = img.shape[:2]
        crop = img[int(ey1 * h):int(ey2 * h), int(ex1 * w):int(ex2 * w)]
        if crop.size == 0:
            return 0.0
        res = self.model(crop, verbose=False)[0]
        probs = res.probs
        if probs is None or self.knife_idx is None:
            return 0.0
        return float(probs.data[self.knife_idx])


class Pose:
    def __init__(self, rel):
        self.path = resolve_weights(rel)
        self.model = None

    def load(self):
        if self.model is None:
            from ultralytics import YOLO  # venv only
            self.model = YOLO(self.path)
        return self.model

    def predict(self, img):
        self.load()
        res = self.model(img, imgsz=IMGSZ, conf=0.25, half=True, device=DEVICE, verbose=False)[0]
        persons = []
        kp = res.keypoints
        if kp is None or kp.xy is None or kp.conf is None:
            return persons
        h, w = img.shape[:2]
        for xy, conf in zip(kp.xy.cpu().numpy(), kp.conf.cpu().numpy()):
            sh_c = min(conf[5], conf[6])
            hip_c = min(conf[11], conf[12])
            if sh_c >= 0.3 and hip_c >= 0.3:
                sh = (xy[5] + xy[6]) / 2
                hip = (xy[11] + xy[12]) / 2
                torso = float(np.linalg.norm(sh - hip))
            else:
                torso = None
            wrists = []
            for idx in (9, 10):
                wrists.append({
                    "xy": [float(xy[idx][0]), float(xy[idx][1])],
                    "conf": float(conf[idx]),
                })
            persons.append({"torso": torso, "wrists": wrists, "h": h, "w": w})
        return persons


def list_gtv2():
    files = sorted(glob.glob(os.path.join(GT_DIR, "**", "*.jpg"), recursive=True))
    files += sorted(glob.glob(os.path.join(GT_DIR, "**", "*.jpeg"), recursive=True))
    files += sorted(glob.glob(os.path.join(GT_DIR, "**", "*.png"), recursive=True))
    return sorted(set(files))


def list_lab():
    """Every sampled test-split lab frame, so the cache survives label fixes; load_cache keeps the labelled ones."""
    items = json.load(open(os.path.join(WORKTREE, "results", "stage1_kd", "lab_label", "lab_items.json")))
    return sorted({os.path.join(WORKTREE, it["img"]) for it in items if it["split"] == "test"})


def gt_boxes_for(img_path, img_dir):
    filename = os.path.splitext(os.path.basename(img_path))[0]
    grandparent = os.path.dirname(os.path.dirname(img_path))
    gt_txt = os.path.join(grandparent, "labels", filename + ".txt")
    boxes = []
    for g in load_ground_truth(gt_txt):
        boxes.append(list(xywhn2xyxy(*g[1:])))
    return boxes


def take_category(stem):
    return re.sub(r"_f\d+$", "", stem)


def cache_set(cache_set_name, limit=None):
    os.makedirs(RULES_DIR, exist_ok=True)
    if cache_set_name == "gtv2":
        image_files = list_gtv2()
        img_dir = GT_DIR
    elif cache_set_name == "lab":
        image_files = list_lab()
        img_dir = LAB_DIR
    else:
        raise ValueError("unknown set: %s (use gtv2 or lab)" % cache_set_name)

    if limit is not None:
        image_files = image_files[:limit]

    print("set=%s images=%d" % (cache_set_name, len(image_files)), flush=True)

    dets = {name: Detector(name, rel) for name, rel in DETECTORS.items()}
    cls_model = None
    pose = Pose(POSE)

    records = []
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            print("skip unreadable: %s" % img_path, flush=True)
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        if cache_set_name == "gtv2":
            category = get_category_from_path(img_path)
        else:
            category = take_category(stem)

        gt = gt_boxes_for(img_path, img_dir)

        det_out = {}
        for name, det in dets.items():
            boxes = det.predict(img)
            det_out[name] = boxes

        if cls_model is None:
            cls_model = Classifier(CLASSIFIER)
            cls_model.load()
        cls_prob = {"student": [None] * len(det_out["student"]),
                    "shipped": [None] * len(det_out["shipped"])}
        for dname in ("student", "shipped"):
            top = det_out[dname][:10]
            for j, b in enumerate(top):
                # find index in full list
                idx = det_out[dname].index(b)
                cls_prob[dname][idx] = cls_model.prob(img, b["xyxy"])

        pose_out = pose.predict(img)

        rec = {
            "path": img_path,
            "set": cache_set_name,
            "category": category,
            "w": w,
            "h": h,
            "gt": gt,
            "det": {name: [{"xyxy": b["xyxy"], "conf": b["conf"], "cls": b["cls"], "name": b["name"]}
                          for b in det_out[name]] for name in det_out},
            "cls_prob": cls_prob,
            "pose": pose_out,
        }
        records.append(rec)
        if (i + 1) % 25 == 0:
            print("  %d/%d" % (i + 1, len(image_files)), flush=True)

    out_path = os.path.join(RULES_DIR, "cache_%s.json" % cache_set_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print("Wrote %d records to %s" % (len(records), out_path), flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Rule evaluation over caches (no ultralytics)
# ---------------------------------------------------------------------------

def load_cache(cache_set_name):
    path = os.path.join(RULES_DIR, "cache_%s.json" % cache_set_name)
    with open(path, "r") as f:
        records = json.load(f)
    if cache_set_name == "lab":  # labels are read now, not at cache time: test.txt lists the frames with a sure label
        keep = {os.path.basename(l.strip()) for l in open(os.path.join(LAB_DIR, "test.txt")) if l.strip()}
        records = [r for r in records if os.path.basename(r["path"]) in keep]
        for r in records:
            r["gt"] = gt_boxes_for(r["path"], LAB_DIR)
    if os.environ.get("DROP_STATIC"):  # DROP_STATIC=<json [[x1,y1,x2,y2] px]>: stand-in for the pipeline's static filter
        static = json.load(open(os.environ["DROP_STATIC"]))
        for r in records:
            for d, boxes in r["det"].items():
                keep = [all(calculate_iou([b["xyxy"][0] * r["w"], b["xyxy"][1] * r["h"], b["xyxy"][2] * r["w"], b["xyxy"][3] * r["h"]], sb) < 0.4
                            for sb in static) for b in boxes]
                r["det"][d] = [b for b, k in zip(boxes, keep) if k]
                if d in r["cls_prob"]:  # classifier scores are stored by box position
                    r["cls_prob"][d] = [c for c, k in zip(r["cls_prob"][d], keep) if k]
    return records


def box_index(det_boxes, xyxy):
    for j, b in enumerate(det_boxes):
        if b["xyxy"] == xyxy:
            return j
    return -1


def filter_boxes(rec, det_name, rule):
    """Return kept boxes (list of xyxy) for a rule on one image record."""
    det_boxes = rec["det"][det_name]
    kept = []
    for b in det_boxes:
        conf = b["conf"]
        if rule["kind"] == "single":
            if conf >= rule["t"]:
                kept.append(b["xyxy"] + [conf])
        elif rule["kind"] == "tier3":  # >= hi kept; [mid, hi) needs a wrist; [lo, mid) needs a closer wrist and the classifier
            if (conf >= rule["hi"]
                    or (conf >= rule["mid"] and gate_passes(rec, b["xyxy"], det_name, {"kind": "wrist@k", "k": rule["k1"]}))
                    or (conf >= rule["lo"] and gate_passes(rec, b["xyxy"], det_name, {"kind": "wrist@k+cls@p", "k": rule["k2"], "p": rule["p"]}))):
                kept.append(b["xyxy"] + [conf])
        elif rule["kind"] == "fuse":  # >= hi kept; else sqrt(conf * classifier) >= t, optionally near a wrist
            if conf >= rule["hi"]:
                kept.append(b["xyxy"] + [conf])
            elif conf >= rule["lo"] and (rule["k"] is None or gate_passes(rec, b["xyxy"], det_name, {"kind": "wrist@k", "k": rule["k"]})):
                probs = rec["cls_prob"][det_name]
                idx = box_index(det_boxes, b["xyxy"])
                pk = probs[idx] if 0 <= idx < len(probs) and probs[idx] is not None else 0.0
                if (conf * pk) ** 0.5 >= rule["t"]:
                    kept.append(b["xyxy"] + [conf])
        else:  # hyst
            keep = False
            if conf >= rule["hi"]:
                keep = True
            elif conf >= rule["lo"]:
                keep = gate_passes(rec, b["xyxy"], det_name, rule["gate"])
            if keep:
                kept.append(b["xyxy"] + [conf])
    return kept


def gate_passes(rec, box, det_name, gate):
    gkind = gate["kind"]
    if gkind == "teacher@c":
        c = gate["c"]
        for tb in rec["det"]["teacher"]:
            if tb["conf"] >= c and calculate_iou(box, tb["xyxy"]) >= IOU_THRESH:
                return True
        return False
    if gkind == "cls@p":
        p = gate["p"]
        if det_name not in ("student", "shipped"):
            return False
        probs = rec["cls_prob"][det_name]
        idx = box_index(rec["det"][det_name], box)
        prob = probs[idx] if 0 <= idx < len(probs) and probs[idx] is not None else 0.0
        return prob >= p
    if gkind == "wrist@k":
        k = gate["k"]
        h, w = rec["h"], rec["w"]
        bx = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]  # box is normalised, wrists and torso are pixels
        for person in rec["pose"]:
            for wr in person["wrists"]:
                if wr["conf"] < 0.3:
                    continue
                length = person["torso"] if person["torso"] is not None else 0.15 * h
                # wrist to box edge (0 inside), as wrist_to_box_distance in src/weapon_gate.py
                dx = max(bx[0] - wr["xy"][0], 0, wr["xy"][0] - bx[2])
                dy = max(bx[1] - wr["xy"][1], 0, wr["xy"][1] - bx[3])
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= k * length:
                    return True
        return False
    if gkind == "wrist@k+cls@p":
        if not gate_passes(rec, box, det_name, {"kind": "wrist@k", "k": gate["k"]}):
            return False
        return gate_passes(rec, box, det_name, {"kind": "cls@p", "p": gate["p"]})
    raise ValueError("unknown gate: %s" % gkind)


def evaluate_rule(rec, det_name, rule):
    pred_boxes = filter_boxes(rec, det_name, rule)
    gt_boxes = rec["gt"]
    preds = [tuple(b) for b in pred_boxes]
    tp, fp, fn = evaluate_matches(preds, gt_boxes)
    return tp, fp, fn


def build_rules(det_name):
    rules = []
    for t in SINGLE_THRESHOLDS:
        rules.append({"kind": "single", "det": det_name, "t": t,
                      "desc": "single %s %.2f" % (det_name, t)})
    cls_dets = ("student", "shipped")
    for hi in HI_VALUES:
        for lo in LO_VALUES:
            if lo >= hi:
                continue
            # teacher gate (all detectors)
            for c in TEACHER_C:
                rules.append({"kind": "hyst", "det": det_name, "hi": hi, "lo": lo,
                              "gate": {"kind": "teacher@c", "c": c},
                              "desc": "hyst %s %.2f %.2f teacher@%.2f" % (det_name, hi, lo, c)})
            # wrist gate (all detectors)
            for k in WRIST_K:
                rules.append({"kind": "hyst", "det": det_name, "hi": hi, "lo": lo,
                              "gate": {"kind": "wrist@k", "k": k},
                              "desc": "hyst %s %.2f %.2f wrist@%.2f" % (det_name, hi, lo, k)})
                if det_name in cls_dets:
                    for p in CLS_P:
                        rules.append({"kind": "hyst", "det": det_name, "hi": hi, "lo": lo,
                                      "gate": {"kind": "wrist@k+cls@p", "k": k, "p": p},
                                      "desc": "hyst %s %.2f %.2f wrist@%.2f+cls@%.2f" % (det_name, hi, lo, k, p)})
            # cls gate (student/shipped only)
            if det_name in cls_dets:
                for p in CLS_P:
                    rules.append({"kind": "hyst", "det": det_name, "hi": hi, "lo": lo,
                                  "gate": {"kind": "cls@p", "p": p},
                                  "desc": "hyst %s %.2f %.2f cls@%.2f" % (det_name, hi, lo, p)})
    if det_name in cls_dets:
        for hi in (0.6, 0.7):
            for mid in (0.25, 0.35, 0.45):
                for lo in (0.01, 0.02, 0.05, 0.1):
                    for k2 in (0.25, 0.5):
                        for p in (0.3, 0.5, 0.7, 0.9):
                            rules.append({"kind": "tier3", "det": det_name, "hi": hi, "mid": mid, "lo": lo, "k1": 0.5, "k2": k2, "p": p,
                                          "desc": "tier3 %s %.2f %.2f %.2f wrist@0.50 | wrist@%.2f+cls@%.2f" % (det_name, hi, mid, lo, k2, p)})
        for hi in (0.6, 0.7, 1.01):
            for lo in (0.01, 0.05):
                for k in (None, 0.25, 0.5):
                    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
                        rules.append({"kind": "fuse", "det": det_name, "hi": hi, "lo": lo, "k": k, "t": t,
                                      "desc": "fuse %s %.2f %.2f %s t%.2f" % (det_name, hi, lo, "any" if k is None else "wrist@%.2f" % k, t)})
    return rules


def pareto_front(rows):
    """rows: list of dicts with precision, recall. Return non-dominated subset."""
    front = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if b["precision"] >= a["precision"] and b["recall"] >= a["recall"] and \
               (b["precision"] > a["precision"] or b["recall"] > a["recall"]):
                dominated = True
                break
        if not dominated:
            front.append(a)
    front.sort(key=lambda r: (-r["precision"], -r["recall"]))
    return front


def sweep_set(cache_set_name, rules_by_det):
    records = load_cache(cache_set_name)
    results = []  # list of rows
    for det_name in rules_by_det:
        agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "fp_special": 0, "n": 0})
        for rec in records:
            special = _is_special(rec, cache_set_name)
            for rule in rules_by_det[det_name]:
                key = rule["desc"]
                tp, fp, fn = evaluate_rule(rec, det_name, rule)
                a = agg[key]
                a["tp"] += tp
                a["fp"] += fp
                a["fn"] += fn
                if special:
                    a["fp_special"] += fp
                a["n"] += 1
        rows = []
        n_gt = sum(len(r["gt"]) for r in records)
        n_img = len(records)
        for key, a in agg.items():
            p, r, f1 = metrics(a["tp"], a["fp"], a["fn"])
            rows.append({
                "set": cache_set_name,
                "det": det_name,
                "desc": key,
                "tp": a["tp"],
                "fp": a["fp"],
                "fn": a["fn"],
                "fp_special": a["fp_special"],
                "precision": p,
                "recall": r,
                "f1": f1,
                "n_images": n_img,
                "n_gt": n_gt,
            })
        results.extend(rows)
    return results


def _is_special(rec, cache_set_name):
    if cache_set_name == "gtv2":
        return rec["category"] == "OOD"
    # lab: images with no GT box
    return len(rec["gt"]) == 0


def print_markdown(all_results, by_set):
    for sname in by_set:
        print("## Set %s" % sname, flush=True)
        print(flush=True)
        for det_name in DETECTORS:
            rows = [r for r in all_results if r["set"] == sname and r["det"] == det_name]
            if not rows:
                continue
            front = pareto_front(rows)
            print("### Detector `%s` -- Pareto front" % det_name, flush=True)
            print("| rule | precision | recall | F1 | TP | FP | FN | FP_special |", flush=True)
            print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
            for r in front:
                print("| %s | %.4f | %.4f | %.4f | %d | %d | %d | %d |" % (
                    r["desc"], r["precision"], r["recall"], r["f1"],
                    r["tp"], r["fp"], r["fn"], r["fp_special"]), flush=True)
            print(flush=True)
            print("### Target precision -> highest-recall rule", flush=True)
            print("| target precision | rule | precision | recall | F1 | TP | FP | FN |", flush=True)
            print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)
            for target in TARGET_PRECISIONS:
                cand = [r for r in rows if r["precision"] >= target - 1e-9]
                if cand:
                    best = max(cand, key=lambda r: (r["recall"], r["precision"]))
                    print("| %.2f | %s | %.4f | %.4f | %.4f | %d | %d | %d |" % (
                        target, best["desc"], best["precision"], best["recall"],
                        best["f1"], best["tp"], best["fp"], best["fn"]), flush=True)
                else:
                    print("| %.2f | (none) | - | - | - | - | - | - |" % target, flush=True)
            print(flush=True)


def write_jsonl(all_results, sname):
    os.makedirs(RULES_DIR, exist_ok=True)
    path = os.path.join(RULES_DIR, "sweep_%s.jsonl" % sname)
    with open(path, "w", encoding="utf-8") as f:
        for r in all_results:
            if r["set"] == sname:
                f.write(json.dumps(r) + "\n")
    print("Wrote %s" % path, flush=True)


def apply_to(setA, setB, min_precision):
    # pick best-recall rule per detector on setA with precision >= min_precision
    # recompute aggregates for setA
    rowsA = sweep_set(setA, {d: build_rules(d) for d in DETECTORS})
    print("== apply-to: setA=%s setB=%s min_precision=%.2f" % (setA, setB, min_precision), flush=True)
    for det_name in DETECTORS:
        cand = [r for r in rowsA if r["set"] == setA and r["det"] == det_name
                and r["precision"] >= min_precision - 1e-9]
        if not cand:
            print("det=%s: no rule meets precision >= %.2f on %s" % (det_name, min_precision, setA), flush=True)
            continue
        best = max(cand, key=lambda r: (r["recall"], r["precision"]))
        # parse rule back from desc is awkward; instead re-find the rule object by desc
        rule = next(r for r in build_rules(det_name) if r["desc"] == best["desc"])
        rowsB = sweep_set(setB, {det_name: [rule]})
        rb = rowsB[0]
        print("det=%s rule=%s -> setB %s: P=%.4f R=%.4f F1=%.4f TP=%d FP=%d FN=%d FP_special=%d" % (
            det_name, best["desc"], setB, rb["precision"], rb["recall"], rb["f1"],
            rb["tp"], rb["fp"], rb["fn"], rb["fp_special"]), flush=True)


def _rule_from_desc(det_name, desc):
    if desc.startswith("single"):
        t = float(desc.split()[-1])
        return {"kind": "single", "det": det_name, "t": t, "desc": desc}
    # hyst ... parse gate
    m = re.match(r"hyst %s ([\d.]+) ([\d.]+) (.+)" % re.escape(det_name), desc)
    hi, lo, gate_s = m.group(1), m.group(2), m.group(3)
    gate = None
    if gate_s.startswith("teacher@"):
        gate = {"kind": "teacher@c", "c": float(gate_s.split("@")[1])}
    elif gate_s.startswith("cls@"):
        gate = {"kind": "cls@p", "p": float(gate_s.split("@")[1])}
    elif gate_s.startswith("wrist@"):
        if "+cls@" in gate_s:
            k_part, p_part = gate_s.split("+cls@")
            gate = {"kind": "wrist@k+cls@p", "k": float(k_part.split("@")[1]), "p": float(p_part)}
        else:
            gate = {"kind": "wrist@k", "k": float(gate_s.split("@")[1])}
    return {"kind": "hyst", "det": det_name, "hi": float(hi), "lo": float(lo),
            "gate": gate, "desc": desc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="Stage 1 rule sweep")
    p.add_argument("worktree")  # repo root (accepted for CLI compatibility; ignored)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache")
    c.add_argument("set_name", choices=["gtv2", "lab"])

    s = sub.add_parser("sweep")
    s.add_argument("sets", nargs="+", choices=["gtv2", "lab"])
    s.add_argument("--apply-to", dest="apply_to", default=None)
    s.add_argument("--min-precision", dest="min_precision", type=float, default=None)
    return p.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)

    if args.cmd == "cache":
        limit = os.environ.get("LIMIT")
        limit = int(limit) if limit else None
        cache_set(args.set_name, limit=limit)
        return 0

    # sweep
    matcher_self_check()
    rules_by_det = {d: build_rules(d) for d in DETECTORS}
    all_results = []
    for sname in args.sets:
        res = sweep_set(sname, rules_by_det)
        all_results.extend(res)
        write_jsonl(all_results, sname)
    print_markdown(all_results, args.sets)

    if args.apply_to:
        if args.min_precision is None:
            print("--min-precision required with --apply-to", file=sys.stderr)
            return 2
        apply_to(args.sets[0], args.apply_to, args.min_precision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
