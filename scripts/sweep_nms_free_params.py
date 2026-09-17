"""
scripts/sweep_nms_free_params.py

Automated Parameter Sweep & Hyperparameter Impact Evaluation for Item 5:
Version 4.50 NMS-Free Real-Time Edge Architecture.

Evaluates:
- One-to-One Bipartite Matching across confidence thresholds [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
- One-to-Many Greedy NMS across confidence [0.20, 0.30, 0.45] and NMS IoU [0.30, 0.45, 0.60, 0.75]
- Zero-Shot RT-DETR-L baseline across confidence [0.10, 0.20, 0.30]

Outputs formatted comparison table and saves results to results/ground_truth_benchmark/nms_free_sweep_results.json.
"""

import os
import sys
import glob
import json
import time
from collections import defaultdict
import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)

from src.inference_nms_free_v4_50 import NMSFreeWeaponDetector, ZeroShotEdgeDETRDetector

GT_DATASET_DIR = os.path.join(REPO_DIR, "data", "Ground_Truth_Dataset")
OUTPUT_DIR = os.path.join(REPO_DIR, "results", "ground_truth_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def xywhn2xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(area1 + area2 - intersection + 1e-6)


def load_ground_truth(txt_path):
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def calculate_metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    return prec, rec, f1, spec, acc


def main():
    print("=" * 100)
    print("STARTING PARAMETER SWEEP: VERSION 4.50 NMS-FREE REAL-TIME ARCHITECTURE")
    print("=" * 100)

    # 1. Collect stratified subset for fine sweep: 50 With_Coat, 40 Without_Coat, 40 OOD = 130 images
    with_coat_imgs = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "With_Coat", "images", "*.jpg")))[:50]
    without_coat_imgs = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "Without_Coat", "images", "*.jpg")))[:40]
    ood_imgs = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "OOD", "images", "*.jpg")))[:40]
    sweep_images = with_coat_imgs + without_coat_imgs + ood_imgs

    print(f"Loaded {len(sweep_images)} stratified benchmark images for parameter sweep:")
    print(f"  With_Coat: {len(with_coat_imgs)}, Without_Coat: {len(without_coat_imgs)}, OOD: {len(ood_imgs)}")

    # Ground truth cache
    gt_cache = {}
    total_gt_targets = 0
    for img_p in sweep_images:
        lbl_p = img_p.replace("images", "labels").replace(".jpg", ".txt")
        boxes = load_ground_truth(lbl_p)
        gt_cache[img_p] = boxes
        total_gt_targets += len(boxes)
    print(f"  Total ground truth weapon targets in sweep set: {total_gt_targets}")

    # 2. Define Configurations to sweep
    configs = []

    # One-to-One Bipartite Matching (NMS-Free)
    for conf in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        configs.append({
            "id": f"O2O_Conf{conf:.2f}",
            "name": f"One-to-One (NMS-Free, Conf={conf:.2f})",
            "model_type": "yolo",
            "mode": "one2one",
            "conf": conf,
            "iou_nms": None,
        })

    # One-to-Many Greedy NMS (at Conf=0.20, 0.30, 0.45 with varying IoU suppression thresholds)
    for conf in [0.20, 0.30, 0.45]:
        for iou_nms in [0.30, 0.45, 0.60, 0.75]:
            configs.append({
                "id": f"O2M_Conf{conf:.2f}_IoU{iou_nms:.2f}",
                "name": f"One-to-Many (NMS IoU={iou_nms:.2f}, Conf={conf:.2f})",
                "model_type": "yolo",
                "mode": "one2many",
                "conf": conf,
                "iou_nms": iou_nms,
            })

    # Zero-Shot RT-DETR-L baseline
    for conf in [0.10, 0.20, 0.30]:
        configs.append({
            "id": f"RTDETR_Conf{conf:.2f}",
            "name": f"RT-DETR-L Zero-Shot (COCO knife, Conf={conf:.2f})",
            "model_type": "rtdetr",
            "mode": "rtdetr",
            "conf": conf,
            "iou_nms": None,
        })

    # Initialize models
    yolo_detector = NMSFreeWeaponDetector()
    try:
        rtdetr_detector = ZeroShotEdgeDETRDetector("rtdetr-l.pt")
    except Exception as e:
        print(f"[WARNING] Could not load RT-DETR-L: {e}")
        rtdetr_detector = None

    sweep_results = {}

    print("\nExecuting configuration sweeps...")
    for cfg in configs:
        cid = cfg["id"]
        cname = cfg["name"]
        mtype = cfg["model_type"]
        mode = cfg["mode"]
        conf = cfg["conf"]
        iou_nms = cfg["iou_nms"]

        if mtype == "rtdetr" and rtdetr_detector is None:
            continue

        tp, fp, fn, tn = 0, 0, 0, 0
        ood_tp, ood_fp, ood_fn, ood_tn = 0, 0, 0, 0
        total_lat_ms = 0.0

        for img_p in sweep_images:
            gt_boxes = gt_cache[img_p]
            is_ood = "OOD" in img_p.replace("\\", "/")

            if mtype == "yolo":
                pred_out = yolo_detector.predict(img_p, conf=conf, iou_nms=iou_nms, mode=mode)
            else:
                pred_out = rtdetr_detector.predict(img_p, conf=conf)

            total_lat_ms += pred_out["latency_ms"]
            pred_boxes = pred_out["normalized_boxes"]

            matched_gt = set()
            for p_box in pred_boxes:
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, g_box_data in enumerate(gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    g_box = xywhn2xyxy(*g_box_data[1:])
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx

                if best_iou >= 0.20:
                    tp += 1
                    matched_gt.add(best_g_idx)
                    if is_ood:
                        ood_tp += 1
                else:
                    fp += 1
                    if is_ood:
                        ood_fp += 1

            unmatched_gt = len(gt_boxes) - len(matched_gt)
            fn += unmatched_gt
            if is_ood:
                ood_fn += unmatched_gt

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                tn += 1
                if is_ood:
                    ood_tn += 1

        prec, rec, f1, spec, acc = calculate_metrics(tp, fp, fn, tn)
        ood_prec, ood_rec, ood_f1, ood_spec, ood_acc = calculate_metrics(ood_tp, ood_fp, ood_fn, ood_tn)
        avg_lat = total_lat_ms / len(sweep_images)

        sweep_results[cid] = {
            "name": cname,
            "mode": mode,
            "conf": conf,
            "iou_nms": iou_nms,
            "overall_f1": round(f1 * 100, 2),
            "threat_recall": round(rec * 100, 2),
            "threat_precision": round(prec * 100, 2),
            "overall_accuracy": round(acc * 100, 2),
            "civilian_specificity": round(spec * 100, 2),
            "ood_false_positives": ood_fp,
            "ood_specificity": round(ood_spec * 100, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        }

        print(f"[{cid:24s}] F1: {f1*100:5.2f}% | Rec: {rec*100:5.2f}% | Prec: {prec*100:5.2f}% | Acc: {acc*100:5.2f}% | OOD FP: {ood_fp:2d} | Lat: {avg_lat:5.2f}ms")

    # Save to JSON
    json_path = os.path.join(OUTPUT_DIR, "nms_free_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved full sweep results to: {json_path}")


if __name__ == "__main__":
    main()
