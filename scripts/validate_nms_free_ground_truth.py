"""
scripts/validate_nms_free_ground_truth.py

Official Ground Truth Validation Script for Item 5:
Version 4.50 NMS-Free Real-Time Edge Architecture.

Evaluates:
1. One-to-One Bipartite Matching (NMS-Free) at Conf=0.45 (Production Operating Point)
2. One-to-One Bipartite Matching (NMS-Free) at Conf=0.20 (High-Sensitivity Security Fallback)
3. One-to-Many with Greedy NMS at Conf=0.45, IoU_NMS=0.45
4. One-to-Many with Greedy NMS at Conf=0.20, IoU_NMS=0.45
5. Zero-Shot RT-DETR-L baseline at Conf=0.20 (COCO knife class 43)

Evaluates all 715 Ground Truth surveillance images at exact IoU=0.20 matching.
Saves comprehensive results to results/ground_truth_benchmark/nms_free_gt_validation_results.json.
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


def get_category_from_path(path):
    p = path.replace("\\", "/")
    if "OOD" in p:
        return "OOD"
    if "With_Coat" in p:
        return "With_Coat"
    if "Without_Coat" in p:
        return "Without_Coat"
    return "General"


def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    total_decisions = tp + tn + fp + fn
    accuracy = (tp + tn) / total_decisions if total_decisions > 0 else 0.0

    if (tp + fn) == 0 and fp == 0 and tn > 0:
        f1 = 1.0
        precision = 1.0
        recall = 1.0

    return precision, recall, f1, specificity, accuracy


def evaluate_configuration(detector, image_files, mode="one2one", conf=0.45, iou_nms=0.45, is_rtdetr=False):
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_images": 0, "inference_time": 0.0})

    for img_path in image_files:
        category = get_category_from_path(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        parent_dir = os.path.dirname(img_path)
        grandparent_dir = os.path.dirname(parent_dir)
        gt_txt_path = os.path.join(grandparent_dir, "labels", f"{filename}.txt")
        gt_boxes = load_ground_truth(gt_txt_path)

        if is_rtdetr:
            pred_out = detector.predict(img_path, conf=conf)
        else:
            pred_out = detector.predict(img_path, conf=conf, iou_nms=iou_nms, mode=mode)

        inf_time = pred_out["latency_ms"]

        stats[category]["inference_time"] += inf_time
        stats[category]["total_images"] += 1
        stats["Overall"]["inference_time"] += inf_time
        stats["Overall"]["total_images"] += 1

        # Normalized boxes are already [x1n, y1n, x2n, y2n]
        pred_boxes = pred_out["normalized_boxes"]

        matched_gt = set()
        matched_pred = set()

        for p_idx, p_box in enumerate(pred_boxes):
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
                stats[category]["TP"] += 1
                stats["Overall"]["TP"] += 1
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)
            else:
                stats[category]["FP"] += 1
                stats["Overall"]["FP"] += 1

        fn_count = len(gt_boxes) - len(matched_gt)
        stats[category]["FN"] += fn_count
        stats["Overall"]["FN"] += fn_count

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            stats[category]["TN"] += 1
            stats["Overall"]["TN"] += 1

    return stats


def main():
    image_files = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))
    print(f"Loaded {len(image_files)} Ground Truth test images from {GT_DATASET_DIR}.")

    # Initialize detectors
    yolo_detector = NMSFreeWeaponDetector()
    try:
        rtdetr_detector = ZeroShotEdgeDETRDetector("rtdetr-l.pt")
    except Exception as e:
        print(f"[WARNING] RT-DETR-L could not be loaded: {e}")
        rtdetr_detector = None

    eval_configs = [
        {
            "id": "O2O_Conf0.45",
            "name": "One-to-One Bipartite Matching (NMS-Free, Conf=0.45)",
            "detector": yolo_detector,
            "mode": "one2one",
            "conf": 0.45,
            "iou_nms": None,
            "is_rtdetr": False,
        },
        {
            "id": "O2O_Conf0.20",
            "name": "One-to-One Bipartite Matching (NMS-Free, Conf=0.20)",
            "detector": yolo_detector,
            "mode": "one2one",
            "conf": 0.20,
            "iou_nms": None,
            "is_rtdetr": False,
        },
        {
            "id": "O2M_Conf0.45_IoU0.45",
            "name": "One-to-Many Greedy NMS (Conf=0.45, IoU_NMS=0.45)",
            "detector": yolo_detector,
            "mode": "one2many",
            "conf": 0.45,
            "iou_nms": 0.45,
            "is_rtdetr": False,
        },
        {
            "id": "O2M_Conf0.20_IoU0.45",
            "name": "One-to-Many Greedy NMS (Conf=0.20, IoU_NMS=0.45)",
            "detector": yolo_detector,
            "mode": "one2many",
            "conf": 0.20,
            "iou_nms": 0.45,
            "is_rtdetr": False,
        },
    ]

    if rtdetr_detector is not None:
        eval_configs.append({
            "id": "RTDETR_Conf0.20",
            "name": "RT-DETR-L Zero-Shot (COCO knife class 43, Conf=0.20)",
            "detector": rtdetr_detector,
            "mode": "rtdetr",
            "conf": 0.20,
            "iou_nms": None,
            "is_rtdetr": True,
        })

    all_results = {}

    for cfg in eval_configs:
        cid = cfg["id"]
        cname = cfg["name"]
        print("\n" + "=" * 90)
        print(f"EVALUATING CONFIGURATION: {cname}")
        print("=" * 90)

        stats = evaluate_configuration(
            cfg["detector"],
            image_files,
            mode=cfg["mode"],
            conf=cfg["conf"],
            iou_nms=cfg["iou_nms"],
            is_rtdetr=cfg["is_rtdetr"],
        )

        cfg_summary = {"name": cname, "categories": {}}

        for cat in ["With_Coat", "Without_Coat", "OOD", "Overall"]:
            d = stats[cat]
            p, r, f1, spec, acc = calculate_metrics(d["TP"], d["FP"], d["FN"], d["TN"])
            avg_t = d["inference_time"] / d["total_images"] if d["total_images"] > 0 else 0

            cfg_summary["categories"][cat] = {
                "accuracy": round(acc * 100, 2),
                "precision": round(p * 100, 2),
                "recall": round(r * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "specificity": round(spec * 100, 2),
                "latency_ms": round(avg_t, 2),
                "TP": d["TP"],
                "FP": d["FP"],
                "FN": d["FN"],
                "TN": d["TN"],
                "total_images": d["total_images"],
            }

            print(
                f"[{cat:12s}] Acc: {acc*100:6.2f}% | P: {p*100:6.2f}% | R: {r*100:6.2f}% | F1: {f1*100:6.2f}% | "
                f"Spec: {spec*100:6.2f}% | Lat: {avg_t:5.2f}ms | (TP={d['TP']}, FP={d['FP']}, FN={d['FN']}, TN={d['TN']})"
            )

        all_results[cid] = cfg_summary

    # Save to JSON
    out_json = os.path.join(OUTPUT_DIR, "nms_free_gt_validation_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSuccessfully saved verified Ground Truth benchmark to: {out_json}")


if __name__ == "__main__":
    main()
