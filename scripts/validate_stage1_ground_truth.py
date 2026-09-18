"""
scripts/validate_stage1_ground_truth.py

Official Ground Truth Evaluation Script for Stage 1 Weapon Detection.
Evaluates models against the 715 Ground Truth surveillance images (data/Ground_Truth_Dataset).
Matches validate_with_Ground_Truth.py protocol with exact IoU=0.20 bounding-box matching.
"""

import os
import sys
import glob
import datetime
from collections import defaultdict
import numpy as np
import torch
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GT_DATASET_DIR = os.path.join(REPO_DIR, "data", "Ground_Truth_Dataset")
OUTPUT_ROOT = os.path.join(REPO_DIR, "results", "ground_truth_benchmark")


def xywhn2xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area + 1e-6)
    return iou


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


def validate_yolo_model(model_path, image_files, conf=0.45, iou_thresh=0.20):
    model = YOLO(model_path, task="detect")
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_images": 0, "inference_time": 0.0})

    for img_path in image_files:
        category = get_category_from_path(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        parent_dir = os.path.dirname(img_path)
        grandparent_dir = os.path.dirname(parent_dir)
        gt_txt_path = os.path.join(grandparent_dir, "labels", f"{filename}.txt")
        gt_boxes = load_ground_truth(gt_txt_path)

        results = model(img_path, conf=conf, imgsz=640, verbose=False)
        inf_time = results[0].speed.get("inference", 0.0)

        stats[category]["inference_time"] += inf_time
        stats[category]["total_images"] += 1
        stats["Overall"]["inference_time"] += inf_time
        stats["Overall"]["total_images"] += 1

        pred_boxes = []
        for box in results[0].boxes:
            x, y, w, h = box.xywhn[0].tolist()
            pred_boxes.append(xywhn2xyxy(x, y, w, h))

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

            if best_iou >= iou_thresh:
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
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Ground Truth Evaluator")
    parser.add_argument("--weights", type=str, default=os.path.join(REPO_DIR, "weights", "yolo_weapon_distilled.engine"),
                        help="Path to YOLO weights (.engine or .pt)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    args = parser.parse_args()

    image_files = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))
    print(f"Loaded {len(image_files)} Ground Truth test images from {GT_DATASET_DIR}.")
    print(f"Target Model: {args.weights}")

    print(f"\n" + "=" * 80)
    print(f"EVALUATING STAGE 1 (Model={os.path.basename(args.weights)}, Conf={args.conf}, IoU=0.20)")
    print("=" * 80)
    stats = validate_yolo_model(args.weights, image_files, conf=args.conf)
    for cat in ["OOD", "With_Coat", "Without_Coat", "Overall"]:
        d = stats[cat]
        p, r, f1, spec, acc = calculate_metrics(d["TP"], d["FP"], d["FN"], d["TN"])
        avg_t = d["inference_time"] / d["total_images"] if d["total_images"] > 0 else 0
        print(f"[{cat:12s}] Acc: {acc*100:6.2f}% | P: {p*100:6.2f}% | R: {r*100:6.2f}% | F1: {f1*100:6.2f}% | Spec: {spec*100:6.2f}% | Lat: {avg_t:5.2f}ms | (TP={d['TP']}, FP={d['FP']}, FN={d['FN']}, TN={d['TN']})")


if __name__ == "__main__":
    main()
