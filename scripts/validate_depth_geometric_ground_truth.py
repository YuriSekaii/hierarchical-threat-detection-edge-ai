"""
scripts/validate_depth_geometric_ground_truth.py

Official Ground Truth Evaluation Script for Version 4.40 (Geometry-Informed Monocular Depth Pipeline).
Evaluates Depth Anything V2 + Metric Surface Discontinuity Filtering against all 715 Ground Truth surveillance images (data/Ground_Truth_Dataset).

Matches exact validate_stage1_ground_truth.py evaluation protocol:
- IoU threshold = 0.20
- Categories: OOD (119/121), With_Coat (351/359), Without_Coat (245/235), Overall (715)
- Metrics: Precision, Recall, F1-Score, Specificity, Accuracy, TP, FP, FN, TN, Latency (ms / FPS)
"""

import os
import sys
import glob
import time
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.inference_depth_geometric_v4_40 import DepthGeometricDetector

GT_DATASET_DIR = os.path.join(REPO_DIR, "data", "Ground_Truth_Dataset")
OUTPUT_JSON_DEFAULT = os.path.join(REPO_DIR, "results", "ground_truth_benchmark", "depth_geometric_gt_validation_results.json")


def xywhn2xyxy(x, y, w, h, img_w=None, img_h=None):
    """Convert normalized xywh to xyxy (normalized or pixel)."""
    if img_w is not None and img_h is not None:
        x1 = (x - w / 2) * img_w
        y1 = (y - h / 2) * img_h
        x2 = (x + w / 2) * img_w
        y2 = (y + h / 2) * img_h
    else:
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
    return x1, y1, x2, y2


def calculate_iou(box1, box2):
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
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
    """Loads ground truth bounding boxes from YOLO annotation txt file."""
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def get_category_from_path(path):
    """Determines category (With_Coat, Without_Coat, OOD) from image path."""
    p = path.replace("\\", "/")
    if "OOD" in p:
        return "OOD"
    if "With_Coat" in p:
        return "With_Coat"
    if "Without_Coat" in p:
        return "Without_Coat"
    return "General"


def calculate_metrics(tp, fp, fn, tn):
    """Computes Precision, Recall, F1, Specificity, and Accuracy."""
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


def run_benchmark(
    detector: DepthGeometricDetector,
    image_files: List[str],
    iou_thresh: float = 0.20,
    weapon_conf: Optional[float] = None,
    min_step: Optional[float] = None,
    min_grad: Optional[float] = None,
    max_var: Optional[float] = None,
) -> Tuple[Dict, Dict]:
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_images": 0, "inference_time": 0.0})
    per_image_results = []

    print(f"Executing Ground Truth validation on {len(image_files)} images...")
    t_start_total = time.perf_counter()

    for idx, img_path in enumerate(image_files):
        category = get_category_from_path(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        parent_dir = os.path.dirname(img_path)
        grandparent_dir = os.path.dirname(parent_dir)
        gt_txt_path = os.path.join(grandparent_dir, "labels", f"{filename}.txt")
        gt_raw = load_ground_truth(gt_txt_path)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]

        gt_boxes = [xywhn2xyxy(b[1], b[2], b[3], b[4], w_img, h_img) for b in gt_raw]

        pred = detector.predict(
            img_bgr,
            weapon_conf_thresh=weapon_conf,
            min_step_thresh=min_step,
            min_grad_thresh=min_grad,
            max_var_thresh=max_var,
        )
        inf_time = pred["latency_ms"]
        pred_boxes = pred["boxes"]

        stats[category]["inference_time"] += inf_time
        stats[category]["total_images"] += 1
        stats["Overall"]["inference_time"] += inf_time
        stats["Overall"]["total_images"] += 1

        matched_gt = set()
        matched_pred = set()

        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
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

        per_image_results.append({
            "image": os.path.basename(img_path),
            "category": category,
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
            "raw_proposals": pred["raw_proposal_count"],
            "tp": len(matched_pred),
            "fp": len(pred_boxes) - len(matched_pred),
            "fn": fn_count,
            "latency_ms": inf_time
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
            elapsed = time.perf_counter() - t_start_total
            print(f"  [{idx + 1}/{len(image_files)}] Elapsed: {elapsed:.1f}s | Current Frame Latency: {inf_time:.2f}ms")

    return dict(stats), {"per_image": per_image_results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Version 4.40 Depth Geometric Pipeline on Official Ground Truth Dataset")
    parser.add_argument("--weapon-conf", type=float, default=0.20, help="Weapon candidate proposal confidence threshold")
    parser.add_argument("--min-step", type=float, default=0.003, help="Minimum depth boundary step discontinuity")
    parser.add_argument("--min-grad", type=float, default=0.020, help="Minimum 90th percentile surface gradient")
    parser.add_argument("--max-var", type=float, default=0.035, help="Maximum internal depth variance (clutter rejection)")
    parser.add_argument("--iou", type=float, default=0.20, help="IoU threshold for evaluation (default: 0.20)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images evaluated for rapid debugging")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON_DEFAULT, help="Path to output JSON summary")
    args = parser.parse_args()

    image_files = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))
    if args.limit:
        image_files = image_files[:args.limit]

    print(f"\n" + "=" * 90)
    print(f"VERSION 4.40 OFFICIAL GROUND TRUTH BENCHMARK: GEOMETRY-INFORMED MONOCULAR DEPTH")
    print(f"Dataset Root: {GT_DATASET_DIR} ({len(image_files)} test images)")
    print(f"Config: Proposal Conf={args.weapon_conf}, Min Step={args.min_step}, Min Grad={args.min_grad}, Max Var={args.max_var}, IoU={args.iou}")
    print("=" * 90 + "\n")

    detector = DepthGeometricDetector(
        weapon_conf_thresh=args.weapon_conf,
        min_step_thresh=args.min_step,
        min_grad_thresh=args.min_grad,
        max_var_thresh=args.max_var,
    )

    stats, details = run_benchmark(
        detector,
        image_files,
        iou_thresh=args.iou,
        weapon_conf=args.weapon_conf,
        min_step=args.min_step,
        min_grad=args.min_grad,
        max_var=args.max_var,
    )

    # Print Report Table
    print("\n" + "=" * 110)
    print(f"{'Category':<14} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'Specificity':<11} | {'Avg Latency':<11} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4}")
    print("-" * 110)

    summary_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "4.40",
        "architecture": "Geometry-Informed Monocular Depth Pipeline (Depth Anything V2)",
        "configuration": {
            "weapon_conf": args.weapon_conf,
            "min_step": args.min_step,
            "min_grad": args.min_grad,
            "max_var": args.max_var,
            "iou_thresh": args.iou,
            "total_images": len(image_files)
        },
        "results": {}
    }

    for cat in ["With_Coat", "Without_Coat", "OOD", "Overall"]:
        d = stats.get(cat, {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_images": 0, "inference_time": 0.0})
        p, r, f1, spec, acc = calculate_metrics(d["TP"], d["FP"], d["FN"], d["TN"])
        avg_t = d["inference_time"] / d["total_images"] if d["total_images"] > 0 else 0.0
        fps = 1000.0 / avg_t if avg_t > 0 else 0.0

        summary_json["results"][cat] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(p * 100, 2),
            "recall": round(r * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "specificity": round(spec * 100, 2),
            "avg_latency_ms": round(avg_t, 2),
            "fps": round(fps, 2),
            "tp": d["TP"],
            "fp": d["FP"],
            "fn": d["FN"],
            "tn": d["TN"],
            "total_images": d["total_images"]
        }

        print(f"{cat:<14} | {acc*100:7.2f}% | {p*100:8.2f}% | {r*100:7.2f}% | {f1*100:7.2f}% | {spec*100:10.2f}% | {avg_t:7.2f}ms ({fps:4.1f} FPS) | {d['TP']:<4} | {d['FP']:<4} | {d['FN']:<4} | {d['TN']:<4}")

    print("=" * 110 + "\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_json["details"] = details["per_image"]
    with open(args.output, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Benchmark artifact successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
