"""
scripts/validate_grounding_dino_ground_truth.py

Official Ground Truth Evaluation Script for Version 4.20 (Grounding DINO 1.5).
Evaluates Grounding DINO against all 715 Ground Truth surveillance images (data/Ground_Truth_Dataset).
Matches exact validate_stage1_ground_truth.py evaluation protocol:
- IoU threshold = 0.20
- Categories: OOD (121), With_Coat (359), Without_Coat (235), Overall (715)
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
from PIL import Image

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.inference_grounding_dino_v4_20 import GroundingDinoDetector, DEFAULT_PROMPT

GT_DATASET_DIR = os.path.join(REPO_DIR, "data", "Ground_Truth_Dataset")
OUTPUT_JSON_DEFAULT = os.path.join(REPO_DIR, "results", "ground_truth_benchmark", "grounding_dino_gt_validation_results.json")


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


def evaluate_dataset(
    detector: GroundingDinoDetector,
    image_files: List[str],
    prompt: Optional[str] = None,
    box_thresh: float = 0.20,
    text_thresh: float = 0.20,
    neg_margin: float = 0.0,
    enable_negative_suppression: bool = True,
    nms_thresh: Optional[float] = 0.50,
    iou_thresh: float = 0.20,
    verbose: bool = True
) -> Dict:
    """Evaluates Grounding DINO on image files using exact IoU=0.20 matching."""
    stats = defaultdict(lambda: {
        "TP": 0, "FP": 0, "FN": 0, "TN": 0,
        "total_images": 0, "inference_time": 0.0,
        "suppressed_count": 0, "raw_candidates": 0
    })

    total_imgs = len(image_files)

    for idx, img_path in enumerate(image_files):
        category = get_category_from_path(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        parent_dir = os.path.dirname(img_path)
        grandparent_dir = os.path.dirname(parent_dir)
        gt_txt_path = os.path.join(grandparent_dir, "labels", f"{filename}.txt")
        gt_boxes = load_ground_truth(gt_txt_path)

        res = detector.detect(
            image=img_path,
            prompt=prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            neg_margin=neg_margin,
            nms_threshold=nms_thresh,
            enable_negative_suppression=enable_negative_suppression
        )

        inf_time = res["latency_ms"]
        pred_boxes_xywhn = res["boxes_xywhn"]

        # Track stats
        for cat_key in [category, "Overall"]:
            stats[cat_key]["inference_time"] += inf_time
            stats[cat_key]["total_images"] += 1
            stats[cat_key]["suppressed_count"] += res["suppressed_boxes_count"]
            stats[cat_key]["raw_candidates"] += res["raw_boxes_count"]

        # Convert pred boxes to normalized xyxy
        pred_boxes_norm = [xywhn2xyxy(*b) for b in pred_boxes_xywhn]

        matched_gt = set()
        matched_pred = set()

        for p_idx, p_box in enumerate(pred_boxes_norm):
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

        if len(gt_boxes) == 0 and len(pred_boxes_norm) == 0:
            stats[category]["TN"] += 1
            stats["Overall"]["TN"] += 1

        if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == total_imgs):
            print(f"  [{idx+1:03d}/{total_imgs}] {category:12s} | preds={len(pred_boxes_norm)}, gt={len(gt_boxes)}, supp={res['suppressed_boxes_count']} ({inf_time:.1f}ms)")

    # Compile results
    results_summary = {}
    for cat in ["OOD", "With_Coat", "Without_Coat", "Overall"]:
        d = stats[cat]
        p, r, f1, spec, acc = calculate_metrics(d["TP"], d["FP"], d["FN"], d["TN"])
        avg_lat = d["inference_time"] / d["total_images"] if d["total_images"] > 0 else 0.0
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        results_summary[cat] = {
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "specificity": round(spec, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "fps": round(fps, 1),
            "tp": d["TP"],
            "fp": d["FP"],
            "fn": d["FN"],
            "tn": d["TN"],
            "suppressed_candidates": d["suppressed_count"],
            "raw_candidates": d["raw_candidates"],
            "total_images": d["total_images"]
        }

    return results_summary


def print_results_table(results_summary: Dict, header_title: str = "BENCHMARK RESULTS"):
    print("\n" + "=" * 95)
    print(f"  {header_title}")
    print("=" * 95)
    print(f"{'Category':14s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s} | {'Specificity':11s} | {'Latency':8s} | {'TP/FP/FN/TN':15s}")
    print("-" * 95)
    for cat in ["OOD", "With_Coat", "Without_Coat", "Overall"]:
        d = results_summary[cat]
        cm = f"{d['tp']}/{d['fp']}/{d['fn']}/{d['tn']}"
        print(f"{cat:14s} | {d['accuracy']*100:7.2f}% | {d['precision']*100:8.2f}% | {d['recall']*100:7.2f}% | {d['f1_score']*100:7.2f}% | {d['specificity']*100:10.2f}% | {d['avg_latency_ms']:6.2f}ms | {cm:15s}")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Grounding DINO on Ground Truth Dataset")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt string")
    parser.add_argument("--box-thresh", type=float, default=0.20, help="Box threshold")
    parser.add_argument("--text-thresh", type=float, default=0.20, help="Text threshold")
    parser.add_argument("--neg-margin", type=float, default=0.0, help="Negative suppression margin")
    parser.add_argument("--nms-thresh", type=float, default=0.50, help="NMS IoU threshold (or None to disable)")
    parser.add_argument("--disable-nms", action="store_true", help="Disable NMS deduplication")
    parser.add_argument("--disable-neg-suppression", action="store_true", help="Disable competing negative suppression")
    parser.add_argument("--stride", type=int, default=1, help="Sampling stride")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to evaluate")
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON_DEFAULT, help="Output JSON path")
    parser.add_argument("--save-json", action="store_true", help="Save results to output JSON")
    args = parser.parse_args()

    image_files = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))
    if args.stride > 1:
        image_files = image_files[::args.stride]
    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"Loaded {len(image_files)} Ground Truth test images from {GT_DATASET_DIR} (Stride: {args.stride}).")

    detector = GroundingDinoDetector(
        box_threshold=args.box_thresh,
        text_threshold=args.text_thresh,
        neg_margin=args.neg_margin,
        nms_threshold=None if args.disable_nms else args.nms_thresh
    )

    supp_mode = "DISABLED" if args.disable_neg_suppression else f"ACTIVE (Margin={args.neg_margin})"
    nms_mode = "DISABLED" if args.disable_nms else f"ACTIVE (IoU={args.nms_thresh})"
    print(f"\nConfiguration: Box Thresh={args.box_thresh}, Text Thresh={args.text_thresh}, Negative Suppression={supp_mode}, NMS={nms_mode}")
    print(f"Prompt: \"{args.prompt}\"")

    results = evaluate_dataset(
        detector=detector,
        image_files=image_files,
        prompt=args.prompt,
        box_thresh=args.box_thresh,
        text_thresh=args.text_thresh,
        neg_margin=args.neg_margin,
        enable_negative_suppression=not args.disable_neg_suppression,
        nms_thresh=None if args.disable_nms else args.nms_thresh,
        iou_thresh=0.20,
        verbose=True
    )

    print_results_table(results, f"GROUNDING DINO v4.20 (Box={args.box_thresh}, Text={args.text_thresh}, IoU=0.20)")

    if args.save_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        payload = {
            "model": "IDEA-Research/grounding-dino-base",
            "prompt": args.prompt,
            "box_threshold": args.box_thresh,
            "text_threshold": args.text_thresh,
            "neg_margin": args.neg_margin,
            "negative_suppression_enabled": not args.disable_neg_suppression,
            "iou_threshold": 0.20,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved benchmark results to: {args.output_json}")


if __name__ == "__main__":
    main()
