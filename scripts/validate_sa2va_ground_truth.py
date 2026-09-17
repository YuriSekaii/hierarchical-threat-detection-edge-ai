"""
scripts/validate_sa2va_ground_truth.py

Evaluates Sa2VA-Qwen2.5-VL-7B against the 715 Ground Truth surveillance images (data/Ground_Truth_Dataset).
Matches the exact validate_with_Ground_Truth.py evaluation protocol:
- IoU threshold = 0.20
- Categories: OOD, With_Coat, Without_Coat, Overall
- Metrics: Precision, Recall, F1-Score, Specificity, Accuracy, TP, FP, FN, TN, Latency
"""

import os
import sys
import glob
import time
import json
from collections import defaultdict
import numpy as np
from PIL import Image
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

GT_DATASET_DIR = os.path.join(REPO_DIR, "data", "Ground_Truth_Dataset")
OUTPUT_JSON = os.path.join(REPO_DIR, "results", "ground_truth_benchmark", "sa2va_gt_validation_results.json")

from transformers import AutoConfig, AutoProcessor, Qwen2Config
from models.sa2va.modeling_sa2va_qwen import Sa2VAChatModelQwen


def xywhn2xyxy(x, y, w, h, img_w=None, img_h=None):
    if img_w is not None and img_h is not None:
        # Denormalize to pixel coordinates
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for testing")
    parser.add_argument("--stride", type=int, default=1, help="Sampling stride")
    args = parser.parse_args()

    image_files = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))
    if args.stride > 1:
        image_files = image_files[::args.stride]
    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"Loaded {len(image_files)} Ground Truth test images (Stride: {args.stride}).")

    device = "cuda:0"
    model_path = "ByteDance/Sa2VA-Qwen2_5-VL-7B"

    print("Loading Sa2VA model...")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg.vocab_size = 151670
    cfg.text_config = Qwen2Config(**cfg.text_config)

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    proc.image_processor.min_pixels = 256 * 28 * 28
    proc.image_processor.max_pixels = 1024 * 28 * 28

    model = Sa2VAChatModelQwen.from_pretrained(
        model_path,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    ).eval()
    print("Sa2VA loaded.")

    prompt = "Please locate and segment any knife, blade, dagger, or handheld weapon in this image."
    iou_thresh = 0.20

    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_images": 0, "inference_time": 0.0})

    for idx, img_path in enumerate(image_files):
        category = get_category_from_path(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        parent_dir = os.path.dirname(img_path)
        grandparent_dir = os.path.dirname(parent_dir)
        gt_txt_path = os.path.join(grandparent_dir, "labels", f"{filename}.txt")
        gt_boxes = load_ground_truth(gt_txt_path)

        pil_img = Image.open(img_path).convert("RGB")
        w_img, h_img = pil_img.size

        t0 = time.perf_counter()
        with torch.inference_mode():
            res = model.predict_forward(image=pil_img, text=prompt, processor=proc)
        inf_time = (time.perf_counter() - t0) * 1000.0

        stats[category]["inference_time"] += inf_time
        stats[category]["total_images"] += 1
        stats["Overall"]["inference_time"] += inf_time
        stats["Overall"]["total_images"] += 1

        pred_boxes = []
        masks = res.get("prediction_masks", [])
        for m in masks:
            m_2d = m[0] if m.ndim == 3 else m
            binary = (m_2d > 0.5).astype(np.uint8)
            active = np.where(binary > 0)
            n_pts = len(active[0])
            pct = n_pts / float(h_img * w_img) if (h_img * w_img) > 0 else 0.0

            if n_pts >= 200 and pct <= 0.03:
                y_min, y_max = int(np.min(active[0])), int(np.max(active[0]))
                x_min, x_max = int(np.min(active[1])), int(np.max(active[1]))
                box_area = (x_max - x_min + 1) * (y_max - y_min + 1)
                solidity = n_pts / float(box_area) if box_area > 0 else 0.0
                if solidity >= 0.15:
                    pred_boxes.append([x_min, y_min, x_max, y_max])

        # Match against ground truth boxes (converted to pixel coordinates)
        matched_gt = set()
        matched_pred = set()

        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_box_data in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                g_box = xywhn2xyxy(*g_box_data[1:], img_w=w_img, img_h=h_img)
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

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
            print(f"  [{idx+1:03d}/{len(image_files)}] {category} | {filename} -> preds={len(pred_boxes)}, gt={len(gt_boxes)} ({inf_time:.0f}ms)")

    print("\n" + "=" * 80)
    print("SA2VA-QWEN2.5-VL-7B GROUND TRUTH VERIFICATION RESULTS (IoU=0.20)")
    print("=" * 80)
    results_dict = {}
    for cat in ["OOD", "With_Coat", "Without_Coat", "Overall"]:
        data = stats[cat]
        p, r, f1, spec, acc = calculate_metrics(data["TP"], data["FP"], data["FN"], data["TN"])
        avg_time = data["inference_time"] / data["total_images"] if data["total_images"] > 0 else 0
        results_dict[cat] = {
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "specificity": round(spec, 4),
            "avg_latency_ms": round(avg_time, 2),
            "tp": data["TP"],
            "fp": data["FP"],
            "fn": data["FN"],
            "tn": data["TN"],
            "total_images": data["total_images"]
        }
        print(f"[{cat:12s}] Acc: {acc*100:6.2f}% | P: {p*100:6.2f}% | R: {r*100:6.2f}% | F1: {f1*100:6.2f}% | Spec: {spec*100:6.2f}% | Lat: {avg_time:5.2f}ms | (TP={data['TP']}, FP={data['FP']}, FN={data['FN']}, TN={data['TN']})")
    print("=" * 80)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved results to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
