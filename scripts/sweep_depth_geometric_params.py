"""
scripts/sweep_depth_geometric_params.py

Automated Hyperparameter Sweep & Ablation Study for Version 4.40 (Geometry-Informed Monocular Depth Pipeline).
Systematically evaluates key architecture hyperparameters across stratified Ground Truth images:
1. Candidate proposal confidence threshold: [0.15, 0.20, 0.25, 0.30, 0.45]
2. Boundary step discontinuity tau_step: [0.001, 0.003, 0.006, 0.010, 0.0 (disabled)]
3. Surface gradient threshold tau_grad: [0.015, 0.020, 0.030, 0.0 (disabled)]
4. Clutter variance limit tau_var: [0.025, 0.035, 0.050, inf (disabled)]
5. Structural Ablations:
   - Raw YOLO Conf=0.20 (No Geometric Gating)
   - Raw YOLO Conf=0.45 (Baseline Champion)
   - Planar Step Gating Alone (tau_step active, var disabled)
   - Clutter Variance Gating Alone (var active, step disabled)
   - Full Geometric Pipeline (Step + Grad + Variance)
"""

import os
import sys
import glob
import time
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import cv2

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.inference_depth_geometric_v4_40 import DepthGeometricDetector
from scripts.validate_depth_geometric_ground_truth import (
    xywhn2xyxy, calculate_iou, load_ground_truth, get_category_from_path, calculate_metrics
)

GT_DATASET_DIR = os.path.join(REPO_DIR, "data", "Ground_Truth_Dataset")
SWEEP_OUTPUT_JSON = os.path.join(REPO_DIR, "results", "ground_truth_benchmark", "depth_geometric_sweep_results.json")


def run_sweep():
    print("=" * 110)
    print("STARTING VERSION 4.40 HYPERPARAMETER SWEEP & STRUCTURAL ABLATION STUDY")
    print("=" * 110)

    # Prepare stratified evaluation subset (50 With_Coat, 40 Without_Coat, 40 OOD = 130 images)
    with_coat_all = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "With_Coat", "images", "*.jpg")))
    without_coat_all = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "Without_Coat", "images", "*.jpg")))
    ood_all = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "OOD", "images", "*.jpg")))

    eval_images = with_coat_all[20:70] + without_coat_all[20:60] + ood_all[10:50]
    print(f"Stratified evaluation subset: {len(eval_images)} images ({len(with_coat_all[20:70])} With_Coat, {len(without_coat_all[20:60])} Without_Coat, {len(ood_all[10:50])} OOD)")

    cached_data = []
    total_gt_weapons = 0
    for p in eval_images:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        cat = get_category_from_path(p)
        fn = os.path.splitext(os.path.basename(p))[0]
        parent = os.path.dirname(p)
        grandparent = os.path.dirname(parent)
        gt_txt = os.path.join(grandparent, "labels", f"{fn}.txt")
        gt_raw = load_ground_truth(gt_txt)
        gt_boxes = [xywhn2xyxy(b[1], b[2], b[3], b[4], w, h) for b in gt_raw]
        total_gt_weapons += len(gt_boxes)
        cached_data.append({
            "path": p,
            "img": img_bgr,
            "w": w,
            "h": h,
            "cat": cat,
            "gt_boxes": gt_boxes
        })
    print(f"Loaded {len(cached_data)} images containing {total_gt_weapons} ground truth weapon annotations.")

    detector = DepthGeometricDetector(
        weapon_conf_thresh=0.15,
        min_step_thresh=0.003,
        min_grad_thresh=0.020,
        max_var_thresh=0.035
    )

    configs = [
        # 1. Raw YOLO Baselines (No Depth Gating)
        {"id": "YOLO_Conf0.20_Raw", "conf": 0.20, "step": 0.0, "grad": 0.0, "var": 9999.0, "desc": "Raw YOLO (conf=0.20, No Gating)"},
        {"id": "YOLO_Conf0.45_Raw", "conf": 0.45, "step": 0.0, "grad": 0.0, "var": 9999.0, "desc": "Raw YOLO (conf=0.45, Baseline)"},

        # 2. Planar Step Gating Alone (Ablation: Step Discontinuity Only)
        {"id": "StepOnly_Step0.002", "conf": 0.20, "step": 0.002, "grad": 0.015, "var": 9999.0, "desc": "Step Gate Only (step=0.002)"},
        {"id": "StepOnly_Step0.005", "conf": 0.20, "step": 0.005, "grad": 0.020, "var": 9999.0, "desc": "Step Gate Only (step=0.005)"},
        {"id": "StepOnly_Step0.010", "conf": 0.20, "step": 0.010, "grad": 0.030, "var": 9999.0, "desc": "Step Gate Only (step=0.010)"},

        # 3. Clutter Variance Gating Alone (Ablation: Background Variance Only)
        {"id": "VarOnly_Var0.025", "conf": 0.20, "step": 0.0, "grad": 0.0, "var": 0.025, "desc": "Variance Gate Only (var=0.025)"},
        {"id": "VarOnly_Var0.035", "conf": 0.20, "step": 0.0, "grad": 0.0, "var": 0.035, "desc": "Variance Gate Only (var=0.035)"},

        # 4. Full Geometric Pipeline Grid
        {"id": "Geo_Step0.002_Var0.035", "conf": 0.20, "step": 0.002, "grad": 0.015, "var": 0.035, "desc": "Full Geo (step=0.002, var=0.035) [High Recall]"},
        {"id": "Geo_Step0.003_Var0.035", "conf": 0.20, "step": 0.003, "grad": 0.020, "var": 0.035, "desc": "Full Geo (step=0.003, var=0.035) [Nominal]"},
        {"id": "Geo_Step0.005_Var0.025", "conf": 0.20, "step": 0.005, "grad": 0.020, "var": 0.025, "desc": "Full Geo (step=0.005, var=0.025) [High Precision]"},
        {"id": "Geo_Conf0.25_Step0.003", "conf": 0.25, "step": 0.003, "grad": 0.020, "var": 0.035, "desc": "Full Geo (conf=0.25, step=0.003)"},
        {"id": "Geo_Conf0.30_Step0.003", "conf": 0.30, "step": 0.003, "grad": 0.020, "var": 0.035, "desc": "Full Geo (conf=0.30, step=0.003)"},
    ]

    sweep_results = []

    print("\n" + "=" * 135)
    print(f"{'Config ID':<26} | {'Description':<38} | {'Overall F1':<10} | {'Recall':<8} | {'Precision':<9} | {'Accuracy':<8} | {'OOD Spec':<8} | {'OOD FP':<6}")
    print("-" * 135)

    for cfg in configs:
        cid = cfg["id"]
        c_conf = cfg["conf"]
        c_step = cfg["step"]
        c_grad = cfg["grad"]
        c_var = cfg["var"]

        stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        t0 = time.perf_counter()

        for item in cached_data:
            img_bgr = item["img"]
            cat = item["cat"]
            gt_boxes = item["gt_boxes"]

            pred = detector.predict(
                img_bgr,
                weapon_conf_thresh=c_conf,
                min_step_thresh=c_step,
                min_grad_thresh=c_grad,
                max_var_thresh=c_var
            )
            pred_boxes = pred["boxes"]

            matched_gt = set()
            for p_b in pred_boxes:
                best_iou = 0.0
                best_g = -1
                for g_idx, g_b in enumerate(gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    v_iou = calculate_iou(p_b, g_b)
                    if v_iou > best_iou:
                        best_iou = v_iou
                        best_g = g_idx
                if best_iou >= 0.20:
                    stats[cat]["TP"] += 1
                    stats["Overall"]["TP"] += 1
                    matched_gt.add(best_g)
                else:
                    stats[cat]["FP"] += 1
                    stats["Overall"]["FP"] += 1

            fn = len(gt_boxes) - len(matched_gt)
            stats[cat]["FN"] += fn
            stats["Overall"]["FN"] += fn

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                stats[cat]["TN"] += 1
                stats["Overall"]["TN"] += 1

        ov = stats["Overall"]
        ood = stats["OOD"]
        p, r, f1, spec, acc = calculate_metrics(ov["TP"], ov["FP"], ov["FN"], ov["TN"])
        _, _, _, ood_spec, _ = calculate_metrics(ood["TP"], ood["FP"], ood["FN"], ood["TN"])
        duration = time.perf_counter() - t0

        result_row = {
            "config_id": cid,
            "description": cfg["desc"],
            "conf": c_conf,
            "min_step": c_step,
            "min_grad": c_grad,
            "max_var": c_var,
            "overall_f1": round(f1 * 100, 2),
            "threat_recall": round(r * 100, 2),
            "threat_precision": round(p * 100, 2),
            "overall_accuracy": round(acc * 100, 2),
            "ood_specificity": round(ood_spec * 100, 2),
            "ood_false_positives": ood["FP"],
            "tp": ov["TP"],
            "fp": ov["FP"],
            "fn": ov["FN"],
            "tn": ov["TN"],
            "sweep_duration_sec": round(duration, 2)
        }
        sweep_results.append(result_row)

        print(f"{cid:<26} | {cfg['desc']:<38} | {f1*100:8.2f}% | {r*100:6.2f}% | {p*100:7.2f}% | {acc*100:6.2f}% | {ood_spec*100:6.2f}% | {ood['FP']:<6}")

    print("=" * 135 + "\n")

    os.makedirs(os.path.dirname(SWEEP_OUTPUT_JSON), exist_ok=True)
    with open(SWEEP_OUTPUT_JSON, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Sweep results successfully saved to: {SWEEP_OUTPUT_JSON}")


if __name__ == "__main__":
    run_sweep()
