"""
scripts/sweep_hoi_contact_params.py

Automated Hyperparameter Sweep & Ablation Study for Version 4.30 (Contact-State HOI Transformer).
Systematically evaluates key architecture hyperparameters across stratified Ground Truth images:
1. Candidate proposal confidence threshold: [0.15, 0.20, 0.25, 0.30]
2. Bipartite spatial interaction radius R_interact: [1.5, 2.0, 2.5, inf (no spatial gate)]
3. DINOv2 contact affinity threshold tau_contact: [0.45, 0.50, 0.52, 0.55, 0.0 (no contact gate)]
4. Structural Ablation Modes:
   - Stream 1 Alone (Raw YOLO, no pose, no DINOv2)
   - Spatial Hand Gating Alone (YOLO + Pose, no DINOv2)
   - Contact Gating Alone (YOLO + DINOv2, no Pose spatial filter)
   - Full Dual-Stream HOI Pipeline (YOLO + Pose + DINOv2)
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

from src.inference_hoi_contact_v4_30 import HOIContactDetector
from scripts.validate_hoi_contact_ground_truth import (
    xywhn2xyxy, calculate_iou, load_ground_truth, get_category_from_path, calculate_metrics
)

GT_DATASET_DIR = os.path.join(REPO_DIR, "data", "Ground_Truth_Dataset")
SWEEP_OUTPUT_JSON = os.path.join(REPO_DIR, "results", "ground_truth_benchmark", "hoi_contact_sweep_results.json")


def run_sweep():
    print("=" * 110)
    print("STARTING VERSION 4.30 HYPERPARAMETER SWEEP & STRUCTURAL ABLATION STUDY")
    print("=" * 110)

    # Prepare stratified evaluation subset (50 With_Coat, 40 Without_Coat, 40 OOD = 130 images)
    with_coat_all = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "With_Coat", "images", "*.jpg")))
    without_coat_all = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "Without_Coat", "images", "*.jpg")))
    ood_all = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "OOD", "images", "*.jpg")))

    eval_images = with_coat_all[20:70] + without_coat_all[20:60] + ood_all[10:50]
    print(f"Stratified evaluation subset: {len(eval_images)} images ({len(with_coat_all[20:70])} With_Coat, {len(without_coat_all[20:60])} Without_Coat, {len(ood_all[10:50])} OOD)")

    # Load images and GT boxes into RAM for rapid multi-config evaluation
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

    # Initialize Base Detector
    detector = HOIContactDetector(
        weapon_conf_thresh=0.15, # allow lowest sweep threshold
        interaction_radius=2.2,
        contact_thresh=0.50
    )

    # Define sweep grid configurations
    configs = [
        # 1. Baseline raw YOLO alone (Ablation: No Pose, No DINOv2)
        {"id": "YOLO_Conf0.20_Raw", "conf": 0.20, "radius": 9999.0, "contact": 0.0, "desc": "Raw YOLO (conf=0.20, No Gating)"},
        {"id": "YOLO_Conf0.45_Raw", "conf": 0.45, "radius": 9999.0, "contact": 0.0, "desc": "Raw YOLO (conf=0.45, Baseline)"},

        # 2. Ablation: Pose Spatial Gating Alone (No DINOv2)
        {"id": "SpatialOnly_R2.0", "conf": 0.20, "radius": 2.0, "contact": 0.0, "desc": "Spatial Hand Gate Only (R=2.0)"},
        {"id": "SpatialOnly_R2.5", "conf": 0.20, "radius": 2.5, "contact": 0.0, "desc": "Spatial Hand Gate Only (R=2.5)"},

        # 3. Ablation: DINOv2 Contact Gating Alone (No Pose Spatial Distance Gate)
        {"id": "ContactOnly_Tau0.50", "conf": 0.20, "radius": 9999.0, "contact": 0.50, "desc": "Contact Gate Only (tau=0.50)"},

        # 4. Full Dual-Stream HOI Pipeline Grid: Varying Interaction Radius R
        {"id": "HOI_R1.8_Tau0.50", "conf": 0.20, "radius": 1.8, "contact": 0.50, "desc": "Full HOI (R=1.8, tau=0.50)"},
        {"id": "HOI_R2.2_Tau0.50", "conf": 0.20, "radius": 2.2, "contact": 0.50, "desc": "Full HOI (R=2.2, tau=0.50) [Nominal]"},
        {"id": "HOI_R2.6_Tau0.50", "conf": 0.20, "radius": 2.6, "contact": 0.50, "desc": "Full HOI (R=2.6, tau=0.50)"},

        # 5. Full Dual-Stream HOI Pipeline Grid: Varying Contact Threshold tau
        {"id": "HOI_R2.2_Tau0.46", "conf": 0.20, "radius": 2.2, "contact": 0.46, "desc": "Full HOI (R=2.2, tau=0.46) [High Recall]"},
        {"id": "HOI_R2.2_Tau0.48", "conf": 0.20, "radius": 2.2, "contact": 0.48, "desc": "Full HOI (R=2.2, tau=0.48)"},
        {"id": "HOI_R2.2_Tau0.52", "conf": 0.20, "radius": 2.2, "contact": 0.52, "desc": "Full HOI (R=2.2, tau=0.52) [High Prec]"},

        # 6. Full Dual-Stream HOI Pipeline Grid: Sensitive Proposal Conf=0.15
        {"id": "HOI_Conf0.15_R2.2_Tau0.50", "conf": 0.15, "radius": 2.2, "contact": 0.50, "desc": "Full HOI (conf=0.15, R=2.2, tau=0.50)"},
    ]

    sweep_results = []

    print("\n" + "=" * 135)
    print(f"{'Config ID':<26} | {'Description':<38} | {'Overall F1':<10} | {'Recall':<8} | {'Precision':<9} | {'Accuracy':<8} | {'OOD Spec':<8} | {'OOD FP':<6}")
    print("-" * 135)

    for cfg in configs:
        cid = cfg["id"]
        c_conf = cfg["conf"]
        c_radius = cfg["radius"]
        c_contact = cfg["contact"]

        stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        t0 = time.perf_counter()

        for item in cached_data:
            img_bgr = item["img"]
            cat = item["cat"]
            gt_boxes = item["gt_boxes"]

            pred = detector.predict(
                img_bgr,
                weapon_conf_thresh=c_conf,
                interaction_radius=c_radius,
                contact_thresh=c_contact
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
            "radius": c_radius,
            "contact_thresh": c_contact,
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
