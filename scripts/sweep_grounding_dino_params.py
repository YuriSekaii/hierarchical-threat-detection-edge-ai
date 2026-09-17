"""
scripts/sweep_grounding_dino_params.py

Hyperparameter Sweep and Sensitivity Analysis for Grounding DINO 1.5 (Version 4.20).
Sweeps:
1. Box confidence threshold: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
2. Competing Negative Suppression: Active (Margin=0.0, 0.05) vs Disabled
3. Prompt formulations:
   - P1: Direct tokens ("knife . blade . dagger . cutter . bare hand . clenched fist . clothing seam . zipper . shadow .")
   - P2: Article-prefixed ("a knife . a blade . a dagger . a cutter . a bare hand . a clenched fist . a clothing seam . a zipper . a shadow .")
   - P3: Targeted weapon ("a knife . a blade . a dagger . a bare hand . a clenched fist . a clothing seam . a zipper . a shadow .")
   - P4: Positive only baseline ("a knife . a blade . a dagger . a cutter .")

Evaluates on stratified surveillance test images from data/Ground_Truth_Dataset/
and generates the mandatory tuning impact table for VERSIONS_PHASE2.md.
"""

import os
import sys
import glob
import time
import json
import argparse
from typing import Dict, List

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.inference_grounding_dino_v4_20 import GroundingDinoDetector
from scripts.validate_grounding_dino_ground_truth import (
    GT_DATASET_DIR, evaluate_dataset, get_category_from_path
)

SWEEP_OUTPUT_JSON = os.path.join(REPO_DIR, "results", "ground_truth_benchmark", "grounding_dino_sweep_results.json")


def run_sweep(stride: int = 5, max_images: int = None):
    image_files = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))
    if stride > 1:
        image_files = image_files[::stride]
    if max_images:
        image_files = image_files[:max_images]

    print(f"Running Grounding DINO parameter sweep on {len(image_files)} stratified test images (Stride: {stride})...")

    # Prompt variations
    prompts = {
        "P1_Direct": "knife . blade . dagger . cutter . bare hand . clenched fist . clothing seam . zipper . shadow .",
        "P2_Articles": "a knife . a blade . a dagger . a cutter . a bare hand . a clenched fist . a clothing seam . a zipper . a shadow .",
        "P3_Targeted": "a knife . a blade . a dagger . a bare hand . a clenched fist . a clothing seam . a zipper . a shadow .",
        "P4_PositiveOnly": "a knife . a blade . a dagger . a cutter .",
    }

    # Configurations to evaluate
    configurations = [
        # Impact of Box Threshold on P2 (Articles + Neg Suppression)
        {"name": "P2_Box0.15_NegSupp", "prompt": prompts["P2_Articles"], "box_thresh": 0.15, "text_thresh": 0.15, "neg_supp": True, "margin": 0.0},
        {"name": "P2_Box0.20_NegSupp", "prompt": prompts["P2_Articles"], "box_thresh": 0.20, "text_thresh": 0.20, "neg_supp": True, "margin": 0.0},
        {"name": "P2_Box0.25_NegSupp", "prompt": prompts["P2_Articles"], "box_thresh": 0.25, "text_thresh": 0.25, "neg_supp": True, "margin": 0.0},
        {"name": "P2_Box0.30_NegSupp", "prompt": prompts["P2_Articles"], "box_thresh": 0.30, "text_thresh": 0.25, "neg_supp": True, "margin": 0.0},
        {"name": "P2_Box0.35_NegSupp", "prompt": prompts["P2_Articles"], "box_thresh": 0.35, "text_thresh": 0.25, "neg_supp": True, "margin": 0.0},

        # Impact of Negative Suppression Ablation (P2 with vs without Neg Suppression)
        {"name": "P2_Box0.25_NoNegSupp", "prompt": prompts["P2_Articles"], "box_thresh": 0.25, "text_thresh": 0.25, "neg_supp": False, "margin": 0.0},
        {"name": "P2_Box0.25_NegMargin0.05", "prompt": prompts["P2_Articles"], "box_thresh": 0.25, "text_thresh": 0.25, "neg_supp": True, "margin": 0.05},

        # Impact of Prompt Formulation
        {"name": "P1_Direct_Box0.20", "prompt": prompts["P1_Direct"], "box_thresh": 0.20, "text_thresh": 0.20, "neg_supp": True, "margin": 0.0},
        {"name": "P3_Targeted_Box0.25", "prompt": prompts["P3_Targeted"], "box_thresh": 0.25, "text_thresh": 0.25, "neg_supp": True, "margin": 0.0},
        {"name": "P4_PosOnly_Box0.25", "prompt": prompts["P4_PositiveOnly"], "box_thresh": 0.25, "text_thresh": 0.25, "neg_supp": False, "margin": 0.0},
    ]

    detector = GroundingDinoDetector()

    sweep_results = []

    print("\n" + "=" * 115)
    print(f"{'Config Name':24s} | {'Overall F1':10s} | {'Recall':8s} | {'Precision':9s} | {'Accuracy':8s} | {'OOD Spec':9s} | {'OOD FP':7s} | {'TP/FP/FN/TN'}")
    print("=" * 115)

    for cfg in configurations:
        t0 = time.time()
        res = evaluate_dataset(
            detector=detector,
            image_files=image_files,
            prompt=cfg["prompt"],
            box_thresh=cfg["box_thresh"],
            text_thresh=cfg["text_thresh"],
            neg_margin=cfg["margin"],
            enable_negative_suppression=cfg["neg_supp"],
            nms_thresh=0.50,
            iou_thresh=0.20,
            verbose=False
        )
        elapsed = time.time() - t0

        ov = res["Overall"]
        ood = res["OOD"]

        cm_str = f"{ov['tp']}/{ov['fp']}/{ov['fn']}/{ov['tn']}"
        print(f"{cfg['name']:24s} | {ov['f1_score']*100:9.2f}% | {ov['recall']*100:7.2f}% | {ov['precision']*100:8.2f}% | {ov['accuracy']*100:7.2f}% | {ood['specificity']*100:8.2f}% | {ood['fp']:7d} | {cm_str}")

        sweep_results.append({
            "config": cfg,
            "overall_f1": ov["f1_score"],
            "recall": ov["recall"],
            "precision": ov["precision"],
            "accuracy": ov["accuracy"],
            "ood_specificity": ood["specificity"],
            "ood_fp": ood["fp"],
            "with_coat_f1": res["With_Coat"]["f1_score"],
            "without_coat_f1": res["Without_Coat"]["f1_score"],
            "latency_ms": ov["avg_latency_ms"],
            "metrics_by_category": res,
            "eval_time_sec": round(elapsed, 1)
        })

    print("=" * 115 + "\n")

    os.makedirs(os.path.dirname(SWEEP_OUTPUT_JSON), exist_ok=True)
    with open(SWEEP_OUTPUT_JSON, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Sweep results successfully saved to: {SWEEP_OUTPUT_JSON}")
    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5, help="Sampling stride for sweep")
    parser.add_argument("--max-images", type=int, default=None, help="Max images for sweep")
    args = parser.parse_args()
    run_sweep(stride=args.stride, max_images=args.max_images)
