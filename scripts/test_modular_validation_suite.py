"""
scripts/test_modular_validation_suite.py

Independent Test Suite for Modular Production Pipeline Verification.
Calls the exact modular components from src/ (Stage2bStaircaseEngine, Stage1WeaponDetector,
Stage2aPoseTracker, ThreadSafeCircularBuffer) and evaluates on the official 77-video validation dataset.

Capabilities:
1. Full 77-Video Suite Evaluation: Evaluates all 33 violent + 44 civilian videos to compute live F1-Score.
2. Random Video Sampling: Randomly selects N videos from the validation set to test inference and inspect verdicts.
3. Raw Video Stream Test: Feeds an actual raw .mp4 video file into the full end-to-end production pipeline.

Usage:
  python scripts/test_modular_validation_suite.py --all
  python scripts/test_modular_validation_suite.py --random 10
  python scripts/test_modular_validation_suite.py --feed-raw-video
"""

import os
import sys
import time
import glob
import random
import argparse
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.stage2b_staircase import Stage2bStaircaseEngine
from src.dataset import resolve_data_paths, ActionDataset, PoseConv3DHeatmapDataset
from src.inference_production_pipeline import ProductionHierarchicalPipeline


def load_validation_records(val_root):
    """Loads all 77 validation video records (coordinates and 3D heatmaps)."""
    val_categories = ["With_Coat", "Without_Coat", "OOD"]
    video_records = []

    for cat in val_categories:
        cat_dir = os.path.join(val_root, cat)
        coord_ds = ActionDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, mode="test")
        hm_ds = PoseConv3DHeatmapDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, heatmap_size=56, mode="test")

        v_coords = defaultdict(list)
        v_hms = defaultdict(list)

        for i in range(len(coord_ds)):
            c, _, src = coord_ds[i]
            v_coords[src].append(c)

        for i in range(len(hm_ds)):
            h, _, src = hm_ds[i]
            v_hms[src].append(h)

        for src in v_coords.keys():
            gt = 1 if cat in ["With_Coat", "Without_Coat"] else 0
            video_records.append({
                "src": src,
                "cat": cat,
                "gt": gt,
                "coords": torch.stack(v_coords[src]),
                "heatmaps": torch.stack(v_hms[src])
            })

    return video_records


def run_benchmark(records, engine, title="MODULAR VALIDATION SUITE"):
    """Evaluates the records through Stage2bStaircaseEngine and computes metrics."""
    print("\n" + "=" * 90)
    print(f"   EVALUATING {title}")
    print(f"   Total Videos to Test: {len(records)}")
    print("=" * 90)

    y_true = []
    y_pred = []
    resolved_tiers = []
    latencies = []

    t_start = time.perf_counter()

    for idx, rec in enumerate(records):
        res = engine.evaluate_video(rec["coords"], rec["heatmaps"])
        y_true.append(rec["gt"])
        y_pred.append(res["verdict"])
        resolved_tiers.append(res["resolved_tier"])
        latencies.append(res["total_video_ms"])

        tag = "VIOLENCE" if rec["gt"] == 1 else "CIVILIAN"
        pred_tag = "ALARM" if res["verdict"] == 1 else "CLEARED"
        correct = (rec["gt"] == res["verdict"])
        status = "OK" if correct else "MISMATCH"
        print(f"[{idx+1:2d}/{len(records):2d}] {rec['cat']:<12} | GT: {tag:<8} -> Pred: {pred_tag:<7} "
              f"| Tier {res['resolved_tier']} (Conf: {res['confidence']:.4f}) | Latency: {res['total_video_ms']:.2f} ms | [{status}]")

    total_time = time.perf_counter() - t_start
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    resolved_tiers = np.array(resolved_tiers)

    # Compute metrics
    prec, rec_score, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    n_total = len(y_true)
    n_t1 = np.sum(resolved_tiers == 1)
    n_t2 = np.sum(resolved_tiers == 2)
    n_t3 = np.sum(resolved_tiers == 3)

    print("\n" + "-" * 90)
    print("   EMPIRICAL VALIDATION METRICS (Modular Stage2bStaircaseEngine)")
    print("-" * 90)
    print(f"Total Videos:         {n_total} (Violent: {tp+fn}, Civilian: {tn+fp})")
    print(f"F1-Score:             {f1:.4f}  ({'*** PERFECT 1.0000 ***' if f1 == 1.0 else 'F1 < 1.0'})")
    print(f"Accuracy:             {acc * 100.0:.2f}%")
    print(f"Threat Recall:        {rec_score * 100.0:.2f}% ({tp}/{tp+fn} attacks caught)")
    print(f"Civilian Precision:   {prec * 100.0:.2f}% ({tp}/{tp+fp} true attacks)")
    print(f"False Positives (FP): {fp} (False alarms on civilian clips)")
    print(f"False Negatives (FN): {fn} (Missed violent attacks)")
    print("-" * 90)
    print("   CASCADE RESOLUTION BREAKDOWN")
    print("-" * 90)
    print(f"Resolved at Tier 1 (PoseConv3D Tier 3):    {n_t1:2d} / {n_total} ({n_t1/n_total*100.0:.1f}%)")
    print(f"Resolved at Tier 2 (PoseConv3D Distilled): {n_t2:2d} / {n_total} ({n_t2/n_total*100.0:.1f}%)")
    print(f"Escalated to Tier 3 (ST-GCN Baseline):     {n_t3:2d} / {n_total} ({n_t3/n_total*100.0:.1f}%)")
    print(f"Mean Action Latency per Video:            {np.mean(latencies):.2f} ms")
    print(f"Total Evaluation Elapsed Time:            {total_time:.2f} s")
    print("=" * 90)

    return f1, acc, tp, fp, fn, tn


def test_raw_video_file(val_root, engine_pipeline=None):
    """Picks a random raw .mp4 video from the dataset and feeds it to the production pipeline."""
    video_files = glob.glob(os.path.join(val_root, "**", "*.mp4"), recursive=True)
    if not video_files:
        print(f"[WARN] No raw .mp4 files found in {val_root}")
        return

    chosen_video = random.choice(video_files)
    print(f"\n[RAW FEED TEST] Randomly Selected Raw Video: {chosen_video}")
    rel_path = os.path.relpath(chosen_video, val_root)
    print(f"                 Relative Path: {rel_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("[CRITICAL] Modular test suite strictly requires an NVIDIA CUDA GPU.")
    pipeline = ProductionHierarchicalPipeline(device="cuda:0")
    print(f"[RAW FEED TEST] Running end-to-end pipeline in headless mode...")
    pipeline.run(source=chosen_video, headless=True)
    print(f"[RAW FEED TEST] Execution completed successfully for {rel_path}!")


def main():
    parser = argparse.ArgumentParser(description="Test Modular Production Pipeline & Verify F1 Score")
    parser.add_argument("--all", action="store_true", default=True, help="Evaluate on all 77 validation videos")
    parser.add_argument("--random", type=int, default=None, help="Sample N random videos instead of all 77")
    parser.add_argument("--feed-raw-video", action="store_true", help="Pick a random raw .mp4 and feed to production pipeline")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device: strictly 'cuda:0'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("[CRITICAL] Modular test suite strictly requires an NVIDIA CUDA GPU.")
    device = torch.device(args.device)
    print(f"[INIT] Initializing Stage2bStaircaseEngine on {device}...")
    engine = Stage2bStaircaseEngine(device=device)

    train_dir, val_root = resolve_data_paths()
    print(f"[INIT] Resolved Validation Directory: {val_root}")

    if args.feed_raw_video:
        test_raw_video_file(val_root)
        return

    all_records = load_validation_records(val_root)

    if args.random is not None and args.random > 0:
        random.seed(args.seed)
        sampled_records = random.sample(all_records, min(args.random, len(all_records)))
        run_benchmark(sampled_records, engine, title=f"RANDOM SAMPLE OF {len(sampled_records)} VIDEOS (Seed={args.seed})")
    else:
        run_benchmark(all_records, engine, title="FULL 77-VIDEO VALIDATION SUITE (33 Violent + 44 Civilian)")


if __name__ == "__main__":
    main()
