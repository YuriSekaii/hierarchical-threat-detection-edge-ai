"""
scripts/compare_champions_live.py

Live Hardware Profiling of Champion 1 vs Champion 2 Cascades.
Directly measures:
1. True live wall-time per tier using synchronized CUDA events (torch.cuda.Event).
2. Exact execution time for Tier 1 only, Tier 1+2, and Tier 1+2+3.
3. Live Accuracy and F1 score for both champions.
Zero hardcoding, pure hardware measurement.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset, PoseConv3DHeatmapDataset
import importlib.util
spec = importlib.util.spec_from_file_location(
    "inference_staircase_cascade_v5_10",
    os.path.join(REPO_DIR, "src", "inference_staircase_cascade_v5.10.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
StaircaseCascadeEngineV510 = mod.StaircaseCascadeEngineV510

def main():
    print("=" * 90)
    print("   LIVE HARDWARE BENCHMARK: CHAMPION 1 vs CHAMPION 2")
    print("   Evaluating True Execution Latency on CUDA with Zero Hardcoding")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir, val_root = resolve_data_paths()
    val_categories = ["With_Coat", "Without_Coat", "OOD"]
    video_records = []
    y_true_list = []

    print("[DATA] Loading 77 validation videos into GPU memory...")
    for cat in val_categories:
        cat_dir = os.path.join(val_root, cat)
        coord_ds = ActionDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, mode="test")
        hm_ds = PoseConv3DHeatmapDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, heatmap_size=56, mode="test")
        v_coords, v_hms = defaultdict(list), defaultdict(list)
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
                "coords": torch.stack(v_coords[src]).to(device),
                "heatmaps": torch.stack(v_hms[src]).to(device)
            })
            y_true_list.append(gt)

    y_true = np.array(y_true_list)
    print(f"[DATA] Successfully cached {len(video_records)} videos in {device} memory.")

    # Instantiate both engines
    # Champion 1: pc3d_tier3 -> stgcn -> pc3d_dt3
    # pl1=0.20, ph1=0.99, pl2=0.35, ph2=0.75, tau3=0.40
    print("\n[INIT] Initializing Champion 1: pc3d_tier3 -> stgcn -> pc3d_dt3...")
    eng1 = StaircaseCascadeEngineV510(
        tier1_key="pc3d_tier3",
        tier2_key="stgcn",
        tier3_key="pc3d_dt3",
        p_low1=0.20,
        p_high1=0.99,
        p_low2=0.35,
        p_high2=0.75,
        tau3=0.40,
        device=device
    )

    # Champion 2: pc3d_tier3 -> pc3d_dt3 -> stgcn
    # pl1=0.20, ph1=0.99, pl2=0.35, ph2=0.95, tau3=0.55
    print("[INIT] Initializing Champion 2: pc3d_tier3 -> pc3d_dt3 -> stgcn...")
    eng2 = StaircaseCascadeEngineV510(
        tier1_key="pc3d_tier3",
        tier2_key="pc3d_dt3",
        tier3_key="stgcn",
        p_low1=0.20,
        p_high1=0.99,
        p_low2=0.35,
        p_high2=0.95,
        tau3=0.55,
        device=device
    )

    # GPU Warmup
    print("[WARMUP] Warming up CUDA pipelines (10 iterations)...")
    for _ in range(10):
        eng1.predict_video(video_records[0]["coords"], video_records[0]["heatmaps"])
        eng2.predict_video(video_records[0]["coords"], video_records[0]["heatmaps"])
    torch.cuda.synchronize()

    # Benchmark Function
    def run_live_eval(engine, name, num_repeats=5):
        print(f"\n>> Benchmarking {name} ({num_repeats} passes over all 77 videos)...")
        all_pass_times = []
        all_resolved_tiers = []
        all_predictions = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for rep in range(num_repeats):
            rep_times = []
            rep_tiers = []
            rep_preds = []

            for rec in video_records:
                torch.cuda.synchronize()
                start_event.record()
                res = engine.predict_video(rec["coords"], rec["heatmaps"])
                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)
                rep_times.append(elapsed_ms)
                rep_tiers.append(res["resolved_tier"])
                rep_preds.append(res["verdict"])

            all_pass_times.append(rep_times)
            if rep == 0:
                all_resolved_tiers = rep_tiers
                all_predictions = rep_preds

        # Compute summary
        mean_video_times = np.mean(all_pass_times, axis=0)
        overall_mean_ms = np.mean(mean_video_times)
        overall_median_ms = np.median(mean_video_times)

        # Metrics
        y_p = np.array(all_predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_p, average="binary")
        acc = accuracy_score(y_true, y_p)
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()

        tiers_arr = np.array(all_resolved_tiers)
        t1_count = int(np.sum(tiers_arr == 1))
        t2_count = int(np.sum(tiers_arr == 2))
        t3_count = int(np.sum(tiers_arr == 3))

        # Time breakdown by resolution tier
        t1_times = mean_video_times[tiers_arr == 1]
        t2_times = mean_video_times[tiers_arr == 2]
        t3_times = mean_video_times[tiers_arr == 3]

        return {
            "name": name,
            "overall_mean_ms": float(overall_mean_ms),
            "overall_median_ms": float(overall_median_ms),
            "f1": float(f1),
            "recall": float(rec * 100.0),
            "precision": float(prec * 100.0),
            "accuracy": float(acc * 100.0),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "t1_count": int(t1_count),
            "t2_count": int(t2_count),
            "t3_count": int(t3_count),
            "pct_esc1": float(((t2_count + t3_count) / len(y_true)) * 100.0),
            "pct_esc2": float((t3_count / len(y_true)) * 100.0),
            "t1_mean_ms": float(np.mean(t1_times)) if len(t1_times) > 0 else 0.0,
            "t2_mean_ms": float(np.mean(t2_times)) if len(t2_times) > 0 else 0.0,
            "t3_mean_ms": float(np.mean(t3_times)) if len(t3_times) > 0 else 0.0,
        }

    res1 = run_live_eval(eng1, "Champion 1 (pc3d_tier3 -> stgcn -> pc3d_dt3)")
    res2 = run_live_eval(eng2, "Champion 2 (pc3d_tier3 -> pc3d_dt3 -> stgcn)")

    print("\n" + "=" * 90)
    print("   DIRECT HEAD-TO-HEAD HARDWARE COMPARISON (CUDA EVENTS)")
    print("=" * 90)
    print(f"{'Metric':<35} | {'Champion 1':<25} | {'Champion 2':<25}")
    print("-" * 90)
    print(f"{'Sequence':<35} | {'pc3d_t3 -> stgcn -> dt3':<25} | {'pc3d_t3 -> dt3 -> stgcn':<25}")
    print(f"{'F1-Score':<35} | {res1['f1']:.4f} {'':<19} | {res2['f1']:.4f} {'':<19}")
    print(f"{'Recall / Precision':<35} | {res1['recall']:.1f}% / {res1['precision']:.1f}% {'':<11} | {res2['recall']:.1f}% / {res2['precision']:.1f}% {'':<11}")
    print(f"{'False Positives / Negatives':<35} | {res1['fp']} FP / {res1['fn']} FN {'':<15} | {res2['fp']} FP / {res2['fn']} FN {'':<15}")
    print("-" * 90)
    print(f"{'Tier 1 Resolved (Early Exit)':<35} | {res1['t1_count']} / 77 ({res1['t1_count']/77*100:.1f}%) {'':<11} | {res2['t1_count']} / 77 ({res2['t1_count']/77*100:.1f}%) {'':<11}")
    print(f"{'Tier 2 Escalation (% ESC_1)':<35} | {res1['pct_esc1']:.2f}% {'':<17} | {res2['pct_esc1']:.2f}% {'':<17}")
    print(f"{'Tier 3 Escalation (% ESC_2)':<35} | {res1['pct_esc2']:.2f}% (only {res1['t3_count']} vids) {'':<4} | {res2['pct_esc2']:.2f}% ({res2['t3_count']} vids) {'':<9}")
    print("-" * 90)
    print(f"{'Live Tier 1 Mean Latency':<35} | {res1['t1_mean_ms']:.2f} ms {'':<17} | {res2['t1_mean_ms']:.2f} ms {'':<17}")
    print(f"{'Live Tier 2 (Escalated) Mean':<35} | {res1['t2_mean_ms']:.2f} ms (runs ST-GCN) {'':<4} | {res2['t2_mean_ms']:.2f} ms (runs DT-3) {'':<5}")
    print(f"{'Live Tier 3 (Escalated) Mean':<35} | {res1['t3_mean_ms']:.2f} ms (runs DT-3) {'':<6} | {res2['t3_mean_ms']:.2f} ms (runs ST-GCN) {'':<4}")
    print("-" * 90)
    print(f"{'Overall Mean Action Latency':<35} | {res1['overall_mean_ms']:.2f} ms / video {'':<8} | {res2['overall_mean_ms']:.2f} ms / video {'':<8}")
    print(f"{'Overall Median Action Latency':<35} | {res1['overall_median_ms']:.2f} ms / video {'':<8} | {res2['overall_median_ms']:.2f} ms / video {'':<8}")
    print("=" * 90)

    # Save to JSON
    out_json = os.path.join(REPO_DIR, "results", "tuning", "compare_champions_live.json")
    with open(out_json, "w") as f:
        json.dump({"champion1": res1, "champion2": res2}, f, indent=2)
    print(f"\n[SAVED] Saved live comparison results to: {out_json}")

if __name__ == "__main__":
    main()
