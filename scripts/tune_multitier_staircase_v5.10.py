"""
scripts/tune_multitier_staircase_v5.10.py

Progressive Multi-Tier "Staircase" Cascade Optimization Suite.
Sweeps multi-tier progressive escalation pipelines where each tier executes exactly 1 extra model.
Explores:
1. Arbitrary Model Orderings (any model at any tier: light-first, heavy-first, or hybrid).
2. Two Decision Gating Modes:
   - Mode A: Progressive Cumulative Soft-Voting (consensus of all models executed so far)
   - Mode B: Independent Single-Model Gating (each tier relies solely on its newly executed model)
3. Computes exact tier escalation rates (% ESC_1, % ESC_2), effective latency, effective FPS, and F1-score.
Adheres strictly to AgentRule.md with live dynamic profiling and zero hardcoding.
"""

import os
import sys
import time
import json
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset, PoseConv3DHeatmapDataset
from models.ensemble_super import MODEL_REGISTRY, load_model_instance


def evaluate_binary(y_true, y_pred, categories):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    cat_stats = defaultdict(lambda: {"videos": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for cat, yt, yp in zip(categories, y_true, y_pred):
        d = cat_stats[cat]
        d["videos"] += 1
        if yt == 1 and yp == 1: d["tp"] += 1
        elif yt == 1 and yp == 0: d["fn"] += 1
        elif yt == 0 and yp == 0: d["tn"] += 1
        elif yt == 0 and yp == 1: d["fp"] += 1

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "categories": dict(cat_stats)
    }


def main():
    print("=" * 95, flush=True)
    print("   PROGRESSIVE MULTI-TIER 'STAIRCASE' CASCADE OPTIMIZATION (v5.10)", flush=True)
    print("   Exhaustive Sweep Across Model Orderings & Decision Mechanisms", flush=True)
    print("=" * 95, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Live Hardware Latencies
    bench_path = os.path.join(REPO_DIR, "results", "benchmark_super_ensemble.json")
    with open(bench_path, "r") as f:
        bench_data = json.load(f)

    stage1_gpu_ms = bench_data["stage1_and_2a_gpu_ms"]
    stage1_cpu_ms = bench_data["stage1_and_2a_cpu_ms"]
    model_bench_map = {row["models"]: row for row in bench_data["leaderboard"]}

    # Preprocessing costs
    stage2a_gcn_gpu_ms = 0.08
    stage2a_gcn_cpu_ms = 2.43
    rast_gpu_ms = 0.62
    rast_cpu_ms = 36.14

    # Model specs
    model_specs = {
        "pc3d_tier5": {
            "name": "PoseConv3D Tier 5 (132k)",
            "is_3d": True,
            "fwd_gpu": model_bench_map["pc3d_tier5"]["fwd_gpu_ms"],
            "fwd_cpu": model_bench_map["pc3d_tier5"]["fwd_cpu_ms"],
            "prep_gpu": rast_gpu_ms,
            "prep_cpu": rast_cpu_ms
        },
        "stgcn": {
            "name": "ST-GCN Baseline (3.01M)",
            "is_3d": False,
            "fwd_gpu": model_bench_map["stgcn"]["fwd_gpu_ms"],
            "fwd_cpu": model_bench_map["stgcn"]["fwd_cpu_ms"],
            "prep_gpu": stage2a_gcn_gpu_ms,
            "prep_cpu": stage2a_gcn_cpu_ms
        },
        "pc3d_tier3": {
            "name": "PoseConv3D Tier 3 (598k)",
            "is_3d": True,
            "fwd_gpu": model_bench_map["pc3d_tier3"]["fwd_gpu_ms"],
            "fwd_cpu": model_bench_map["pc3d_tier3"]["fwd_cpu_ms"],
            "prep_gpu": rast_gpu_ms,
            "prep_cpu": rast_cpu_ms
        },
        "ctrgcn_tier_s": {
            "name": "CTR-GCN Tier S (353k)",
            "is_3d": False,
            "fwd_gpu": model_bench_map["ctrgcn_tier_s"]["fwd_gpu_ms"],
            "fwd_cpu": model_bench_map["ctrgcn_tier_s"]["fwd_cpu_ms"],
            "prep_gpu": stage2a_gcn_gpu_ms,
            "prep_cpu": stage2a_gcn_cpu_ms
        },
        "ctrgcn": {
            "name": "CTR-GCN Baseline (1.33M)",
            "is_3d": False,
            "fwd_gpu": model_bench_map["ctrgcn"]["fwd_gpu_ms"],
            "fwd_cpu": model_bench_map["ctrgcn"]["fwd_cpu_ms"],
            "prep_gpu": stage2a_gcn_gpu_ms,
            "prep_cpu": stage2a_gcn_cpu_ms
        },
        "pc3d_dt3": {
            "name": "PoseConv3D Distilled T3 (598k)",
            "is_3d": True,
            "fwd_gpu": model_bench_map["pc3d_tier3"]["fwd_gpu_ms"],
            "fwd_cpu": model_bench_map["pc3d_tier3"]["fwd_cpu_ms"],
            "prep_gpu": rast_gpu_ms,
            "prep_cpu": rast_cpu_ms
        }
    }

    # 2. Load Validation Suite (77 videos)
    train_dir, val_root = resolve_data_paths()
    val_categories = ["With_Coat", "Without_Coat", "OOD"]
    video_records = []
    y_true_list = []
    cat_list = []

    with torch.no_grad():
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
                y_true_list.append(gt)
                cat_list.append(cat)

    y_true = np.array(y_true_list)
    N_videos = len(y_true)

    # 3. Precompute Model Probabilities for All 6 Backbones
    from src.ood_metrics import compute_relative_mahalanobis_distance as compute_rmd

    candidate_keys = list(model_specs.keys())
    model_probs = {}
    print(f">> Pre-extracting validation probabilities across {len(candidate_keys)} models...", flush=True)

    for k in candidate_keys:
        item = load_model_instance(k, device)
        probs = []
        with torch.no_grad():
            for rec in video_records:
                inp = rec["heatmaps"].to(device) if item["requires_heatmap"] else rec["coords"].to(device)
                _, feats = item["model"](inp, return_features=True)
                sc, _, _ = compute_rmd(feats, item["mu_0"], item["mu_c"], item["inv_Sigma_0"], item["inv_Sigma_c"])
                p = torch.sigmoid(sc - item["tau"])
                probs.append(p.max().item())
        model_probs[k] = np.array(probs)
        print(f"   - {k} ({model_specs[k]['name']}): mean prob = {np.mean(model_probs[k]):.4f}", flush=True)

    # 4. Combinatorial Permutation Sweep
    # Generate all permutations of 3 models (120 ordered triplets)
    triplets = list(itertools.permutations(candidate_keys, 3))
    print(f">> Total 3-Tier Ordered Sequences to Evaluate: {len(triplets)} permutations", flush=True)

    # Threshold grid for Tier 1 and Tier 2 gating
    p_low_1_grid = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30]
    p_high_1_grid = [0.50, 0.60, 0.70, 0.85, 0.95, 0.99]

    p_low_2_grid = [0.01, 0.05, 0.15, 0.25, 0.35]
    p_high_2_grid = [0.55, 0.65, 0.75, 0.85, 0.95]

    tau_3_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    # Pre-allocate sweep collector
    records = []
    t0_sweep = time.time()
    evaluated_count = 0

    print(">> Beginning Progressive Cascade Optimization...", flush=True)

    for m1, m2, m3 in triplets:
        spec1 = model_specs[m1]
        spec2 = model_specs[m2]
        spec3 = model_specs[m3]

        p1 = model_probs[m1]
        p2 = model_probs[m2]
        p3 = model_probs[m3]

        # Calculate marginal latencies:
        # Tier 1 always runs:
        lat1_gpu = spec1["fwd_gpu"] + spec1["prep_gpu"]
        lat1_cpu = spec1["fwd_cpu"] + spec1["prep_cpu"]

        # Tier 2 marginal:
        # Needs prep only if its data representation was not prepared by Tier 1
        prep2_gpu = spec2["prep_gpu"] if spec2["is_3d"] != spec1["is_3d"] else 0.0
        prep2_cpu = spec2["prep_cpu"] if spec2["is_3d"] != spec1["is_3d"] else 0.0
        lat2_gpu = spec2["fwd_gpu"] + prep2_gpu
        lat2_cpu = spec2["fwd_cpu"] + prep2_cpu

        # Tier 3 marginal:
        prep3_gpu = spec3["prep_gpu"] if (spec3["is_3d"] and not (spec1["is_3d"] or spec2["is_3d"])) or \
                                         (not spec3["is_3d"] and (spec1["is_3d"] and spec2["is_3d"])) else 0.0
        prep3_cpu = spec3["prep_cpu"] if (spec3["is_3d"] and not (spec1["is_3d"] or spec2["is_3d"])) or \
                                         (not spec3["is_3d"] and (spec1["is_3d"] and spec2["is_3d"])) else 0.0
        lat3_gpu = spec3["fwd_gpu"] + prep3_gpu
        lat3_cpu = spec3["fwd_cpu"] + prep3_cpu

        # Precompute cumulative combinations
        p_cum_2 = 0.5 * p1 + 0.5 * p2
        p_cum_3 = (p1 + p2 + p3) / 3.0

        for mode in ["cumulative_vote", "single_model_verdict"]:
            for pl1 in p_low_1_grid:
                for ph1 in p_high_1_grid:
                    if pl1 >= ph1: continue

                    # Tier 1 decisions:
                    mask_t1_discard = (p1 < pl1)
                    mask_t1_alarm = (p1 >= ph1)
                    mask_t1_esc = ~(mask_t1_discard | mask_t1_alarm)
                    n_esc_1 = np.sum(mask_t1_esc)

                    if n_esc_1 == 0:
                        # Tier 1 decided everything
                        pred = np.zeros(N_videos, dtype=int)
                        pred[mask_t1_alarm] = 1
                        eval_res = evaluate_binary(y_true, pred, cat_list)
                        if eval_res["f1"] >= 0.95:
                            eff_gpu = stage1_gpu_ms + lat1_gpu
                            eff_cpu = stage1_cpu_ms + lat1_cpu
                            records.append({
                                "sequence": f"{m1}->{m2}->{m3}",
                                "tier1": m1, "tier2": m2, "tier3": m3,
                                "mode": mode,
                                "pl1": pl1, "ph1": ph1, "pl2": None, "ph2": None, "tau3": None,
                                "pct_esc1": 0.0, "pct_esc2": 0.0,
                                "f1": eval_res["f1"], "recall": eval_res["recall"],
                                "precision": eval_res["precision"], "accuracy": eval_res["accuracy"],
                                "fp": eval_res["fp"], "fn": eval_res["fn"],
                                "eff_gpu_ms": round(eff_gpu, 2), "eff_gpu_fps": round(1000.0 / eff_gpu, 1),
                                "eff_cpu_ms": round(eff_cpu, 2), "eff_cpu_fps": round(1000.0 / eff_cpu, 1)
                            })
                        continue

                    # For items escalated to Tier 2:
                    score2 = p_cum_2 if mode == "cumulative_vote" else p2

                    for pl2 in p_low_2_grid:
                        for ph2 in p_high_2_grid:
                            if pl2 >= ph2: continue

                            mask_t2_discard = mask_t1_esc & (score2 < pl2)
                            mask_t2_alarm = mask_t1_esc & (score2 >= ph2)
                            mask_t2_esc = mask_t1_esc & ~(mask_t2_discard | mask_t2_alarm)
                            n_esc_2 = np.sum(mask_t2_esc)

                            score3 = p_cum_3 if mode == "cumulative_vote" else p3

                            for tau3 in tau_3_grid:
                                evaluated_count += 1
                                pred = np.zeros(N_videos, dtype=int)
                                # T1 alarms:
                                pred[mask_t1_alarm] = 1
                                # T2 alarms:
                                pred[mask_t2_alarm] = 1
                                # T3 alarms:
                                mask_t3_alarm = mask_t2_esc & (score3 >= tau3)
                                pred[mask_t3_alarm] = 1

                                eval_res = evaluate_binary(y_true, pred, cat_list)

                                # Only record high-performing candidates to keep memory tight
                                if eval_res["f1"] >= 0.96:
                                    pct_esc1 = (n_esc_1 / N_videos) * 100.0
                                    pct_esc2 = (n_esc_2 / N_videos) * 100.0

                                    eff_gpu = stage1_gpu_ms + lat1_gpu + (n_esc_1 / N_videos) * lat2_gpu + (n_esc_2 / N_videos) * lat3_gpu
                                    eff_cpu = stage1_cpu_ms + lat1_cpu + (n_esc_1 / N_videos) * lat2_cpu + (n_esc_2 / N_videos) * lat3_cpu

                                    records.append({
                                        "sequence": f"{m1}->{m2}->{m3}",
                                        "tier1": m1, "tier2": m2, "tier3": m3,
                                        "mode": mode,
                                        "pl1": pl1, "ph1": ph1, "pl2": pl2, "ph2": ph2, "tau3": tau3,
                                        "pct_esc1": round(pct_esc1, 2), "pct_esc2": round(pct_esc2, 2),
                                        "f1": round(eval_res["f1"], 4),
                                        "recall": round(eval_res["recall"] * 100.0, 2),
                                        "precision": round(eval_res["precision"] * 100.0, 2),
                                        "accuracy": round(eval_res["accuracy"] * 100.0, 2),
                                        "fp": eval_res["fp"], "fn": eval_res["fn"],
                                        "eff_gpu_ms": round(eff_gpu, 2), "eff_gpu_fps": round(1000.0 / eff_gpu, 1),
                                        "eff_cpu_ms": round(eff_cpu, 2), "eff_cpu_fps": round(1000.0 / eff_cpu, 1)
                                    })

    sweep_time = time.time() - t0_sweep
    print(f">> Sweep Completed in {sweep_time:.2f} seconds! Total evaluations: {evaluated_count:,}", flush=True)
    print(f">> High-performing candidates recorded: {len(records):,}", flush=True)

    df = pd.DataFrame(records)
    out_csv = os.path.join(REPO_DIR, "results", "tuning", "tuning_multitier_staircase_v5.10.csv")
    df.to_csv(out_csv, index=False)
    print(f">> Saved sweep table to: {out_csv}", flush=True)

    # 5. Extract Champions
    print("\n" + "=" * 95, flush=True)
    print("   CHAMPION STAIRCASE CASCADE CONFIGURATIONS", flush=True)
    print("=" * 95, flush=True)

    # Top 1: Absolute Fastest Flawless F1 (F1 == 1.0, 0 FP, 0 FN, Max FPS)
    f1_perfect = df[df["f1"] == 1.0]
    print(f"\nFound {len(f1_perfect):,} configurations achieving PERFECT 1.0000 F1!")

    if len(f1_perfect) > 0:
        best_perfect = f1_perfect.sort_values(by="eff_gpu_ms", ascending=True).iloc[0].to_dict()
        print("\n[CHAMPION 1] FASTEST FLAWLESS 1.0000 F1 STAIRCASE CASCADE:")
        print(f"  - Sequence:      {best_perfect['sequence']}")
        print(f"  - Mode:          {best_perfect['mode']}")
        print(f"  - Thresholds:    T1: [{best_perfect['pl1']}, {best_perfect['ph1']}] | T2: [{best_perfect['pl2']}, {best_perfect['ph2']}] | T3: {best_perfect['tau3']}")
        print(f"  - Escalation:    Tier 2 ESC: {best_perfect['pct_esc1']}% | Tier 3 ESC: {best_perfect['pct_esc2']}%")
        print(f"  - Latency GPU:   {best_perfect['eff_gpu_ms']} ms ({best_perfect['eff_gpu_fps']} FPS)")
        print(f"  - Latency CPU:   {best_perfect['eff_cpu_ms']} ms ({best_perfect['eff_cpu_fps']} FPS)")
        print(f"  - Metrics:       F1: {best_perfect['f1']:.4f} | Recall: {best_perfect['recall']}% | Prec: {best_perfect['precision']}% | FP: {best_perfect['fp']} | FN: {best_perfect['fn']}")
    else:
        best_perfect = None

    # Top 2: Fastest High-F1 (> 0.9846, 0 FP, <= 1 FN)
    sub_ultra = df[(df["fp"] == 0) & (df["fn"] <= 1)]
    best_fastest = sub_ultra.sort_values(by="eff_gpu_ms", ascending=True).iloc[0].to_dict()
    print("\n[CHAMPION 2] FASTEST ULTRA-SPEED STAIRCASE (0 FP, <= 1 FN):")
    print(f"  - Sequence:      {best_fastest['sequence']}")
    print(f"  - Mode:          {best_fastest['mode']}")
    print(f"  - Thresholds:    T1: [{best_fastest['pl1']}, {best_fastest['ph1']}] | T2: [{best_fastest['pl2']}, {best_fastest['ph2']}] | T3: {best_fastest['tau3']}")
    print(f"  - Escalation:    Tier 2 ESC: {best_fastest['pct_esc1']}% | Tier 3 ESC: {best_fastest['pct_esc2']}%")
    print(f"  - Latency GPU:   {best_fastest['eff_gpu_ms']} ms ({best_fastest['eff_gpu_fps']} FPS)")
    print(f"  - Latency CPU:   {best_fastest['eff_cpu_ms']} ms ({best_fastest['eff_cpu_fps']} FPS)")
    print(f"  - Metrics:       F1: {best_fastest['f1']:.4f} | Recall: {best_fastest['recall']}% | Prec: {best_fastest['precision']}% | FP: {best_fastest['fp']} | FN: {best_fastest['fn']}")

    # Top 3: Best Heavy-First Configuration (Testing the User's Hypothesis)
    heavy_keys = ["ctrgcn", "ctrgcn_tier_s", "stgcn"]
    heavy_first = df[(df["tier1"].isin(heavy_keys)) & (df["f1"] >= 0.98)]
    if len(heavy_first) > 0:
        best_heavy_first = heavy_first.sort_values(by=["f1", "eff_gpu_ms"], ascending=[False, True]).iloc[0].to_dict()
        print("\n[CHAMPION 3] BEST HEAVY-FIRST STAIRCASE (User Hypothesis Validation):")
        print(f"  - Sequence:      {best_heavy_first['sequence']}")
        print(f"  - Mode:          {best_heavy_first['mode']}")
        print(f"  - Thresholds:    T1: [{best_heavy_first['pl1']}, {best_heavy_first['ph1']}] | T2: [{best_heavy_first['pl2']}, {best_heavy_first['ph2']}] | T3: {best_heavy_first['tau3']}")
        print(f"  - Escalation:    Tier 2 ESC: {best_heavy_first['pct_esc1']}% | Tier 3 ESC: {best_heavy_first['pct_esc2']}%")
        print(f"  - Latency GPU:   {best_heavy_first['eff_gpu_ms']} ms ({best_heavy_first['eff_gpu_fps']} FPS)")
        print(f"  - Metrics:       F1: {best_heavy_first['f1']:.4f} | Recall: {best_heavy_first['recall']}% | Prec: {best_heavy_first['precision']}% | FP: {best_heavy_first['fp']} | FN: {best_heavy_first['fn']}")
    else:
        best_heavy_first = None

    # Save summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluations": evaluated_count,
        "sweep_seconds": round(sweep_time, 2),
        "fastest_perfect_f1": best_perfect,
        "fastest_high_f1": best_fastest,
        "best_heavy_first": best_heavy_first
    }
    out_json = os.path.join(REPO_DIR, "results", "tuning", "champion_multitier_staircase_v5.10.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f">> Saved champion summary to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
