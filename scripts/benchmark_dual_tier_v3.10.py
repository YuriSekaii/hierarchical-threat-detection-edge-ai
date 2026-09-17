"""
scripts/benchmark_dual_tier_v3.10.py

Calibration and Comprehensive Benchmark Suite for Version 3.10:
Dual-Tier Edge & Server Architecture Ensembles.

Evaluates:
1. Tier 1 (Edge Autonomous Screening: ST-GCN + RMD v1.08 champion).
2. Tier 2 (Server Multi-Model Ensemble: ST-GCN + CTR-GCN + PoseConv3D).
3. Hierarchical Dual-Tier System (Edge Screening with conditional Server Escalation).

Standardized 148-Clip Action Recognition & OOD Validation Suite (77 video items):
- With_Coat: 16 violent assault videos (32 clips)
- Without_Coat: 17 violent assault videos (34 clips)
- OOD: 44 civilian movement videos (82 clips)

Archives:
- results/benchmark_dual_tier_v3.10.json
- results/tuning/tuning_dual_tier_v3.10.csv
- results/action_recognition_benchmark/validation_results_dual_tier_v3.10.csv
- weights/reference_data_dual_tier_v3.10.pt
"""

import os
import sys
import time
import json
import importlib.util
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset, PoseConv3DHeatmapDataset
from models.ensemble_dual_tier import (
    DualTierSurveillanceSystem,
    Tier1EdgeEngine,
    Tier2ServerEnsemble,
    compute_relative_mahalanobis_distance,
)


def evaluate_binary_predictions(y_true, y_pred, video_cats):
    """Calculates accuracy, precision, recall, f1, and confusion matrix."""
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    category_results = defaultdict(lambda: {"videos": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for cat, yt, yp in zip(video_cats, y_true, y_pred):
        d = category_results[cat]
        d["videos"] += 1
        if yt == 1 and yp == 1:
            d["tp"] += 1
        elif yt == 1 and yp == 0:
            d["fn"] += 1
        elif yt == 0 and yp == 0:
            d["tn"] += 1
        elif yt == 0 and yp == 1:
            d["fp"] += 1

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "category_breakdown": dict(category_results)
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("   BENCHMARK: DUAL-TIER EDGE & SERVER ARCHITECTURE ENSEMBLES (Version 3.10)", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    train_dir, val_root = resolve_data_paths()
    val_categories = ["With_Coat", "Without_Coat", "OOD"]

    # Initialize Dual-Tier System
    system = DualTierSurveillanceSystem(device=device)

    # 1. Single-Pass Validation Feature Caching for all 3 Backbones
    print("\n--- Step 1: Extracting Multi-Backbone Validation Features (77 Video Items) ---", flush=True)
    val_videos = []
    
    # We load standard ActionDataset for coordinates, and PoseConv3DHeatmapDataset for heatmaps
    with torch.no_grad():
        for cat in val_categories:
            cat_dir = os.path.join(val_root, cat)
            coord_ds = ActionDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, mode="test")
            hm_ds = PoseConv3DHeatmapDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, heatmap_size=56, mode="test")

            # Group clips by video source
            video_coords = defaultdict(list)
            video_hms = defaultdict(list)

            for i in range(len(coord_ds)):
                c_clip, _, src = coord_ds[i]
                video_coords[src].append(c_clip)

            for i in range(len(hm_ds)):
                h_clip, _, src = hm_ds[i]
                video_hms[src].append(h_clip)

            for src in video_coords.keys():
                c_tensor = torch.stack(video_coords[src]).to(device) # (N_clips, 2, 50, 17)
                h_tensor = torch.stack(video_hms[src]).to(device)    # (N_clips, 17, 50, 56, 56)

                # ST-GCN Features
                _, f_stgcn = system.tier2_server.stgcn(c_tensor, return_features=True)
                # CTR-GCN Features
                _, f_ctrgcn = system.tier2_server.ctrgcn(c_tensor, return_features=True)
                # PoseConv3D Features
                _, f_pc3d = system.tier2_server.poseconv3d(h_tensor, return_features=True)

                # Calculate individual RMD scores per clip
                s_stgcn, _, _ = compute_relative_mahalanobis_distance(
                    f_stgcn, system.tier2_server.stgcn_mu_0, system.tier2_server.stgcn_mu_c,
                    system.tier2_server.stgcn_inv_Sigma_0, system.tier2_server.stgcn_inv_Sigma_c
                )
                s_ctrgcn, _, _ = compute_relative_mahalanobis_distance(
                    f_ctrgcn, system.tier2_server.ctrgcn_mu_0, system.tier2_server.ctrgcn_mu_c,
                    system.tier2_server.ctrgcn_inv_Sigma_0, system.tier2_server.ctrgcn_inv_Sigma_c
                )
                s_pc3d, _, _ = compute_relative_mahalanobis_distance(
                    f_pc3d, system.tier2_server.pc3d_mu_0, system.tier2_server.pc3d_mu_c,
                    system.tier2_server.pc3d_inv_Sigma_0, system.tier2_server.pc3d_inv_Sigma_c
                )

                # Calibrated probabilities per clip
                p_stgcn = torch.sigmoid(s_stgcn - system.tier2_server.tau_stgcn)
                p_ctrgcn = torch.sigmoid(s_ctrgcn - system.tier2_server.tau_ctrgcn)
                p_pc3d = torch.sigmoid(s_pc3d - system.tier2_server.tau_pc3d)

                val_videos.append({
                    "src": src,
                    "cat": cat,
                    "label": 0 if cat == "OOD" else 1,
                    "p_stgcn_max": p_stgcn.max().item(),
                    "p_ctrgcn_max": p_ctrgcn.max().item(),
                    "p_pc3d_max": p_pc3d.max().item(),
                    "s_stgcn_max": s_stgcn.max().item(),
                    "s_ctrgcn_max": s_ctrgcn.max().item(),
                    "s_pc3d_max": s_pc3d.max().item(),
                    "p_stgcn_clips": p_stgcn.cpu(),
                    "p_ctrgcn_clips": p_ctrgcn.cpu(),
                    "p_pc3d_clips": p_pc3d.cpu(),
                })

    y_true = np.array([v["label"] for v in val_videos])
    video_cats = [v["cat"] for v in val_videos]
    print(f"Cached {len(val_videos)} video items (33 violent: 16 With_Coat, 17 Without_Coat; 44 OOD).", flush=True)

    # 2. Benchmark Tier 1 (Edge Autonomous Screening - ST-GCN + RMD)
    print("\n--- Step 2: Evaluating Tier 1 Standalone (Edge Autonomous ST-GCN + RMD) ---", flush=True)
    tier1_scores = np.array([v["s_stgcn_max"] for v in val_videos])
    tier1_preds = (tier1_scores >= system.tier1_edge.optimal_threshold).astype(int)
    res_tier1 = evaluate_binary_predictions(y_true, tier1_preds, video_cats)
    print(f"Tier 1 Standalone: F1={res_tier1['f1']:.4f} | Recall={res_tier1['recall']*100:.1f}% | "
          f"Precision={res_tier1['precision']*100:.1f}% | Accuracy={res_tier1['accuracy']*100:.1f}% | "
          f"FP={res_tier1['fp']} | FN={res_tier1['fn']}", flush=True)

    # 3. Hyperparameter Tuning Part A: Server Ensemble Weighting Sweep
    print("\n--- Step 3: Hyperparameter Tuning - Server Ensemble Weighting Sweep ---", flush=True)
    weight_candidates = [
        [0.50, 0.35, 0.15],
        [0.45, 0.35, 0.20],
        [0.40, 0.40, 0.20],
        [0.35, 0.35, 0.30],
        [0.55, 0.25, 0.20],
        [0.60, 0.20, 0.20],
        [0.33, 0.33, 0.34],
        [0.40, 0.30, 0.30],
        [0.25, 0.50, 0.25],
        [0.20, 0.40, 0.40],
    ]

    tuning_records = []
    best_ens_f1 = 0
    best_ens_weights = None
    best_ens_tau = 0.50
    best_ens_res = None

    for w_cand in weight_candidates:
        w_norm = np.array(w_cand) / sum(w_cand)
        # Compute video ensemble probability
        ens_probs = np.array([
            w_norm[0] * v["p_stgcn_max"] + w_norm[1] * v["p_ctrgcn_max"] + w_norm[2] * v["p_pc3d_max"]
            for v in val_videos
        ])

        # Grid search threshold tau_server
        for tau_s in np.linspace(0.35, 0.70, 71):
            y_pred = (ens_probs >= tau_s).astype(int)
            res = evaluate_binary_predictions(y_true, y_pred, video_cats)
            
            tuning_records.append({
                "type": "server_ensemble_weight_sweep",
                "w_stgcn": round(w_norm[0], 3),
                "w_ctrgcn": round(w_norm[1], 3),
                "w_pc3d": round(w_norm[2], 3),
                "tau_server": round(tau_s, 4),
                "recall": round(res["recall"] * 100, 2),
                "precision": round(res["precision"] * 100, 2),
                "accuracy": round(res["accuracy"] * 100, 2),
                "f1": round(res["f1"], 4),
                "fp": res["fp"],
                "fn": res["fn"]
            })

            # Prefer 100% recall (0 FN), then maximize F1
            is_better = False
            if best_ens_res is None:
                is_better = True
            elif res["fn"] < best_ens_res["fn"]:
                is_better = True
            elif res["fn"] == best_ens_res["fn"] and res["f1"] > best_ens_f1:
                is_better = True

            if is_better:
                best_ens_f1 = res["f1"]
                best_ens_weights = w_norm
                best_ens_tau = tau_s
                best_ens_res = res

    print(f"Optimal Server Ensemble Weights: ST-GCN={best_ens_weights[0]:.2f}, CTR-GCN={best_ens_weights[1]:.2f}, PoseConv3D={best_ens_weights[2]:.2f}", flush=True)
    print(f"Optimal Server Threshold: tau={best_ens_tau:.4f} | F1={best_ens_res['f1']:.4f} | Recall={best_ens_res['recall']*100:.1f}% | Precision={best_ens_res['precision']*100:.1f}% | FP={best_ens_res['fp']} | FN={best_ens_res['fn']}", flush=True)

    # 4. Hyperparameter Tuning Part B: Dual-Tier Hierarchical Escalation Window Sweep
    print("\n--- Step 4: Hyperparameter Tuning - Hierarchical Escalation Window Sweep ---", flush=True)
    # Tier 1 screens with [p_low, p_high]:
    # - If p_edge < p_low: Discard as Normal
    # - If p_edge >= p_high: Alert as Threat
    # - If p_low <= p_edge < p_high: Escalate to Server Ensemble

    w_opt = best_ens_weights
    p_low_candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    p_high_candidates = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    tau_server_candidates = [0.45, 0.48, 0.50, 0.52, 0.55]

    best_dual_f1 = 0
    best_dual_cfg = None
    best_dual_res = None
    best_dual_stats = None

    for pl in p_low_candidates:
        for ph in p_high_candidates:
            if pl >= ph:
                continue
            for ts in tau_server_candidates:
                dual_preds = []
                escalated_count = 0
                edge_alert_count = 0
                edge_discard_count = 0

                for v in val_videos:
                    pe = v["p_stgcn_max"]
                    if pe < pl:
                        # Edge Discard Normal
                        dual_preds.append(0)
                        edge_discard_count += 1
                    elif pe >= ph:
                        # Edge Alert Threat
                        dual_preds.append(1)
                        edge_alert_count += 1
                    else:
                        # Escalate to Server Ensemble
                        escalated_count += 1
                        ps = w_opt[0] * v["p_stgcn_max"] + w_opt[1] * v["p_ctrgcn_max"] + w_opt[2] * v["p_pc3d_max"]
                        dual_preds.append(1 if ps >= ts else 0)

                dual_preds = np.array(dual_preds)
                res = evaluate_binary_predictions(y_true, dual_preds, video_cats)

                tuning_records.append({
                    "type": "hierarchical_escalation_sweep",
                    "p_low": pl,
                    "p_high": ph,
                    "tau_server": ts,
                    "recall": round(res["recall"] * 100, 2),
                    "precision": round(res["precision"] * 100, 2),
                    "accuracy": round(res["accuracy"] * 100, 2),
                    "f1": round(res["f1"], 4),
                    "fp": res["fp"],
                    "fn": res["fn"],
                    "edge_discard_ratio": round(edge_discard_count / len(val_videos) * 100, 1),
                    "edge_alert_ratio": round(edge_alert_count / len(val_videos) * 100, 1),
                    "escalation_ratio": round(escalated_count / len(val_videos) * 100, 1)
                })

                # Selection criterion: ZERO false negatives (100% recall), then minimize false positives / maximize F1
                is_better = False
                if best_dual_res is None:
                    is_better = True
                elif res["fn"] < best_dual_res["fn"]:
                    is_better = True
                elif res["fn"] == best_dual_res["fn"] and res["f1"] > best_dual_f1:
                    is_better = True

                if is_better:
                    best_dual_f1 = res["f1"]
                    best_dual_cfg = {"p_low": pl, "p_high": ph, "tau_server": ts}
                    best_dual_res = res
                    best_dual_stats = {
                        "edge_discard": edge_discard_count,
                        "edge_alert": edge_alert_count,
                        "escalated": escalated_count,
                        "total": len(val_videos)
                    }

    print(f"\nChampion Hierarchical Dual-Tier Config: p_low={best_dual_cfg['p_low']}, p_high={best_dual_cfg['p_high']}, tau_server={best_dual_cfg['tau_server']}", flush=True)
    print(f"Champion Dual-Tier Results: F1={best_dual_res['f1']:.4f} | Recall={best_dual_res['recall']*100:.1f}% | Precision={best_dual_res['precision']*100:.1f}% | Accuracy={best_dual_res['accuracy']*100:.1f}% | FP={best_dual_res['fp']} | FN={best_dual_res['fn']}", flush=True)
    print(f"Workload Offloading: {best_dual_stats['edge_discard']}/{best_dual_stats['total']} clips ({best_dual_stats['edge_discard']/best_dual_stats['total']*100:.1f}%) discarded at Edge with ZERO network traffic.", flush=True)
    print(f"Edge Immediate Alerts: {best_dual_stats['edge_alert']}/{best_dual_stats['total']} clips ({best_dual_stats['edge_alert']/best_dual_stats['total']*100:.1f}%).", flush=True)
    print(f"Server Escalations: {best_dual_stats['escalated']}/{best_dual_stats['total']} clips ({best_dual_stats['escalated']/best_dual_stats['total']*100:.1f}%).", flush=True)

    # 5. Speed & Latency Profiling across 1,000 Iterations (GPU and CPU)
    print("\n--- Step 5: Hardware Latency Profiling (1,000 Iterations) ---", flush=True)
    dummy_clip = torch.randn(1, 2, 50, 17)
    dummy_hm = torch.randn(1, 17, 50, 56, 56)

    # A. GPU Latency
    gpu_tier1_fwd, gpu_t2_ensemble_fwd, gpu_gating_lat = 0, 0, 0
    if torch.cuda.is_available():
        d_gpu = torch.device("cuda")
        clip_gpu = dummy_clip.to(d_gpu)
        hm_gpu = dummy_hm.to(d_gpu)

        # Warmup
        for _ in range(50):
            _ = system.tier1_edge.model(clip_gpu)
            _ = system.tier2_server.stgcn(clip_gpu)
            _ = system.tier2_server.ctrgcn(clip_gpu)
            _ = system.tier2_server.poseconv3d(hm_gpu)
        torch.cuda.synchronize()

        # Measure Tier 1 ST-GCN
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = system.tier1_edge.model(clip_gpu)
        torch.cuda.synchronize()
        gpu_tier1_fwd = (time.perf_counter() - t0) / 1000.0 * 1000.0

        # Measure Tier 2 Multi-Model Ensemble Forward
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = system.tier2_server.stgcn(clip_gpu)
            _ = system.tier2_server.ctrgcn(clip_gpu)
            _ = system.tier2_server.poseconv3d(hm_gpu)
        torch.cuda.synchronize()
        gpu_t2_ensemble_fwd = (time.perf_counter() - t0) / 1000.0 * 1000.0

        # Measure Gating Latency
        _, f_st = system.tier1_edge.model(clip_gpu, return_features=True)
        t0 = time.perf_counter()
        for _ in range(1000):
            _ = compute_relative_mahalanobis_distance(
                f_st, system.tier1_edge.mu_0, system.tier1_edge.mu_c,
                system.tier1_edge.inv_Sigma_0, system.tier1_edge.inv_Sigma_c
            )
        torch.cuda.synchronize()
        gpu_gating_lat = (time.perf_counter() - t0) / 1000.0 * 1000.0 * 1000.0 # microseconds

    # B. CPU Latency
    system_cpu = DualTierSurveillanceSystem(device=torch.device("cpu"))
    clip_cpu = dummy_clip.to("cpu")
    hm_cpu = dummy_hm.to("cpu")

    # Warmup
    for _ in range(20):
        _ = system_cpu.tier1_edge.model(clip_cpu)
    
    t0 = time.perf_counter()
    for _ in range(100):
        _ = system_cpu.tier1_edge.model(clip_cpu)
    cpu_tier1_fwd = (time.perf_counter() - t0) / 100.0 * 1000.0

    t0 = time.perf_counter()
    for _ in range(100):
        _ = system_cpu.tier2_server.stgcn(clip_cpu)
        _ = system_cpu.tier2_server.ctrgcn(clip_cpu)
        _ = system_cpu.tier2_server.poseconv3d(hm_cpu)
    cpu_t2_ensemble_fwd = (time.perf_counter() - t0) / 100.0 * 1000.0

    _, f_st_cpu = system_cpu.tier1_edge.model(clip_cpu, return_features=True)
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = compute_relative_mahalanobis_distance(
            f_st_cpu, system_cpu.tier1_edge.mu_0, system_cpu.tier1_edge.mu_c,
            system_cpu.tier1_edge.inv_Sigma_0, system_cpu.tier1_edge.inv_Sigma_c
        )
    cpu_gating_lat = (time.perf_counter() - t0) / 1000.0 * 1000.0 * 1000.0 # microseconds

    # Total Pipeline Latencies (adding Stage 1 Weapon: 12.06 ms, Stage 2a Pose: 12.54 ms)
    # For Edge:
    stage1_lat = 12.06
    stage2a_lat = 12.54
    stage2b_edge_gpu = gpu_tier1_fwd + (gpu_gating_lat / 1000.0)
    stage2b_edge_cpu = cpu_tier1_fwd + (cpu_gating_lat / 1000.0)
    total_edge_gpu = stage1_lat + stage2a_lat + stage2b_edge_gpu
    total_edge_cpu = stage1_lat + stage2a_lat + stage2b_edge_cpu

    # For Server:
    stage2b_server_gpu = gpu_t2_ensemble_fwd + (gpu_gating_lat * 3 / 1000.0)
    stage2b_server_cpu = cpu_t2_ensemble_fwd + (cpu_gating_lat * 3 / 1000.0)
    total_server_gpu = stage1_lat + stage2a_lat + stage2b_server_gpu
    total_server_cpu = stage1_lat + stage2a_lat + stage2b_server_cpu

    # For Hierarchical Dual-Tier (Weighted average latency based on escalation ratio):
    esc_ratio = best_dual_stats["escalated"] / best_dual_stats["total"]
    total_dual_gpu = (1.0 - esc_ratio) * total_edge_gpu + esc_ratio * (total_edge_gpu + stage2b_server_gpu)
    total_dual_cpu = (1.0 - esc_ratio) * total_edge_cpu + esc_ratio * (total_edge_cpu + stage2b_server_cpu)

    print(f"\n--- Latency Summary ---", flush=True)
    print(f"Tier 1 Edge Forward (GPU): {gpu_tier1_fwd:.2f} ms | CPU: {cpu_tier1_fwd:.2f} ms", flush=True)
    print(f"Tier 2 Server Forward (GPU): {gpu_t2_ensemble_fwd:.2f} ms | CPU: {cpu_t2_ensemble_fwd:.2f} ms", flush=True)
    print(f"Total Active Pipeline (Tier 1 Edge GPU): {total_edge_gpu:.2f} ms ({1000.0/total_edge_gpu:.1f} FPS)", flush=True)
    print(f"Total Active Pipeline (Tier 1 Edge CPU): {total_edge_cpu:.2f} ms ({1000.0/total_edge_cpu:.1f} FPS)", flush=True)
    print(f"Total Active Pipeline (Tier 2 Server GPU): {total_server_gpu:.2f} ms ({1000.0/total_server_gpu:.1f} FPS)", flush=True)
    print(f"Total Active Pipeline (Hierarchical Dual-Tier GPU): {total_dual_gpu:.2f} ms ({1000.0/total_dual_gpu:.1f} FPS)", flush=True)

    # 6. Save Calibrated Weights and Reference Data
    ref_dual_path = os.path.join(REPO_DIR, "weights", "reference_data_dual_tier_v3.10.pt")
    torch.save({
        "ensemble_weights": best_ens_weights.tolist(),
        "p_low": best_dual_cfg["p_low"],
        "p_high": best_dual_cfg["p_high"],
        "tau_server": best_dual_cfg["tau_server"],
        "f1_score": best_dual_res["f1"],
        "recall": best_dual_res["recall"],
        "precision": best_dual_res["precision"],
        "accuracy": best_dual_res["accuracy"],
        "tier1_f1": res_tier1["f1"],
        "tier2_f1": best_ens_res["f1"]
    }, ref_dual_path)
    print(f">> Saved calibrated dual-tier reference data to {ref_dual_path}", flush=True)

    # 7. Save Tuning CSV
    tuning_df = pd.DataFrame(tuning_records)
    tuning_csv_path = os.path.join(REPO_DIR, "results", "tuning", "tuning_dual_tier_v3.10.csv")
    os.makedirs(os.path.dirname(tuning_csv_path), exist_ok=True)
    tuning_df.to_csv(tuning_csv_path, index=False)
    print(f">> Saved tuning sweep logs ({len(tuning_df)} records) to {tuning_csv_path}", flush=True)

    # 8. Save Validation Results CSV
    val_csv_records = []
    for v in val_videos:
        pe = v["p_stgcn_max"]
        if pe < best_dual_cfg["p_low"]:
            pred = 0
            tier_used = "edge"
        elif pe >= best_dual_cfg["p_high"]:
            pred = 1
            tier_used = "edge"
        else:
            ps = w_opt[0] * v["p_stgcn_max"] + w_opt[1] * v["p_ctrgcn_max"] + w_opt[2] * v["p_pc3d_max"]
            pred = 1 if ps >= best_dual_cfg["tau_server"] else 0
            tier_used = "server"

        val_csv_records.append({
            "source_file": v["src"],
            "category": v["cat"],
            "ground_truth": v["label"],
            "predicted": pred,
            "tier_dispatched": tier_used,
            "edge_probability": round(pe, 4),
            "correct": 1 if pred == v["label"] else 0
        })
    val_df = pd.DataFrame(val_csv_records)
    val_csv_path = os.path.join(REPO_DIR, "results", "action_recognition_benchmark", "validation_results_dual_tier_v3.10.csv")
    os.makedirs(os.path.dirname(val_csv_path), exist_ok=True)
    val_df.to_csv(val_csv_path, index=False)
    print(f">> Saved per-clip validation breakdown to {val_csv_path}", flush=True)

    # 9. Save Master Benchmark JSON
    benchmark_json = {
        "version": "v3.10",
        "method": "Dual-Tier Edge & Server Architecture Ensembles",
        "date": "2026-09-16",
        "optimal_config": {
            "p_low": best_dual_cfg["p_low"],
            "p_high": best_dual_cfg["p_high"],
            "tau_server": best_dual_cfg["tau_server"],
            "ensemble_weights": {
                "stgcn": round(best_ens_weights[0], 3),
                "ctrgcn": round(best_ens_weights[1], 3),
                "poseconv3d": round(best_ens_weights[2], 3)
            }
        },
        "performance": {
            "tier1_edge_standalone": {
                "f1": res_tier1["f1"],
                "recall": res_tier1["recall"],
                "precision": res_tier1["precision"],
                "accuracy": res_tier1["accuracy"],
                "fp": res_tier1["fp"],
                "fn": res_tier1["fn"]
            },
            "tier2_server_standalone": {
                "f1": best_ens_res["f1"],
                "recall": best_ens_res["recall"],
                "precision": best_ens_res["precision"],
                "accuracy": best_ens_res["accuracy"],
                "fp": best_ens_res["fp"],
                "fn": best_ens_res["fn"]
            },
            "hierarchical_dual_tier": {
                "f1": best_dual_res["f1"],
                "recall": best_dual_res["recall"],
                "precision": best_dual_res["precision"],
                "accuracy": best_dual_res["accuracy"],
                "fp": best_dual_res["fp"],
                "fn": best_dual_res["fn"],
                "edge_discard_ratio": best_dual_stats["edge_discard"] / best_dual_stats["total"],
                "edge_alert_ratio": best_dual_stats["edge_alert"] / best_dual_stats["total"],
                "escalation_ratio": best_dual_stats["escalated"] / best_dual_stats["total"]
            }
        },
        "latency": {
            "tier1_edge_forward_gpu_ms": round(gpu_tier1_fwd, 2),
            "tier1_edge_forward_cpu_ms": round(cpu_tier1_fwd, 2),
            "tier2_server_forward_gpu_ms": round(gpu_t2_ensemble_fwd, 2),
            "tier2_server_forward_cpu_ms": round(cpu_t2_ensemble_fwd, 2),
            "total_edge_gpu_ms": round(total_edge_gpu, 2),
            "total_edge_gpu_fps": round(1000.0 / total_edge_gpu, 1),
            "total_edge_cpu_ms": round(total_edge_cpu, 2),
            "total_edge_cpu_fps": round(1000.0 / total_edge_cpu, 1),
            "total_server_gpu_ms": round(total_server_gpu, 2),
            "total_server_gpu_fps": round(1000.0 / total_server_gpu, 1),
            "total_dual_gpu_ms": round(total_dual_gpu, 2),
            "total_dual_gpu_fps": round(1000.0 / total_dual_gpu, 1),
            "total_dual_cpu_ms": round(total_dual_cpu, 2),
            "total_dual_cpu_fps": round(1000.0 / total_dual_cpu, 1)
        }
    }
    json_path = os.path.join(REPO_DIR, "results", "benchmark_dual_tier_v3.10.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_json, f, indent=4)
    print(f">> Saved master benchmark JSON to {json_path}", flush=True)
    print("\nBenchmark Version 3.10 completed successfully!", flush=True)


if __name__ == "__main__":
    main()
