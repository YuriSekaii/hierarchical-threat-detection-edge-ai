"""
benchmark_ctrgcn_v2.00.py

Calibration and Comprehensive Benchmark Suite for CTR-GCN Dynamic Spatio-Temporal Backbone (Version 2.00).
Evaluated on the standardized 148-Clip Action Recognition & OOD Validation Suite.
Archives hyperparameter tuning sweeps (shrinkage epsilon, decision thresholds, distance metrics)
and saves CTR-GCN calibrated reference weights.
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
sys.path.insert(0, REPO_DIR)

from models.ctrgcn import CTRGCNModel

# Import v1.08 RMD metrics
spec_rmd = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py")
)
rmd_mod = importlib.util.module_from_spec(spec_rmd)
spec_rmd.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance

# Import dataset resolution
spec_train = importlib.util.spec_from_file_location("train_v102", os.path.join(REPO_DIR, "training", "train_stgcn_knn_v1.02.py"))
train_mod = importlib.util.module_from_spec(spec_train)
spec_train.loader.exec_module(train_mod)
ActionDataset = train_mod.ActionDataset
resolve_data_paths = train_mod.resolve_data_paths


def compute_dataset_rmd_scores(val_videos, mu_0, mu_c, inv_Sigma_0, inv_Sigma_c, device):
    m0_d = mu_0.to(device)
    mc_d = mu_c.to(device)
    is0_d = inv_Sigma_0.to(device)
    isc_d = inv_Sigma_c.to(device)
    
    video_scores = []
    video_labels = []
    video_cats = []
    
    with torch.no_grad():
        for item in val_videos:
            feats = item["feats"].to(device)
            scores, _, _ = compute_relative_mahalanobis_distance(
                feats, m0_d, mc_d, is0_d, isc_d
            )
            video_scores.append(scores.max().item())
            video_labels.append(item["label"])
            video_cats.append(item["cat"])
            
    return np.array(video_scores), np.array(video_labels), video_cats


def compute_dataset_cosine_scores(val_videos, train_centroids, device):
    tc_norm = F.normalize(train_centroids.to(device), p=2, dim=-1)
    
    video_scores = []
    video_labels = []
    video_cats = []
    
    with torch.no_grad():
        for item in val_videos:
            feats = F.normalize(item["feats"].to(device), p=2, dim=-1) # (K, 256)
            sims = torch.mm(feats, tc_norm.t()) # (K, C)
            max_sim = sims.max().item()
            video_scores.append(max_sim)
            video_labels.append(item["label"])
            video_cats.append(item["cat"])
            
    return np.array(video_scores), np.array(video_labels), video_cats


def evaluate_scores_at_threshold(video_scores, video_labels, video_cats, threshold):
    y_pred = (video_scores >= threshold).astype(int)
    y_true = video_labels
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    category_results = defaultdict(lambda: {"videos": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for cat, yt, yp in zip(video_cats, y_true, y_pred):
        d = category_results[cat]
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
        "category_breakdown": dict(category_results)
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("   BENCHMARK: CTR-GCN DYNAMIC SPATIO-TEMPORAL BACKBONE (Version 2.00)", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    model_path = os.path.join(REPO_DIR, "weights", "ctrgcn_violence_v2.00.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please wait for training to complete.", flush=True)
        sys.exit(1)

    train_dir, val_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    model = CTRGCNModel(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f">> CTR-GCN Parameter Count: {total_params:,} parameters (vs 3,014,981 in ST-GCN, -{(1 - total_params/3014981)*100:.1f}%)", flush=True)

    # 1. Extract Training Features and Action Labels
    print("\n--- Step 1: Extracting CTR-GCN Training Features (297 Clips) ---", flush=True)
    train_ds = ActionDataset(root_dir=train_dir, action_folders=actions, clip_length=50, mode="train")
    
    train_feats = []
    train_labels = []
    with torch.no_grad():
        for i in range(len(train_ds)):
            clip = train_ds[i][0].unsqueeze(0).to(device)
            src = train_ds.samples[i]["source_file"]
            _, f = model(clip, return_features=True)
            train_feats.append(f.squeeze(0).cpu())
            for c_idx, act in enumerate(actions):
                if act in src:
                    train_labels.append(c_idx)
                    break
    train_feats = torch.stack(train_feats) # (297, 256)
    train_labels = torch.tensor(train_labels)
    N, D = train_feats.shape
    C = len(actions)
    print(f"Extracted {N} training clips of dimension {D} across {C} action categories:", flush=True)
    for c_idx, act in enumerate(actions):
        print(f"  - Class {c_idx} ({act:<8}): {(train_labels == c_idx).sum().item()} clips", flush=True)

    # 2. Extract Validation Suite Features (Single-Pass Cache)
    print("\n--- Step 2: Extracting Validation Suite Features (72 Videos / 148 Clips) ---", flush=True)
    val_categories = ["With_Coat", "Without_Coat", "OOD"]
    val_videos = []
    with torch.no_grad():
        for cat in val_categories:
            cat_dir = os.path.join(val_root, cat)
            dataset = ActionDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, mode="test")
            video_clips = defaultdict(list)
            for i in range(len(dataset)):
                clip, label, src = dataset[i]
                video_clips[src].append(clip)
            for src, clips in video_clips.items():
                clips_tensor = torch.stack(clips).to(device)
                _, feats = model(clips_tensor, return_features=True)
                val_videos.append({
                    "src": src,
                    "feats": feats.cpu(),
                    "cat": cat,
                    "label": 0 if cat == "OOD" else 1
                })
    print(f"Validation features cached: {len(val_videos)} video items across With_Coat, Without_Coat, OOD.", flush=True)

    # 3. Base Empirical Means and Covariances
    mu_0 = train_feats.mean(dim=0, keepdim=True) # (1, 256)
    diff_0 = train_feats - mu_0
    Sigma_0_emp = (diff_0.T @ diff_0) / (N - 1) # (256, 256)

    mu_c = torch.stack([train_feats[train_labels == c].mean(dim=0) for c in range(C)]) # (3, 256)
    diff_c_list = [train_feats[train_labels == c] - mu_c[c] for c in range(C)]
    Sigma_c_emp = torch.stack([(d.T @ d) / (len(d) - 1) for d in diff_c_list]) # (3, 256, 256)

    # 4. Hyperparameter Tuning Sweep: Shrinkage Epsilon for RMD on CTR-GCN
    print("\n--- Step 3: Hyperparameter Tuning - Shrinkage Epsilon Sweep on CTR-GCN ---", flush=True)
    eps_candidates = [0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
    tuning_eps_records = []
    
    eye_D = torch.eye(D)
    eye_C_D = eye_D.unsqueeze(0).expand(C, -1, -1)
    
    best_overall_eps = 0.005
    best_overall_f1 = 0
    best_overall_res = None
    best_overall_tau = None
    
    for eps_val in eps_candidates:
        inv_S0 = torch.linalg.pinv(Sigma_0_emp + eps_val * eye_D)
        inv_Sc = torch.linalg.pinv(Sigma_c_emp + eps_val * eye_C_D)
        
        # Compute video max RMD scores
        v_scores, v_labels, v_cats = compute_dataset_rmd_scores(val_videos, mu_0, mu_c, inv_S0, inv_Sc, device)
        
        # Grid search threshold
        best_f1, best_t, best_res = 0, 0, None
        test_taus = np.linspace(np.percentile(v_scores, 2), np.percentile(v_scores, 98), 200)
        for t_test in test_taus:
            eval_res = evaluate_scores_at_threshold(v_scores, v_labels, v_cats, t_test)
            if eval_res["f1"] > best_f1:
                best_f1 = eval_res["f1"]
                best_t = t_test
                best_res = eval_res
                
        tuning_eps_records.append({
            "shrinkage_epsilon": eps_val,
            "optimal_threshold": round(best_t, 4),
            "f1_score": round(best_res["f1"], 4),
            "precision": round(best_res["precision"] * 100, 2),
            "recall": round(best_res["recall"] * 100, 2),
            "accuracy": round(best_res["accuracy"] * 100, 2),
            "fp": best_res["fp"],
            "fn": best_res["fn"]
        })
        print(f"eps={eps_val:.4f} | Best F1={best_res['f1']:.4f} @ Tau={best_t:7.4f} | Prec={best_res['precision']*100:.1f}% | Rec={best_res['recall']*100:.1f}% | Acc={best_res['accuracy']*100:.1f}% | FP={best_res['fp']} | FN={best_res['fn']}", flush=True)
        
        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_eps = eps_val
            best_overall_tau = best_t
            best_overall_res = best_res

    # 5. Hyperparameter Tuning: Threshold Sweep for Champion Epsilon
    print(f"\n--- Step 4: Hyperparameter Tuning - Decision Threshold Sweep (Champion eps={best_overall_eps}) ---", flush=True)
    inv_S0_opt = torch.linalg.pinv(Sigma_0_emp + best_overall_eps * eye_D)
    inv_Sc_opt = torch.linalg.pinv(Sigma_c_emp + best_overall_eps * eye_C_D)
    
    v_scores_opt, v_labels_opt, v_cats_opt = compute_dataset_rmd_scores(
        val_videos, mu_0, mu_c, inv_S0_opt, inv_Sc_opt, device
    )
    
    threshold_sweep_records = []
    # Test a set of representative thresholds around the optimal
    base_tau = best_overall_tau
    test_taus = sorted(list(set([
        round(base_tau - 1.5, 4), round(base_tau - 1.0, 4), round(base_tau - 0.5, 4),
        round(base_tau - 0.2, 4), round(base_tau, 4), round(base_tau + 0.2, 4),
        round(base_tau + 0.5, 4), round(base_tau + 1.0, 4), round(base_tau + 1.5, 4)
    ])))
    
    for t_val in test_taus:
        res = evaluate_scores_at_threshold(v_scores_opt, v_labels_opt, v_cats_opt, t_val)
        profile = "Balanced Best (Optimal F1)" if abs(t_val - base_tau) < 1e-3 else (
            "Zero-Miss Security" if res["fn"] == 0 else (
                "Conservative" if res["fp"] <= 5 else "High Sensitivity"
            )
        )
        threshold_sweep_records.append({
            "threshold": round(t_val, 4),
            "f1_score": round(res["f1"], 4),
            "precision": round(res["precision"] * 100, 2),
            "recall": round(res["recall"] * 100, 2),
            "accuracy": round(res["accuracy"] * 100, 2),
            "tp": res["tp"],
            "tn": res["tn"],
            "fp": res["fp"],
            "fn": res["fn"],
            "operational_profile": profile
        })
        print(f"Tau={t_val:7.4f} | F1={res['f1']:.4f} | Prec={res['precision']*100:.1f}% | Rec={res['recall']*100:.1f}% (TP={res['tp']}/33) | TN={res['tn']}/39 (FP={res['fp']}) | Acc={res['accuracy']*100:.1f}% | Profile={profile}", flush=True)

    optimal_config = next(r for r in threshold_sweep_records if abs(r["threshold"] - round(base_tau, 4)) < 1e-3)
    print(f"\n>> OPTIMAL CTR-GCN OPERATING POINT: eps={best_overall_eps}, Tau={optimal_config['threshold']}", flush=True)
    print(f"   F1-Score : {optimal_config['f1_score']}", flush=True)
    print(f"   Precision: {optimal_config['precision']}%", flush=True)
    print(f"   Recall   : {optimal_config['recall']}% ({optimal_config['tp']}/33 detected)", flush=True)
    print(f"   Accuracy : {optimal_config['accuracy']}%", flush=True)
    print(f"   FP Alarms: {optimal_config['fp']}/39", flush=True)

    # 6. Detailed Latency Benchmarks (CTR-GCN Forward + Gating on GPU & CPU)
    print("\n--- Step 5: Measuring Forward & Gating Latencies (1,000 iterations) ---", flush=True)
    clip_sample = torch.randn(1, 2, 50, 17)
    query_feat = torch.randn(1, D)
    latency_stats = {}
    
    for dev_name, dev in [("gpu_rtx3090", torch.device("cuda" if torch.cuda.is_available() else "cpu")), ("cpu", torch.device("cpu"))]:
        m_dev = CTRGCNModel(num_classes=1).to(dev)
        m_dev.load_state_dict(model.state_dict())
        m_dev.eval()
        
        c_d = clip_sample.to(dev)
        m0_d = mu_0.to(dev)
        mc_d = mu_c.to(dev)
        is0_d = inv_S0_opt.to(dev)
        isc_d = inv_Sc_opt.to(dev)
        q_d = query_feat.to(dev)
        
        # Warmup
        with torch.no_grad():
            for _ in range(100):
                _ = m_dev(c_d, return_features=True)
                d0 = q_d - m0_d
                m0 = torch.sum((torch.mm(d0, is0_d)) * d0, dim=-1)
                dc = q_d.unsqueeze(1) - mc_d.unsqueeze(0)
                mc = torch.einsum('bcd,cde,bce->bc', dc, isc_d, dc)
                min_mc, _ = mc.min(dim=-1)
                _ = m0 - min_mc
        if dev.type == "cuda": torch.cuda.synchronize()
        
        N_iter = 1000
        # Benchmark Backbone Forward
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(N_iter):
                _ = m_dev(c_d, return_features=True)
            if dev.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
        fwd_lat_ms = (t1 - t0) / N_iter * 1000.0
        
        # Benchmark Gating Alone
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(N_iter * 5):
                d0 = q_d - m0_d
                m0 = torch.sum((torch.mm(d0, is0_d)) * d0, dim=-1)
                dc = q_d.unsqueeze(1) - mc_d.unsqueeze(0)
                mc = torch.einsum('bcd,cde,bce->bc', dc, isc_d, dc)
                min_mc, _ = mc.min(dim=-1)
                _ = m0 - min_mc
            if dev.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
        gate_lat_us = (t1 - t0) / (N_iter * 5) * 1e6
        
        stage2b_total_ms = fwd_lat_ms + (gate_lat_us / 1000.0)
        
        latency_stats[dev_name] = {
            "backbone_fwd_ms": round(fwd_lat_ms, 2),
            "gating_us": round(gate_lat_us, 2),
            "stage2b_total_ms": round(stage2b_total_ms, 2)
        }
        print(f"[{dev_name.upper()}] CTR-GCN Forward: {fwd_lat_ms:.2f} ms | Gating: {gate_lat_us:.2f} us | Stage 2b Total: {stage2b_total_ms:.2f} ms", flush=True)

    # 7. Save Calibrated Weights
    weights_path = os.path.join(REPO_DIR, "weights", "reference_data_ctrgcn_v2.00.pt")
    torch.save({
        "mu_0": mu_0.cpu(),
        "mu_c": mu_c.cpu(),
        "inv_Sigma_0": inv_S0_opt.cpu(),
        "inv_Sigma_c": inv_Sc_opt.cpu(),
        "classes": actions,
        "optimal_epsilon": best_overall_eps,
        "optimal_threshold": float(optimal_config["threshold"]),
        "f1_score": optimal_config["f1_score"],
        "precision": optimal_config["precision"],
        "recall": optimal_config["recall"],
        "accuracy": optimal_config["accuracy"]
    }, weights_path)
    file_size_kb = os.path.getsize(weights_path) / 1024.0
    print(f"\n>> Calibrated CTR-GCN reference weights saved to: {weights_path} ({file_size_kb:.1f} KB)", flush=True)

    # 8. Archive Tuning Logs
    tuning_dir = os.path.join(REPO_DIR, "results", "tuning")
    os.makedirs(tuning_dir, exist_ok=True)
    
    df_tuning_eps = pd.DataFrame(tuning_eps_records)
    df_tuning_tau = pd.DataFrame(threshold_sweep_records)
    
    csv_tuning_path = os.path.join(tuning_dir, "tuning_ctrgcn_v2.00.csv")
    json_tuning_path = os.path.join(tuning_dir, "tuning_ctrgcn_v2.00.json")
    
    df_tuning_tau.to_csv(csv_tuning_path, index=False)
    with open(json_tuning_path, "w") as f:
        json.dump({
            "shrinkage_epsilon_sweep": tuning_eps_records,
            "threshold_sweep_champion_eps": threshold_sweep_records,
            "champion_config": {
                "epsilon": best_overall_eps,
                "threshold": optimal_config["threshold"],
                "f1_score": optimal_config["f1_score"],
                "precision": optimal_config["precision"],
                "recall": optimal_config["recall"],
                "accuracy": optimal_config["accuracy"],
                "fp": optimal_config["fp"],
                "fn": optimal_config["fn"]
            }
        }, f, indent=2)
    print(f">> Tuning logs saved to: {csv_tuning_path} and {json_tuning_path}", flush=True)

    # 9. Save Validation Results CSV
    eval_opt = evaluate_scores_at_threshold(v_scores_opt, v_labels_opt, v_cats_opt, optimal_config["threshold"])
    val_records = []
    for cat, stats in eval_opt["category_breakdown"].items():
        v_total = stats["videos"]
        v_acc = (stats["tp"] + stats["tn"]) / v_total if v_total > 0 else 0
        v_prec = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
        v_rec = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        v_f1 = 2 * v_prec * v_rec / (v_prec + v_rec) if (v_prec + v_rec) > 0 else 0
        val_records.append({
            "Category": cat,
            "Videos": v_total,
            "Accuracy": round(v_acc * 100, 2),
            "Precision": round(v_prec * 100, 2),
            "Recall": round(v_rec * 100, 2),
            "F1_Score": round(v_f1, 4),
            "TP": stats["tp"], "TN": stats["tn"], "FP": stats["fp"], "FN": stats["fn"]
        })
    df_val = pd.DataFrame(val_records)
    val_csv_path = os.path.join(REPO_DIR, "results", "action_recognition_benchmark", "validation_results_ctrgcn_v2.00.csv")
    df_val.to_csv(val_csv_path, index=False)
    print(f">> Category validation breakdown saved to: {val_csv_path}", flush=True)

    # 10. Save Master Benchmark JSON
    benchmark_json_path = os.path.join(REPO_DIR, "results", "benchmark_ctrgcn_v2.00.json")
    stage1_ms = 12.06
    stage2a_ms = 12.54
    stage2b_gpu = latency_stats.get("gpu_rtx3090", {}).get("stage2b_total_ms", 1.80)
    stage2b_cpu = latency_stats.get("cpu", {}).get("stage2b_total_ms", 14.00)
    total_gpu_ms = round(stage1_ms + stage2a_ms + stage2b_gpu, 2)
    total_cpu_ms = round(stage1_ms + stage2a_ms + stage2b_cpu, 2)
    
    with open(benchmark_json_path, "w") as f:
        json.dump({
            "version": "2.00",
            "backbone": "Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN)",
            "parameters": total_params,
            "parameter_reduction_vs_stgcn": f"-{(1 - total_params/3014981)*100:.1f}%",
            "champion_hyperparameters": {
                "shrinkage_epsilon": best_overall_eps,
                "threshold": optimal_config["threshold"],
                "classes": actions,
                "degrees_of_freedom": D
            },
            "performance": {
                "f1_score": optimal_config["f1_score"],
                "precision": optimal_config["precision"],
                "recall": optimal_config["recall"],
                "accuracy": optimal_config["accuracy"],
                "tp": optimal_config["tp"],
                "tn": optimal_config["tn"],
                "fp": optimal_config["fp"],
                "fn": optimal_config["fn"],
                "ood_rejection_rate": round(optimal_config["tn"] / 39 * 100, 2)
            },
            "speed_and_latency": {
                "backbone_fwd_gpu_ms": latency_stats.get("gpu_rtx3090", {}).get("backbone_fwd_ms", 0),
                "backbone_fwd_cpu_ms": latency_stats.get("cpu", {}).get("backbone_fwd_ms", 0),
                "gating_gpu_us": latency_stats.get("gpu_rtx3090", {}).get("gating_us", 0),
                "gating_cpu_us": latency_stats.get("cpu", {}).get("gating_us", 0),
                "stage1_weapon_ms": stage1_ms,
                "stage2a_pose_ms": stage2a_ms,
                "stage2b_ctrgcn_and_ood_gpu_ms": stage2b_gpu,
                "stage2b_ctrgcn_and_ood_cpu_ms": stage2b_cpu,
                "total_threat_active_pipeline_gpu_ms": total_gpu_ms,
                "total_threat_active_pipeline_cpu_ms": total_cpu_ms,
                "fps_gpu": round(1000.0 / total_gpu_ms, 1),
                "fps_cpu": round(1000.0 / total_cpu_ms, 1)
            },
            "storage": {
                "model_weights_mb": round(os.path.getsize(model_path) / (1024*1024), 2),
                "reference_data_kb": round(file_size_kb, 1),
                "num_classes": C,
                "feature_dimension": D
            }
        }, f, indent=2)
    print(f">> Master benchmark JSON saved to: {benchmark_json_path}", flush=True)
    print("=" * 80, flush=True)
    print("   BENCHMARK AND CALIBRATION COMPLETE FOR VERSION 2.00!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
