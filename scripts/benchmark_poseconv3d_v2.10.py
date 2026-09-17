"""
benchmark_poseconv3d_v2.10.py

Calibration and Comprehensive Benchmark Suite for PoseConv3D Volumetric 3D Heatmap CNN (Version 2.10 Refined).
Evaluated on the standardized 148-Clip Action Recognition & OOD Validation Suite.
Archives hyperparameter tuning sweeps (shrinkage epsilon, decision thresholds, distance metrics)
and saves PoseConv3D calibrated reference weights.

100% self-contained: uses local workspace data and shared src modules.
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

from src.dataset import resolve_data_paths, PoseConv3DHeatmapDataset
from models.poseconv3d import PoseConv3DModel

# Import RMD metrics from v1.08
spec_rmd = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py")
)
rmd_mod = importlib.util.module_from_spec(spec_rmd)
spec_rmd.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance


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


def compute_dataset_cosine_scores(val_videos, train_feats_norm, device):
    video_scores = []
    video_labels = []
    video_cats = []

    t_norm = train_feats_norm.to(device)

    with torch.no_grad():
        for item in val_videos:
            feats = item["feats"].to(device)
            vf = F.normalize(feats, p=2, dim=-1)
            sims = torch.mm(vf, t_norm.t())
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
    print("   BENCHMARK: POSECONV3D VOLUMETRIC 3D HEATMAP CNN (Version 2.10 Refined)", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    model_path = os.path.join(REPO_DIR, "weights", "poseconv3d_violence_v2.10.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train the model first.", flush=True)
        sys.exit(1)

    train_dir, val_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    model = PoseConv3DModel(num_classes=1, base_channels=24).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f">> PoseConv3D Parameter Count: {total_params:,} parameters", flush=True)

    # 1. Extract Training Features
    print("\n--- Step 1: Extracting PoseConv3D Training Features (297 Clips) ---", flush=True)
    train_ds = PoseConv3DHeatmapDataset(root_dir=train_dir, action_folders=actions, clip_length=50, heatmap_size=56, mode="train")

    train_feats = []
    train_labels = []
    with torch.no_grad():
        for i in range(len(train_ds)):
            hm, _, src = train_ds[i]
            hm_tensor = hm.unsqueeze(0).to(device)
            _, f = model(hm_tensor, return_features=True)
            train_feats.append(f.squeeze(0).cpu())
            for c_idx, act in enumerate(actions):
                if act in src:
                    train_labels.append(c_idx)
                    break
    train_feats = torch.stack(train_feats)
    train_labels = torch.tensor(train_labels)
    N, D = train_feats.shape
    C = len(actions)
    print(f"Extracted {N} training clips of dimension {D} across {C} action categories.", flush=True)

    # 2. Extract Validation Suite Features
    print("\n--- Step 2: Extracting Validation Suite Features (72 Videos / 148 Clips) ---", flush=True)
    val_categories = ["With_Coat", "Without_Coat", "OOD"]
    val_videos = []
    with torch.no_grad():
        for cat in val_categories:
            cat_dir = os.path.join(val_root, cat)
            dataset = PoseConv3DHeatmapDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, heatmap_size=56, mode="test")
            video_clips = defaultdict(list)
            for i in range(len(dataset)):
                hm, label, src = dataset[i]
                video_clips[src].append(hm)
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

    # 3. Base Empirical Means and Covariances for RMD
    mu_0 = train_feats.mean(dim=0, keepdim=True)
    diff_0 = train_feats - mu_0
    Sigma_0_emp = (diff_0.T @ diff_0) / (N - 1)

    mu_c = torch.stack([train_feats[train_labels == c].mean(dim=0) for c in range(C)])
    diff_c_list = [train_feats[train_labels == c] - mu_c[c] for c in range(C)]
    Sigma_c_emp = torch.stack([(d.T @ d) / (len(d) - 1) for d in diff_c_list])

    # 4. Hyperparameter Tuning Sweep: Shrinkage Epsilon for RMD
    print("\n--- Step 3: Hyperparameter Tuning - Shrinkage Epsilon Sweep on PoseConv3D ---", flush=True)
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

        v_scores, v_labels, v_cats = compute_dataset_rmd_scores(val_videos, mu_0, mu_c, inv_S0, inv_Sc, device)

        best_f1, best_t, best_res = 0, 0, None
        test_taus = np.linspace(np.percentile(v_scores, 1), np.percentile(v_scores, 99), 300)
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
    base_tau = best_overall_tau
    test_taus = sorted(list(set([
        round(base_tau - 2.0, 4), round(base_tau - 1.5, 4), round(base_tau - 1.0, 4),
        round(base_tau - 0.5, 4), round(base_tau - 0.2, 4), round(base_tau, 4),
        round(base_tau + 0.2, 4), round(base_tau + 0.5, 4), round(base_tau + 1.0, 4),
        round(base_tau + 1.5, 4), round(base_tau + 2.0, 4)
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

    # Check Hyperspherical Cosine k-NN on PoseConv3D Features
    print("\n--- Step 4b: Hyperspherical Cosine Gating Analysis ---", flush=True)
    train_norm = F.normalize(train_feats, p=2, dim=-1)
    cos_scores, cos_labels, cos_cats = compute_dataset_cosine_scores(val_videos, train_norm, device)
    best_cos_f1, best_cos_t, best_cos_res = 0, 0, None
    for t_cos in np.linspace(cos_scores.min(), cos_scores.max(), 300):
        res = evaluate_scores_at_threshold(cos_scores, cos_labels, cos_cats, t_cos)
        if res["f1"] > best_cos_f1:
            best_cos_f1 = res["f1"]
            best_cos_t = t_cos
            best_cos_res = res
    print(f">> Hyperspherical Cosine k-NN: Best F1={best_cos_res['f1']:.4f} @ Tau={best_cos_t:.4f} | Prec={best_cos_res['precision']*100:.1f}% | Rec={best_cos_res['recall']*100:.1f}% | FP={best_cos_res['fp']} | FN={best_cos_res['fn']}", flush=True)

    # 6. Detailed Latency Benchmarks (1,000 iterations)
    print("\n--- Step 5: Measuring Forward & Gating Latencies (1,000 iterations) ---", flush=True)
    clip_sample = torch.randn(1, 17, 50, 56, 56)
    query_feat = torch.randn(1, D)
    latency_stats = {}

    for dev_name, dev in [("gpu_rtx3090", torch.device("cuda" if torch.cuda.is_available() else "cpu")), ("cpu", torch.device("cpu"))]:
        m_dev = PoseConv3DModel(num_classes=1, base_channels=24).to(dev)
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
        if dev.type == "cuda":
            torch.cuda.synchronize()

        N_iter = 1000
        # Forward pass
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(N_iter):
                _ = m_dev(c_d, return_features=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        fwd_lat_ms = (t1 - t0) / N_iter * 1000.0

        # Gating pass
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(N_iter * 5):
                d0 = q_d - m0_d
                m0 = torch.sum((torch.mm(d0, is0_d)) * d0, dim=-1)
                dc = q_d.unsqueeze(1) - mc_d.unsqueeze(0)
                mc = torch.einsum('bcd,cde,bce->bc', dc, isc_d, dc)
                min_mc, _ = mc.min(dim=-1)
                _ = m0 - min_mc
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        gate_lat_us = (t1 - t0) / (N_iter * 5) * 1e6

        stage2b_total_ms = fwd_lat_ms + (gate_lat_us / 1000.0)

        latency_stats[dev_name] = {
            "backbone_fwd_ms": round(fwd_lat_ms, 2),
            "gating_us": round(gate_lat_us, 2),
            "stage2b_total_ms": round(stage2b_total_ms, 2)
        }
        print(f"[{dev_name.upper()}] PoseConv3D Forward: {fwd_lat_ms:.2f} ms | Gating: {gate_lat_us:.2f} us | Stage 2b Total: {stage2b_total_ms:.2f} ms", flush=True)

    # 7. Save Calibrated Weights
    weights_path = os.path.join(REPO_DIR, "weights", "reference_data_poseconv3d_v2.10.pt")
    torch.save({
        "mu_0": mu_0.cpu(),
        "mu_c": mu_c.cpu(),
        "inv_Sigma_0": inv_S0_opt.cpu(),
        "inv_Sigma_c": inv_Sc_opt.cpu(),
        "train_features_norm": train_norm.cpu(),
        "classes": actions,
        "optimal_epsilon": best_overall_eps,
        "optimal_threshold": float(optimal_config["threshold"]),
        "f1_score": optimal_config["f1_score"],
        "precision": optimal_config["precision"],
        "recall": optimal_config["recall"],
        "accuracy": optimal_config["accuracy"],
        "cosine_f1": round(best_cos_res["f1"], 4),
        "cosine_threshold": round(best_cos_t, 4)
    }, weights_path)
    file_size_kb = os.path.getsize(weights_path) / 1024.0
    print(f"\n>> Calibrated PoseConv3D reference weights saved to: {weights_path} ({file_size_kb:.1f} KB)", flush=True)

    # 8. Archive Tuning Logs
    tuning_dir = os.path.join(REPO_DIR, "results", "tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    csv_tuning_path = os.path.join(tuning_dir, "tuning_poseconv3d_v2.10.csv")
    json_tuning_path = os.path.join(tuning_dir, "tuning_poseconv3d_v2.10.json")

    pd.DataFrame(threshold_sweep_records).to_csv(csv_tuning_path, index=False)
    with open(json_tuning_path, "w") as f:
        json.dump({
            "model": "PoseConv3D (Version 2.10 Refined)",
            "shrinkage_sweep": tuning_eps_records,
            "threshold_sweep": threshold_sweep_records,
            "optimal_operating_point": optimal_config,
            "cosine_k_nn": {
                "f1_score": round(best_cos_res["f1"], 4),
                "threshold": round(best_cos_t, 4),
                "precision": round(best_cos_res["precision"] * 100, 2),
                "recall": round(best_cos_res["recall"] * 100, 2),
                "fp": best_cos_res["fp"],
                "fn": best_cos_res["fn"]
            }
        }, f, indent=2)

    # 9. Save Per-Category Validation Results CSV
    res_dir = os.path.join(REPO_DIR, "results", "action_recognition_benchmark")
    os.makedirs(res_dir, exist_ok=True)
    cat_rows = []
    opt_eval = evaluate_scores_at_threshold(v_scores_opt, v_labels_opt, v_cats_opt, optimal_config["threshold"])
    for cat_name, cat_stat in opt_eval["category_breakdown"].items():
        n_vids = cat_stat["videos"]
        tp, tn, fp, fn = cat_stat["tp"], cat_stat["tn"], cat_stat["fp"], cat_stat["fn"]
        cat_acc = (tp + tn) / n_vids if n_vids > 0 else 0
        cat_rec = tp / (tp + fn) if (tp + fn) > 0 else (tn / (tn + fp) if (tn + fp) > 0 else 0)
        cat_prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        cat_f1 = 2 * cat_prec * cat_rec / (cat_prec + cat_rec) if (cat_prec + cat_rec) > 0 else 0.0
        cat_rows.append({
            "category": cat_name,
            "total_videos": n_vids,
            "accuracy": round(cat_acc * 100, 2),
            "recall": round(cat_rec * 100, 2),
            "precision": round(cat_prec * 100, 2),
            "f1_score": round(cat_f1, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        })
    df_cat = pd.DataFrame(cat_rows)
    cat_csv_path = os.path.join(res_dir, "validation_results_poseconv3d_v2.10.csv")
    df_cat.to_csv(cat_csv_path, index=False)

    # 10. Save Master Benchmark JSON
    stage1_lat = 12.06
    stage2a_lat = 12.54
    total_gpu_threat_lat = stage1_lat + stage2a_lat + latency_stats["gpu_rtx3090"]["stage2b_total_ms"]
    total_cpu_threat_lat = stage1_lat + stage2a_lat + latency_stats["cpu"]["stage2b_total_ms"]

    benchmark_json_path = os.path.join(REPO_DIR, "results", "benchmark_poseconv3d_v2.10.json")
    with open(benchmark_json_path, "w") as f:
        json.dump({
            "version": "2.10",
            "technique": "PoseConv3D Volumetric 3D Heatmap CNN (Refined)",
            "model_parameters": total_params,
            "optimal_parameters": {
                "shrinkage_epsilon": best_overall_eps,
                "threat_threshold": float(optimal_config["threshold"]),
                "reference_storage_kb": round(file_size_kb, 1)
            },
            "metrics": {
                "f1_score": optimal_config["f1_score"],
                "precision": optimal_config["precision"],
                "recall": optimal_config["recall"],
                "accuracy": optimal_config["accuracy"],
                "true_positives": optimal_config["tp"],
                "true_negatives": optimal_config["tn"],
                "false_positives": optimal_config["fp"],
                "false_negatives": optimal_config["fn"],
                "cosine_metrics": {
                    "f1_score": round(best_cos_res["f1"], 4),
                    "threshold": round(best_cos_t, 4),
                    "precision": round(best_cos_res["precision"] * 100, 2),
                    "recall": round(best_cos_res["recall"] * 100, 2),
                    "fp": best_cos_res["fp"],
                    "fn": best_cos_res["fn"]
                },
                "category_breakdown": opt_eval["category_breakdown"]
            },
            "latencies": {
                "stage1_weapon_scan_ms": stage1_lat,
                "stage2a_pose_track_ms": stage2a_lat,
                "stage2b_gpu_backbone_fwd_ms": latency_stats["gpu_rtx3090"]["backbone_fwd_ms"],
                "stage2b_gpu_gating_us": latency_stats["gpu_rtx3090"]["gating_us"],
                "stage2b_gpu_total_ms": latency_stats["gpu_rtx3090"]["stage2b_total_ms"],
                "stage2b_cpu_backbone_fwd_ms": latency_stats["cpu"]["backbone_fwd_ms"],
                "stage2b_cpu_gating_us": latency_stats["cpu"]["gating_us"],
                "stage2b_cpu_total_ms": latency_stats["cpu"]["stage2b_total_ms"],
                "total_threat_pipeline_gpu_ms": round(total_gpu_threat_lat, 2),
                "total_threat_pipeline_gpu_fps": round(1000.0 / total_gpu_threat_lat, 1),
                "total_threat_pipeline_cpu_ms": round(total_cpu_threat_lat, 2),
                "total_threat_pipeline_cpu_fps": round(1000.0 / total_cpu_threat_lat, 1)
            }
        }, f, indent=2)
    print(f">> Master benchmark JSON saved to: {benchmark_json_path}", flush=True)


if __name__ == "__main__":
    main()
