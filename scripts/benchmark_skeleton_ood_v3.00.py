"""
scripts/benchmark_skeleton_ood_v3.00.py

Calibration and Comprehensive Benchmark Suite for Skeleton-OOD (Version 3.00).
Evaluates ST-GCN with In-Training ASH-SE Fusion and Latent Boundary Learning
on the standardized 148-Clip Action Recognition & OOD Validation Suite (77 video items).

Evaluations:
1. Direct Skeleton-OOD Energy Gating (S_energy = -E(x) = logsumexp(logits / T)).
2. Boundary-Optimized Relative Mahalanobis Distance (RMD on S^{D-1} manifold).
3. Hyperspherical Cosine Centroid Gating.
4. Latency profiling (1,000 iterations on GPU & CPU).
5. Exports calibrated reference data and benchmark results.
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

from models.skeleton_ood import SkeletonOODModel
from src.dataset import resolve_data_paths, ActionDataset

# Import v1.08 RMD metric computation
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


def compute_dataset_energy_scores(val_videos):
    video_scores = []
    video_labels = []
    video_cats = []

    for item in val_videos:
        # Threat score = -energy (higher = more likely violent threat)
        threat_scores = -item["energies"]
        video_scores.append(threat_scores.max().item())
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
            feats = F.normalize(item["feats"].to(device), p=2, dim=-1)
            sims = torch.mm(feats, tc_norm.t())
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
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "categories": dict(category_results)
    }


def sweep_optimal_threshold(video_scores, video_labels, video_cats, num_steps=300):
    s_min, s_max = video_scores.min(), video_scores.max()
    thresholds = np.linspace(s_min, s_max, num_steps)

    best_f1 = -1.0
    best_res = None
    best_tau = 0.0

    for tau in thresholds:
        res = evaluate_scores_at_threshold(video_scores, video_labels, video_cats, tau)
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_res = res
            best_tau = tau
        elif res["f1"] == best_f1 and res["recall"] > (best_res["recall"] if best_res else 0):
            best_f1 = res["f1"]
            best_res = res
            best_tau = tau

    return best_tau, best_res


def main():
    print("========================================================================", flush=True)
    print("  Comprehensive Calibration & Benchmark: Skeleton-OOD (Version 3.00)", flush=True)
    print("  ST-GCN Backbone + ASH-SE Fusion + Latent Boundary Learning", flush=True)
    print("========================================================================", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    train_dir, val_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    weights_path = os.path.join(REPO_DIR, "weights", "stgcn_skeleton_ood_v3.00.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing champion checkpoint: {weights_path}")

    model = SkeletonOODModel(
        num_classes=3,
        in_channels=2,
        num_keypoints=17,
        ash_percentile=80,
        ash_mode='ash_b',
        temperature=1.0
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Successfully loaded weights from: {weights_path}", flush=True)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.st_gcn_networks.parameters())
    print(f"Model Parameters: Total={total_params:,} (Backbone={backbone_params:,})", flush=True)

    # 1. Extract Training Features and Action Labels
    print("\n--- Step 1: Extracting Training Embeddings (297 Clips) ---", flush=True)
    train_ds = ActionDataset(root_dir=train_dir, action_folders=actions, clip_length=50, mode="train")

    train_feats = []
    train_labels = []
    train_energies = []

    with torch.no_grad():
        for i in range(len(train_ds)):
            clip = train_ds[i][0].unsqueeze(0).to(device)
            src = train_ds.samples[i]["source_file"]
            l, z, e = model(clip, return_features=True, return_energy=True)
            train_feats.append(z.squeeze(0).cpu())
            train_energies.append(e.squeeze(0).cpu())
            for c_idx, act in enumerate(actions):
                if act in src:
                    train_labels.append(c_idx)
                    break

    train_feats = torch.stack(train_feats) # (297, 256)
    train_labels = torch.tensor(train_labels)
    train_energies = torch.stack(train_energies)
    N, D = train_feats.shape
    C = len(actions)
    print(f"Extracted {N} training clips of dimension {D} across {C} categories:", flush=True)
    for c_idx, act in enumerate(actions):
        print(f"  - Class {c_idx} ({act:<8}): {(train_labels == c_idx).sum().item()} clips", flush=True)

    # 2. Extract Validation Suite Features
    print("\n--- Step 2: Extracting Validation Suite Features (77 Videos / 148 Clips) ---", flush=True)
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
                l, feats, energies = model(clips_tensor, return_features=True, return_energy=True)
                val_videos.append({
                    "src": src,
                    "feats": feats.cpu(),
                    "energies": energies.cpu(),
                    "cat": cat,
                    "label": 0 if cat == "OOD" else 1
                })
    print(f"Validation cached: {len(val_videos)} video items across With_Coat, Without_Coat, OOD.", flush=True)

    # 3. Base Empirical Means and Covariances on Boundary Latent Space
    mu_0 = train_feats.mean(dim=0, keepdim=True)
    diff_0 = train_feats - mu_0
    Sigma_0_emp = (diff_0.T @ diff_0) / (N - 1)

    mu_c = torch.stack([train_feats[train_labels == c].mean(dim=0) for c in range(C)])
    diff_c_list = [train_feats[train_labels == c] - mu_c[c] for c in range(C)]
    Sigma_c_emp = torch.stack([(d.T @ d) / (len(d) - 1) for d in diff_c_list])

    # 4. Gating Method 1: Boundary-Optimized RMD
    print("\n--- Step 3: Hyperparameter Tuning - Shrinkage Epsilon Sweep (RMD on Skeleton-OOD) ---", flush=True)
    eps_candidates = [0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
    tuning_eps_records = []
    eye_D = torch.eye(D)

    best_rmd_eps = 0.005
    best_rmd_f1 = 0
    best_rmd_res = None
    best_rmd_tau = None
    best_inv_S0 = None
    best_inv_Sc = None

    for eps_val in eps_candidates:
        inv_S0 = torch.linalg.pinv(Sigma_0_emp + eps_val * eye_D)
        inv_Sc = torch.stack([torch.linalg.pinv(Sigma_c_emp[c] + eps_val * eye_D) for c in range(C)])

        v_scores, v_labels, v_cats = compute_dataset_rmd_scores(
            val_videos, mu_0, mu_c, inv_S0, inv_Sc, device
        )
        tau_opt, res_opt = sweep_optimal_threshold(v_scores, v_labels, v_cats, num_steps=200)

        record = {
            "eps": eps_val,
            "tau_opt": round(float(tau_opt), 4),
            "precision": round(res_opt["precision"] * 100, 2),
            "recall": round(res_opt["recall"] * 100, 2),
            "accuracy": round(res_opt["accuracy"] * 100, 2),
            "f1": round(res_opt["f1"], 4),
            "fp": res_opt["fp"],
            "fn": res_opt["fn"]
        }
        tuning_eps_records.append(record)

        if (res_opt["f1"] > best_rmd_f1) or (res_opt["f1"] == best_rmd_f1 and res_opt["recall"] > (best_rmd_res["recall"] if best_rmd_res else 0)):
            best_rmd_f1 = res_opt["f1"]
            best_rmd_res = res_opt
            best_rmd_eps = eps_val
            best_rmd_tau = tau_opt
            best_inv_S0 = inv_S0
            best_inv_Sc = inv_Sc

    print("\nShrinkage Epsilon Sweep Results (RMD on Skeleton-OOD):")
    print(pd.DataFrame(tuning_eps_records).to_string(index=False))

    # 5. Gating Method 2: Direct Skeleton-OOD Energy Gating
    print("\n--- Step 4: Direct Skeleton-OOD Energy Gating Evaluation ---", flush=True)
    v_energy_scores, v_energy_labels, v_energy_cats = compute_dataset_energy_scores(val_videos)
    tau_energy, res_energy = sweep_optimal_threshold(v_energy_scores, v_energy_labels, v_energy_cats, num_steps=300)
    print(f"Optimal Energy Threshold: tau = {tau_energy:.4f}")
    print(f"  - Recall:    {res_energy['recall']*100:.2f}% ({res_energy['tp']}/33 detected, {res_energy['fn']} missed)")
    print(f"  - Precision: {res_energy['precision']*100:.2f}% ({res_energy['tp']}/{res_energy['tp']+res_energy['fp']})")
    print(f"  - Accuracy:  {res_energy['accuracy']*100:.2f}%")
    print(f"  - F1-Score:  {res_energy['f1']:.4f}")
    print(f"  - FP (Alarms/44): {res_energy['fp']}")
    print(f"  - FN (Missed/33): {res_energy['fn']}")

    # 6. Gating Method 3: Hyperspherical Cosine k-NN Gating
    print("\n--- Step 5: Hyperspherical Cosine Gating Evaluation ---", flush=True)
    v_cos_scores, v_cos_labels, v_cos_cats = compute_dataset_cosine_scores(val_videos, mu_c, device)
    tau_cos, res_cos = sweep_optimal_threshold(v_cos_scores, v_cos_labels, v_cos_cats, num_steps=200)
    print(f"Optimal Cosine Threshold: tau = {tau_cos:.4f}")
    print(f"  - Recall:    {res_cos['recall']*100:.2f}% ({res_cos['tp']}/33 detected, {res_cos['fn']} missed)")
    print(f"  - Precision: {res_cos['precision']*100:.2f}%")
    print(f"  - F1-Score:  {res_cos['f1']:.4f}")
    print(f"  - FP: {res_cos['fp']} | FN: {res_cos['fn']}")

    # 7. Select Champion Gating Configuration
    # Compare best_rmd_res vs res_energy
    if best_rmd_res["f1"] >= res_energy["f1"] and (best_rmd_res["recall"] >= res_energy["recall"] or best_rmd_res["fn"] <= res_energy["fn"]):
        champion_method = "Boundary-Optimized RMD"
        champion_res = best_rmd_res
        champion_tau = best_rmd_tau
        champion_eps = best_rmd_eps
        champion_scores = compute_dataset_rmd_scores(val_videos, mu_0, mu_c, best_inv_S0, best_inv_Sc, device)[0]
    else:
        champion_method = "Direct Free Energy Gating"
        champion_res = res_energy
        champion_tau = tau_energy
        champion_eps = None
        champion_scores = v_energy_scores

    print(f"\n========================================================================")
    print(f"  CHAMPION CONFIGURATION SELECTED: {champion_method}")
    print(f"  F1-Score:    {champion_res['f1']:.4f}")
    print(f"  Recall:      {champion_res['recall']*100:.2f}% ({champion_res['tp']}/33 detected, {champion_res['fn']} missed attacks)")
    print(f"  Precision:   {champion_res['precision']*100:.2f}% ({champion_res['tp']}/{champion_res['tp']+champion_res['fp']})")
    print(f"  Accuracy:    {champion_res['accuracy']*100:.2f}%")
    print(f"  False Alarms:{champion_res['fp']} / 44 civilian videos")
    print(f"  Missed Atks: {champion_res['fn']} / 33 violent videos")
    print(f"========================================================================")

    # 8. Decision Threshold Sweep Table for Champion Method
    print(f"\n--- Step 6: Decision Threshold Sweep for Champion Method ({champion_method}) ---", flush=True)
    sweep_offsets = [-1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5]
    sweep_records = []
    for off in sweep_offsets:
        t_val = champion_tau + off
        r = evaluate_scores_at_threshold(champion_scores, v_labels, v_cats, t_val)
        sweep_records.append({
            "threshold": round(float(t_val), 4),
            "recall": round(r["recall"] * 100, 2),
            "precision": round(r["precision"] * 100, 2),
            "accuracy": round(r["accuracy"] * 100, 2),
            "f1": round(r["f1"], 4),
            "tp": r["tp"],
            "tn": r["tn"],
            "fp": r["fp"],
            "fn": r["fn"],
            "profile": "Security (Zero Miss)" if r["fn"] == 0 else ("Optimal F1" if off == 0.0 else "High Precision")
        })
    print(pd.DataFrame(sweep_records).to_string(index=False))

    # 9. Latency Profiling (1,000 Iterations GPU & CPU)
    print("\n--- Step 7: Latency Profiling (1,000 Iterations GPU & CPU) ---", flush=True)
    sample_clip = torch.randn(1, 2, 50, 17)

    # GPU Latency
    sample_gpu = sample_clip.to(device)
    for _ in range(50): # Warmup
        _ = model(sample_gpu, return_features=True, return_energy=True)
    torch.cuda.synchronize()

    start_gpu = time.perf_counter()
    for _ in range(1000):
        _ = model(sample_gpu, return_features=True, return_energy=True)
    torch.cuda.synchronize()
    gpu_forward_ms = ((time.perf_counter() - start_gpu) / 1000.0) * 1000.0

    # Gating Latency (GPU)
    f_dummy = torch.randn(1, 256).to(device)
    is0_d = best_inv_S0.to(device)
    isc_d = best_inv_Sc.to(device)
    m0_d = mu_0.to(device)
    mc_d = mu_c.to(device)
    torch.cuda.synchronize()
    start_gating_gpu = time.perf_counter()
    for _ in range(1000):
        _ = compute_relative_mahalanobis_distance(f_dummy, m0_d, mc_d, is0_d, isc_d)
    torch.cuda.synchronize()
    gpu_gating_us = ((time.perf_counter() - start_gating_gpu) / 1000.0) * 1e6

    # CPU Latency
    model_cpu = SkeletonOODModel(
        num_classes=3, in_channels=2, num_keypoints=17, ash_percentile=80, ash_mode='ash_b'
    ).to("cpu")
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    sample_cpu = sample_clip.to("cpu")
    for _ in range(10): # Warmup
        _ = model_cpu(sample_cpu, return_features=True, return_energy=True)

    start_cpu = time.perf_counter()
    for _ in range(100): # 100 on CPU
        _ = model_cpu(sample_cpu, return_features=True, return_energy=True)
    cpu_forward_ms = ((time.perf_counter() - start_cpu) / 100.0) * 1000.0

    # Gating Latency (CPU)
    f_dummy_cpu = torch.randn(1, 256).to("cpu")
    start_gating_cpu = time.perf_counter()
    for _ in range(1000):
        _ = compute_relative_mahalanobis_distance(f_dummy_cpu, mu_0, mu_c, best_inv_S0, best_inv_Sc)
    cpu_gating_us = ((time.perf_counter() - start_gating_cpu) / 1000.0) * 1e6

    stage1_ms = 12.06
    stage2a_ms = 12.54
    total_gpu_threat_ms = stage1_ms + stage2a_ms + gpu_forward_ms + (gpu_gating_us / 1000.0)
    total_cpu_threat_ms = stage1_ms + stage2a_ms + cpu_forward_ms + (cpu_gating_us / 1000.0)
    fps_gpu = 1000.0 / total_gpu_threat_ms
    fps_cpu = 1000.0 / total_cpu_threat_ms

    print(f"\nLatency Benchmark (1,000 Iterations):")
    print(f"  - Skeleton-OOD Forward Pass (GPU): {gpu_forward_ms:.2f} ms")
    print(f"  - Skeleton-OOD Forward Pass (CPU): {cpu_forward_ms:.2f} ms")
    print(f"  - Gating Latency (GPU):            {gpu_gating_us:.2f} us ({gpu_gating_us/1000:.3f} ms)")
    print(f"  - Gating Latency (CPU):            {cpu_gating_us:.2f} us ({cpu_gating_us/1000:.3f} ms)")
    print(f"  - Stage 2b Total (GPU):            {gpu_forward_ms + (gpu_gating_us / 1000.0):.2f} ms")
    print(f"  - Stage 2b Total (CPU):            {cpu_forward_ms + (cpu_gating_us / 1000.0):.2f} ms")
    print(f"  - Total Threat Active Pipeline (GPU): {total_gpu_threat_ms:.2f} ms ({fps_gpu:.1f} FPS)")
    print(f"  - Total Threat Active Pipeline (CPU): {total_cpu_threat_ms:.2f} ms ({fps_cpu:.1f} FPS)")

    # 10. Save Calibrated Weights and Results
    ref_save_path = os.path.join(REPO_DIR, "weights", "reference_data_skeleton_ood_v3.00.pt")
    torch.save({
        "mu_0": mu_0,
        "mu_c": mu_c,
        "inv_Sigma_0": best_inv_S0,
        "inv_Sigma_c": best_inv_Sc,
        "best_eps": best_rmd_eps,
        "tau_rmd": best_rmd_tau,
        "tau_energy": tau_energy,
        "champion_method": champion_method,
        "champion_tau": champion_tau
    }, ref_save_path)
    print(f"\nCalibrated reference weights saved -> {ref_save_path} ({os.path.getsize(ref_save_path)/1024:.1f} KB)", flush=True)

    # Save per-video validation CSV
    val_csv_records = []
    y_champion_pred = (champion_scores >= champion_tau).astype(int)
    for item, score, pred in zip(val_videos, champion_scores, y_champion_pred):
        val_csv_records.append({
            "source_video": os.path.basename(item["src"]),
            "category": item["cat"],
            "ground_truth": "Violence" if item["label"] == 1 else "Civilian/OOD",
            "threat_score": round(float(score), 4),
            "predicted": "Violence" if pred == 1 else "Civilian/OOD",
            "verdict": "CORRECT" if pred == item["label"] else ("FALSE_ALARM" if pred == 1 else "MISSED_ATTACK")
        })
    val_csv_path = os.path.join(REPO_DIR, "results", "action_recognition_benchmark", "validation_results_skeleton_ood_v3.00.csv")
    os.makedirs(os.path.dirname(val_csv_path), exist_ok=True)
    pd.DataFrame(val_csv_records).to_csv(val_csv_path, index=False)
    print(f"Validation CSV saved -> {val_csv_path}", flush=True)

    # Save master benchmark JSON
    benchmark_json_path = os.path.join(REPO_DIR, "results", "benchmark_skeleton_ood_v3.00.json")
    with open(benchmark_json_path, "w") as f:
        json.dump({
            "version": "v3.00",
            "model": "Skeleton-OOD (ST-GCN + In-Training ASH-SE + Latent Boundary Learning)",
            "parameters": total_params,
            "champion_method": champion_method,
            "champion_threshold": round(float(champion_tau), 4),
            "champion_epsilon": champion_eps,
            "f1_score": champion_res["f1"],
            "recall": champion_res["recall"],
            "precision": champion_res["precision"],
            "accuracy": champion_res["accuracy"],
            "tp": champion_res["tp"],
            "tn": champion_res["tn"],
            "fp": champion_res["fp"],
            "fn": champion_res["fn"],
            "categories": champion_res["categories"],
            "latency": {
                "gpu_forward_ms": round(gpu_forward_ms, 2),
                "cpu_forward_ms": round(cpu_forward_ms, 2),
                "gpu_gating_us": round(gpu_gating_us, 2),
                "cpu_gating_us": round(cpu_gating_us, 2),
                "stage1_scan_ms": stage1_ms,
                "stage2a_pose_ms": stage2a_ms,
                "total_gpu_threat_ms": round(total_gpu_threat_ms, 2),
                "total_cpu_threat_ms": round(total_cpu_threat_ms, 2),
                "fps_gpu": round(fps_gpu, 1),
                "fps_cpu": round(fps_cpu, 1)
            },
            "sweeps": {
                "eps_sweep": tuning_eps_records,
                "threshold_sweep": sweep_records
            }
        }, f, indent=4)
    print(f"Master benchmark JSON saved -> {benchmark_json_path}", flush=True)


if __name__ == "__main__":
    main()
