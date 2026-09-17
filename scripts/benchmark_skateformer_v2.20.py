"""
benchmark_skateformer_v2.20.py

Calibration and Comprehensive Benchmark Suite for SkateFormer Partitioned Skeletal-Temporal Vision Transformer (Version 2.20).
Evaluated on the standardized 148-Clip Action Recognition & OOD Validation Suite.
Archives hyperparameter tuning sweeps (shrinkage epsilon, decision thresholds, distance metrics)
and saves SkateFormer calibrated reference weights.
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

from models.skateformer import SkateFormerModel
from src.dataset import resolve_data_paths, ActionDataset

# Import v1.08 RMD metrics
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
    print("   BENCHMARK: SKATEFORMER PARTITIONED SPATIAL-TEMPORAL ViT (Version 2.20)", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    model_path = os.path.join(REPO_DIR, "weights", "skateformer_violence_v2.20.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please wait for training to complete.", flush=True)
        sys.exit(1)

    train_dir, val_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    model = SkateFormerModel(
        num_classes=1,
        in_channels=2,
        num_frames=50,
        num_keypoints=17,
        embed_dim=128,
        num_blocks=4,
        dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f">> SkateFormer Parameter Count: {total_params:,} parameters", flush=True)
    print(f"   vs ST-GCN (3,014,981): -{(1 - total_params/3014981)*100:.1f}% reduction", flush=True)
    print(f"   vs CTR-GCN (1,333,369): -{(1 - total_params/1333369)*100:.1f}% reduction", flush=True)
    print(f"   vs PoseConv3D (765,529): -{(1 - total_params/765529)*100:.1f}% reduction", flush=True)

    # 1. Extract Training Features and Action Labels
    print("\n--- Step 1: Extracting SkateFormer Training Features (297 Clips) ---", flush=True)
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
    print("\n--- Step 2: Extracting Validation Suite Features (Standardized Validation Suite) ---", flush=True)
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

    # 4. Hyperparameter Tuning Sweep: Shrinkage Epsilon for RMD on SkateFormer
    print("\n--- Step 3: Hyperparameter Tuning - Shrinkage Epsilon Sweep on SkateFormer ---", flush=True)
    eps_candidates = [0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
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

    # 5. Hyperparameter Tuning: Decision Threshold Sweep for Champion Epsilon
    print(f"\n--- Step 4: Hyperparameter Tuning - Decision Threshold Sweep (Champion eps={best_overall_eps}) ---", flush=True)
    inv_S0_opt = torch.linalg.pinv(Sigma_0_emp + best_overall_eps * eye_D)
    inv_Sc_opt = torch.linalg.pinv(Sigma_c_emp + best_overall_eps * eye_C_D)

    v_scores_opt, v_labels_opt, v_cats_opt = compute_dataset_rmd_scores(
        val_videos, mu_0, mu_c, inv_S0_opt, inv_Sc_opt, device
    )

    threshold_sweep_records = []
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
            "operational_profile": profile,
            "category_breakdown": res["category_breakdown"]
        })
        print(f"Tau={t_val:7.4f} | F1={res['f1']:.4f} | Prec={res['precision']*100:.1f}% | Rec={res['recall']*100:.1f}% (TP={res['tp']}/33) | TN={res['tn']} (FP={res['fp']}) | Acc={res['accuracy']*100:.1f}% | Profile={profile}", flush=True)

    optimal_config = next(r for r in threshold_sweep_records if abs(r["threshold"] - round(base_tau, 4)) < 1e-3)
    print(f"\n>> OPTIMAL SKATEFORMER OPERATING POINT: eps={best_overall_eps}, Tau={optimal_config['threshold']}", flush=True)
    print(f"   F1-Score : {optimal_config['f1_score']}", flush=True)
    print(f"   Precision: {optimal_config['precision']}%", flush=True)
    print(f"   Recall   : {optimal_config['recall']}% ({optimal_config['tp']}/33 detected)", flush=True)
    print(f"   Accuracy : {optimal_config['accuracy']}%", flush=True)
    print(f"   FP Alarms: {optimal_config['fp']}", flush=True)

    # 6. Comparison against Hyperspherical Cosine k-NN on SkateFormer embeddings
    print("\n--- Step 5: Hyperspherical Cosine k-NN Comparison on SkateFormer ---", flush=True)
    cos_scores, cos_labels, cos_cats = compute_dataset_cosine_scores(val_videos, mu_c, device)
    best_cos_f1, best_cos_t, best_cos_res = 0, 0, None
    for t_test in np.linspace(np.percentile(cos_scores, 2), np.percentile(cos_scores, 98), 200):
        eval_res = evaluate_scores_at_threshold(cos_scores, cos_labels, cos_cats, t_test)
        if eval_res["f1"] > best_cos_f1:
            best_cos_f1 = eval_res["f1"]
            best_cos_t = t_test
            best_cos_res = eval_res
    print(f"Cosine k-NN on SkateFormer: F1={best_cos_res['f1']:.4f} @ Tau={best_cos_t:.4f} | Prec={best_cos_res['precision']*100:.1f}% | Rec={best_cos_res['recall']*100:.1f}% | FP={best_cos_res['fp']} | FN={best_cos_res['fn']}", flush=True)

    # 7. Detailed Latency Benchmarks (SkateFormer Forward + Gating on GPU & CPU)
    print("\n--- Step 6: Measuring Forward & Gating Latencies (1,000 iterations) ---", flush=True)
    clip_sample = torch.randn(1, 2, 50, 17, 1)
    query_feat = torch.randn(1, D)
    latency_stats = {}

    for dev_name, dev in [("gpu_rtx3090", torch.device("cuda" if torch.cuda.is_available() else "cpu")), ("cpu", torch.device("cpu"))]:
        m_dev = SkateFormerModel(
            num_classes=1, in_channels=2, num_frames=50, num_keypoints=17, embed_dim=128, num_blocks=4, dropout=0.1
        ).to(dev)
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
            for _ in range(50):
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
        # Benchmark Backbone Forward
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(N_iter):
                _ = m_dev(c_d, return_features=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
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
        print(f"[{dev_name.upper()}] Forward: {fwd_lat_ms:.2f} ms | Gating: {gate_lat_us:.2f} us | Stage 2b Total: {stage2b_total_ms:.2f} ms", flush=True)

    # Pipeline Totals (Stage 1: 12.06 ms GPU / 66.60 ms CPU; Stage 2a: 12.54 ms GPU / 70.86 ms CPU)
    gpu_total_ms = round(12.06 + 12.54 + latency_stats["gpu_rtx3090"]["stage2b_total_ms"], 2)
    gpu_fps = round(1000.0 / gpu_total_ms, 1)
    cpu_total_ms = round(66.60 + 70.86 + latency_stats["cpu"]["stage2b_total_ms"], 2)
    cpu_fps = round(1000.0 / cpu_total_ms, 1)

    print(f"\n>> FULL THREAT ACTIVE PIPELINE LATENCY:")
    print(f"   GPU Total (RTX 3090): {gpu_total_ms} ms ({gpu_fps} FPS)")
    print(f"   CPU Total (Host Sim): {cpu_total_ms} ms ({cpu_fps} FPS)")

    # 8. Save Artifacts & Reference Data
    ref_data = {
        "mu_0": mu_0.cpu(),
        "mu_c": mu_c.cpu(),
        "inv_Sigma_0": inv_S0_opt.cpu(),
        "inv_Sigma_c": inv_Sc_opt.cpu(),
        "optimal_epsilon": best_overall_eps,
        "optimal_threshold": float(optimal_config["threshold"]),
        "model_architecture": "SkateFormer (Version 2.20)",
        "feature_dim": D,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    ref_path = os.path.join(REPO_DIR, "weights", "reference_data_skateformer_v2.20.pt")
    torch.save(ref_data, ref_path)
    print(f"\n>> Reference Data Saved to: {ref_path} ({os.path.getsize(ref_path) / 1024:.1f} KB)", flush=True)

    # Save tuning records
    tuning_dir = os.path.join(REPO_DIR, "results", "tuning")
    os.makedirs(tuning_dir, exist_ok=True)
    df_eps = pd.DataFrame(tuning_eps_records)
    df_eps.to_csv(os.path.join(tuning_dir, "tuning_skateformer_v2.20.csv"), index=False)
    with open(os.path.join(tuning_dir, "tuning_skateformer_v2.20.json"), "w") as f:
        json.dump({
            "architecture": "SkateFormer (Version 2.20)",
            "epsilon_sweep": tuning_eps_records,
            "threshold_sweep": threshold_sweep_records,
            "champion_epsilon": best_overall_eps,
            "optimal_threshold": float(optimal_config["threshold"])
        }, f, indent=2)

    # Save category validation breakdown CSV
    val_out_dir = os.path.join(REPO_DIR, "results", "action_recognition_benchmark")
    os.makedirs(val_out_dir, exist_ok=True)
    cat_rows = []
    for cat, metrics in optimal_config["category_breakdown"].items():
        cat_rows.append({
            "category": cat,
            "videos": metrics["videos"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"]
        })
    pd.DataFrame(cat_rows).to_csv(os.path.join(val_out_dir, "validation_results_skateformer_v2.20.csv"), index=False)

    # Master Benchmark JSON
    benchmark_data = {
        "version": "v2.20",
        "technique": "SkateFormer Partitioned Skeletal-Temporal Vision Transformer Backbone",
        "date": time.strftime("%Y-%m-%d"),
        "total_parameters": total_params,
        "champion_epsilon": best_overall_eps,
        "optimal_threshold": float(optimal_config["threshold"]),
        "f1_score": float(optimal_config["f1_score"]),
        "precision": float(optimal_config["precision"]),
        "recall": float(optimal_config["recall"]),
        "accuracy": float(optimal_config["accuracy"]),
        "false_positives": int(optimal_config["fp"]),
        "false_negatives": int(optimal_config["fn"]),
        "category_breakdown": optimal_config["category_breakdown"],
        "cosine_knn_comparison": {
            "f1_score": float(best_cos_res["f1"]),
            "threshold": float(best_cos_t),
            "precision": float(best_cos_res["precision"] * 100),
            "recall": float(best_cos_res["recall"] * 100),
            "fp": int(best_cos_res["fp"]),
            "fn": int(best_cos_res["fn"])
        },
        "latency_stats": latency_stats,
        "pipeline_latency": {
            "gpu_total_ms": gpu_total_ms,
            "gpu_fps": gpu_fps,
            "cpu_total_ms": cpu_total_ms,
            "cpu_fps": cpu_fps
        }
    }
    bench_path = os.path.join(REPO_DIR, "results", "benchmark_skateformer_v2.20.json")
    with open(bench_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f">> Master Benchmark Summary Saved to: {bench_path}", flush=True)


if __name__ == "__main__":
    main()
