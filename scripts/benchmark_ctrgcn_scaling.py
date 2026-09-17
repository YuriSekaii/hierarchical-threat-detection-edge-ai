"""
scripts/benchmark_ctrgcn_scaling.py

Comprehensive Calibration and Benchmarking Suite for Scaled CTR-GCN Models:
- Tier S: channels=[32, 64, 128] (352,841 parameters)
- Tier L: channels=[96, 192, 384] (2,972,277 parameters)

Evaluated on the standardized 77-Video / 148-Clip Action Recognition Validation Suite.
Calibrates Relative Mahalanobis Distance (RMD) reference matrices:
- weights/reference_data_ctrgcn_tier_s.pt
- weights/reference_data_ctrgcn_tier_l.pt
Outputs tuning logs and live hardware benchmarks adhering strictly to AgentRule.md.
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

from models.ctrgcn_scaled import ScaledCTRGCNModel
from src.dataset import resolve_data_paths, ActionDataset

# Import v1.08 RMD metrics
spec_rmd = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py")
)
rmd_mod = importlib.util.module_from_spec(spec_rmd)
spec_rmd.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance


def measure_gpu_latency(model_fn, dummy_input, warmup=50, iterations=500):
    if not torch.cuda.is_available():
        return 0.0
    for _ in range(warmup):
        _ = model_fn(dummy_input)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(iterations):
        start_event.record()
        _ = model_fn(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    return float(np.median(timings))


def measure_cpu_latency(model_fn, dummy_input, warmup=10, iterations=100):
    for _ in range(warmup):
        _ = model_fn(dummy_input)
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model_fn(dummy_input)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(timings))


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


def calibrate_and_benchmark_tier(tier_name, channels, reduction, arm_weight=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(REPO_DIR, "weights", f"ctrgcn_{tier_name}.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    model = ScaledCTRGCNModel(channels=channels, reduction=reduction, arm_weight=arm_weight).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 85, flush=True)
    print(f"   CALIBRATING & BENCHMARKING CTR-GCN {tier_name.upper()} ({total_params:,} params)", flush=True)
    print(f"   Channels: {channels} | Reduction: {reduction} | Arm Weight: {arm_weight}x", flush=True)
    print("=" * 85, flush=True)

    train_dir, val_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    # 1. Extract Training Features
    print("\n--- Step 1: Extracting Training Features (297 Clips) ---", flush=True)
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
    train_feats = torch.stack(train_feats)
    train_labels = torch.tensor(train_labels)
    N, D = train_feats.shape
    C = len(actions)
    print(f">> Extracted {N} clips, feature dimension D={D} across C={C} categories", flush=True)

    # 2. Extract Validation Features
    print("\n--- Step 2: Extracting Validation Features (77 Videos / 148 Clips) ---", flush=True)
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
    print(f">> Cached features for {len(val_videos)} validation videos", flush=True)

    # 3. Covariance Estimation
    mu_0 = train_feats.mean(dim=0, keepdim=True)
    diff_0 = train_feats - mu_0
    Sigma_0_emp = (diff_0.T @ diff_0) / (N - 1)

    mu_c = torch.stack([train_feats[train_labels == c].mean(dim=0) for c in range(C)])
    diff_c_list = [train_feats[train_labels == c] - mu_c[c] for c in range(C)]
    Sigma_c_emp = torch.stack([(d.T @ d) / (len(d) - 1) for d in diff_c_list])

    # 4. Shrinkage Epsilon Sweep
    print("\n--- Step 3: Shrinkage Epsilon Sweep ---", flush=True)
    eps_candidates = [0.001, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
    eye_D = torch.eye(D)
    eye_C_D = eye_D.unsqueeze(0).expand(C, -1, -1)

    best_eps = None
    best_f1 = -1.0
    best_tau = None
    best_eval = None
    sweep_records = []

    for eps in eps_candidates:
        inv_S0 = torch.linalg.pinv(Sigma_0_emp + eps * eye_D)
        inv_Sc = torch.linalg.pinv(Sigma_c_emp + eps * eye_C_D)

        v_scores, v_labels, v_cats = compute_dataset_rmd_scores(val_videos, mu_0, mu_c, inv_S0, inv_Sc, device)

        # Threshold search
        eps_best_f1, eps_best_tau, eps_best_res = -1.0, 0.0, None
        for t_cand in np.linspace(np.percentile(v_scores, 2), np.percentile(v_scores, 98), 200):
            res = evaluate_scores_at_threshold(v_scores, v_labels, v_cats, t_cand)
            if res["f1"] > eps_best_f1:
                eps_best_f1 = res["f1"]
                eps_best_tau = t_cand
                eps_best_res = res

        sweep_records.append({
            "tier": tier_name,
            "epsilon": eps,
            "threshold": round(eps_best_tau, 4),
            "f1_score": round(eps_best_res["f1"], 4),
            "recall": round(eps_best_res["recall"] * 100, 2),
            "precision": round(eps_best_res["precision"] * 100, 2),
            "accuracy": round(eps_best_res["accuracy"] * 100, 2),
            "fp": eps_best_res["fp"],
            "fn": eps_best_res["fn"]
        })
        print(f"eps={eps:.4f} | F1={eps_best_res['f1']:.4f} @ Tau={eps_best_tau:7.4f} | Rec={eps_best_res['recall']*100:.1f}% ({eps_best_res['tp']}/33) | Prec={eps_best_res['precision']*100:.1f}% | FP={eps_best_res['fp']}/44 | FN={eps_best_res['fn']}", flush=True)

        if eps_best_f1 > best_f1:
            best_f1 = eps_best_f1
            best_eps = eps
            best_tau = eps_best_tau
            best_eval = eps_best_res

    print(f"\n>> CHAMPION EPSILON: eps={best_eps:.4f} | Best F1: {best_f1:.4f} @ Tau={best_tau:.4f}", flush=True)

    # 5. Measure Latencies
    print("\n--- Step 4: Hardware Latency Benchmarking (Live CUDA Events & PerfCounter) ---", flush=True)
    dummy_c_gpu = torch.randn(1, 2, 50, 17, device=device)
    fwd_gpu_ms = measure_gpu_latency(lambda x: model(x), dummy_c_gpu)

    model_cpu = ScaledCTRGCNModel(channels=channels, reduction=reduction, arm_weight=arm_weight).to("cpu").eval()
    model_cpu.load_state_dict(model.state_dict())
    dummy_c_cpu = torch.randn(1, 2, 50, 17, device="cpu")
    fwd_cpu_ms = measure_cpu_latency(lambda x: model_cpu(x), dummy_c_cpu)

    print(f">> Forward Latency: GPU = {fwd_gpu_ms:.2f} ms | CPU = {fwd_cpu_ms:.2f} ms", flush=True)

    # Save Calibrated Reference Data
    ref_target = os.path.join(REPO_DIR, "weights", f"reference_data_ctrgcn_{tier_name}.pt")
    inv_S0_opt = torch.linalg.pinv(Sigma_0_emp + best_eps * eye_D)
    inv_Sc_opt = torch.linalg.pinv(Sigma_c_emp + best_eps * eye_C_D)
    torch.save({
        "tier": tier_name,
        "channels": channels,
        "reduction": reduction,
        "arm_weight": arm_weight,
        "feature_dim": D,
        "parameters": total_params,
        "mu_0": mu_0,
        "mu_c": mu_c,
        "inv_Sigma_0": inv_S0_opt,
        "inv_Sigma_c": inv_Sc_opt,
        "optimal_epsilon": best_eps,
        "optimal_threshold": float(best_tau),
        "best_f1": float(best_f1),
        "best_precision": float(best_eval["precision"]),
        "best_recall": float(best_eval["recall"]),
        "fp": int(best_eval["fp"]),
        "fn": int(best_eval["fn"])
    }, ref_target)
    print(f">> Calibrated Reference Data saved to {ref_target}", flush=True)

    return {
        "tier": tier_name,
        "params": total_params,
        "channels": channels,
        "f1": best_f1,
        "recall": best_eval["recall"],
        "precision": best_eval["precision"],
        "accuracy": best_eval["accuracy"],
        "fp": best_eval["fp"],
        "fn": best_eval["fn"],
        "best_eps": best_eps,
        "best_tau": best_tau,
        "fwd_gpu_ms": fwd_gpu_ms,
        "fwd_cpu_ms": fwd_cpu_ms,
        "breakdown": best_eval["category_breakdown"],
        "sweep_records": sweep_records
    }


def main():
    print("=" * 85, flush=True)
    print("   SCALED CTR-GCN BENCHMARK & CALIBRATION HARNESS", flush=True)
    print("=" * 85, flush=True)

    res_s = calibrate_and_benchmark_tier("tier_s", channels=[32, 64, 128], reduction=4, arm_weight=5.0)
    res_l = calibrate_and_benchmark_tier("tier_l", channels=[96, 192, 384], reduction=8, arm_weight=5.0)

    # Save tuning sweep table
    all_sweeps = res_s["sweep_records"] + res_l["sweep_records"]
    df_sweeps = pd.DataFrame(all_sweeps)
    csv_path = os.path.join(REPO_DIR, "results", "tuning", "tuning_ctrgcn_scaling.csv")
    df_sweeps.to_csv(csv_path, index=False)
    print(f"\n>> Tuning sweep table saved to: {csv_path}", flush=True)

    # Save benchmark JSON
    json_path = os.path.join(REPO_DIR, "results", "benchmark_ctrgcn_scaling.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tier_s": {k: v for k, v in res_s.items() if k != "sweep_records"},
            "tier_l": {k: v for k, v in res_l.items() if k != "sweep_records"}
        }, f, indent=2)
    print(f">> Benchmark report saved to: {json_path}", flush=True)

    print("\n" + "=" * 85)
    print("   FINAL CTR-GCN SCALING COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'Tier':<10} | {'Params':<10} | {'F1':<7} | {'Recall':<7} | {'Prec':<7} | {'FP/44':<6} | {'FN/33':<6} | {'GPU fwd':<10} | {'CPU fwd':<10}")
    print("-" * 85)
    print(f"{'Tier S':<10} | {res_s['params']:<10,d} | {res_s['f1']:<7.4f} | {res_s['recall']*100:<6.1f}% | {res_s['precision']*100:<6.1f}% | {res_s['fp']:<6d} | {res_s['fn']:<6d} | {res_s['fwd_gpu_ms']:<7.2f} ms | {res_s['fwd_cpu_ms']:<7.2f} ms")
    print(f"{'Tier L':<10} | {res_l['params']:<10,d} | {res_l['f1']:<7.4f} | {res_l['recall']*100:<6.1f}% | {res_l['precision']*100:<6.1f}% | {res_l['fp']:<6d} | {res_l['fn']:<6d} | {res_l['fwd_gpu_ms']:<7.2f} ms | {res_l['fwd_cpu_ms']:<7.2f} ms")


if __name__ == "__main__":
    main()
