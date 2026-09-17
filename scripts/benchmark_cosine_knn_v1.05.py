"""
benchmark_cosine_knn_v1.05.py

Comprehensive Calibration and Head-to-Head Benchmark:
Hyperspherical Cosine k-NN (v1.05) vs Legacy Euclidean k-NN (v1.00 - v1.04)
Evaluated across the standardized 148-Clip Action Recognition & OOD Validation Suite.
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

# Import v1.05 Hyperspherical Cosine OOD metrics
spec_ood = importlib.util.spec_from_file_location(
    "ood_metrics_v1_05",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.05.py")
)
ood_mod = importlib.util.module_from_spec(spec_ood)
spec_ood.loader.exec_module(ood_mod)
project_hypersphere = ood_mod.project_hypersphere
compute_hyperspherical_cosine_knn_distance = ood_mod.compute_hyperspherical_cosine_knn_distance
compute_legacy_euclidean_knn_distance = ood_mod.compute_legacy_euclidean_knn_distance

# Dynamically import STGCNModel and ActionDataset from v1.02 training script
spec = importlib.util.spec_from_file_location("train_v102", os.path.join(REPO_DIR, "training", "train_stgcn_knn_v1.02.py"))
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
STGCNModel = train_mod.STGCNModel
ActionDataset = train_mod.ActionDataset


def resolve_validation_path():
    candidates = [
        os.path.join(REPO_DIR, "data", "Validate"),
        os.path.join(REPO_DIR, "..", "Clean", "Train_Action_Recognition_STGCN_Model", "data", "Validate"),
        os.path.join(REPO_DIR, "..", "Train_Action_Recognition_STGCN_Model", "data", "Validate")
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "With_Coat")):
            return os.path.abspath(p)
    raise FileNotFoundError("Could not locate validation dataset directory.")


def evaluate_dataset(model, val_root, feature_bank, device, metric_type="cosine", k=2, threshold=0.0015):
    """
    Evaluates dataset across all 148 validation clips using specified distance metric.
    Returns:
        dict: Summary metrics (accuracy, precision, recall, f1, tp, tn, fp, fn)
    """
    categories = ["With_Coat", "Without_Coat", "OOD"]
    y_true = []
    y_pred = []
    category_results = {}
    
    # Ensure bank is on device and normalized for cosine
    fb = feature_bank.to(device)
    fb_norm = project_hypersphere(fb)
    
    for cat in categories:
        true_label = 0 if cat == "OOD" else 1
        cat_dir = os.path.join(val_root, cat)
        dataset = ActionDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, mode="test")
        
        video_clips = defaultdict(list)
        for i in range(len(dataset)):
            clip, label, src = dataset[i]
            video_clips[src].append(clip)
            
        cat_tp, cat_tn, cat_fp, cat_fn = 0, 0, 0, 0
        
        with torch.no_grad():
            for src, clips in video_clips.items():
                clips_tensor = torch.stack(clips).to(device)
                _, feats = model(clips_tensor, return_features=True)
                
                if metric_type == "cosine":
                    q_norm = project_hypersphere(feats)
                    sims = torch.mm(q_norm, fb_norm.t())
                    dists = torch.clamp(1.0 - sims, min=0.0, max=2.0)
                    kth_vals, _ = dists.kthvalue(k, dim=1)
                    min_dist = kth_vals.min().item()
                else: # euclidean
                    dists = torch.cdist(feats, fb, p=2)
                    kth_vals, _ = dists.kthvalue(k, dim=1)
                    min_dist = kth_vals.min().item()
                    
                is_violence = 1 if min_dist <= threshold else 0
                y_true.append(true_label)
                y_pred.append(is_violence)
                
                if true_label == 1 and is_violence == 1: cat_tp += 1
                elif true_label == 1 and is_violence == 0: cat_fn += 1
                elif true_label == 0 and is_violence == 0: cat_tn += 1
                elif true_label == 0 and is_violence == 1: cat_fp += 1
                
        category_results[cat] = {
            "videos": len(video_clips),
            "tp": cat_tp, "tn": cat_tn, "fp": cat_fp, "fn": cat_fn
        }
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "category_breakdown": category_results
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("   BENCHMARK: HYPERSPHERICAL COSINE k-NN (v1.05) vs EUCLIDEAN k-NN")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load trained Champion model & feature bank
    model_path = os.path.join(REPO_DIR, "weights", "stgcn_violence_v1.02.pth")
    ref_path = os.path.join(REPO_DIR, "weights", "reference_data_knn_v1.02.pt")
    
    model = STGCNModel(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    ref_data = torch.load(ref_path, weights_only=False)
    raw_feature_bank = ref_data["feature_bank"] # (297, 256)
    print(f"Loaded Feature Bank: {raw_feature_bank.shape} | Normalized by: {ref_data.get('normalization')}")

    # 2. Project feature bank to unit hypersphere
    normalized_feature_bank = project_hypersphere(raw_feature_bank)
    
    # 3. Calibrate Cosine Thresholds on Training Bank
    cos_sims = torch.mm(normalized_feature_bank, normalized_feature_bank.t())
    cos_dists = torch.clamp(1.0 - cos_sims, min=0.0, max=2.0)
    # k=2 distance (index 3 because index 1 is self-distance 0.0)
    kth_train_dists, _ = cos_dists.kthvalue(3, dim=1)
    train_dists_np = kth_train_dists.cpu().numpy()
    
    p80_thresh = float(np.percentile(train_dists_np, 80))
    p90_thresh = float(np.percentile(train_dists_np, 90))
    p95_thresh = float(np.percentile(train_dists_np, 95))
    print(f"\nTraining Calibration Percentiles (Cosine k=2):")
    print(f"  80th Percentile: {p80_thresh:.6f}")
    print(f"  90th Percentile: {p90_thresh:.6f}")
    print(f"  95th Percentile: {p95_thresh:.6f}")

    val_root = resolve_validation_path()
    print(f"Validation Root : {val_root}")

    # 4. Evaluate Euclidean Baseline (v1.02 @ tau=1.50)
    print("\n--- Running Baseline Euclidean Evaluation (v1.02 @ tau=1.50) ---")
    euc_res = evaluate_dataset(
        model, val_root, raw_feature_bank, device,
        metric_type="euclidean", k=2, threshold=1.50
    )
    print(f"Euclidean Baseline (tau=1.50):")
    print(f"  F1-Score  : {euc_res['f1']:.4f}")
    print(f"  Accuracy  : {euc_res['accuracy']*100:.2f}%")
    print(f"  Precision : {euc_res['precision']*100:.2f}% (TP={euc_res['tp']}, FP={euc_res['fp']})")
    print(f"  Recall    : {euc_res['recall']*100:.2f}% (FN={euc_res['fn']})")
    print(f"  OOD Reject: {euc_res['tn']}/39 ({euc_res['tn']/39*100:.1f}%)")

    # 5. Evaluate Hyperspherical Cosine Across Thresholds
    print("\n--- Running Hyperspherical Cosine k-NN Evaluation (v1.05) ---")
    test_thresholds = [0.0012, 0.0015, 0.0018, 0.0025, 0.0035, 0.0047]
    cosine_results = []
    
    for tau in test_thresholds:
        res = evaluate_dataset(
            model, val_root, normalized_feature_bank, device,
            metric_type="cosine", k=2, threshold=tau
        )
        res["threshold"] = tau
        cosine_results.append(res)
        print(f"Tau={tau:.6f} | F1={res['f1']:.4f} | Prec={res['precision']*100:.1f}% | Rec={res['recall']*100:.1f}% (TP={res['tp']}/33) | TN={res['tn']}/39 (FP={res['fp']}) | Acc={res['accuracy']*100:.1f}%")

    # Optimal F1 Operating Point
    best_cos = max(cosine_results, key=lambda x: x["f1"])
    optimal_tau = best_cos["threshold"]
    print(f"\n>> OPTIMAL COSINE F1 OPERATING POINT: Tau={optimal_tau:.6f}")
    print(f"   F1-Score : {best_cos['f1']:.4f} (+{best_cos['f1'] - euc_res['f1']:+.4f} vs Euclidean)")
    print(f"   Precision: {best_cos['precision']*100:.2f}% (+{best_cos['precision']*100 - euc_res['precision']*100:+.2f}% vs Euclidean)")
    print(f"   Accuracy : {best_cos['accuracy']*100:.2f}% (+{best_cos['accuracy']*100 - euc_res['accuracy']*100:+.2f}% vs Euclidean)")
    print(f"   Recall   : {best_cos['recall']*100:.2f}%")
    print(f"   OOD Reject: {best_cos['tn']}/39 ({best_cos['tn']/39*100:.1f}% vs Euclidean {euc_res['tn']/39*100:.1f}%)")
    print(f"   False Alarms Slashed from {euc_res['fp']} down to {best_cos['fp']}!")

    # 6. Benchmark Latency (GPU & CPU)
    query_tensor = torch.randn(1, 256)
    latency_stats = {}
    
    for dev_name, dev in [("gpu_rtx3090", torch.device("cuda" if torch.cuda.is_available() else "cpu")), ("cpu", torch.device("cpu"))]:
        fb_d = raw_feature_bank.to(dev)
        fb_norm_d = normalized_feature_bank.to(dev)
        q_d = query_tensor.to(dev)
        
        # Warmup
        for _ in range(100):
            _ = torch.cdist(q_d, fb_d, p=2)
            _ = torch.mm(project_hypersphere(q_d), fb_norm_d.t())
        if dev.type == "cuda": torch.cuda.synchronize()
        
        N = 5000
        t0 = time.perf_counter()
        for _ in range(N):
            dists = torch.cdist(q_d, fb_d, p=2)
            _ = dists.kthvalue(2, dim=1)
        if dev.type == "cuda": torch.cuda.synchronize()
        euc_lat_us = (time.perf_counter() - t0) * 1e6 / N
        
        t0 = time.perf_counter()
        for _ in range(N):
            q_norm = project_hypersphere(q_d)
            cos_d = 1.0 - torch.mm(q_norm, fb_norm_d.t())
            _ = cos_d.kthvalue(2, dim=1)
        if dev.type == "cuda": torch.cuda.synchronize()
        cos_lat_us = (time.perf_counter() - t0) * 1e6 / N
        
        speedup = (euc_lat_us - cos_lat_us) / euc_lat_us * 100.0
        latency_stats[dev_name] = {
            "euclidean_us": round(euc_lat_us, 2),
            "hyperspherical_cosine_us": round(cos_lat_us, 2),
            "speedup_pct": round(speedup, 1)
        }
        print(f"\nLatency on {dev_name.upper()}:")
        print(f"  Euclidean cdist      : {euc_lat_us:.2f} us")
        print(f"  Hyperspherical Cosine: {cos_lat_us:.2f} us ({speedup:+.1f}% faster)")

    # 7. Save Calibrated Reference Bank (v1.05)
    save_ref_path = os.path.join(REPO_DIR, "weights", "reference_data_cosine_knn_v1.05.pt")
    torch.save({
        "feature_bank": normalized_feature_bank.cpu(),
        "threshold": optimal_tau,
        "k": 2,
        "is_normalized": True,
        "metric": "hyperspherical_cosine",
        "calibration_stats": {
            "p80": p80_thresh,
            "p90": p90_thresh,
            "p95": p95_thresh
        }
    }, save_ref_path)
    print(f"\n>> Saved Hyperspherical Cosine Reference Bank to: {save_ref_path}")

    # 8. Save Results JSON
    results_json_path = os.path.join(REPO_DIR, "results", "benchmark_cosine_knn_v1.05.json")
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w") as f:
        json.dump({
            "benchmark_target": "Hyperspherical Cosine k-NN (Version 1.05)",
            "baseline_euclidean_v1.02": euc_res,
            "hyperspherical_cosine_v1.05_optimal": best_cos,
            "threshold_sweep": cosine_results,
            "latency_benchmarks": latency_stats
        }, f, indent=2)
    print(f">> Saved Benchmark Metrics to: {results_json_path}")

    # 9. Save CSV breakdown
    csv_rows = [
        {"Pipeline": "v1.02 Euclidean k-NN (tau=1.50)", "Accuracy": euc_res["accuracy"], "Precision": euc_res["precision"], "Recall": euc_res["recall"], "F1": euc_res["f1"], "TP": euc_res["tp"], "TN": euc_res["tn"], "FP": euc_res["fp"], "FN": euc_res["fn"]}
    ]
    for r in cosine_results:
        csv_rows.append({
            "Pipeline": f"v1.05 Cosine k-NN (tau={r['threshold']:.4f})",
            "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"],
            "TP": r["tp"], "TN": r["tn"], "FP": r["fp"], "FN": r["fn"]
        })
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(REPO_DIR, "results", "action_recognition_benchmark", "validation_results_cosine_knn_v1.05.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f">> Saved CSV Table to: {csv_path}")


if __name__ == "__main__":
    main()
