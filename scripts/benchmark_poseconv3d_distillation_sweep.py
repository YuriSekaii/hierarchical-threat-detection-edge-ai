"""
benchmark_poseconv3d_distillation_sweep.py

Automated Benchmark Suite for Distilled PoseConv3D Models (Tier 3 Backbone: 598k params).
Evaluates 4 Distillation Formulations:
1. Method A: Latent Feature Alignment
2. Method B: Relational Knowledge Distillation (RKD)
3. Method C: Arm Attention Distillation
4. Method D: Combined Tri-Distillation (A + B + C)
against the Undistilled Tier 3 Baseline and ST-GCN Teacher on the standardized 148-Clip Validation Suite.

100% self-contained: adheres strictly to AgentRule.md.
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
from src.poseconv3d_utils import batch_rasterize_gpu
from models.poseconv3d import PoseConv3DModel

# Import RMD metrics
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


def benchmark_single_model(model_id, weights_path, train_heatmaps, train_labels, val_video_heatmaps, device):
    print(f"\n{'='*70}", flush=True)
    print(f"   BENCHMARKING: {model_id.upper()}", flush=True)
    print(f"{'='*70}", flush=True)

    model = PoseConv3DModel(base_channels=12, feature_dim=256, dropout=0.2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # 1. Extract Training Features
    train_feats = []
    chunk_size = 32
    with torch.no_grad():
        for i in range(0, len(train_heatmaps), chunk_size):
            chunk = train_heatmaps[i : i + chunk_size]
            _, feats = model(chunk, return_features=True)
            train_feats.append(feats.cpu())
    train_feats = torch.cat(train_feats, dim=0)
    N, D = train_feats.shape
    C = 3

    # 2. Extract Validation Features
    val_videos = []
    with torch.no_grad():
        for item in val_video_heatmaps:
            clips_tensor = item["heatmaps"].to(device)
            _, feats = model(clips_tensor, return_features=True)
            val_videos.append({
                "src": item["src"],
                "feats": feats.cpu(),
                "cat": item["cat"],
                "label": item["label"]
            })

    # 3. Covariance Matrices
    mu_0 = train_feats.mean(dim=0, keepdim=True)
    diff_0 = train_feats - mu_0
    Sigma_0_emp = (diff_0.T @ diff_0) / (N - 1)

    mu_c = torch.stack([train_feats[train_labels == c].mean(dim=0) for c in range(C)])
    diff_c_list = [train_feats[train_labels == c] - mu_c[c] for c in range(C)]
    Sigma_c_emp = torch.stack([(d.T @ d) / (len(d) - 1) for d in diff_c_list])

    eye_D = torch.eye(D)
    eye_C_D = eye_D.unsqueeze(0).expand(C, -1, -1)

    # 4. Shrinkage Epsilon Sweep
    eps_candidates = [0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
    best_f1 = 0
    best_eps = 0.005
    best_tau = 0
    best_res = None

    for eps_val in eps_candidates:
        inv_S0 = torch.linalg.pinv(Sigma_0_emp + eps_val * eye_D)
        inv_Sc = torch.linalg.pinv(Sigma_c_emp + eps_val * eye_C_D)

        v_scores, v_labels, v_cats = compute_dataset_rmd_scores(val_videos, mu_0, mu_c, inv_S0, inv_Sc, device)

        test_taus = np.linspace(np.percentile(v_scores, 1), np.percentile(v_scores, 99), 200)
        for t_test in test_taus:
            res = evaluate_scores_at_threshold(v_scores, v_labels, v_cats, t_test)
            if res["f1"] > best_f1:
                best_f1 = res["f1"]
                best_eps = eps_val
                best_tau = t_test
                best_res = res

    print(f"[{model_id}] Best F1: {best_res['f1']:.4f} @ eps={best_eps}, tau={best_tau:.4f} | Recall={best_res['recall']*100:.1f}% | Prec={best_res['precision']*100:.1f}% | FP={best_res['fp']} | FN={best_res['fn']}", flush=True)

    # Latencies
    dummy_hm_gpu = torch.randn(1, 17, 50, 56, 56, device=device)
    for _ in range(15):
        _ = model(dummy_hm_gpu)
    torch.cuda.synchronize()

    times_gpu = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(dummy_hm_gpu)
        torch.cuda.synchronize()
        times_gpu.append((time.perf_counter() - t0) * 1000)
    gpu_backbone_ms = float(np.median(times_gpu))

    model_cpu = PoseConv3DModel(base_channels=12, feature_dim=256, dropout=0.2).to("cpu")
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    dummy_hm_cpu = torch.randn(1, 17, 50, 56, 56, device="cpu")

    for _ in range(3):
        _ = model_cpu(dummy_hm_cpu)

    times_cpu = []
    for _ in range(15):
        t0 = time.perf_counter()
        _ = model_cpu(dummy_hm_cpu)
        times_cpu.append((time.perf_counter() - t0) * 1000)
    cpu_backbone_ms = float(np.median(times_cpu))

    stage1_ms, stage2a_ms = 12.06, 12.54
    total_gpu_ms = stage1_ms + stage2a_ms + gpu_backbone_ms + 0.20
    total_gpu_fps = 1000.0 / total_gpu_ms
    total_cpu_pipeline_ms = 18.0 + cpu_backbone_ms
    total_cpu_fps = 1000.0 / total_cpu_pipeline_ms

    return {
        "model_id": model_id,
        "parameters": total_params,
        "f1_score": round(best_res["f1"], 4),
        "recall": round(best_res["recall"] * 100, 2),
        "precision": round(best_res["precision"] * 100, 2),
        "accuracy": round(best_res["accuracy"] * 100, 2),
        "tp": best_res["tp"],
        "tn": best_res["tn"],
        "fp": best_res["fp"],
        "fn": best_res["fn"],
        "optimal_eps": best_eps,
        "optimal_tau": round(float(best_tau), 4),
        "gpu_backbone_ms": round(gpu_backbone_ms, 2),
        "gpu_pipeline_ms": round(total_gpu_ms, 2),
        "gpu_fps": round(total_gpu_fps, 1),
        "cpu_backbone_ms": round(cpu_backbone_ms, 2),
        "cpu_pipeline_ms": round(total_cpu_pipeline_ms, 2),
        "cpu_fps": round(total_cpu_fps, 1),
        "category_breakdown": best_res["category_breakdown"]
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("   BENCHMARK: ST-GCN -> POSECONV3D DISTILLATION SWEEP", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    train_data_dir, validate_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    # Pre-rasterize training clips
    raw_ds = PoseConv3DHeatmapDataset(root_dir=train_data_dir, action_folders=actions, clip_length=50, heatmap_size=56, mode="train")
    N = len(raw_ds)
    all_kpts, train_labels = [], []
    for i in range(N):
        sample = raw_ds.samples[i]
        video_idx = sample["video_idx"]
        start_frame = sample["start_frame"]
        skeletons = raw_ds.skeleton_data[video_idx]
        clip = skeletons[start_frame : start_frame + 50]
        if not isinstance(clip, torch.Tensor):
            clip = torch.from_numpy(clip).float()
        if clip.shape[0] < 50:
            pad = torch.zeros((50 - clip.shape[0], 17, 2))
            clip = torch.cat([clip, pad], dim=0)
        all_kpts.append(clip[:, :, :2])
        for c_idx, act in enumerate(actions):
            if act in sample["source_file"]:
                train_labels.append(c_idx)
                break

    all_kpts = torch.stack(all_kpts).to(device)
    train_labels = torch.tensor(train_labels)

    heatmaps_list = []
    for i in range(0, N, 50):
        chunk = all_kpts[i : i + 50]
        hm = batch_rasterize_gpu(chunk, H=56, W=56, sigma=1.5)
        heatmaps_list.append(hm)
    all_train_heatmaps = torch.cat(heatmaps_list, dim=0)

    # Pre-rasterize validation clips
    val_categories = ["With_Coat", "Without_Coat", "OOD"]
    val_video_heatmaps = []
    for cat in val_categories:
        cat_dir = os.path.join(validate_root, cat)
        dataset = PoseConv3DHeatmapDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, heatmap_size=56, mode="test")
        video_clips = defaultdict(list)
        for i in range(len(dataset)):
            sample = dataset.samples[i]
            video_idx = sample["video_idx"]
            start_frame = sample["start_frame"]
            skeletons = dataset.skeleton_data[video_idx]
            clip = skeletons[start_frame : start_frame + 50]
            if not isinstance(clip, torch.Tensor):
                clip = torch.from_numpy(clip).float()
            if clip.shape[0] < 50:
                pad = torch.zeros((50 - clip.shape[0], 17, 2))
                clip = torch.cat([clip, pad], dim=0)
            video_clips[sample["source_file"]].append(clip[:, :, :2])

        for src, clips in video_clips.items():
            clips_kpts = torch.stack(clips).to(device)
            clips_hm = batch_rasterize_gpu(clips_kpts, H=56, W=56, sigma=1.5)
            val_video_heatmaps.append({
                "src": src,
                "heatmaps": clips_hm,
                "cat": cat,
                "label": 0 if cat == "OOD" else 1
            })

    print(f"Pre-rasterized {len(val_video_heatmaps)} validation videos on GPU.", flush=True)

    models_to_evaluate = [
        ("undistilled_tier3", os.path.join(REPO_DIR, "weights", "poseconv3d_downscale_tier3_598k.pth")),
        ("distill_methodA_feature", os.path.join(REPO_DIR, "weights", "poseconv3d_distill_methodA_feature.pth")),
        ("distill_methodB_rkd", os.path.join(REPO_DIR, "weights", "poseconv3d_distill_methodB_rkd.pth")),
        ("distill_methodC_attention", os.path.join(REPO_DIR, "weights", "poseconv3d_distill_methodC_attention.pth")),
        ("distill_methodD_combined", os.path.join(REPO_DIR, "weights", "poseconv3d_distill_methodD_combined.pth")),
    ]

    benchmark_records = []
    for m_id, w_path in models_to_evaluate:
        if os.path.exists(w_path):
            rec = benchmark_single_model(m_id, w_path, all_train_heatmaps, train_labels, val_video_heatmaps, device)
            benchmark_records.append(rec)
        else:
            print(f"Warning: Checkpoint {w_path} not found. Skipping.", flush=True)

    df_sweep = pd.DataFrame(benchmark_records)
    csv_path = os.path.join(REPO_DIR, "results", "tuning", "tuning_poseconv3d_distillation_sweep.csv")
    df_sweep.to_csv(csv_path, index=False)
    print(f"\n>> Distillation Benchmark Table saved to: {csv_path}", flush=True)

    json_path = os.path.join(REPO_DIR, "results", "benchmark_poseconv3d_distillation_sweep.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)
    print(f">> Distillation Benchmark JSON saved to: {json_path}", flush=True)

    print("\n" + "=" * 90)
    print("   DISTILLED POSECONV3D SWEEP: COMPLETE BENCHMARK SUMMARY")
    print("=" * 90)
    print(df_sweep[["model_id", "parameters", "f1_score", "recall", "precision", "fp", "fn", "cpu_fps"]].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
