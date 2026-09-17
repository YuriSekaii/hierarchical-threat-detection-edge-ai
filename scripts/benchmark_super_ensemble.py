"""
scripts/benchmark_super_ensemble.py

Official Comprehensive Benchmark Harness for Multi-Model Super-Ensembles.
Measures live GPU and CPU execution latencies with zero hardcoded numbers.
Adheres strictly to AgentRule.md and provides per-category breakdowns across all 77 validation videos.
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset, PoseConv3DHeatmapDataset
from src.poseconv3d_utils import rasterize_keypoints_to_heatmap
from src.skeleton_utils import normalize_skeleton_clip, smooth_kinematics
from models.ensemble_super import MODEL_REGISTRY, load_model_instance, SuperEnsembleSystem


def measure_gpu_latency(fn, warmup=50, iterations=500):
    if not torch.cuda.is_available():
        return 0.0
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(iterations):
        start_event.record()
        _ = fn()
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    return float(np.median(timings))


def measure_cpu_latency(fn, warmup=10, iterations=100):
    for _ in range(warmup):
        _ = fn()
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(timings))


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
    print("=" * 90, flush=True)
    print("   OFFICIAL LIVE BENCHMARK OF SUPER-ENSEMBLE ARCHITECTURES", flush=True)
    print("   Live Timers - Zero Hardcoded Numbers - Standardized 77-Video Suite", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Primary Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load Data for 77 Validation Videos
    train_dir, val_root = resolve_data_paths()
    val_categories = ["With_Coat", "Without_Coat", "OOD"]

    print("\n[STEP 1] Loading Standardized 77-Video Validation Suite...", flush=True)
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
    print(f">> Loaded {len(video_records)} validation videos (Positives: {y_true.sum()}, Negatives: {len(y_true) - y_true.sum()})", flush=True)

    # 2. Stage 1 & Stage 2a Hardware Latency Dynamic Profiling (Live Timers - Zero Hardcoding)
    print("\n[STEP 2] Dynamically Measuring Stage 1 & Stage 2a Component Latencies...", flush=True)
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    rect_imgsz = (384, 640)
    dummy_kpts = np.random.randn(50, 17, 2).astype(np.float32) * 50.0 + 300.0

    # Stage 1 Weapon
    weapon_weights = os.path.join(REPO_DIR, "weights", "yolo_weapon_distilled.pt")
    weapon_model = YOLO(weapon_weights)

    for _ in range(5): _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False)
    torch.cuda.synchronize()

    t_s1_gpu = []
    ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(30):
        ev_s.record(); _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False); ev_e.record()
        torch.cuda.synchronize()
        t_s1_gpu.append(ev_s.elapsed_time(ev_e))
    stage1_w_gpu_ms = float(np.median(t_s1_gpu))

    for _ in range(2): _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
    t_s1_cpu = []
    for _ in range(5):
        t0 = time.perf_counter(); _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
        t_s1_cpu.append((time.perf_counter() - t0) * 1000.0)
    stage1_w_cpu_ms = float(np.median(t_s1_cpu))

    # Stage 2a Pose
    pose_weights = os.path.join(REPO_DIR, "weights", "yolo26s-pose.pt")
    pose_model = YOLO(pose_weights)

    for _ in range(5): _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False)
    torch.cuda.synchronize()

    t_s2a_gpu = []
    for _ in range(30):
        ev_s.record(); _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False); ev_e.record()
        torch.cuda.synchronize()
        t_s2a_gpu.append(ev_s.elapsed_time(ev_e))
    stage2a_pose_gpu_ms = float(np.median(t_s2a_gpu))

    for _ in range(2): _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
    t_s2a_cpu = []
    for _ in range(5):
        t0 = time.perf_counter(); _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
        t_s2a_cpu.append((time.perf_counter() - t0) * 1000.0)
    stage2a_pose_cpu_ms = float(np.median(t_s2a_cpu))

    # Preprocessing
    t_gcn_gpu = []
    for _ in range(30):
        ev_s.record()
        sn = normalize_skeleton_clip(dummy_kpts)
        t = torch.from_numpy(sn).permute(2, 0, 1).unsqueeze(0).float().cuda()
        ev_e.record(); torch.cuda.synchronize()
        t_gcn_gpu.append(ev_s.elapsed_time(ev_e))
    stage2a_gcn_gpu_ms = float(np.median(t_gcn_gpu))

    t_gcn_cpu = []
    for _ in range(30):
        t0 = time.perf_counter()
        s = smooth_kinematics(dummy_kpts)
        sn = normalize_skeleton_clip(s)
        t = torch.from_numpy(sn).permute(2, 0, 1).unsqueeze(0).float()
        t_gcn_cpu.append((time.perf_counter() - t0) * 1000.0)
    stage2a_gcn_cpu_ms = float(np.median(t_gcn_cpu))

    t_rast_gpu = []
    for _ in range(30):
        ev_s.record()
        _ = rasterize_keypoints_to_heatmap(dummy_kpts, H=56, W=56, device="cuda")
        ev_e.record(); torch.cuda.synchronize()
        t_rast_gpu.append(ev_s.elapsed_time(ev_e))
    rast_gpu_ms = float(np.median(t_rast_gpu))

    t_rast_cpu = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = rasterize_keypoints_to_heatmap(dummy_kpts, H=56, W=56, device="cpu")
        t_rast_cpu.append((time.perf_counter() - t0) * 1000.0)
    rast_cpu_ms = float(np.median(t_rast_cpu))

    stage1_gpu_ms = stage1_w_gpu_ms + stage2a_pose_gpu_ms
    stage1_cpu_ms = stage1_w_cpu_ms + stage2a_pose_cpu_ms

    print(f">> Stage 1 Weapon: GPU = {stage1_w_gpu_ms:.2f} ms | CPU = {stage1_w_cpu_ms:.2f} ms", flush=True)
    print(f">> Stage 2a Pose:  GPU = {stage2a_pose_gpu_ms:.2f} ms | CPU = {stage2a_pose_cpu_ms:.2f} ms", flush=True)
    print(f">> Stage 1+2a Ingestion: GPU = {stage1_gpu_ms:.2f} ms | CPU = {stage1_cpu_ms:.2f} ms", flush=True)

    # 3. Pre-extract Model Probabilities for All 9 Models
    all_keys = list(MODEL_REGISTRY.keys())
    print("\n[STEP 3] Loading Models and Extracting Validation Probabilities...", flush=True)
    loaded_models_gpu = {}
    loaded_models_cpu = {}
    model_probs = {}

    import importlib.util
    spec_rmd = importlib.util.spec_from_file_location("ood_metrics_v1_08", os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py"))
    rmd_mod = importlib.util.module_from_spec(spec_rmd)
    spec_rmd.loader.exec_module(rmd_mod)
    compute_rmd = rmd_mod.compute_relative_mahalanobis_distance

    dummy_c_gpu = torch.randn(1, 2, 50, 17, device=device)
    dummy_h_gpu = torch.randn(1, 17, 50, 56, 56, device=device)
    dummy_c_cpu = torch.randn(1, 2, 50, 17, device="cpu")
    dummy_h_cpu = torch.randn(1, 17, 50, 56, 56, device="cpu")

    for k in all_keys:
        item_gpu = load_model_instance(k, device)
        item_cpu = load_model_instance(k, "cpu")
        loaded_models_gpu[k] = item_gpu
        loaded_models_cpu[k] = item_cpu

        probs = []
        with torch.no_grad():
            for rec in video_records:
                inp = rec["heatmaps"].to(device) if item_gpu["requires_heatmap"] else rec["coords"].to(device)
                _, feats = item_gpu["model"](inp, return_features=True)
                sc, _, _ = compute_rmd(feats, item_gpu["mu_0"], item_gpu["mu_c"], item_gpu["inv_Sigma_0"], item_gpu["inv_Sigma_c"])
                p = torch.sigmoid(sc - item_gpu["tau"])
                probs.append(p.max().item())
        model_probs[k] = np.array(probs)

    # 4. Define Super-Ensemble Configurations to Benchmark
    # Load optimal configurations from tuning summary
    tuning_json_path = os.path.join(REPO_DIR, "results", "tuning", "champion_super_ensembles.json")
    with open(tuning_json_path, "r") as f:
        tuning_data = json.load(f)

    champions_by_size = tuning_data["best_per_size"]

    print("\n[STEP 4] Profiling Live Hardware Latency & Accuracy for Champion Super-Ensembles...", flush=True)

    leaderboard = []

    # Add Standalone Champions for Reference
    standalone_keys = ["stgcn", "ctrgcn", "ctrgcn_tier_s", "pc3d_tier3", "pc3d_tier5"]
    for k in standalone_keys:
        item = loaded_models_gpu[k]
        item_cpu = loaded_models_cpu[k]
        inp_gpu = dummy_h_gpu if item["requires_heatmap"] else dummy_c_gpu
        inp_cpu = dummy_h_cpu if item["requires_heatmap"] else dummy_c_cpu

        fwd_gpu = measure_gpu_latency(lambda: item["model"](inp_gpu))
        fwd_cpu = measure_cpu_latency(lambda: item_cpu["model"](inp_cpu))

        act_gpu = (rast_gpu_ms if item["requires_heatmap"] else stage2a_gcn_gpu_ms) + fwd_gpu
        act_cpu = (rast_cpu_ms if item["requires_heatmap"] else stage2a_gcn_cpu_ms) + fwd_cpu
        tot_gpu = stage1_gpu_ms + act_gpu
        tot_cpu = stage1_cpu_ms + act_cpu

        preds = (model_probs[k] >= 0.50).astype(int)
        metrics = evaluate_binary(y_true, preds, cat_list)

        leaderboard.append({
            "system": f"Standalone: {item['name']}",
            "size": 1,
            "models": k,
            "params": item["params"],
            "f1": metrics["f1"],
            "recall": metrics["recall"] * 100,
            "precision": metrics["precision"] * 100,
            "accuracy": metrics["accuracy"] * 100,
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "with_coat_tp": metrics["categories"]["With_Coat"]["tp"],
            "without_coat_tp": metrics["categories"]["Without_Coat"]["tp"],
            "ood_fp": metrics["categories"]["OOD"]["fp"],
            "fwd_gpu_ms": fwd_gpu,
            "fwd_cpu_ms": fwd_cpu,
            "tot_gpu_ms": tot_gpu,
            "tot_cpu_ms": tot_cpu,
            "fps_gpu": 1000.0 / tot_gpu,
            "fps_cpu": 1000.0 / tot_cpu
        })

    # Benchmark Champion Super-Ensembles for sizes m = 2, 3, 4, 5, 6, 7, 8, 9
    for m_str, cfg in champions_by_size.items():
        m = int(m_str)
        combo_keys = cfg["models"].split("+")
        w_vec = np.array(cfg["weights"], dtype=np.float32)
        tau = float(cfg["threshold"])

        # Compute combined predictions
        p_matrix = np.stack([model_probs[k] for k in combo_keys])
        p_ens = (w_vec[:, None] * p_matrix).sum(axis=0)

        if cfg.get("veto", "none") != "none" and "pc3d_tier5" in model_probs:
            theta_veto = float(cfg["veto"].split(">=")[1])
            preds = ((p_ens >= tau) & (model_probs["pc3d_tier5"] >= theta_veto)).astype(int)
        else:
            preds = (p_ens >= tau).astype(int)

        metrics = evaluate_binary(y_true, preds, cat_list)

        # Profile live forward pass for the ensemble
        requires_gcn = any(not loaded_models_gpu[k]["requires_heatmap"] for k in combo_keys)
        requires_hm = any(loaded_models_gpu[k]["requires_heatmap"] for k in combo_keys)

        def ens_fwd_gpu():
            for k in combo_keys:
                inp = dummy_h_gpu if loaded_models_gpu[k]["requires_heatmap"] else dummy_c_gpu
                _ = loaded_models_gpu[k]["model"](inp)

        def ens_fwd_cpu():
            for k in combo_keys:
                inp = dummy_h_cpu if loaded_models_cpu[k]["requires_heatmap"] else dummy_c_cpu
                _ = loaded_models_cpu[k]["model"](inp)

        fwd_gpu = measure_gpu_latency(ens_fwd_gpu)
        fwd_cpu = measure_cpu_latency(ens_fwd_cpu)

        prep_gpu = (stage2a_gcn_gpu_ms if requires_gcn else 0.0) + (rast_gpu_ms if requires_hm else 0.0)
        prep_cpu = (stage2a_gcn_cpu_ms if requires_gcn else 0.0) + (rast_cpu_ms if requires_hm else 0.0)

        act_gpu = prep_gpu + fwd_gpu
        act_cpu = prep_cpu + fwd_cpu
        tot_gpu = stage1_gpu_ms + act_gpu
        tot_cpu = stage1_cpu_ms + act_cpu

        total_params = sum(loaded_models_gpu[k]["params"] for k in combo_keys)

        leaderboard.append({
            "system": f"Super-Ensemble m={m}",
            "size": m,
            "models": "+".join(combo_keys),
            "weights": cfg["weights"],
            "threshold": tau,
            "params": total_params,
            "f1": metrics["f1"],
            "recall": metrics["recall"] * 100,
            "precision": metrics["precision"] * 100,
            "accuracy": metrics["accuracy"] * 100,
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "with_coat_tp": metrics["categories"]["With_Coat"]["tp"],
            "without_coat_tp": metrics["categories"]["Without_Coat"]["tp"],
            "ood_fp": metrics["categories"]["OOD"]["fp"],
            "fwd_gpu_ms": fwd_gpu,
            "fwd_cpu_ms": fwd_cpu,
            "tot_gpu_ms": tot_gpu,
            "tot_cpu_ms": tot_cpu,
            "fps_gpu": 1000.0 / tot_gpu,
            "fps_cpu": 1000.0 / tot_cpu
        })

    # Save CSV
    df_lead = pd.DataFrame(leaderboard)
    csv_out = os.path.join(REPO_DIR, "results", "benchmark_super_ensemble.csv")
    df_lead.to_csv(csv_out, index=False)
    print(f"\n>> Master Leaderboard CSV saved to: {csv_out}", flush=True)

    # Save JSON
    json_out = os.path.join(REPO_DIR, "results", "benchmark_super_ensemble.json")
    with open(json_out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "stage1_and_2a_gpu_ms": stage1_gpu_ms,
            "stage1_and_2a_cpu_ms": stage1_cpu_ms,
            "leaderboard": leaderboard
        }, f, indent=2)
    print(f">> Master Benchmark JSON saved to: {json_out}", flush=True)

    # Display Master Table
    print("\n" + "=" * 115)
    print("   MASTER PERFORMANCE LEADERBOARD: STANDALONE VS SUPER-ENSEMBLES (m = 2 to 9)")
    print("=" * 115)
    print(f"{'System':<24} | {'Params':<10} | {'F1':<7} | {'Recall':<7} | {'Prec':<7} | {'FP':<4} | {'FN':<4} | {'Tot GPU':<9} | {'GPU FPS':<7} | {'Tot CPU':<9} | {'CPU FPS'}")
    print("-" * 115)
    for row in leaderboard:
        print(f"{row['system']:<24} | {row['params']:<10,d} | {row['f1']:<7.4f} | {row['recall']:<6.1f}% | {row['precision']:<6.1f}% | {row['fp']:<4d} | {row['fn']:<4d} | {row['tot_gpu_ms']:<6.2f} ms | {row['fps_gpu']:<7.1f} | {row['tot_cpu_ms']:<6.2f} ms | {row['fps_cpu']:<5.1f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
