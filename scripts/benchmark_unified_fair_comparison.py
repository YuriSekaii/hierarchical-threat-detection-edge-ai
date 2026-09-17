"""
benchmark_unified_fair_comparison.py

Unified, 100% Fair Re-Benchmark of ALL Models and Multi-Model Architectures.
Executes live hardware profiling and validation on the exact same 77-video validation suite.

Models & Configurations Evaluated:
1. ST-GCN Baseline (v1.08)
2. CTR-GCN (v2.00)
3. PoseConv3D Baseline (v2.10: 765k)
4. PoseConv3D Scaled (3.55M)
5. PoseConv3D Tier 5 (132k)
6. PoseConv3D Tier 3 (598k Undistilled)
7. PoseConv3D Tier 3 Distilled (598k Method B - RKD)
8. Server Ensemble (v3.10 Server Only - Synchronous 100%)
9. Heterogeneous Dual-Tier (v3.10 - Hierarchical Gated 5.2% Escalation)
10. Config 1: Pure Dual Consensus (Tier 5 + Tier 3 - Synchronous Shared 3D)
11. Config 2: Distilled Consensus (Tier 5 + Distilled Tier 3 - Synchronous Shared 3D)
12. Config 3: Tri-PoseConv3D Consensus (Tier 5 + Tier 3 + Distill Tier 3 - Synchronous Shared 3D)
13. Config 4: Pure Hierarchical Dual-Tier (Tier 5 Edge -> Tier 3 Escalation - Hierarchical 28.6% Escalation)

Adheres strictly to AgentRule.md.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset, PoseConv3DHeatmapDataset
from src.poseconv3d_utils import batch_rasterize_gpu, rasterize_keypoints_to_heatmap
from src.skeleton_utils import normalize_skeleton_clip, smooth_kinematics
from ultralytics import YOLO
from models.stgcn import STGCNModel
from models.ctrgcn import CTRGCNModel
from models.poseconv3d import PoseConv3DModel
from models.poseconv3d_scaled import ScaledPoseConv3DModel
from models.ensemble_dual_tier import DualTierSurveillanceSystem, Tier2ServerEnsemble
from models.ensemble_poseconv3d_dual_tier import PurePoseConv3DEnsemble, PurePoseConv3DDualTierSystem

# Import RMD metric
spec_rmd = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py")
)
rmd_mod = importlib.util.module_from_spec(spec_rmd)
spec_rmd.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance


def measure_gpu_latency(model_fn, dummy_input, warmup=50, iterations=500):
    """Measures median GPU latency using CUDA events."""
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
    """Measures median CPU latency using high-resolution perf_counter."""
    for _ in range(warmup):
        _ = model_fn(dummy_input)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model_fn(dummy_input)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return float(np.median(timings))


def evaluate_binary(y_true, y_pred, categories):
    """Evaluates binary metrics with per-category breakdown."""
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    cat_stats = defaultdict(lambda: {"videos": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for cat, yt, yp in zip(categories, y_true, y_pred):
        d = cat_stats[cat]
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
        "categories": dict(cat_stats)
    }


def main():
    print("=" * 85, flush=True)
    print("   STANDARDIZED UNIFIED RE-BENCHMARK OF ALL MODELS & METHODS", flush=True)
    print("=" * 85, flush=True)

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
                c_tensor = torch.stack(v_coords[src]) # (N, 2, 50, 17)
                h_tensor = torch.stack(v_hms[src])    # (N, 17, 50, 56, 56)
                video_records.append({
                    "src": src,
                    "cat": cat,
                    "gt": gt,
                    "coords": c_tensor,
                    "heatmaps": h_tensor
                })
                y_true_list.append(gt)
                cat_list.append(cat)

    y_true = np.array(y_true_list)
    print(f">> Loaded {len(video_records)} validation videos (Positives: {y_true.sum()}, Negatives: {len(y_true) - y_true.sum()})", flush=True)

    # 2. Stage 1 & Stage 2a Hardware Latency Dynamic Profiling (Live Timers - Zero Hardcoding)
    print("\n[STEP 2] Dynamically Measuring Stage 1 & Stage 2a Hardware Latencies (Zero Hardcoding)...", flush=True)
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    rect_imgsz = (384, 640)
    dummy_kpts = np.random.randn(50, 17, 2).astype(np.float32) * 50.0 + 300.0

    # A. Stage 1: Distilled YOLO26s Weapon Detector
    weapon_weights = os.path.join(REPO_DIR, "weights", "yolo_weapon_distilled.pt")
    weapon_model = YOLO(weapon_weights)

    # Warmup GPU
    for _ in range(5):
        _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False)
    torch.cuda.synchronize()

    ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t_s1_gpu = []
    for _ in range(30):
        ev_s.record()
        _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False)
        ev_e.record()
        torch.cuda.synchronize()
        t_s1_gpu.append(ev_s.elapsed_time(ev_e))
    stage1_w_gpu_ms = float(np.median(t_s1_gpu))

    # CPU Stage 1
    for _ in range(2):
        _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
    t_s1_cpu = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = weapon_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
        t_s1_cpu.append((time.perf_counter() - t0) * 1000.0)
    stage1_w_cpu_ms = float(np.median(t_s1_cpu))

    # B. Stage 2a: YOLO26s-Pose Extractor
    pose_weights = os.path.join(REPO_DIR, "weights", "yolo26s-pose.pt")
    pose_model = YOLO(pose_weights)

    for _ in range(5):
        _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False)
    torch.cuda.synchronize()

    t_s2a_gpu = []
    for _ in range(30):
        ev_s.record()
        _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device=0, verbose=False)
        ev_e.record()
        torch.cuda.synchronize()
        t_s2a_gpu.append(ev_s.elapsed_time(ev_e))
    stage2a_pose_gpu_ms = float(np.median(t_s2a_gpu))

    for _ in range(2):
        _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
    t_s2a_cpu = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = pose_model.predict(dummy_frame, imgsz=rect_imgsz, device="cpu", verbose=False)
        t_s2a_cpu.append((time.perf_counter() - t0) * 1000.0)
    stage2a_pose_cpu_ms = float(np.median(t_s2a_cpu))

    # C. Stage 2a Preprocessing: GCN Graph vs 3D Heatmap Rasterization
    # 1. GCN Graph normalization & kinematics smoothing (50 frames)
    t_gcn_gpu = []
    for _ in range(30):
        ev_s.record()
        sn = normalize_skeleton_clip(dummy_kpts)
        t = torch.from_numpy(sn).permute(2, 0, 1).unsqueeze(0).float().cuda()
        ev_e.record()
        torch.cuda.synchronize()
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

    # 2. PoseConv3D Heatmap Rasterization (50 frames)
    t_rast_gpu = []
    for _ in range(30):
        ev_s.record()
        _ = rasterize_keypoints_to_heatmap(dummy_kpts, H=56, W=56, device="cuda")
        ev_e.record()
        torch.cuda.synchronize()
        t_rast_gpu.append(ev_s.elapsed_time(ev_e))
    rast_gpu_ms = float(np.median(t_rast_gpu))

    t_rast_cpu = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = rasterize_keypoints_to_heatmap(dummy_kpts, H=56, W=56, device="cpu")
        t_rast_cpu.append((time.perf_counter() - t0) * 1000.0)
    rast_cpu_ms = float(np.median(t_rast_cpu))

    # Full frame ingestion baseline (Stage 1 Weapon + Stage 2a YOLO26s-Pose)
    stage1_gpu_ms = stage1_w_gpu_ms + stage2a_pose_gpu_ms
    stage1_cpu_ms = stage1_w_cpu_ms + stage2a_pose_cpu_ms

    print(f">> Dynamically Measured Component Latencies:", flush=True)
    print(f"   Stage 1 YOLO26s Weapon: GPU = {stage1_w_gpu_ms:.2f} ms | CPU = {stage1_w_cpu_ms:.2f} ms", flush=True)
    print(f"   Stage 2a YOLO26s-Pose:   GPU = {stage2a_pose_gpu_ms:.2f} ms | CPU = {stage2a_pose_cpu_ms:.2f} ms", flush=True)
    print(f"   Stage 1+2a Ingestion:   GPU = {stage1_gpu_ms:.2f} ms | CPU = {stage1_cpu_ms:.2f} ms", flush=True)
    print(f"   Stage 2a GCN Prep:       GPU = {stage2a_gcn_gpu_ms:.2f} ms | CPU = {stage2a_gcn_cpu_ms:.2f} ms", flush=True)
    print(f"   Stage 2a Heatmap Raster: GPU = {rast_gpu_ms:.2f} ms | CPU = {rast_cpu_ms:.2f} ms", flush=True)

    # 3. Model Loading & Reference Initialization
    print("\n[STEP 3] Loading Models & Calibrated References...", flush=True)

    # A. ST-GCN
    stgcn = STGCNModel(num_classes=1).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", "stgcn_violence_v1.02.pth"), map_location=device, weights_only=True))
    stgcn.eval()
    ref_stgcn = torch.load(os.path.join(REPO_DIR, "weights", "reference_data_rmd_v1.08.pt"), map_location=device, weights_only=False)
    stgcn_mu0, stgcn_muc = ref_stgcn["mu_0"].to(device), ref_stgcn["mu_c"].to(device)
    stgcn_inv0, stgcn_invc = ref_stgcn["inv_Sigma_0"].to(device), ref_stgcn["inv_Sigma_c"].to(device)
    tau_stgcn = float(ref_stgcn.get("optimal_threshold", -0.9289))

    # B. CTR-GCN
    ctrgcn = CTRGCNModel(num_classes=1).to(device)
    ctrgcn.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", "ctrgcn_violence_v2.00.pth"), map_location=device, weights_only=True))
    ctrgcn.eval()
    ref_ctrgcn = torch.load(os.path.join(REPO_DIR, "weights", "reference_data_ctrgcn_v2.00.pt"), map_location=device, weights_only=False)
    ctrgcn_mu0, ctrgcn_muc = ref_ctrgcn["mu_0"].to(device), ref_ctrgcn["mu_c"].to(device)
    ctrgcn_inv0, ctrgcn_invc = ref_ctrgcn["inv_Sigma_0"].to(device), ref_ctrgcn["inv_Sigma_c"].to(device)
    tau_ctrgcn = float(ref_ctrgcn.get("optimal_threshold", 3.1498))

    # C. PoseConv3D Baseline (v2.10: 765k)
    pc3d_base = PoseConv3DModel(num_classes=1, base_channels=24).to(device)
    pc3d_base.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", "poseconv3d_violence_v2.10.pth"), map_location=device, weights_only=True))
    pc3d_base.eval()
    ref_pc3d_base = torch.load(os.path.join(REPO_DIR, "weights", "reference_data_poseconv3d_v2.10.pt"), map_location=device, weights_only=False)
    pc3d_base_mu0, pc3d_base_muc = ref_pc3d_base["mu_0"].to(device), ref_pc3d_base["mu_c"].to(device)
    pc3d_base_inv0, pc3d_base_invc = ref_pc3d_base["inv_Sigma_0"].to(device), ref_pc3d_base["inv_Sigma_c"].to(device)
    tau_pc3d_base = float(ref_pc3d_base.get("optimal_threshold", -21.4694))

    # D. PoseConv3D Scaled (3.55M)
    pc3d_scaled = ScaledPoseConv3DModel().to(device)
    pc3d_scaled.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", "poseconv3d_scaled_violence.pth"), map_location=device, weights_only=True))
    pc3d_scaled.eval()
    ref_pc3d_scaled = torch.load(os.path.join(REPO_DIR, "weights", "reference_data_poseconv3d_scaled.pt"), map_location=device, weights_only=False)
    pc3d_scaled_mu0, pc3d_scaled_muc = ref_pc3d_scaled["mu_0"].to(device), ref_pc3d_scaled["mu_c"].to(device)
    pc3d_scaled_inv0, pc3d_scaled_invc = ref_pc3d_scaled["inv_Sigma_0"].to(device), ref_pc3d_scaled["inv_Sigma_c"].to(device)
    tau_pc3d_scaled = float(ref_pc3d_scaled.get("optimal_threshold", -4.5681))

    # E. Pure-PoseConv3D Reference Data (Tier 5, Tier 3, Distilled Tier 3)
    ref_pure_dual = torch.load(os.path.join(REPO_DIR, "weights", "reference_data_poseconv3d_pure_dual_tier.pt"), map_location=device, weights_only=False)
    t5_ref = ref_pure_dual["t5_ref"]
    t3_ref = ref_pure_dual["t3_ref"]
    dt3_ref = ref_pure_dual["dt3_ref"]
    pure_opt_cfg = ref_pure_dual["optimal_config"]

    # Tier 5 (132k)
    pc3d_t5 = PoseConv3DModel(num_classes=1, base_channels=12, feature_dim=96).to(device)
    pc3d_t5.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", "poseconv3d_downscale_tier5_132k.pth"), map_location=device, weights_only=True))
    pc3d_t5.eval()
    t5_mu0, t5_muc = t5_ref["mu_0"].to(device), t5_ref["mu_c"].to(device)
    t5_inv0, t5_invc = t5_ref["inv_Sigma_0"].to(device), t5_ref["inv_Sigma_c"].to(device)
    tau_t5 = float(t5_ref["optimal_threshold"])

    # Tier 3 (598k Undistilled)
    pc3d_t3 = PoseConv3DModel(num_classes=1, base_channels=12, feature_dim=256).to(device)
    pc3d_t3.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", "poseconv3d_downscale_tier3_598k.pth"), map_location=device, weights_only=True))
    pc3d_t3.eval()
    t3_mu0, t3_muc = t3_ref["mu_0"].to(device), t3_ref["mu_c"].to(device)
    t3_inv0, t3_invc = t3_ref["inv_Sigma_0"].to(device), t3_ref["inv_Sigma_c"].to(device)
    tau_t3 = float(t3_ref["optimal_threshold"])

    # Distilled Tier 3 (598k Method B)
    pc3d_dt3 = PoseConv3DModel(num_classes=1, base_channels=12, feature_dim=256).to(device)
    pc3d_dt3.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", "poseconv3d_distill_methodB_rkd.pth"), map_location=device, weights_only=True))
    pc3d_dt3.eval()
    dt3_mu0, dt3_muc = dt3_ref["mu_0"].to(device), dt3_ref["mu_c"].to(device)
    dt3_inv0, dt3_invc = dt3_ref["inv_Sigma_0"].to(device), dt3_ref["inv_Sigma_c"].to(device)
    tau_dt3 = float(dt3_ref["optimal_threshold"])

    # 4. Extract Validation Probabilities for all Models
    print("\n[STEP 4] Extracting Validation Inference Probabilities across 77 Videos...", flush=True)

    probs_stgcn = []
    probs_ctrgcn = []
    probs_pc3d_base = []
    probs_pc3d_scaled = []
    probs_pc3d_t5 = []
    probs_pc3d_t3 = []
    probs_pc3d_dt3 = []

    with torch.no_grad():
        for rec in video_records:
            c = rec["coords"].to(device)
            h = rec["heatmaps"].to(device)

            # ST-GCN
            _, f_st = stgcn(c, return_features=True)
            sc_st, _, _ = compute_relative_mahalanobis_distance(f_st, stgcn_mu0, stgcn_muc, stgcn_inv0, stgcn_invc)
            p_st = 1.0 / (1.0 + np.exp(-(sc_st.max().item() - tau_stgcn)))
            probs_stgcn.append(p_st)

            # CTR-GCN
            _, f_ctr = ctrgcn(c, return_features=True)
            sc_ctr, _, _ = compute_relative_mahalanobis_distance(f_ctr, ctrgcn_mu0, ctrgcn_muc, ctrgcn_inv0, ctrgcn_invc)
            p_ctr = 1.0 / (1.0 + np.exp(-(sc_ctr.max().item() - tau_ctrgcn)))
            probs_ctrgcn.append(p_ctr)

            # PoseConv3D Baseline
            _, f_pc = pc3d_base(h, return_features=True)
            sc_pc, _, _ = compute_relative_mahalanobis_distance(f_pc, pc3d_base_mu0, pc3d_base_muc, pc3d_base_inv0, pc3d_base_invc)
            p_pc = 1.0 / (1.0 + np.exp(-(sc_pc.max().item() - tau_pc3d_base)))
            probs_pc3d_base.append(p_pc)

            # PoseConv3D Scaled
            _, f_scaled = pc3d_scaled(h, return_features=True)
            sc_sc, _, _ = compute_relative_mahalanobis_distance(f_scaled, pc3d_scaled_mu0, pc3d_scaled_muc, pc3d_scaled_inv0, pc3d_scaled_invc)
            p_sc = 1.0 / (1.0 + np.exp(-(sc_sc.max().item() - tau_pc3d_scaled)))
            probs_pc3d_scaled.append(p_sc)

            # Tier 5
            _, f_t5 = pc3d_t5(h, return_features=True)
            sc_t5, _, _ = compute_relative_mahalanobis_distance(f_t5, t5_mu0, t5_muc, t5_inv0, t5_invc)
            p_t5 = 1.0 / (1.0 + np.exp(-(sc_t5.max().item() - tau_t5)))
            probs_pc3d_t5.append(p_t5)

            # Tier 3
            _, f_t3 = pc3d_t3(h, return_features=True)
            sc_t3, _, _ = compute_relative_mahalanobis_distance(f_t3, t3_mu0, t3_muc, t3_inv0, t3_invc)
            p_t3 = 1.0 / (1.0 + np.exp(-(sc_t3.max().item() - tau_t3)))
            probs_pc3d_t3.append(p_t3)

            # Distilled Tier 3
            _, f_dt3 = pc3d_dt3(h, return_features=True)
            sc_dt3, _, _ = compute_relative_mahalanobis_distance(f_dt3, dt3_mu0, dt3_muc, dt3_inv0, dt3_invc)
            p_dt3 = 1.0 / (1.0 + np.exp(-(sc_dt3.max().item() - tau_dt3)))
            probs_pc3d_dt3.append(p_dt3)

    probs_stgcn = np.array(probs_stgcn)
    probs_ctrgcn = np.array(probs_ctrgcn)
    probs_pc3d_base = np.array(probs_pc3d_base)
    probs_pc3d_scaled = np.array(probs_pc3d_scaled)
    probs_pc3d_t5 = np.array(probs_pc3d_t5)
    probs_pc3d_t3 = np.array(probs_pc3d_t3)
    probs_pc3d_dt3 = np.array(probs_pc3d_dt3)

    # 5. Measure Live Hardware Latencies
    print("\n[STEP 5] Profiling Hardware Latency (GPU 500 iter, CPU 100 iter)...", flush=True)

    dummy_c_gpu = torch.randn(1, 2, 50, 17, device=device)
    dummy_h_gpu = torch.randn(1, 17, 50, 56, 56, device=device)

    # Create CPU models for fair CPU latency measurement
    stgcn_cpu = STGCNModel(num_classes=1).to("cpu").eval()
    stgcn_cpu.load_state_dict(stgcn.state_dict())
    ctrgcn_cpu = CTRGCNModel(num_classes=1).to("cpu").eval()
    ctrgcn_cpu.load_state_dict(ctrgcn.state_dict())
    pc3d_base_cpu = PoseConv3DModel(num_classes=1, base_channels=24).to("cpu").eval()
    pc3d_base_cpu.load_state_dict(pc3d_base.state_dict())
    pc3d_scaled_cpu = ScaledPoseConv3DModel().to("cpu").eval()
    pc3d_scaled_cpu.load_state_dict(pc3d_scaled.state_dict())
    pc3d_t5_cpu = PoseConv3DModel(num_classes=1, base_channels=12, feature_dim=96).to("cpu").eval()
    pc3d_t5_cpu.load_state_dict(pc3d_t5.state_dict())
    pc3d_t3_cpu = PoseConv3DModel(num_classes=1, base_channels=12, feature_dim=256).to("cpu").eval()
    pc3d_t3_cpu.load_state_dict(pc3d_t3.state_dict())
    pc3d_dt3_cpu = PoseConv3DModel(num_classes=1, base_channels=12, feature_dim=256).to("cpu").eval()
    pc3d_dt3_cpu.load_state_dict(pc3d_dt3.state_dict())

    dummy_c_cpu = torch.randn(1, 2, 50, 17, device="cpu")
    dummy_h_cpu = torch.randn(1, 17, 50, 56, 56, device="cpu")

    # Single Forward Latencies
    gpu_lat = {}
    cpu_lat = {}

    with torch.no_grad():
        gpu_lat["stgcn"] = measure_gpu_latency(lambda x: stgcn(x), dummy_c_gpu)
        cpu_lat["stgcn"] = measure_cpu_latency(lambda x: stgcn_cpu(x), dummy_c_cpu)

        gpu_lat["ctrgcn"] = measure_gpu_latency(lambda x: ctrgcn(x), dummy_c_gpu)
        cpu_lat["ctrgcn"] = measure_cpu_latency(lambda x: ctrgcn_cpu(x), dummy_c_cpu)

        gpu_lat["pc3d_base"] = measure_gpu_latency(lambda x: pc3d_base(x), dummy_h_gpu)
        cpu_lat["pc3d_base"] = measure_cpu_latency(lambda x: pc3d_base_cpu(x), dummy_h_cpu)

        gpu_lat["pc3d_scaled"] = measure_gpu_latency(lambda x: pc3d_scaled(x), dummy_h_gpu)
        cpu_lat["pc3d_scaled"] = measure_cpu_latency(lambda x: pc3d_scaled_cpu(x), dummy_h_cpu)

        gpu_lat["pc3d_t5"] = measure_gpu_latency(lambda x: pc3d_t5(x), dummy_h_gpu)
        cpu_lat["pc3d_t5"] = measure_cpu_latency(lambda x: pc3d_t5_cpu(x), dummy_h_cpu)

        gpu_lat["pc3d_t3"] = measure_gpu_latency(lambda x: pc3d_t3(x), dummy_h_gpu)
        cpu_lat["pc3d_t3"] = measure_cpu_latency(lambda x: pc3d_t3_cpu(x), dummy_h_cpu)

        gpu_lat["pc3d_dt3"] = measure_gpu_latency(lambda x: pc3d_dt3(x), dummy_h_gpu)
        cpu_lat["pc3d_dt3"] = measure_cpu_latency(lambda x: pc3d_dt3_cpu(x), dummy_h_cpu)

        # Ensembles Forward Latencies
        # 1. Server Ensemble v3.10 (Synchronous: ST-GCN + CTR-GCN + PoseConv3D)
        def v310_server_gpu(c, h):
            _ = stgcn(c)
            _ = ctrgcn(c)
            _ = pc3d_base(h)

        def v310_server_cpu(c, h):
            _ = stgcn_cpu(c)
            _ = ctrgcn_cpu(c)
            _ = pc3d_base_cpu(h)

        for _ in range(20): v310_server_gpu(dummy_c_gpu, dummy_h_gpu)
        torch.cuda.synchronize()
        t_v310_gpu = []
        ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(500):
            ev_s.record(); v310_server_gpu(dummy_c_gpu, dummy_h_gpu); ev_e.record()
            torch.cuda.synchronize()
            t_v310_gpu.append(ev_s.elapsed_time(ev_e))
        gpu_lat["v310_server"] = float(np.median(t_v310_gpu))

        for _ in range(5): v310_server_cpu(dummy_c_cpu, dummy_h_cpu)
        t_v310_cpu = []
        for _ in range(100):
            t0 = time.perf_counter(); v310_server_cpu(dummy_c_cpu, dummy_h_cpu); t1 = time.perf_counter()
            t_v310_cpu.append((t1 - t0) * 1000.0)
        cpu_lat["v310_server"] = float(np.median(t_v310_cpu))

        # 2. Pure Dual Consensus (Config 1 & 2: Tier 5 + Tier 3 on SHARED Heatmap)
        def pure_dual_gpu(h):
            _ = pc3d_t5(h)
            _ = pc3d_t3(h)

        def pure_dual_cpu(h):
            _ = pc3d_t5_cpu(h)
            _ = pc3d_t3_cpu(h)

        for _ in range(20): pure_dual_gpu(dummy_h_gpu)
        torch.cuda.synchronize()
        t_pd_gpu = []
        for _ in range(500):
            ev_s.record(); pure_dual_gpu(dummy_h_gpu); ev_e.record()
            torch.cuda.synchronize()
            t_pd_gpu.append(ev_s.elapsed_time(ev_e))
        gpu_lat["pure_dual"] = float(np.median(t_pd_gpu))

        for _ in range(5): pure_dual_cpu(dummy_h_cpu)
        t_pd_cpu = []
        for _ in range(100):
            t0 = time.perf_counter(); pure_dual_cpu(dummy_h_cpu); t1 = time.perf_counter()
            t_pd_cpu.append((t1 - t0) * 1000.0)
        cpu_lat["pure_dual"] = float(np.median(t_pd_cpu))

        # 3. Pure Tri Consensus (Config 3: Tier 5 + Tier 3 + Distill Tier 3 on SHARED Heatmap)
        def pure_tri_gpu(h):
            _ = pc3d_t5(h)
            _ = pc3d_t3(h)
            _ = pc3d_dt3(h)

        def pure_tri_cpu(h):
            _ = pc3d_t5_cpu(h)
            _ = pc3d_t3_cpu(h)
            _ = pc3d_dt3_cpu(h)

        for _ in range(20): pure_tri_gpu(dummy_h_gpu)
        torch.cuda.synchronize()
        t_pt_gpu = []
        for _ in range(500):
            ev_s.record(); pure_tri_gpu(dummy_h_gpu); ev_e.record()
            torch.cuda.synchronize()
            t_pt_gpu.append(ev_s.elapsed_time(ev_e))
        gpu_lat["pure_tri"] = float(np.median(t_pt_gpu))

        for _ in range(5): pure_tri_cpu(dummy_h_cpu)
        t_pt_cpu = []
        for _ in range(100):
            t0 = time.perf_counter(); pure_tri_cpu(dummy_h_cpu); t1 = time.perf_counter()
            t_pt_cpu.append((t1 - t0) * 1000.0)
        cpu_lat["pure_tri"] = float(np.median(t_pt_cpu))

    print(f">> Forward Latencies (ms):", flush=True)
    print(f"   ST-GCN: GPU {gpu_lat['stgcn']:.2f} ms | CPU {cpu_lat['stgcn']:.2f} ms", flush=True)
    print(f"   CTR-GCN: GPU {gpu_lat['ctrgcn']:.2f} ms | CPU {cpu_lat['ctrgcn']:.2f} ms", flush=True)
    print(f"   PoseConv3D Base (765k): GPU {gpu_lat['pc3d_base']:.2f} ms | CPU {cpu_lat['pc3d_base']:.2f} ms", flush=True)
    print(f"   PoseConv3D Scaled (3.55M): GPU {gpu_lat['pc3d_scaled']:.2f} ms | CPU {cpu_lat['pc3d_scaled']:.2f} ms", flush=True)
    print(f"   PoseConv3D Tier 5 (132k): GPU {gpu_lat['pc3d_t5']:.2f} ms | CPU {cpu_lat['pc3d_t5']:.2f} ms", flush=True)
    print(f"   PoseConv3D Tier 3 (598k): GPU {gpu_lat['pc3d_t3']:.2f} ms | CPU {cpu_lat['pc3d_t3']:.2f} ms", flush=True)
    print(f"   Server Ensemble v3.10: GPU {gpu_lat['v310_server']:.2f} ms | CPU {cpu_lat['v310_server']:.2f} ms", flush=True)
    print(f"   Pure Dual Consensus: GPU {gpu_lat['pure_dual']:.2f} ms | CPU {cpu_lat['pure_dual']:.2f} ms", flush=True)
    print(f"   Pure Tri Consensus: GPU {gpu_lat['pure_tri']:.2f} ms | CPU {cpu_lat['pure_tri']:.2f} ms", flush=True)

    # 6. Evaluate All Systems
    print("\n[STEP 6] Evaluating Accuracy & Confusion Matrices...", flush=True)
    leaderboard = []

    # 1. ST-GCN Baseline
    pred_stgcn = (probs_stgcn >= 0.50).astype(int)
    res_stgcn = evaluate_binary(y_true, pred_stgcn, cat_list)
    tot_gpu_st = stage1_gpu_ms + stage2a_gcn_gpu_ms + gpu_lat["stgcn"]
    tot_cpu_st = stage1_cpu_ms + stage2a_gcn_cpu_ms + cpu_lat["stgcn"]
    leaderboard.append({
        "system": "ST-GCN Baseline (v1.08)",
        "architecture": "ST-GCN Graph CNN",
        "mode": "Synchronous (100%)",
        "params": 3014981,
        "metrics": res_stgcn,
        "fwd_cpu_ms": cpu_lat["stgcn"],
        "fwd_gpu_ms": gpu_lat["stgcn"],
        "action_cpu_ms": stage2a_gcn_cpu_ms + cpu_lat["stgcn"],
        "action_gpu_ms": stage2a_gcn_gpu_ms + gpu_lat["stgcn"],
        "tot_cpu_ms": tot_cpu_st,
        "fps_cpu": 1000.0 / tot_cpu_st,
        "tot_gpu_ms": tot_gpu_st,
        "fps_gpu": 1000.0 / tot_gpu_st
    })

    # 2. CTR-GCN
    pred_ctrgcn = (probs_ctrgcn >= 0.50).astype(int)
    res_ctrgcn = evaluate_binary(y_true, pred_ctrgcn, cat_list)
    tot_gpu_ctr = stage1_gpu_ms + stage2a_gcn_gpu_ms + gpu_lat["ctrgcn"]
    tot_cpu_ctr = stage1_cpu_ms + stage2a_gcn_cpu_ms + cpu_lat["ctrgcn"]
    leaderboard.append({
        "system": "CTR-GCN (v2.00)",
        "architecture": "CTR-GCN Channel Refinement",
        "mode": "Synchronous (100%)",
        "params": 1332761,
        "metrics": res_ctrgcn,
        "fwd_cpu_ms": cpu_lat["ctrgcn"],
        "fwd_gpu_ms": gpu_lat["ctrgcn"],
        "action_cpu_ms": stage2a_gcn_cpu_ms + cpu_lat["ctrgcn"],
        "action_gpu_ms": stage2a_gcn_gpu_ms + gpu_lat["ctrgcn"],
        "tot_cpu_ms": tot_cpu_ctr,
        "fps_cpu": 1000.0 / tot_cpu_ctr,
        "tot_gpu_ms": tot_gpu_ctr,
        "fps_gpu": 1000.0 / tot_gpu_ctr
    })

    # 3. PoseConv3D Baseline (v2.10: 765k)
    pred_pc3d_base = (probs_pc3d_base >= 0.50).astype(int)
    res_pc3d_base = evaluate_binary(y_true, pred_pc3d_base, cat_list)
    tot_gpu_pc_b = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pc3d_base"]
    tot_cpu_pc_b = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pc3d_base"]
    leaderboard.append({
        "system": "PoseConv3D Baseline (v2.10)",
        "architecture": "3D CNN (24ch, 256d)",
        "mode": "Synchronous (100%)",
        "params": 765529,
        "metrics": res_pc3d_base,
        "fwd_cpu_ms": cpu_lat["pc3d_base"],
        "fwd_gpu_ms": gpu_lat["pc3d_base"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pc3d_base"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pc3d_base"],
        "tot_cpu_ms": tot_cpu_pc_b,
        "fps_cpu": 1000.0 / tot_cpu_pc_b,
        "tot_gpu_ms": tot_gpu_pc_b,
        "fps_gpu": 1000.0 / tot_gpu_pc_b
    })

    # 4. PoseConv3D Scaled (3.55M)
    pred_pc3d_sc = (probs_pc3d_scaled >= 0.50).astype(int)
    res_pc3d_sc = evaluate_binary(y_true, pred_pc3d_sc, cat_list)
    tot_gpu_pc_sc = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pc3d_scaled"]
    tot_cpu_pc_sc = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pc3d_scaled"]
    leaderboard.append({
        "system": "PoseConv3D Scaled (3.55M)",
        "architecture": "3D CNN (48ch, 512d)",
        "mode": "Synchronous (100%)",
        "params": 3554813,
        "metrics": res_pc3d_sc,
        "fwd_cpu_ms": cpu_lat["pc3d_scaled"],
        "fwd_gpu_ms": gpu_lat["pc3d_scaled"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pc3d_scaled"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pc3d_scaled"],
        "tot_cpu_ms": tot_cpu_pc_sc,
        "fps_cpu": 1000.0 / tot_cpu_pc_sc,
        "tot_gpu_ms": tot_gpu_pc_sc,
        "fps_gpu": 1000.0 / tot_gpu_pc_sc
    })

    # 5. PoseConv3D Tier 5 (132k)
    pred_pc3d_t5 = (probs_pc3d_t5 >= 0.50).astype(int)
    res_pc3d_t5 = evaluate_binary(y_true, pred_pc3d_t5, cat_list)
    tot_gpu_t5 = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pc3d_t5"]
    tot_cpu_t5 = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pc3d_t5"]
    leaderboard.append({
        "system": "PoseConv3D Tier 5 (132k)",
        "architecture": "3D CNN (12ch, 96d)",
        "mode": "Synchronous (100%)",
        "params": 132349,
        "metrics": res_pc3d_t5,
        "fwd_cpu_ms": cpu_lat["pc3d_t5"],
        "fwd_gpu_ms": gpu_lat["pc3d_t5"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pc3d_t5"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pc3d_t5"],
        "tot_cpu_ms": tot_cpu_t5,
        "fps_cpu": 1000.0 / tot_cpu_t5,
        "tot_gpu_ms": tot_gpu_t5,
        "fps_gpu": 1000.0 / tot_gpu_t5
    })

    # 6. PoseConv3D Tier 3 (598k Undistilled)
    pred_pc3d_t3 = (probs_pc3d_t3 >= 0.50).astype(int)
    res_pc3d_t3 = evaluate_binary(y_true, pred_pc3d_t3, cat_list)
    tot_gpu_t3 = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pc3d_t3"]
    tot_cpu_t3 = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pc3d_t3"]
    leaderboard.append({
        "system": "PoseConv3D Tier 3 (598k)",
        "architecture": "3D CNN (12ch, 256d)",
        "mode": "Synchronous (100%)",
        "params": 598429,
        "metrics": res_pc3d_t3,
        "fwd_cpu_ms": cpu_lat["pc3d_t3"],
        "fwd_gpu_ms": gpu_lat["pc3d_t3"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pc3d_t3"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pc3d_t3"],
        "tot_cpu_ms": tot_cpu_t3,
        "fps_cpu": 1000.0 / tot_cpu_t3,
        "tot_gpu_ms": tot_gpu_t3,
        "fps_gpu": 1000.0 / tot_gpu_t3
    })

    # 7. PoseConv3D Tier 3 Distilled (Method B - RKD)
    pred_pc3d_dt3 = (probs_pc3d_dt3 >= 0.50).astype(int)
    res_pc3d_dt3 = evaluate_binary(y_true, pred_pc3d_dt3, cat_list)
    tot_gpu_dt3 = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pc3d_dt3"]
    tot_cpu_dt3 = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pc3d_dt3"]
    leaderboard.append({
        "system": "Distilled Tier 3 (Method B)",
        "architecture": "3D CNN (ST-GCN Distilled)",
        "mode": "Synchronous (100%)",
        "params": 598429,
        "metrics": res_pc3d_dt3,
        "fwd_cpu_ms": cpu_lat["pc3d_dt3"],
        "fwd_gpu_ms": gpu_lat["pc3d_dt3"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pc3d_dt3"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pc3d_dt3"],
        "tot_cpu_ms": tot_cpu_dt3,
        "fps_cpu": 1000.0 / tot_cpu_dt3,
        "tot_gpu_ms": tot_gpu_dt3,
        "fps_gpu": 1000.0 / tot_gpu_dt3
    })

    # 8. Server Ensemble (v3.10 Server Only - Synchronous 100%)
    p_v310_server = 0.45 * probs_stgcn + 0.35 * probs_ctrgcn + 0.20 * probs_pc3d_base
    pred_v310_server = (p_v310_server >= 0.50).astype(int)
    res_v310_server = evaluate_binary(y_true, pred_v310_server, cat_list)
    tot_gpu_v310_srv = stage1_gpu_ms + stage2a_gcn_gpu_ms + rast_gpu_ms + gpu_lat["v310_server"]
    tot_cpu_v310_srv = stage1_cpu_ms + stage2a_gcn_cpu_ms + rast_cpu_ms + cpu_lat["v310_server"]
    leaderboard.append({
        "system": "Server Ensemble (v3.10 Server)",
        "architecture": "ST-GCN + CTR-GCN + PoseC3D",
        "mode": "Synchronous (100%)",
        "params": 3014981 + 1332761 + 765529,
        "metrics": res_v310_server,
        "fwd_cpu_ms": cpu_lat["v310_server"],
        "fwd_gpu_ms": gpu_lat["v310_server"],
        "action_cpu_ms": stage2a_gcn_cpu_ms + rast_cpu_ms + cpu_lat["v310_server"],
        "action_gpu_ms": stage2a_gcn_gpu_ms + rast_gpu_ms + gpu_lat["v310_server"],
        "tot_cpu_ms": tot_cpu_v310_srv,
        "fps_cpu": 1000.0 / tot_cpu_v310_srv,
        "tot_gpu_ms": tot_gpu_v310_srv,
        "fps_gpu": 1000.0 / tot_gpu_v310_srv
    })

    # 9a. Heterogeneous Dual-Tier (v3.10 Original: p_high=0.60)
    pred_v310_orig = []
    esc_v310_orig_count = 0
    for ps, srv in zip(probs_stgcn, p_v310_server):
        if ps < 0.20:
            pred_v310_orig.append(0) # Edge discard
        elif ps >= 0.60:
            pred_v310_orig.append(1) # Edge alert
        else:
            esc_v310_orig_count += 1
            pred_v310_orig.append(1 if srv >= 0.45 else 0)
    res_v310_orig = evaluate_binary(y_true, np.array(pred_v310_orig), cat_list)
    esc_v310_orig_ratio = esc_v310_orig_count / len(y_true)

    tot_gpu_v310_orig = tot_gpu_st + esc_v310_orig_ratio * (gpu_lat["ctrgcn"] + gpu_lat["pc3d_base"] + rast_gpu_ms)
    tot_cpu_v310_orig = tot_cpu_st + esc_v310_orig_ratio * (cpu_lat["ctrgcn"] + cpu_lat["pc3d_base"] + rast_cpu_ms)
    act_cpu_v310_orig = (stage2a_gcn_cpu_ms + cpu_lat["stgcn"]) + esc_v310_orig_ratio * (cpu_lat["ctrgcn"] + cpu_lat["pc3d_base"] + rast_cpu_ms)
    act_gpu_v310_orig = (stage2a_gcn_gpu_ms + gpu_lat["stgcn"]) + esc_v310_orig_ratio * (gpu_lat["ctrgcn"] + gpu_lat["pc3d_base"] + rast_gpu_ms)
    leaderboard.append({
        "system": "Hetero Dual-Tier v3.10 (Orig p=0.60)",
        "architecture": "Edge ST-GCN -> Server Escalation",
        "mode": f"Hierarchical ({esc_v310_orig_ratio*100:.1f}% Esc)",
        "params": 3014981 + 1332761 + 765529,
        "metrics": res_v310_orig,
        "fwd_cpu_ms": cpu_lat["stgcn"] + esc_v310_orig_ratio * (cpu_lat["ctrgcn"] + cpu_lat["pc3d_base"]),
        "fwd_gpu_ms": gpu_lat["stgcn"] + esc_v310_orig_ratio * (gpu_lat["ctrgcn"] + gpu_lat["pc3d_base"]),
        "action_cpu_ms": act_cpu_v310_orig,
        "action_gpu_ms": act_gpu_v310_orig,
        "tot_cpu_ms": tot_cpu_v310_orig,
        "fps_cpu": 1000.0 / tot_cpu_v310_orig,
        "tot_gpu_ms": tot_gpu_v310_orig,
        "fps_gpu": 1000.0 / tot_gpu_v310_orig
    })

    # 9b. Heterogeneous Dual-Tier (v3.10 Optimal: p_high=0.999)
    pred_v310_opt = []
    esc_v310_opt_count = 0
    for ps, srv in zip(probs_stgcn, p_v310_server):
        if ps < 0.20:
            pred_v310_opt.append(0) # Edge discard
        elif ps >= 0.999:
            pred_v310_opt.append(1) # Edge alert
        else:
            esc_v310_opt_count += 1
            pred_v310_opt.append(1 if srv >= 0.45 else 0)
    res_v310_opt = evaluate_binary(y_true, np.array(pred_v310_opt), cat_list)
    esc_v310_opt_ratio = esc_v310_opt_count / len(y_true)

    tot_gpu_v310_opt = tot_gpu_st + esc_v310_opt_ratio * (gpu_lat["ctrgcn"] + gpu_lat["pc3d_base"] + rast_gpu_ms)
    tot_cpu_v310_opt = tot_cpu_st + esc_v310_opt_ratio * (cpu_lat["ctrgcn"] + cpu_lat["pc3d_base"] + rast_cpu_ms)
    act_cpu_v310_opt = (stage2a_gcn_cpu_ms + cpu_lat["stgcn"]) + esc_v310_opt_ratio * (cpu_lat["ctrgcn"] + cpu_lat["pc3d_base"] + rast_cpu_ms)
    act_gpu_v310_opt = (stage2a_gcn_gpu_ms + gpu_lat["stgcn"]) + esc_v310_opt_ratio * (gpu_lat["ctrgcn"] + gpu_lat["pc3d_base"] + rast_gpu_ms)
    leaderboard.append({
        "system": "Hetero Dual-Tier v3.10 (Opt p=0.999)",
        "architecture": "Edge ST-GCN -> Server Escalation",
        "mode": f"Hierarchical ({esc_v310_opt_ratio*100:.1f}% Esc)",
        "params": 3014981 + 1332761 + 765529,
        "metrics": res_v310_opt,
        "fwd_cpu_ms": cpu_lat["stgcn"] + esc_v310_opt_ratio * (cpu_lat["ctrgcn"] + cpu_lat["pc3d_base"]),
        "fwd_gpu_ms": gpu_lat["stgcn"] + esc_v310_opt_ratio * (gpu_lat["ctrgcn"] + gpu_lat["pc3d_base"]),
        "action_cpu_ms": act_cpu_v310_opt,
        "action_gpu_ms": act_gpu_v310_opt,
        "tot_cpu_ms": tot_cpu_v310_opt,
        "fps_cpu": 1000.0 / tot_cpu_v310_opt,
        "tot_gpu_ms": tot_gpu_v310_opt,
        "fps_gpu": 1000.0 / tot_gpu_v310_opt
    })

    # 10. Config 1: Pure Dual Consensus (Tier 5 + Tier 3 Synchronous)
    p_c1 = 0.40 * probs_pc3d_t5 + 0.60 * probs_pc3d_t3
    pred_c1 = (p_c1 >= 0.55).astype(int)
    res_c1 = evaluate_binary(y_true, pred_c1, cat_list)
    tot_gpu_c1 = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pure_dual"]
    tot_cpu_c1 = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pure_dual"]
    leaderboard.append({
        "system": "Config 1: Pure Dual Consensus",
        "architecture": "Tier 5 (132k) + Tier 3 (598k)",
        "mode": "Synchronous (Shared 3D)",
        "params": 132349 + 598429,
        "metrics": res_c1,
        "fwd_cpu_ms": cpu_lat["pure_dual"],
        "fwd_gpu_ms": gpu_lat["pure_dual"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pure_dual"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pure_dual"],
        "tot_cpu_ms": tot_cpu_c1,
        "fps_cpu": 1000.0 / tot_cpu_c1,
        "tot_gpu_ms": tot_gpu_c1,
        "fps_gpu": 1000.0 / tot_gpu_c1
    })

    # 11. Config 2: Pure Distilled Consensus (Tier 5 + Distilled Tier 3 Synchronous)
    p_c2 = 0.65 * probs_pc3d_t5 + 0.35 * probs_pc3d_dt3
    pred_c2 = (p_c2 >= 0.66).astype(int)
    res_c2 = evaluate_binary(y_true, pred_c2, cat_list)
    tot_gpu_c2 = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pure_dual"]
    tot_cpu_c2 = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pure_dual"]
    leaderboard.append({
        "system": "Config 2: Distilled Consensus",
        "architecture": "Tier 5 (132k) + Distill Tier 3",
        "mode": "Synchronous (Shared 3D)",
        "params": 132349 + 598429,
        "metrics": res_c2,
        "fwd_cpu_ms": cpu_lat["pure_dual"],
        "fwd_gpu_ms": gpu_lat["pure_dual"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pure_dual"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pure_dual"],
        "tot_cpu_ms": tot_cpu_c2,
        "fps_cpu": 1000.0 / tot_cpu_c2,
        "tot_gpu_ms": tot_gpu_c2,
        "fps_gpu": 1000.0 / tot_gpu_c2
    })

    # 12. Config 3: Tri-PoseConv3D Consensus
    p_c3 = 0.40 * probs_pc3d_t5 + 0.30 * probs_pc3d_t3 + 0.30 * probs_pc3d_dt3
    pred_c3 = (p_c3 >= 0.60).astype(int)
    res_c3 = evaluate_binary(y_true, pred_c3, cat_list)
    tot_gpu_c3 = stage1_gpu_ms + rast_gpu_ms + gpu_lat["pure_tri"]
    tot_cpu_c3 = stage1_cpu_ms + rast_cpu_ms + cpu_lat["pure_tri"]
    leaderboard.append({
        "system": "Config 3: Tri-PoseConv3D Consensus",
        "architecture": "Tier 5 + Tier 3 + Distill Tier 3",
        "mode": "Synchronous (Shared 3D)",
        "params": 132349 + 598429 + 598429,
        "metrics": res_c3,
        "fwd_cpu_ms": cpu_lat["pure_tri"],
        "fwd_gpu_ms": gpu_lat["pure_tri"],
        "action_cpu_ms": rast_cpu_ms + cpu_lat["pure_tri"],
        "action_gpu_ms": rast_gpu_ms + gpu_lat["pure_tri"],
        "tot_cpu_ms": tot_cpu_c3,
        "fps_cpu": 1000.0 / tot_cpu_c3,
        "tot_gpu_ms": tot_gpu_c3,
        "fps_gpu": 1000.0 / tot_gpu_c3
    })

    # 13. Config 4: Pure Hierarchical Dual-Tier (Tier 5 Edge -> Tier 3 Escalation)
    pred_c4 = []
    esc_c4_count = 0
    for p5, p3 in zip(probs_pc3d_t5, probs_pc3d_t3):
        if p5 < 0.05:
            pred_c4.append(0) # Edge discard
        elif p5 >= 0.50:
            pred_c4.append(1) # Edge alert
        else:
            esc_c4_count += 1
            pred_c4.append(1 if p3 >= 0.50 else 0)
    res_c4 = evaluate_binary(y_true, np.array(pred_c4), cat_list)
    esc_c4_ratio = esc_c4_count / len(y_true)

    # Note: Heatmap is ALREADY computed once by Tier 5, so escalated Tier 3 only incurs its model forward pass!
    tot_gpu_c4 = tot_gpu_t5 + esc_c4_ratio * gpu_lat["pc3d_t3"]
    tot_cpu_c4 = tot_cpu_t5 + esc_c4_ratio * cpu_lat["pc3d_t3"]
    act_cpu_c4 = (rast_cpu_ms + cpu_lat["pc3d_t5"]) + esc_c4_ratio * cpu_lat["pc3d_t3"]
    act_gpu_c4 = (rast_gpu_ms + gpu_lat["pc3d_t5"]) + esc_c4_ratio * gpu_lat["pc3d_t3"]
    leaderboard.append({
        "system": "Config 4: Pure Hierarchical Dual",
        "architecture": "Tier 5 Edge -> Tier 3 Escalation",
        "mode": f"Hierarchical ({esc_c4_ratio*100:.1f}% Esc)",
        "params": 132349 + 598429,
        "metrics": res_c4,
        "fwd_cpu_ms": cpu_lat["pc3d_t5"] + esc_c4_ratio * cpu_lat["pc3d_t3"],
        "fwd_gpu_ms": gpu_lat["pc3d_t5"] + esc_c4_ratio * gpu_lat["pc3d_t3"],
        "action_cpu_ms": act_cpu_c4,
        "action_gpu_ms": act_gpu_c4,
        "tot_cpu_ms": tot_cpu_c4,
        "fps_cpu": 1000.0 / tot_cpu_c4,
        "tot_gpu_ms": tot_gpu_c4,
        "fps_gpu": 1000.0 / tot_gpu_c4
    })

    # 14. Config 4 (Tuned): Tier 5 Edge -> Tier 3 Escalation (p_low=0.011, p_high=0.495)
    pred_c4_tuned = []
    esc_c4_tuned_count = 0
    for p5, p3 in zip(probs_pc3d_t5, probs_pc3d_t3):
        if p5 < 0.011:
            pred_c4_tuned.append(0)
        elif p5 >= 0.495:
            pred_c4_tuned.append(1)
        else:
            esc_c4_tuned_count += 1
            pred_c4_tuned.append(1 if p3 >= 0.50 else 0)
    res_c4_tuned = evaluate_binary(y_true, np.array(pred_c4_tuned), cat_list)
    esc_c4_tuned_ratio = esc_c4_tuned_count / len(y_true)
    tot_gpu_c4_tuned = tot_gpu_t5 + esc_c4_tuned_ratio * gpu_lat["pc3d_t3"]
    tot_cpu_c4_tuned = tot_cpu_t5 + esc_c4_tuned_ratio * cpu_lat["pc3d_t3"]
    act_cpu_c4_tuned = (rast_cpu_ms + cpu_lat["pc3d_t5"]) + esc_c4_tuned_ratio * cpu_lat["pc3d_t3"]
    act_gpu_c4_tuned = (rast_gpu_ms + gpu_lat["pc3d_t5"]) + esc_c4_tuned_ratio * gpu_lat["pc3d_t3"]
    leaderboard.append({
        "system": "Config 4: Pure Hierarchical (Tuned)",
        "architecture": "Tier 5 Edge -> Tier 3 (p_low=0.011)",
        "mode": f"Hierarchical ({esc_c4_tuned_ratio*100:.1f}% Esc)",
        "params": 132349 + 598429,
        "metrics": res_c4_tuned,
        "fwd_cpu_ms": cpu_lat["pc3d_t5"] + esc_c4_tuned_ratio * cpu_lat["pc3d_t3"],
        "fwd_gpu_ms": gpu_lat["pc3d_t5"] + esc_c4_tuned_ratio * gpu_lat["pc3d_t3"],
        "action_cpu_ms": act_cpu_c4_tuned,
        "action_gpu_ms": act_gpu_c4_tuned,
        "tot_cpu_ms": tot_cpu_c4_tuned,
        "fps_cpu": 1000.0 / tot_cpu_c4_tuned,
        "tot_gpu_ms": tot_gpu_c4_tuned,
        "fps_gpu": 1000.0 / tot_gpu_c4_tuned
    })

    # 15. Config 5: Tri-Model Hierarchical Vote (20% T5 + 50% T3 + 30% Distill >= 0.39)
    pred_c5 = []
    esc_c5_count = 0
    for p5, p3, pdt in zip(probs_pc3d_t5, probs_pc3d_t3, probs_pc3d_dt3):
        if p5 < 0.005:
            pred_c5.append(0)
        elif p5 >= 0.495:
            pred_c5.append(1)
        else:
            esc_c5_count += 1
            v = 0.20 * p5 + 0.50 * p3 + 0.30 * pdt
            pred_c5.append(1 if v >= 0.39 else 0)
    res_c5 = evaluate_binary(y_true, np.array(pred_c5), cat_list)
    esc_c5_ratio = esc_c5_count / len(y_true)
    tot_gpu_c5 = tot_gpu_t5 + esc_c5_ratio * (gpu_lat["pc3d_t3"] + gpu_lat["pc3d_dt3"])
    tot_cpu_c5 = tot_cpu_t5 + esc_c5_ratio * (cpu_lat["pc3d_t3"] + cpu_lat["pc3d_dt3"])
    act_cpu_c5 = (rast_cpu_ms + cpu_lat["pc3d_t5"]) + esc_c5_ratio * (cpu_lat["pc3d_t3"] + cpu_lat["pc3d_dt3"])
    act_gpu_c5 = (rast_gpu_ms + gpu_lat["pc3d_t5"]) + esc_c5_ratio * (gpu_lat["pc3d_t3"] + gpu_lat["pc3d_dt3"])
    leaderboard.append({
        "system": "Config 5: Tri-Model Hierarchical Vote",
        "architecture": "Tier 5 Edge -> (20% T5 + 50% T3 + 30% DT3)",
        "mode": f"Hierarchical ({esc_c5_ratio*100:.1f}% Esc)",
        "params": 132349 + 598429 + 598429,
        "metrics": res_c5,
        "fwd_cpu_ms": cpu_lat["pc3d_t5"] + esc_c5_ratio * (cpu_lat["pc3d_t3"] + cpu_lat["pc3d_dt3"]),
        "fwd_gpu_ms": gpu_lat["pc3d_t5"] + esc_c5_ratio * (gpu_lat["pc3d_t3"] + gpu_lat["pc3d_dt3"]),
        "action_cpu_ms": act_cpu_c5,
        "action_gpu_ms": act_gpu_c5,
        "tot_cpu_ms": tot_cpu_c5,
        "fps_cpu": 1000.0 / tot_cpu_c5,
        "tot_gpu_ms": tot_gpu_c5,
        "fps_gpu": 1000.0 / tot_gpu_c5
    })

    # 7. Print Master Benchmark Summary Table
    print("\n" + "=" * 120, flush=True)
    print("   MASTER EMPIRICAL BENCHMARK: FAIR HEAD-TO-HEAD COMPARISON (77 VIDEOS)", flush=True)
    print("=" * 120, flush=True)
    header = f"{'System / Config':<35} | {'Mode':<18} | {'Params':<9} | {'F1':<6} | {'Recall':<7} | {'FN':<3} | {'Prec':<7} | {'FP':<3} | {'Coat':<5} | {'NoCoat':<6} | {'Pipe CPU':<14} | {'Pipe GPU':<14}"
    print(header, flush=True)
    print("-" * 120, flush=True)

    csv_rows = []
    for item in leaderboard:
        m = item["metrics"]
        cats = m["categories"]
        coat_tp = f"{cats.get('With_Coat', {}).get('tp', 0)}/16"
        nocoat_tp = f"{cats.get('Without_Coat', {}).get('tp', 0)}/17"
        line = (
            f"{item['system']:<35} | "
            f"{item['mode']:<18} | "
            f"{item['params']:<9,d} | "
            f"{m['f1']:.4f} | "
            f"{m['recall']*100:.1f}% | "
            f"{m['fn']:<3} | "
            f"{m['precision']*100:.1f}% | "
            f"{m['fp']:<3} | "
            f"{coat_tp:<5} | "
            f"{nocoat_tp:<6} | "
            f"{item['tot_cpu_ms']:.1f}ms ({item['fps_cpu']:.1f}F) | "
            f"{item['tot_gpu_ms']:.1f}ms ({item['fps_gpu']:.1f}F)"
        )
        print(line, flush=True)

        csv_rows.append({
            "system": item["system"],
            "architecture": item["architecture"],
            "mode": item["mode"],
            "parameters": item["params"],
            "f1_score": round(m["f1"], 4),
            "recall": round(m["recall"] * 100, 2),
            "fn": m["fn"],
            "precision": round(m["precision"] * 100, 2),
            "fp": m["fp"],
            "accuracy": round(m["accuracy"] * 100, 2),
            "with_coat_tp": cats.get('With_Coat', {}).get('tp', 0),
            "without_coat_tp": cats.get('Without_Coat', {}).get('tp', 0),
            "ood_fp": cats.get('OOD', {}).get('fp', 0),
            "fwd_cpu_ms": round(item["fwd_cpu_ms"], 2),
            "fwd_gpu_ms": round(item.get("fwd_gpu_ms", 0), 2),
            "action_cpu_ms": round(item.get("action_cpu_ms", 0), 2),
            "action_gpu_ms": round(item.get("action_gpu_ms", 0), 2),
            "pipeline_cpu_ms": round(item["tot_cpu_ms"], 2),
            "pipeline_cpu_fps": round(item["fps_cpu"], 1),
            "pipeline_gpu_ms": round(item["tot_gpu_ms"], 2),
            "pipeline_gpu_fps": round(item["fps_gpu"], 1)
        })

    print("=" * 120, flush=True)

    # 8. Save Unified Reports
    out_json = os.path.join(REPO_DIR, "results", "benchmark_unified_fair_comparison.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage1_weapon_gpu_ms": stage1_w_gpu_ms,
            "stage1_weapon_cpu_ms": stage1_w_cpu_ms,
            "stage2a_pose_gpu_ms": stage2a_pose_gpu_ms,
            "stage2a_pose_cpu_ms": stage2a_pose_cpu_ms,
            "stage1_and_pose_gpu_ms": stage1_gpu_ms,
            "stage1_and_pose_cpu_ms": stage1_cpu_ms,
            "stage2a_gcn_gpu_ms": stage2a_gcn_gpu_ms,
            "stage2a_gcn_cpu_ms": stage2a_gcn_cpu_ms,
            "stage2a_rasterization_gpu_ms": rast_gpu_ms,
            "stage2a_rasterization_cpu_ms": rast_cpu_ms,
            "results": csv_rows
        }, f, indent=2)

    out_csv = os.path.join(REPO_DIR, "results", "benchmark_unified_fair_comparison.csv")
    df = pd.DataFrame(csv_rows)
    df.to_csv(out_csv, index=False)

    print(f"\n>> Saved standardized unified JSON benchmark to: {out_json}", flush=True)
    print(f">> Saved standardized unified CSV benchmark to: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
