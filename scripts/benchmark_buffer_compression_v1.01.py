"""
Benchmark Suite for In-Memory JPEG Buffer Compression (v1.01 vs v1.00 Baseline)
Evaluates:
1. Memory Consumption: Raw 1200-frame circular buffer (v1.00) vs JPEG-compressed buffer (v1.01).
2. Speed & Latency: Ingestion time (encode), retrieval time (decode), and throughput (FPS).
3. Detection & Keypoint Fidelity: Impact of Quality=85 JPEG compression on YOLO Weapon Detection,
   YOLO-Pose keypoint estimation, and ST-GCN k-NN action feature embeddings.
"""

import os
import sys
import gc
import time
import json
import argparse
from collections import deque
import cv2
import numpy as np
import torch
import psutil
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stgcn import STGCNModel
from src.skeleton_utils import interpolate_missing_joints, smooth_kinematics, normalize_skeleton_clip
from src.ood_metrics import compute_knn_distance


# Import v1.00 FrameItem
from src.inference_realtime import FrameItem as FrameItemV1_00

# Dynamically import v1.01 FrameItem
import importlib.util
spec = importlib.util.spec_from_file_location(
    "inference_realtime_v1_01",
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'inference_realtime_v1.01.py'))
)
v1_01_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1_01_mod)
FrameItemV1_01 = v1_01_mod.FrameItem


def get_process_memory_mb():
    """Return process RSS in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def load_sample_frames(video_path, target_count=1200, target_size=(640, 480)):
    """Load frames from video, resizing to standard surveillance resolution and looping if needed."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {video_path}")

    raw_frames = []
    while len(raw_frames) < target_count:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        resized = cv2.resize(frame, target_size)
        raw_frames.append(resized)
        if len(raw_frames) >= target_count:
            break

    cap.release()
    return raw_frames


def benchmark_memory_and_throughput(frames, buffer_capacity=1200, jpeg_quality=85):
    """Benchmark memory footprint and ingestion/retrieval throughput for v1.00 vs v1.01."""
    num_frames = len(frames)
    results = {}

    print(f"\n[BENCHMARK 1/3] Memory & Speed (1,200 frames @ {frames[0].shape[1]}x{frames[0].shape[0]}x3)")
    print("-" * 75)

    # -------------------------------------------------------------
    # 1. Baseline v1.00 (Raw Uncompressed NumPy Buffer)
    # -------------------------------------------------------------
    gc.collect()
    time.sleep(0.5)
    rss_before_v1_00 = get_process_memory_mb()

    buf_v1_00 = deque(maxlen=buffer_capacity)
    t0 = time.perf_counter()
    for idx, f in enumerate(frames):
        buf_v1_00.append(FrameItemV1_00(f, time.time(), idx))
    ingest_time_v1_00 = time.perf_counter() - t0

    rss_after_v1_00 = get_process_memory_mb()
    payload_mb_v1_00 = sum(it.frame.nbytes for it in buf_v1_00) / (1024.0 * 1024.0)
    rss_delta_v1_00 = max(0.0, rss_after_v1_00 - rss_before_v1_00)

    # Retrieval / Copy Speed
    t0 = time.perf_counter()
    for it in buf_v1_00:
        _ = it.frame.copy()
    retrieval_time_v1_00 = time.perf_counter() - t0

    del buf_v1_00
    gc.collect()
    time.sleep(0.5)

    # -------------------------------------------------------------
    # 2. Optimized v1.01 (In-Memory JPEG Compressed Buffer)
    # -------------------------------------------------------------
    rss_before_v1_01 = get_process_memory_mb()

    buf_v1_01 = deque(maxlen=buffer_capacity)
    t0 = time.perf_counter()
    for idx, f in enumerate(frames):
        buf_v1_01.append(FrameItemV1_01(f, time.time(), idx, jpeg_quality=jpeg_quality))
    ingest_time_v1_01 = time.perf_counter() - t0

    rss_after_v1_01 = get_process_memory_mb()
    payload_mb_v1_01 = sum(it.nbytes for it in buf_v1_01) / (1024.0 * 1024.0)
    rss_delta_v1_01 = max(0.0, rss_after_v1_01 - rss_before_v1_01)

    # Retrieval / Decompression Speed
    t0 = time.perf_counter()
    for it in buf_v1_01:
        _ = it.get_frame()
    retrieval_time_v1_01 = time.perf_counter() - t0

    # Calculate metrics
    reduction_ratio = payload_mb_v1_00 / max(0.001, payload_mb_v1_01)
    savings_pct = (1.0 - (payload_mb_v1_01 / payload_mb_v1_00)) * 100.0

    ingest_latency_v1_00_ms = (ingest_time_v1_00 / num_frames) * 1000.0
    ingest_latency_v1_01_ms = (ingest_time_v1_01 / num_frames) * 1000.0
    ingest_fps_v1_01 = num_frames / ingest_time_v1_01

    decode_latency_v1_01_ms = (retrieval_time_v1_01 / num_frames) * 1000.0
    retrieval_fps_v1_01 = num_frames / retrieval_time_v1_01

    results['memory'] = {
        'buffer_capacity': buffer_capacity,
        'frame_resolution': f"{frames[0].shape[1]}x{frames[0].shape[0]}",
        'v1_00_payload_mb': round(payload_mb_v1_00, 2),
        'v1_01_payload_mb': round(payload_mb_v1_01, 2),
        'v1_00_rss_delta_mb': round(rss_delta_v1_00, 2),
        'v1_01_rss_delta_mb': round(rss_delta_v1_01, 2),
        'compression_ratio': round(reduction_ratio, 2),
        'savings_percentage': round(savings_pct, 2)
    }

    results['speed'] = {
        'v1_00_ingest_latency_ms': round(ingest_latency_v1_00_ms, 4),
        'v1_01_ingest_latency_ms': round(ingest_latency_v1_01_ms, 4),
        'v1_01_max_ingest_fps': round(ingest_fps_v1_01, 1),
        'v1_00_retrieval_latency_ms': round((retrieval_time_v1_00 / num_frames) * 1000.0, 4),
        'v1_01_decode_latency_ms': round(decode_latency_v1_01_ms, 4),
        'v1_01_max_decode_fps': round(retrieval_fps_v1_01, 1)
    }

    print(f"  [v1.00 Baseline] Buffer Payload : {payload_mb_v1_00:.2f} MB | Process RSS Delta: {rss_delta_v1_00:.2f} MB")
    print(f"  [v1.01 Compressed] Buffer Payload: {payload_mb_v1_01:.2f} MB | Process RSS Delta: {rss_delta_v1_01:.2f} MB")
    print(f"  -> Memory Reduction Factor      : {reduction_ratio:.2f}x ({savings_pct:.1f}% RAM savings)")
    print(f"  -> Ingestion (Encode) Latency   : {ingest_latency_v1_01_ms:.3f} ms/frame ({ingest_fps_v1_01:.1f} FPS capacity)")
    print(f"  -> Retrieval (Decode) Latency   : {decode_latency_v1_01_ms:.3f} ms/frame ({retrieval_fps_v1_01:.1f} FPS capacity)")

    return results, buf_v1_01


def calculate_iou(box1, box2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def benchmark_model_fidelity(test_frames, weights_dir, device='cuda', num_eval_frames=100):
    """Verify detection and pose extraction fidelity between raw frames and Quality=85 decoded frames."""
    print(f"\n[BENCHMARK 2/3] Model Accuracy & Fidelity on Raw vs Decoded Frames ({num_eval_frames} frames)")
    print("-" * 75)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    weapon_model_path = os.path.join(weights_dir, 'yolo_weapon_distilled.pt')
    pose_model_path = os.path.join(weights_dir, 'yolo26s-pose.pt')
    action_model_path = os.path.join(weights_dir, 'stgcn_violence_fold2.pth')
    ref_weights_path = os.path.join(weights_dir, 'reference_data_knn.pt')

    weapon_model = YOLO(weapon_model_path)
    pose_model = YOLO(pose_model_path)

    action_model = STGCNModel(num_classes=1, in_channels=2, num_keypoints=17).to(device)
    ckpt = torch.load(action_model_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    action_model.load_state_dict(state_dict, strict=False)
    action_model.eval()

    ref_data = torch.load(ref_weights_path, map_location='cpu', weights_only=False)
    knn_bank = ref_data['feature_bank'].float().cpu()
    knn_threshold = float(ref_data['threshold'])
    knn_k = int(ref_data.get('k', 2))

    eval_frames = test_frames[:num_eval_frames]

    # Pre-generate JPEG compressed/decompressed pairs
    pairs = []
    for f in eval_frames:
        _, enc = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        pairs.append((f, dec))

    # --- Stage 1: Weapon Detection Fidelity ---
    print("  -> Evaluating Stage 1: Distilled YOLO Weapon Detection...")
    weapon_agreements = 0
    box_ious = []
    conf_diffs = []

    for raw, dec in pairs:
        res_raw = weapon_model(raw, conf=0.45, verbose=False)[0]
        res_dec = weapon_model(dec, conf=0.45, verbose=False)[0]

        has_raw = len(res_raw.boxes) > 0
        has_dec = len(res_dec.boxes) > 0

        if has_raw == has_dec:
            weapon_agreements += 1

        if has_raw and has_dec:
            b_raw = res_raw.boxes.xyxy[0].cpu().numpy()
            b_dec = res_dec.boxes.xyxy[0].cpu().numpy()
            c_raw = float(res_raw.boxes.conf[0].cpu().numpy())
            c_dec = float(res_dec.boxes.conf[0].cpu().numpy())

            iou = calculate_iou(b_raw, b_dec)
            box_ious.append(iou)
            conf_diffs.append(abs(c_raw - c_dec))

    stage1_agreement_pct = (weapon_agreements / len(pairs)) * 100.0
    mean_box_iou = float(np.mean(box_ious)) if box_ious else 1.0
    mean_conf_diff = float(np.mean(conf_diffs)) if conf_diffs else 0.0

    print(f"     * Weapon Presence Decision Agreement : {stage1_agreement_pct:.1f}%")
    print(f"     * Bounding Box Mean IoU               : {mean_box_iou:.4f}")
    print(f"     * Mean Absolute Confidence Delta      : {mean_conf_diff:.4f}")

    # --- Stage 2: YOLO26s Pose Keypoint Extraction Fidelity ---
    print("  -> Evaluating Stage 2: YOLO26s Pose Keypoint Estimation...")
    kpt_pixel_errors = []
    pck_hits = 0
    total_kpts = 0

    raw_skeletons = []
    dec_skeletons = []

    for raw, dec in pairs:
        res_raw = pose_model(raw, verbose=False)[0]
        res_dec = pose_model(dec, verbose=False)[0]

        kpts_raw = np.zeros((17, 2), dtype=np.float32)
        kpts_dec = np.zeros((17, 2), dtype=np.float32)

        if hasattr(res_raw, 'keypoints') and res_raw.keypoints is not None and len(res_raw.keypoints.data) > 0:
            kpts_raw = res_raw.keypoints.data[0].cpu().numpy()[:, :2]
        if hasattr(res_dec, 'keypoints') and res_dec.keypoints is not None and len(res_dec.keypoints.data) > 0:
            kpts_dec = res_dec.keypoints.data[0].cpu().numpy()[:, :2]

        raw_skeletons.append(kpts_raw)
        dec_skeletons.append(kpts_dec)

        if np.any(kpts_raw > 0) and np.any(kpts_dec > 0):
            # Compute Euclidean error per joint
            diffs = np.linalg.norm(kpts_raw - kpts_dec, axis=-1)
            kpt_pixel_errors.extend(diffs.tolist())

            # PCK@0.05 (threshold = 5% of bounding box diagonal or frame height)
            diag = 0.05 * np.sqrt(raw.shape[0]**2 + raw.shape[1]**2)
            pck_hits += np.sum(diffs <= diag)
            total_kpts += len(diffs)

    mean_kpt_mae = float(np.mean(kpt_pixel_errors)) if kpt_pixel_errors else 0.0
    pck_score = (pck_hits / max(1, total_kpts)) * 100.0

    print(f"     * Joint Keypoint Mean Pixel Error     : {mean_kpt_mae:.3f} px")
    print(f"     * Joint PCK@0.05 Consistency          : {pck_score:.2f}%")

    # --- Stage 3: ST-GCN + Deep k-NN Embedding Fidelity ---
    print("  -> Evaluating Stage 3: ST-GCN Action Classifier & Deep k-NN Manifold...")
    clip_len = 50
    stgcn_cos_sims = []
    knn_dist_deltas = []
    violence_agreements = 0
    clip_eval_count = 0

    for start_idx in range(0, len(raw_skeletons) - clip_len + 1, 10):
        sub_raw = raw_skeletons[start_idx : start_idx + clip_len]
        sub_dec = dec_skeletons[start_idx : start_idx + clip_len]

        if not np.any(np.array(sub_raw) > 0) or not np.any(np.array(sub_dec) > 0):
            continue

        clip_eval_count += 1

        # Preprocess both sequences
        seq_raw = normalize_skeleton_clip(smooth_kinematics(interpolate_missing_joints(np.array(sub_raw))))
        seq_dec = normalize_skeleton_clip(smooth_kinematics(interpolate_missing_joints(np.array(sub_dec))))

        t_raw = torch.from_numpy(seq_raw).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(device)
        t_dec = torch.from_numpy(seq_dec).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            _, feat_raw = action_model(t_raw, return_features=True)
            _, feat_dec = action_model(t_dec, return_features=True)

        f_raw_cpu = feat_raw.cpu()
        f_dec_cpu = feat_dec.cpu()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(f_raw_cpu, f_dec_cpu).item()
        stgcn_cos_sims.append(cos_sim)

        # k-NN distance
        d_raw = compute_knn_distance(f_raw_cpu, knn_bank, k=knn_k, mode='kth')
        d_dec = compute_knn_distance(f_dec_cpu, knn_bank, k=knn_k, mode='kth')
        knn_dist_deltas.append(abs(d_raw - d_dec))

        v_raw = (d_raw <= knn_threshold)
        v_dec = (d_dec <= knn_threshold)
        if v_raw == v_dec:
            violence_agreements += 1

    mean_cos_sim = float(np.mean(stgcn_cos_sims)) if stgcn_cos_sims else 1.0
    mean_dist_delta = float(np.mean(knn_dist_deltas)) if knn_dist_deltas else 0.0
    action_agree_pct = (violence_agreements / max(1, clip_eval_count)) * 100.0

    print(f"     * ST-GCN Embedding Cosine Similarity : {mean_cos_sim:.5f}")
    print(f"     * Mean k-NN Manifold Distance Delta  : {mean_dist_delta:.4f}")
    print(f"     * Final Violence Decision Agreement  : {action_agree_pct:.1f}%")

    fidelity_results = {
        'weapon_decision_agreement_pct': round(stage1_agreement_pct, 2),
        'weapon_box_iou': round(mean_box_iou, 4),
        'weapon_conf_delta': round(mean_conf_diff, 4),
        'pose_joint_mae_pixels': round(mean_kpt_mae, 3),
        'pose_pck_05_pct': round(pck_score, 2),
        'stgcn_embedding_cosine_similarity': round(mean_cos_sim, 5),
        'knn_manifold_distance_delta': round(mean_dist_delta, 4),
        'violence_decision_agreement_pct': round(action_agree_pct, 2)
    }

    return fidelity_results


def benchmark_realtime_duty_cycle(speed_metrics):
    """Calculate duty cycle and edge CPU/GPU overhead in real-time execution."""
    print("\n[BENCHMARK 3/3] Real-Time Edge Deployment Duty Cycle Analysis")
    print("-" * 75)

    ingest_ms = speed_metrics['v1_01_ingest_latency_ms']
    decode_ms = speed_metrics['v1_01_decode_latency_ms']

    # At 30 FPS webcam ingestion:
    # Camera thread encodes 30 frames/sec:
    cam_cpu_overhead_pct = (30 * ingest_ms) / 1000.0 * 100.0

    # Weapon scanning thread runs at 3 FPS:
    weapon_scan_overhead_pct = (3 * decode_ms) / 1000.0 * 100.0

    # Action worker (when threat is active): strides 10 frames every ~0.33s (10 FPS stride):
    action_decode_overhead_pct = (10 * decode_ms) / 1000.0 * 100.0

    total_duty_overhead_pct = cam_cpu_overhead_pct + weapon_scan_overhead_pct

    print(f"  -> Ingestion CPU Load @ 30 FPS stream : {cam_cpu_overhead_pct:.2f}% of single CPU core ({30 * ingest_ms:.2f} ms/sec)")
    print(f"  -> Screening Decompression @ 3 FPS     : {weapon_scan_overhead_pct:.2f}% of single CPU core ({3 * decode_ms:.2f} ms/sec)")
    print(f"  -> Action Worker Pose Decode @ Threat  : {action_decode_overhead_pct:.2f}% of single CPU core ({10 * decode_ms:.2f} ms/sec)")
    print(f"  -> Total Idle Surveillance Duty Load   : {total_duty_overhead_pct:.2f}% CPU overhead (Negligible)")

    duty_results = {
        'cam_ingest_30fps_cpu_overhead_pct': round(cam_cpu_overhead_pct, 2),
        'screening_decode_3fps_cpu_overhead_pct': round(weapon_scan_overhead_pct, 2),
        'action_decode_threat_cpu_overhead_pct': round(action_decode_overhead_pct, 2),
        'total_idle_duty_cpu_overhead_pct': round(total_duty_overhead_pct, 2)
    }
    return duty_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark In-Memory JPEG Buffer Compression (v1.01)")
    parser.add_argument('--video', type=str, default=None, help='Sample video path')
    parser.add_argument('--frames', type=int, default=1200, help='Number of frames to buffer (default: 1200)')
    parser.add_argument('--quality', type=int, default=85, help='JPEG compression quality (default: 85)')
    args = parser.parse_args()

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weights_dir = os.path.join(repo_dir, 'weights')

    # Resolve test video
    test_video = args.video
    if test_video is None or not os.path.exists(test_video):
        candidate_videos = [
            os.path.join(repo_dir, '..', 'Clean', 'Violence_Detection_RealTime_Interface', 'sample_videos', '2025-09-27 11-50-49.mp4'),
            os.path.join(repo_dir, '..', 'Violence_Detection_RealTime_Interface', 'Sample dataset video', '2025-09-27 11-50-49.mp4'),
            os.path.join(repo_dir, '..', 'Clean', 'Train_Action_Recognition_STGCN_Model', 'data', 'Validate', 'OOD', 'Chop_Wood.mp4'),
        ]
        for cv in candidate_videos:
            if os.path.exists(cv):
                test_video = cv
                break

    if test_video is None or not os.path.exists(test_video):
        raise FileNotFoundError("No valid test video found for benchmarking.")

    print(f"\n=========================================================================")
    print(f"       BENCHMARK REPORT: IN-MEMORY JPEG BUFFER COMPRESSION (v1.01)")
    print(f"=========================================================================")
    print(f"Target Buffer Size : {args.frames} frames")
    print(f"JPEG Quality       : {args.quality}")
    print(f"Test Video Source  : {test_video}")

    # Load 1,200 sample frames
    print(f"Loading {args.frames} surveillance frames...")
    frames = load_sample_frames(test_video, target_count=args.frames, target_size=(640, 480))
    print(f"Loaded {len(frames)} frames successfully.")

    # 1. Run Memory & Throughput Benchmark
    mem_speed_results, _ = benchmark_memory_and_throughput(frames, buffer_capacity=args.frames, jpeg_quality=args.quality)

    # 2. Run Model Fidelity Benchmark
    fidelity_results = benchmark_model_fidelity(frames, weights_dir, num_eval_frames=100)

    # 3. Run Duty Cycle Analysis
    duty_results = benchmark_realtime_duty_cycle(mem_speed_results['speed'])

    # Compile Final Benchmark Summary
    summary = {
        'benchmark_target': 'In-Memory JPEG Buffer Compression (Option A)',
        'baseline_version': 'v1.00 (Uncompressed NumPy FrameItem)',
        'evaluated_version': 'v1.01 (JPEG Quality=85 FrameItem)',
        'memory_metrics': mem_speed_results['memory'],
        'speed_metrics': mem_speed_results['speed'],
        'fidelity_metrics': fidelity_results,
        'duty_metrics': duty_results
    }

    out_path = os.path.join(repo_dir, 'results', 'benchmark_buffer_compression_v1.01.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OUTPUT] Detailed benchmark metrics exported to: {out_path}")
    print("=" * 75)


if __name__ == '__main__':
    main()
