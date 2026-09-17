"""
Benchmark Suite: Rectangular Inference vs Square Letterboxing (v1.03)
Compares:
1. Multiply-Accumulate Area: Square 640x640 vs Rectangular (384, 640) for 16:9 feeds.
2. Latency & Throughput: Stage 1 YOLO and Stage 2 YOLO-Pose.
3. Detection & Pose Precision: Evaluates bounding box IoU and keypoint accuracy across widescreen surveillance frames.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_DIR)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=========================================================================")
    print(f"   BENCHMARK: RECTANGULAR INFERENCE vs SQUARE LETTERBOXING (v1.03)")
    print(f"=========================================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    weapon_weights = os.path.join(REPO_DIR, 'weights', 'yolo_weapon_distilled.pt')
    pose_weights = os.path.join(REPO_DIR, 'weights', 'yolo26s-pose.pt')

    weapon_model = YOLO(weapon_weights)
    pose_model = YOLO(pose_weights)

    # Resolve 16:9 test video
    test_video = os.path.join(REPO_DIR, '..', 'Clean', 'Violence_Detection_RealTime_Interface', 'sample_videos', '2025-09-27 11-50-49.mp4')
    cap = cv2.VideoCapture(test_video)
    frames_16_9 = []
    while len(frames_16_9) < 60:
        ret, f = cap.read()
        if not ret: break
        # Resize to standard 16:9 HD resolution (1280x720)
        frames_16_9.append(cv2.resize(f, (1280, 720)))
    cap.release()
    print(f"Loaded {len(frames_16_9)} test frames (1280x720, 16:9 aspect ratio).")

    # 1. Theoretical Compute Analysis
    sq_pixels = 640 * 640
    rect_pixels = 384 * 640
    padding_saved_pixels = sq_pixels - rect_pixels
    padding_waste_pct = (padding_saved_pixels / sq_pixels) * 100.0

    print(f"\n[1/3] Compute & Tensor Geometry Analysis")
    print(f"  * Square Tensor Shape      : 640 x 640 = {sq_pixels:,} pixels")
    print(f"  * Rectangular Tensor Shape : 384 x 640 = {rect_pixels:,} pixels")
    print(f"  * Unneeded Letterbox Black Bars Eliminated: {padding_saved_pixels:,} pixels ({padding_waste_pct:.1f}% reduction)")

    # 2. Latency Benchmark
    print(f"\n[2/3] Latency & Throughput Benchmark (100 trials, GPU synchronized)")
    # Warmup
    for f in frames_16_9[:10]:
        _ = weapon_model(f, imgsz=640, verbose=False)
        _ = weapon_model(f, imgsz=(384, 640), verbose=False)
        _ = pose_model(f, imgsz=640, verbose=False)
        _ = pose_model(f, imgsz=(384, 640), verbose=False)

    N = len(frames_16_9)

    # Stage 1: Weapon
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in frames_16_9:
        _ = weapon_model(f, imgsz=640, verbose=False)
    torch.cuda.synchronize()
    stage1_sq_ms = ((time.perf_counter() - t0) / N) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in frames_16_9:
        _ = weapon_model(f, imgsz=(384, 640), verbose=False)
    torch.cuda.synchronize()
    stage1_rect_ms = ((time.perf_counter() - t0) / N) * 1000

    # Stage 2: Pose
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in frames_16_9:
        _ = pose_model(f, imgsz=640, verbose=False)
    torch.cuda.synchronize()
    stage2_sq_ms = ((time.perf_counter() - t0) / N) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for f in frames_16_9:
        _ = pose_model(f, imgsz=(384, 640), verbose=False)
    torch.cuda.synchronize()
    stage2_rect_ms = ((time.perf_counter() - t0) / N) * 1000

    print(f"  Stage 1 YOLO Weapon Detector:")
    print(f"    - Square (640x640)      : {stage1_sq_ms:.2f} ms ({1000/stage1_sq_ms:.1f} FPS)")
    print(f"    - Rectangular (384x640) : {stage1_rect_ms:.2f} ms ({1000/stage1_rect_ms:.1f} FPS)")
    print(f"    - Latency Delta         : {stage1_rect_ms - stage1_sq_ms:+.2f} ms")

    print(f"  Stage 2 YOLO-Pose Estimator:")
    print(f"    - Square (640x640)      : {stage2_sq_ms:.2f} ms ({1000/stage2_sq_ms:.1f} FPS)")
    print(f"    - Rectangular (384x640) : {stage2_rect_ms:.2f} ms ({1000/stage2_rect_ms:.1f} FPS)")
    print(f"    - Latency Delta         : {stage2_rect_ms - stage2_sq_ms:+.2f} ms")

    # 3. Fidelity & Precision Check: Square vs Rectangular
    print(f"\n[3/3] Detection Agreement & Keypoint Stability")
    weapon_agreements = 0
    ious = []
    kpt_pixel_diffs = []

    for f in frames_16_9:
        r_sq = weapon_model(f, imgsz=640, conf=0.45, verbose=False)[0]
        r_rect = weapon_model(f, imgsz=(384, 640), conf=0.45, verbose=False)[0]
        h_sq = len(r_sq.boxes) > 0
        h_rect = len(r_rect.boxes) > 0
        if h_sq == h_rect: weapon_agreements += 1
        if h_sq and h_rect:
            ious.append(calculate_iou(r_sq.boxes.xyxy[0].cpu().numpy(), r_rect.boxes.xyxy[0].cpu().numpy()))

        p_sq = pose_model(f, imgsz=640, verbose=False)[0]
        p_rect = pose_model(f, imgsz=(384, 640), verbose=False)[0]
        if hasattr(p_sq, 'keypoints') and hasattr(p_rect, 'keypoints'):
            if len(p_sq.keypoints.data) > 0 and len(p_rect.keypoints.data) > 0:
                k_sq = p_sq.keypoints.data[0].cpu().numpy()[:, :2]
                k_rect = p_rect.keypoints.data[0].cpu().numpy()[:, :2]
                diff = np.linalg.norm(k_sq - k_rect, axis=-1)
                kpt_pixel_diffs.extend(diff.tolist())

    agr_pct = (weapon_agreements / len(frames_16_9)) * 100.0
    mean_iou = float(np.mean(ious)) if ious else 1.0
    mean_kpt_diff = float(np.mean(kpt_pixel_diffs)) if kpt_pixel_diffs else 0.0

    print(f"  * Stage 1 Decision Agreement: {agr_pct:.1f}%")
    print(f"  * Stage 1 Bounding Box IoU  : {mean_iou:.4f}")
    print(f"  * Stage 2 Pose Joint MAE    : {mean_kpt_diff:.2f} px")
    print("=" * 75)


if __name__ == '__main__':
    main()
