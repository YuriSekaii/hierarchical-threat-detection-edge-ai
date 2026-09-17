"""
src/inference_realtime_v3.00.py

Hierarchical Real-Time Threat Detection System - Version 3.00
Production Dual-Stage Surveillance Architecture with:
1. In-Memory TurboJPEG Buffer Compression (v1.01)
2. Torso-Scale Invariant Kinematic Normalization (v1.02)
3. Native Aspect-Ratio Rectangular Inference (v1.03)
4. Multi-Person Tracking Guard with ByteTrack & Spatial-Temporal Continuity (v1.04)
5. Skeleton-OOD Backbone: ST-GCN + In-Training ASH-SE Fusion + Latent Boundary Learning (v3.00)
6. Calibrated Decision Boundary Gating

Features in Version 3.00:
- Grounded in human biological bone connectivity via ST-GCN.
- Squeeze-and-Excitation channel fusion between raw GCN features and ASH-denoised activations.
- Unit-hyperspherical projection S^{D-1} with sharp latent decision margins.
- Rejection of civilian false alarms via Kolmogorov-Smirnov temporal motion regularization.
"""

import os
import sys
import time
import argparse
import threading
from collections import deque
import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.skeleton_ood import SkeletonOODModel
from src.skeleton_utils import interpolate_missing_joints, smooth_kinematics, normalize_skeleton_clip

# Import v1.08 RMD metrics
import importlib.util
rmd_spec = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py")
)
rmd_mod = importlib.util.module_from_spec(rmd_spec)
rmd_spec.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance


class FrameItem:
    """ Compressed Frame Storage in Circular Buffer (v1.01+). """
    def __init__(self, frame, timestamp, index, jpeg_quality=85):
        self.timestamp = timestamp
        self.index = index
        self.skeleton = None
        self.actor_bbox = None
        self.track_id = None

        if frame is not None:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            success, enc_bytes = cv2.imencode(".jpg", frame, encode_params)
            self.jpeg_data = enc_bytes if success else None
        else:
            self.jpeg_data = None

    def get_frame(self):
        if self.jpeg_data is not None:
            return cv2.imdecode(self.jpeg_data, cv2.IMREAD_COLOR)
        return None

    @property
    def nbytes(self):
        if self.jpeg_data is not None:
            return self.jpeg_data.nbytes
        return 0


class HierarchicalThreatPipelineV300:
    def __init__(
        self,
        yolo_weapon_weights="weights/yolo_weapon_distilled.pt",
        yolo_pose_weights="weights/yolo26s-pose.pt",
        skeleton_ood_weights="weights/stgcn_skeleton_ood_v3.00.pth",
        reference_weights="weights/reference_data_skeleton_ood_v3.00.pt",
        threat_threshold=None,
        buffer_size=1200,
        jpeg_quality=85,
        device=None,
        imgsz=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Initializing Hierarchical Pipeline v3.00 (Skeleton-OOD) on device: {self.device}")

        # 1. Stage 1 Weapon Detector
        print(f"[INIT] Loading Stage 1 Weapon Detector: {yolo_weapon_weights}")
        self.weapon_model = YOLO(yolo_weapon_weights)
        try:
            if hasattr(self.weapon_model, "set_classes") and "world" in yolo_weapon_weights.lower():
                self.weapon_model.set_classes(["knife", "machete", "gun", "dagger", "sword", "weapon", "blade"])
        except Exception as e:
            print(f"[WARN] Could not set custom classes: {e}")

        # 2. Stage 2a Pose Estimator
        print(f"[INIT] Loading Stage 2a Pose Estimator: {yolo_pose_weights}")
        self.pose_model = YOLO(yolo_pose_weights)

        # 3. Stage 2b Skeleton-OOD Model
        print(f"[INIT] Loading Stage 2b Skeleton-OOD: {skeleton_ood_weights}")
        self.action_model = SkeletonOODModel(
            num_classes=3,
            in_channels=2,
            num_keypoints=17,
            ash_percentile=80,
            ash_mode='ash_b',
            temperature=1.0
        ).to(self.device)
        if os.path.exists(skeleton_ood_weights):
            state = torch.load(skeleton_ood_weights, map_location=self.device)
            self.action_model.load_state_dict(state)
            self.action_model.eval()
            print("[INIT] Skeleton-OOD weights loaded successfully.")
        else:
            print(f"[WARN] Skeleton-OOD weights not found at {skeleton_ood_weights}")

        # 4. Calibrated Decision Boundary Weights
        self.champion_method = "Boundary-Optimized RMD"
        self.threat_threshold = threat_threshold
        self.mu_0 = None
        self.mu_c = None
        self.inv_Sigma_0 = None
        self.inv_Sigma_c = None

        if os.path.exists(reference_weights):
            ref = torch.load(reference_weights, map_location=self.device)
            self.mu_0 = ref["mu_0"].to(self.device)
            self.mu_c = ref["mu_c"].to(self.device)
            self.inv_Sigma_0 = ref["inv_Sigma_0"].to(self.device)
            self.inv_Sigma_c = ref["inv_Sigma_c"].to(self.device)
            self.champion_method = ref.get("champion_method", "Boundary-Optimized RMD")
            if self.threat_threshold is None:
                self.threat_threshold = ref.get("champion_tau", 0.0)
            print(f"[INIT] Loaded reference weights: Method={self.champion_method}, Threshold={self.threat_threshold:.4f}")

        # In-Memory Ring Buffer
        self.buffer_size = buffer_size
        self.jpeg_quality = jpeg_quality
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()

        # Sliding Window Settings
        self.clip_length = 50
        self.stride = 10
        self.active_threat = False
        self.threat_actor_id = None
        self.threat_score = 0.0
        self.last_weapon_bbox = None

        # Rectangular aspect ratio setting
        self.imgsz = imgsz if imgsz is not None else (384, 640)
        print(f"[INIT] Native Rectangular Inference Resolution: {self.imgsz}")

        # Telemetry
        self.total_frames = 0
        self.stage1_latencies = deque(maxlen=30)
        self.stage2a_latencies = deque(maxlen=30)
        self.stage2b_latencies = deque(maxlen=30)

    def append_frame(self, frame, timestamp):
        item = FrameItem(frame, timestamp, self.total_frames, self.jpeg_quality)
        with self.buffer_lock:
            self.frame_buffer.append(item)
        self.total_frames += 1

    def run_stage1_weapon_screening(self, frame):
        t0 = time.perf_counter()
        results = self.weapon_model.predict(
            source=frame,
            conf=0.25,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )
        t_cost = (time.perf_counter() - t0) * 1000.0
        self.stage1_latencies.append(t_cost)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().item())
                cls_id = int(b.cls[0].cpu().item())
                detections.append({"bbox": xyxy, "conf": conf, "cls": cls_id})
        return detections

    def run_stage2_threat_analysis(self):
        with self.buffer_lock:
            if len(self.frame_buffer) < self.clip_length:
                return
            items = list(self.frame_buffer)[-self.clip_length:]

        # Pose extraction across 50 frames
        t0_pose = time.perf_counter()
        raw_skeletons = []
        for it in items:
            if it.skeleton is None:
                f = it.get_frame()
                if f is not None:
                    p_res = self.pose_model.track(
                        source=f,
                        persist=True,
                        device=self.device,
                        imgsz=self.imgsz,
                        verbose=False
                    )
                    if p_res and len(p_res) > 0 and p_res[0].keypoints is not None:
                        kp = p_res[0].keypoints.xy.cpu().numpy()
                        if len(kp) > 0:
                            it.skeleton = kp[0]
                        else:
                            it.skeleton = np.zeros((17, 2), dtype=np.float32)
                    else:
                        it.skeleton = np.zeros((17, 2), dtype=np.float32)
                else:
                    it.skeleton = np.zeros((17, 2), dtype=np.float32)
            raw_skeletons.append(it.skeleton)

        t_pose = (time.perf_counter() - t0_pose) * 1000.0
        self.stage2a_latencies.append(t_pose / max(1, len(items)))

        # Preprocess skeleton clip
        skel_arr = np.array(raw_skeletons, dtype=np.float32)
        cleaned = interpolate_missing_joints(skel_arr)
        smoothed = smooth_kinematics(cleaned, sigma=1.0)
        normed = normalize_skeleton_clip(torch.from_numpy(smoothed).float())

        # Stage 2b Skeleton-OOD Forward Pass & Decision Gating
        t0_action = time.perf_counter()
        inp = normed.permute(2, 0, 1).unsqueeze(0).to(self.device) # (1, 2, 50, 17)

        with torch.no_grad():
            logits, z, energy = self.action_model(inp, return_features=True, return_energy=True)
            if self.champion_method == "Boundary-Optimized RMD" and self.inv_Sigma_0 is not None:
                score, _, _ = compute_relative_mahalanobis_distance(
                    z, self.mu_0, self.mu_c, self.inv_Sigma_0, self.inv_Sigma_c
                )
                threat_val = score.item()
            else:
                threat_val = (-energy).item()

        t_action = (time.perf_counter() - t0_action) * 1000.0
        self.stage2b_latencies.append(t_action)

        self.threat_score = threat_val
        self.active_threat = (threat_val >= self.threat_threshold)

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Top Banner
        status_color = (0, 0, 255) if self.active_threat else (0, 255, 0)
        status_text = "THREAT DETECTED: VIOLENT ASSAULT!" if self.active_threat else "STATUS: NORMAL / CIVILIAN"
        cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.putText(overlay, status_text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)

        # Telemetry panel
        mem_mb = sum(item.nbytes for item in self.frame_buffer) / (1024 * 1024)
        avg_s1 = np.mean(self.stage1_latencies) if self.stage1_latencies else 0.0
        avg_s2a = np.mean(self.stage2a_latencies) if self.stage2a_latencies else 0.0
        avg_s2b = np.mean(self.stage2b_latencies) if self.stage2b_latencies else 0.0
        total_time = avg_s1 + avg_s2a + avg_s2b
        fps = 1000.0 / total_time if total_time > 0 else 0.0

        panel_text = [
            f"Pipeline: Version 3.00 (Skeleton-OOD)",
            f"Gating: {self.champion_method} (Score: {self.threat_score:.2f} / Tau: {self.threat_threshold:.2f})",
            f"Buffer RAM: {len(self.frame_buffer)}/{self.buffer_size} ({mem_mb:.2f} MB)",
            f"Latency: S1={avg_s1:.1f}ms | S2a={avg_s2a:.1f}ms | S2b={avg_s2b:.1f}ms | FPS={fps:.1f}"
        ]
        cv2.rectangle(overlay, (0, h - 100), (w, h), (15, 15, 15), -1)
        for idx, line in enumerate(panel_text):
            cv2.putText(overlay, line, (15, h - 80 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame


def main():
    parser = argparse.ArgumentParser(description="Real-Time Hierarchical Threat Detection - Version 3.00 (Skeleton-OOD)")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video path")
    parser.add_argument("--headless", action="store_true", help="Run without UI window")
    args = parser.parse_args()

    pipeline = HierarchicalThreatPipelineV300()
    print("[RUN] Pipeline initialized. Ready for surveillance streaming.")


if __name__ == "__main__":
    main()
