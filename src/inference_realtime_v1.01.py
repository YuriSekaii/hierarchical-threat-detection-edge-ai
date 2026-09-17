"""
Hierarchical Real-Time Threat Detection System - Version 1.01
Dual-Stage Surveillance Architecture with In-Memory JPEG Buffer Compression (Option A)

Changes in Version 1.01:
- Implemented in-memory JPEG frame compression in FrameItem (cv2.imencode Quality=85).
- Implemented lazy on-demand frame decompression in FrameItem.get_frame() (cv2.imdecode).
- Slashes circular buffer memory footprint from ~1.1 GB down to ~45 MB (~24x reduction).
- Added dynamic buffer memory telemetry (MB) to HUD overlay and status monitor.
- Added dual input source parsing (accepts webcam index or video file paths) and optional headless mode.
- Preserved baseline file intact (inference_realtime.py v1.00).
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stgcn import STGCNModel
from src.skeleton_utils import interpolate_missing_joints, smooth_kinematics, normalize_skeleton_clip
from src.ood_metrics import compute_knn_distance


class FrameItem:
    """Compressed Frame Storage in Circular Buffer (v1.01).
    Stores SIMD JPEG-compressed bytes instead of raw 3-channel uncompressed NumPy arrays.
    """
    def __init__(self, frame, timestamp, index, jpeg_quality=85):
        self.timestamp = timestamp
        self.index = index
        self.skeleton = None

        if frame is not None:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            success, enc_bytes = cv2.imencode('.jpg', frame, encode_params)
            if success:
                self.jpeg_data = enc_bytes
                self.nbytes = enc_bytes.nbytes
            else:
                self.jpeg_data = None
                self.nbytes = 0
        else:
            self.jpeg_data = None
            self.nbytes = 0

    def get_frame(self):
        """On-demand lazy decompression into standard BGR NumPy array."""
        if self.jpeg_data is not None:
            return cv2.imdecode(self.jpeg_data, cv2.IMREAD_COLOR)
        return None


class HierarchicalThreatDetector:
    def __init__(self, weapon_model_path, pose_model_path, action_model_path, ref_data_path,
                 device='cuda', jpeg_quality=85):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.jpeg_quality = jpeg_quality
        print(f"[INIT] Initializing Hierarchical Detector (v1.01 - JPEG Quality={self.jpeg_quality}) on {self.device}...")

        # 1. Load Object & Pose Models
        print(f"  [1/4] Loading Distilled YOLO Weapon Detector: {os.path.basename(weapon_model_path)}")
        self.weapon_model = YOLO(weapon_model_path)

        print(f"  [2/4] Loading YOLO26s Pose Estimator: {os.path.basename(pose_model_path)}")
        self.pose_model = YOLO(pose_model_path)
        
        # 2. Load ST-GCN Action Model
        print(f"  [3/4] Loading ST-GCN Violence Model: {os.path.basename(action_model_path)}")
        self.action_model = STGCNModel(num_classes=1, in_channels=2, num_keypoints=17).to(self.device)
        ckpt = torch.load(action_model_path, map_location=self.device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        self.action_model.load_state_dict(state_dict, strict=False)
        self.action_model.eval()

        # 3. Load Deep k-NN Reference Bank
        print(f"  [4/4] Loading Deep k-NN Reference Weights: {os.path.basename(ref_data_path)}")
        ref_data = torch.load(ref_data_path, map_location='cpu', weights_only=False)
        self.knn_feature_bank = ref_data['feature_bank'].float().cpu()
        self.knn_threshold = float(ref_data['threshold'])
        self.knn_k = int(ref_data.get('k', 2))
        print(f"        -> Deep k-NN Initialized: k={self.knn_k}, threshold={self.knn_threshold:.4f}, bank={self.knn_feature_bank.shape}")

        # 4. Zero-Drop Buffer Architecture with In-Memory JPEG Compression (1200 frames ~ 40s at 30 FPS)
        self.buffer_size = 1200
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        self.is_running = False
        self.frame_counter = 0

        # State flags & Two-Tier Alert System
        self.weapon_active_until = 0.0
        self.last_analyzed_idx = 0
        self.clip_stride = 10  # 10-frame slide = 80% overlap between consecutive ST-GCN evaluations
        self.clip_length = 50
        self.status_text = "SCANNING (IDLE)"
        self.status_color = (0, 255, 0)  # Green
        self.is_violent_alert = False
        self.alert_until = 0.0
        self.pending_audit_frames = 0
        self.buffer_ram_mb = 0.0

    def weapon_scanner_loop(self):
        """Thread 2: 3 FPS lightweight gating loop (~90% GPU duty reduction).
        Decodes the latest frame from buffer on-demand for Tier 1 weapon detection.
        """
        target_delay = 1.0 / 3.0
        while self.is_running:
            start_t = time.time()
            frame_copy = None

            with self.buffer_lock:
                if len(self.frame_buffer) > 0:
                    frame_copy = self.frame_buffer[-1].get_frame()

            if frame_copy is not None:
                # Production calibrated threshold: Conf = 0.45
                results = self.weapon_model(frame_copy, conf=0.45, verbose=False)
                has_weapon = len(results[0].boxes) > 0

                if has_weapon:
                    # Hold threat monitoring window for 3.0 seconds after last weapon sighting
                    self.weapon_active_until = time.time() + 3.0
                    if not self.is_violent_alert:
                        self.status_text = "WEAPON DETECTED"
                        self.status_color = (0, 165, 255)  # Orange

            elapsed = time.time() - start_t
            sleep_time = max(0.01, target_delay - elapsed)
            time.sleep(sleep_time)

    def action_recognition_loop(self):
        """Thread 3: Zero-Drop gapless sliding-window kinematic evaluator.
        Audits 100% of contiguous frames during threat events, caching extracted poses.
        """
        while self.is_running:
            now = time.time()
            is_active = (now < self.weapon_active_until)

            if not is_active:
                if self.is_violent_alert and now > self.alert_until:
                    self.is_violent_alert = False
                    self.status_text = "SCANNING (IDLE)"
                    self.status_color = (0, 255, 0)
                elif not self.is_violent_alert:
                    self.status_text = "SCANNING (IDLE)"
                    self.status_color = (0, 255, 0)
                self.pending_audit_frames = 0
                time.sleep(0.05)
                continue

            # Extract next contiguous window from 1200-frame buffer
            window = []
            with self.buffer_lock:
                if len(self.frame_buffer) >= self.clip_length:
                    buf_list = list(self.frame_buffer)
                    min_idx = buf_list[0].index
                    max_idx = buf_list[-1].index

                    # Keep pointer within available memory window
                    if self.last_analyzed_idx < min_idx:
                        self.last_analyzed_idx = min_idx

                    # Check if we have 50 frames starting from last_analyzed_idx
                    target_start = self.last_analyzed_idx
                    target_end = target_start + self.clip_length

                    if target_end <= max_idx + 1:
                        start_offset = target_start - min_idx
                        window = buf_list[start_offset : start_offset + self.clip_length]
                        self.last_analyzed_idx += self.clip_stride
                        self.pending_audit_frames = max(0, max_idx - target_end)

            if len(window) < self.clip_length:
                # Buffer has caught up with live webcam, wait briefly for new frames
                time.sleep(0.02)
                continue

            # Step 1: Extract Skeletons (On-demand decode for new stride frames)
            for item in window:
                if item.skeleton is None:
                    raw_frame = item.get_frame()
                    kpts = np.zeros((17, 2), dtype=np.float32)
                    if raw_frame is not None:
                        res = self.pose_model(raw_frame, verbose=False)
                        if hasattr(res[0], 'keypoints') and res[0].keypoints is not None:
                            if len(res[0].keypoints.data) > 0:
                                kpts = res[0].keypoints.data[0].cpu().numpy()[:, :2]
                    item.skeleton = kpts

            # Step 2: Temporal smoothing and normalization
            raw_seq = np.array([item.skeleton for item in window])  # (50, 17, 2)
            cleaned = interpolate_missing_joints(raw_seq)
            smoothed = smooth_kinematics(cleaned, sigma=1.0)
            normalized = normalize_skeleton_clip(smoothed)

            # Step 3: ST-GCN Feature Extraction
            clip_tensor = torch.from_numpy(normalized).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                _, features = self.action_model(clip_tensor, return_features=True)

            # Step 4: Out-of-Distribution Threat Discrimination (Deep k-NN)
            feats = features.cpu()
            dist = compute_knn_distance(feats, self.knn_feature_bank, k=self.knn_k, mode='kth')
            is_violence = (dist <= self.knn_threshold)
            detail_str = f"kNN Dist: {dist:.3f} (Thresh: {self.knn_threshold:.3f})"

            if is_violence:
                self.is_violent_alert = True
                self.alert_until = time.time() + 5.0  # Hold alert display for 5s
                self.status_text = f"VIOLENCE DETECTED! ({detail_str})"
                self.status_color = (0, 0, 255)  # Red
                print(f"[ACTION] !!! VIOLENCE DETECTED ({detail_str}) !!!")
            else:
                if not self.is_violent_alert:
                    self.status_text = "WEAPON DETECTED"
                    self.status_color = (0, 165, 255)  # Orange
                    print(f"[ACTION] Non-violent motion verified ({detail_str})")

    def run(self, source=0, headless=False):
        """Main thread: Ingestion loop, buffer telemetry, and UI display."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {source}")
            return

        self.is_running = True

        # Launch background threads
        t_weapon = threading.Thread(target=self.weapon_scanner_loop, daemon=True)
        t_action = threading.Thread(target=self.action_recognition_loop, daemon=True)
        t_weapon.start()
        t_action.start()

        print("[RUNNING] Threat detection system v1.01 started. Press 'q' to exit.")

        total_bytes = 0
        frame_idx = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # Buffer frame into 1200-frame circular buffer using In-Memory JPEG Compression
            new_item = FrameItem(frame, time.time(), self.frame_counter, jpeg_quality=self.jpeg_quality)
            with self.buffer_lock:
                if len(self.frame_buffer) == self.buffer_size:
                    # Pop oldest item's bytes from running estimate
                    old_item = self.frame_buffer[0]
                    total_bytes -= getattr(old_item, 'nbytes', 0)
                self.frame_buffer.append(new_item)
                total_bytes += new_item.nbytes
                self.frame_counter += 1

            frame_idx += 1
            if frame_idx % 30 == 0:
                self.buffer_ram_mb = total_bytes / (1024.0 * 1024.0)

            if not headless:
                # Render UI HUD overlay
                cv2.rectangle(frame, (10, 10), (740, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"STATUS: {self.status_text}", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.status_color, 2)
                
                # Show buffer & memory telemetry
                buf_len = len(self.frame_buffer)
                telemetry = f"Buffer: {buf_len}/1200 (~{self.buffer_ram_mb:.1f} MB) | Backlog: {self.pending_audit_frames} frames"
                cv2.putText(frame, telemetry, (20, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                cv2.imshow("Hierarchical Threat Detection (v1.01 Edge AI)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.is_running = False
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def parse_source(src_str):
    """Parse source string to int (device index) or path."""
    try:
        return int(src_str)
    except ValueError:
        return src_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical Edge-AI Threat & Violence Detection (v1.01)")
    parser.add_argument('--source', type=str, default="0",
                        help='Webcam ID (0) or video file path (default: 0)')
    parser.add_argument('--camera-id', type=int, default=None,
                        help='Legacy camera ID argument (overrides --source if provided)')
    parser.add_argument('--quality', type=int, default=85,
                        help='In-memory JPEG buffer compression quality (default: 85)')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode without cv2.imshow GUI display')
    args = parser.parse_args()

    active_source = args.camera_id if args.camera_id is not None else parse_source(args.source)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weapon_weights = os.path.join(base_dir, 'weights', 'yolo_weapon_distilled.pt')
    pose_weights = os.path.join(base_dir, 'weights', 'yolo26s-pose.pt')
    action_weights = os.path.join(base_dir, 'weights', 'stgcn_violence_fold2.pth')
    ref_weights = os.path.join(base_dir, 'weights', 'reference_data_knn.pt')

    detector = HierarchicalThreatDetector(
        weapon_model_path=weapon_weights,
        pose_model_path=pose_weights,
        action_model_path=action_weights,
        ref_data_path=ref_weights,
        jpeg_quality=args.quality
    )
    detector.run(source=active_source, headless=args.headless)
