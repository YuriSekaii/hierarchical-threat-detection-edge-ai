"""
Hierarchical Real-Time Threat Detection System.
Multi-threaded architecture:
- Thread 1: Ingests 30 FPS video into a 60-frame thread-safe deque.
- Thread 2 (Gate): 3 FPS YOLO weapon scanning (~90% compute reduction).
- Thread 3 (Action): Triggered on-demand ST-GCN + Pose + OOD classification.

Supports dual OOD modes:
1. Deep k-NN (Default, exact mode tested in Real_Time.py with reference_data_knn.pt)
2. Mahalanobis Distance (Mode evaluated in Verify_RealTime_Logic.py & diagrammed in Workflow.md)
"""

import os
import sys
import time
import json
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
from src.ood_metrics import compute_knn_distance, compute_mahalanobis_distance


class FrameItem:
    def __init__(self, frame, timestamp):
        self.frame = frame
        self.timestamp = timestamp
        self.skeleton = None


class HierarchicalThreatDetector:
    def __init__(self, weapon_model_path, pose_model_path, action_model_path, ref_data_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[INIT] Initializing Hierarchical Detector on {self.device}...")

        # 1. Load Object & Pose Models
        print(f"  [1/4] Loading YOLO P2 Weapon Detector: {os.path.basename(weapon_model_path)}")
        self.weapon_model = YOLO(weapon_model_path)

        print(f"  [2/4] Loading YOLOv8s Pose Estimator: {os.path.basename(pose_model_path)}")
        self.pose_model = YOLO(pose_model_path)
        
        # 2. Load ST-GCN Action Model
        print(f"  [3/4] Loading ST-GCN Violence Model: {os.path.basename(action_model_path)}")
        self.action_model = STGCNModel(num_classes=1, in_channels=2, num_keypoints=17).to(self.device)
        ckpt = torch.load(action_model_path, map_location=self.device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        self.action_model.load_state_dict(state_dict, strict=False)
        self.action_model.eval()

        # 3. Load OOD Reference Data (Auto-detects k-NN .pt or Mahalanobis .json)
        print(f"  [4/4] Loading OOD Reference Weights: {os.path.basename(ref_data_path)}")
        if ref_data_path.endswith('.pt'):
            # Deep k-NN Mode (Exact match to Real_Time.py)
            self.ood_mode = 'knn'
            ref_data = torch.load(ref_data_path, map_location='cpu', weights_only=False)
            self.knn_feature_bank = ref_data['feature_bank'].float().cpu()
            self.knn_threshold = float(ref_data['threshold'])
            self.knn_k = int(ref_data.get('k', 5))
            print(f"        -> Mode: Deep k-NN (k={self.knn_k}, threshold={self.knn_threshold:.4f}, bank={self.knn_feature_bank.shape})")
        elif ref_data_path.endswith('.json'):
            # Mahalanobis Distance Mode
            self.ood_mode = 'mahalanobis'
            with open(ref_data_path, 'r') as f:
                ref_json = json.load(f)
            self.maha_mean = np.array(ref_json['mean'], dtype=np.float32)
            self.maha_inv_cov = np.array(ref_json['inv_cov'], dtype=np.float32)
            self.maha_threshold = float(ref_json.get('threshold', 7.60))
            print(f"        -> Mode: Mahalanobis Distance (threshold={self.maha_threshold:.2f}, dim={len(self.maha_mean)})")
        else:
            raise ValueError(f"Unsupported reference data format: {ref_data_path}. Expected .pt or .json")

        # Threading buffers & State
        self.frame_buffer = deque(maxlen=60)
        self.buffer_lock = threading.Lock()
        self.is_running = False

        # State flags
        self.stage2_active = False
        self.status_text = "SCANNING (IDLE)"
        self.status_color = (0, 255, 0)  # Green

    def weapon_scanner_loop(self):
        """Thread 2: 3 FPS lightweight gating loop (~90% GPU duty reduction)."""
        target_delay = 1.0 / 3.0
        while self.is_running:
            start_t = time.time()
            frame_copy = None

            with self.buffer_lock:
                if len(self.frame_buffer) > 0:
                    frame_copy = self.frame_buffer[-1].frame.copy()

            if frame_copy is not None:
                # Run lightweight weapon detection
                results = self.weapon_model(frame_copy, conf=0.5, verbose=False)
                has_weapon = len(results[0].boxes) > 0

                if has_weapon:
                    self.stage2_active = True
                    self.status_text = "WEAPON DETECTED - ANALYZING MOTION..."
                    self.status_color = (0, 165, 255)  # Orange

            elapsed = time.time() - start_t
            sleep_time = max(0.01, target_delay - elapsed)
            time.sleep(sleep_time)

    def action_recognition_loop(self):
        """Thread 3: On-demand kinematic action evaluation with OOD gating."""
        while self.is_running:
            if not self.stage2_active:
                time.sleep(0.05)
                continue

            # Extract window from buffer (30 frames)
            window = []
            with self.buffer_lock:
                if len(self.frame_buffer) >= 30:
                    window = list(self.frame_buffer)[-30:]

            if len(window) < 30:
                time.sleep(0.05)
                continue

            # Step 1: Extract Skeletons for frames lacking pose
            for item in window:
                if item.skeleton is None:
                    res = self.pose_model(item.frame, verbose=False)
                    kpts = np.zeros((17, 2), dtype=np.float32)
                    if hasattr(res[0], 'keypoints') and res[0].keypoints is not None:
                        if len(res[0].keypoints.data) > 0:
                            kpts = res[0].keypoints.data[0].cpu().numpy()[:, :2]
                    item.skeleton = kpts

            # Step 2: Temporal smoothing and normalization
            raw_seq = np.array([item.skeleton for item in window])  # (T, 17, 2)
            cleaned = interpolate_missing_joints(raw_seq)
            smoothed = smooth_kinematics(cleaned, sigma=1.0)
            normalized = normalize_skeleton_clip(smoothed)

            # Step 3: ST-GCN Feature Extraction
            clip_tensor = torch.from_numpy(normalized).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                _, features = self.action_model(clip_tensor, return_features=True)

            # Step 4: Out-of-Distribution / Threat Discrimination
            if self.ood_mode == 'knn':
                feats = features.cpu()
                dist = compute_knn_distance(feats, self.knn_feature_bank, k=self.knn_k, mode='kth')
                is_violence = (dist <= self.knn_threshold)
                detail_str = f"kNN Dist: {dist:.3f} (Thresh: {self.knn_threshold:.3f})"
            else:
                feat_np = features[0].cpu().numpy()
                dist = compute_mahalanobis_distance(feat_np, self.maha_mean, self.maha_inv_cov)
                is_violence = (dist < self.maha_threshold)
                detail_str = f"Maha Dist: {dist:.2f} (Thresh: {self.maha_threshold:.2f})"

            if is_violence:
                self.status_text = f"ALERT: VIOLENCE CONFIRMED! [{detail_str}]"
                self.status_color = (0, 0, 255)  # Red
                print(f"[ACTION] !!! VIOLENCE CONFIRMED ({detail_str}) !!!")
            else:
                self.status_text = f"WEAPON SEEN, PASSIVE MOTION [{detail_str}]"
                self.status_color = (0, 165, 255)  # Orange
                print(f"[ACTION] False Alarm / Passive Motion Filtered ({detail_str})")

            # Reset gate after window completes
            time.sleep(0.5)
            self.stage2_active = False

    def run(self, source=0):
        """Main thread: Ingestion loop and UI display."""
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

        print("[RUNNING] Threat detection system started. Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Buffer frame
            with self.buffer_lock:
                self.frame_buffer.append(FrameItem(frame, time.time()))

            # Render UI overlay
            cv2.rectangle(frame, (10, 10), (620, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"STATUS: {self.status_text}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.status_color, 2)

            cv2.imshow("Hierarchical Threat Detection (Edge AI)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical Edge-AI Threat & Violence Detection")
    parser.add_argument('--camera-id', type=int, default=0, help='Webcam device ID (default: 0)')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weapon_weights = os.path.join(base_dir, 'weights', 'yolo_weapon_p2.pt')
    pose_weights = os.path.join(base_dir, 'weights', 'yolo26s-pose.pt')
    action_weights = os.path.join(base_dir, 'weights', 'stgcn_violence_fold2.pth')
    ref_weights = os.path.join(base_dir, 'weights', 'reference_data_knn.pt')

    detector = HierarchicalThreatDetector(
        weapon_model_path=weapon_weights,
        pose_model_path=pose_weights,
        action_model_path=action_weights,
        ref_data_path=ref_weights
    )
    detector.run(source=args.camera_id)
