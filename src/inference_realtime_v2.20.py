"""
Hierarchical Real-Time Threat Detection System - Version 2.20
Dual-Stage Surveillance Architecture with:
1. In-Memory JPEG Buffer Compression (v1.01)
2. Torso-Scale Invariant Kinematic Normalization (v1.02)
3. Native Aspect-Ratio Rectangular Inference (v1.03)
4. Multi-Person Tracking Guard with ByteTrack & Spatial-Temporal Continuity (v1.04)
5. SkateFormer Partitioned Skeletal-Temporal Vision Transformer Backbone (v2.20)
6. Manifold-Aware Likelihood-Ratio Threat Gating (RMD / Relative Mahalanobis Distance)

Key Advances in Version 2.20:
- Coordinate-First Vision Transformer: Replaces heatmaps and graph convolutions with SkateFormer
  operating directly on normalized 2D coordinates, preserving sharp knife thrust kinematics.
- Skate-Embedding: Injects spatial joint positional embeddings and temporal position encodings.
- Partitioned Multi-Head Self-Attention (Skate-MSA): 4 interaction regimes (S_near/T_near,
  S_near/T_dist, S_dist/T_near, S_dist/T_dist) capturing fine micro-kinematics and macro confrontation.
- 447,205 Model Parameters: -85.2% vs ST-GCN (3.01M), -66.5% vs CTR-GCN (1.33M), -41.6% vs PoseConv3D (765k).
- Arm-Weighted Spatial Attention: 5x weighting on strike joints (shoulders, elbows, wrists).
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

from models.skateformer import SkateFormerModel
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
    """Compressed Frame Storage in Circular Buffer (v1.01+)."""
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


class HierarchicalThreatPipelineV220:
    def __init__(
        self,
        yolo_weapon_weights="weights/yolo_weapon_distilled.pt",
        yolo_pose_weights="weights/yolo26s-pose.pt",
        skateformer_weights="weights/skateformer_violence_v2.20.pth",
        reference_weights="weights/reference_data_skateformer_v2.20.pt",
        threat_threshold=None,
        buffer_size=1200,
        jpeg_quality=85,
        device=None,
        imgsz=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Initializing Hierarchical Pipeline v2.20 (SkateFormer) on device: {self.device}")

        # 1. Stage 1 Weapon Detector (Distilled YOLO / YOLO-World)
        print(f"[INIT] Loading Stage 1 Detector: {yolo_weapon_weights}")
        self.weapon_model = YOLO(yolo_weapon_weights)
        try:
            if hasattr(self.weapon_model, "set_classes") and "world" in yolo_weapon_weights.lower():
                self.weapon_model.set_classes(["knife", "machete", "gun", "dagger", "sword", "weapon", "blade"])
        except Exception as e:
            print(f"[WARN] set_classes skipped: {e}")

        # 2. Stage 2a Pose Extractor (YOLO26s-pose)
        print(f"[INIT] Loading Stage 2a Pose Extractor: {yolo_pose_weights}")
        self.pose_model = YOLO(yolo_pose_weights)

        # 3. Stage 2b Action Recognition Model (SkateFormer v2.20)
        print(f"[INIT] Loading Stage 2b SkateFormer Backbone: {skateformer_weights}")
        self.action_model = SkateFormerModel(
            num_classes=1, in_channels=2, num_frames=50, num_keypoints=17, embed_dim=128, num_blocks=4, dropout=0.1
        ).to(self.device)
        if os.path.exists(skateformer_weights):
            self.action_model.load_state_dict(torch.load(skateformer_weights, map_location=self.device, weights_only=True))
        self.action_model.eval()

        # 4. Calibrated Manifold Reference Bank (RMD)
        self.has_ref_data = False
        if os.path.exists(reference_weights):
            print(f"[INIT] Loading v2.20 Calibrated Reference Bank: {reference_weights}")
            ref_data = torch.load(reference_weights, map_location=self.device, weights_only=False)
            self.rmd_mu_0 = ref_data.get("mu_0", None)
            if self.rmd_mu_0 is not None:
                self.rmd_mu_0 = self.rmd_mu_0.to(self.device)
            self.rmd_mu_c = ref_data.get("mu_c", None)
            if self.rmd_mu_c is not None:
                self.rmd_mu_c = self.rmd_mu_c.to(self.device)
            self.rmd_inv_Sigma_0 = ref_data.get("inv_Sigma_0", None)
            if self.rmd_inv_Sigma_0 is not None:
                self.rmd_inv_Sigma_0 = self.rmd_inv_Sigma_0.to(self.device)
            self.rmd_inv_Sigma_c = ref_data.get("inv_Sigma_c", None)
            if self.rmd_inv_Sigma_c is not None:
                self.rmd_inv_Sigma_c = self.rmd_inv_Sigma_c.to(self.device)
            self.threat_threshold = threat_threshold if threat_threshold is not None else float(ref_data.get("optimal_threshold", 0.0))
            self.has_ref_data = True
            print(f"[INIT] Calibrated Parameters Loaded (Threshold: {self.threat_threshold:.4f})")
        else:
            self.threat_threshold = threat_threshold if threat_threshold is not None else 0.0
            print(f"[INIT] Reference file {reference_weights} not found yet; will use threshold {self.threat_threshold}")

        # Buffer and Concurrency Settings
        self.buffer_size = buffer_size
        self.jpeg_quality = jpeg_quality
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

        # Multi-Person Tracking Guard state (v1.04+)
        self.active_threat_actor_id = None
        self.last_known_actor_bbox = None
        self.last_known_keypoints = None

        # System State
        self.threat_active = False
        self.threat_event = threading.Event()
        self.running = False
        self.frame_idx = 0
        self.active_weapon_box = None
        self.status_text = "System Initialized - Standby (SkateFormer v2.20)"
        self.status_color = (0, 255, 0)
        self.is_violent_alert = False
        self.alert_until = 0.0
        self.current_fps = 0.0

        # Aspect-Ratio Aware Resolution
        self.configured_imgsz = imgsz
        self.native_imgsz = None

        # Threads
        self.weapon_thread = threading.Thread(target=self._weapon_scanner_worker, daemon=True)
        self.action_thread = threading.Thread(target=self._action_worker, daemon=True)

    def _determine_rectangular_imgsz(self, frame_w, frame_h, stride=32):
        if self.configured_imgsz is not None:
            return self.configured_imgsz
        long_side = 640
        if frame_w >= frame_h:
            w = long_side
            h = int(round((long_side * frame_h / frame_w) / stride) * stride)
        else:
            h = long_side
            w = int(round((long_side * frame_w / frame_h) / stride) * stride)
        return (h, w)

    def _select_actor_keypoints(self, pose_result):
        """ByteTrack Tracking Guard (v1.04+)."""
        if pose_result.boxes is None or len(pose_result.boxes) == 0:
            return None, None, None

        boxes = pose_result.boxes.xyxy.cpu().numpy()
        keypoints = pose_result.keypoints.data.cpu().numpy()
        track_ids = pose_result.boxes.id.int().cpu().tolist() if pose_result.boxes.id is not None else None

        chosen_idx = None

        # 1. Continuity Match by Track ID
        if self.active_threat_actor_id is not None and track_ids is not None:
            for idx, tid in enumerate(track_ids):
                if tid == self.active_threat_actor_id:
                    chosen_idx = idx
                    break

        # 2. Spatial Overlap (IoU) with Weapon Box
        if chosen_idx is None and self.active_weapon_box is not None:
            wx1, wy1, wx2, wy2 = self.active_weapon_box
            best_iou = -1.0
            for idx, (bx1, by1, bx2, by2) in enumerate(boxes):
                ix1, iy1 = max(wx1, bx1), max(wy1, by1)
                ix2, iy2 = min(wx2, bx2), min(wy2, by2)
                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter_area = inter_w * inter_h
                w_area = (wx2 - wx1) * (wy2 - wy1)
                b_area = (bx2 - bx1) * (by2 - by1)
                union_area = w_area + b_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    chosen_idx = idx

        # 3. Spatial Proximity to Last Known BBox
        if chosen_idx is None and self.last_known_actor_bbox is not None:
            lx1, ly1, lx2, ly2 = self.last_known_actor_bbox
            l_cx, l_cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            min_dist = float("inf")
            for idx, (bx1, by1, bx2, by2) in enumerate(boxes):
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = np.hypot(cx - l_cx, cy - l_cy)
                if dist < min_dist:
                    min_dist = dist
                    chosen_idx = idx

        if chosen_idx is None:
            chosen_idx = 0

        chosen_tid = track_ids[chosen_idx] if track_ids is not None else None
        chosen_bbox = boxes[chosen_idx]
        chosen_kps = keypoints[chosen_idx]

        self.active_threat_actor_id = chosen_tid
        self.last_known_actor_bbox = chosen_bbox
        self.last_known_keypoints = chosen_kps

        return chosen_kps, chosen_bbox, chosen_tid

    def _weapon_scanner_worker(self):
        while self.running:
            with self.lock:
                if len(self.buffer) == 0:
                    time.sleep(0.01)
                    continue
                latest_item = self.buffer[-1]

            frame = latest_item.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            if self.native_imgsz is None:
                self.native_imgsz = self._determine_rectangular_imgsz(w, h)

            results = self.weapon_model.predict(
                frame,
                conf=0.25,
                verbose=False,
                device=self.device,
                imgsz=self.native_imgsz
            )

            weapon_detected = False
            detected_box = None
            if len(results) > 0 and len(results[0].boxes) > 0:
                weapon_detected = True
                detected_box = results[0].boxes.xyxy[0].cpu().numpy()

            if weapon_detected:
                self.active_weapon_box = detected_box
                if not self.threat_active:
                    print("[STAGE 1] Weapon detected! Triggering Threat Active Mode (SkateFormer v2.20)...")
                    self.threat_active = True
                    self.threat_event.set()
            else:
                self.active_weapon_box = None

            time.sleep(0.33)

    def _action_worker(self):
        clip_length = 50
        stride = 10

        while self.running:
            self.threat_event.wait()
            if not self.running:
                break

            with self.lock:
                current_len = len(self.buffer)

            if current_len < clip_length:
                time.sleep(0.05)
                continue

            with self.lock:
                recent_items = [self.buffer[i] for i in range(current_len - clip_length, current_len)]

            skeletons = []
            for item in recent_items:
                if item.skeleton is None:
                    fr = item.get_frame()
                    if fr is not None:
                        h, w = fr.shape[:2]
                        if self.native_imgsz is None:
                            self.native_imgsz = self._determine_rectangular_imgsz(w, h)

                        p_res = self.pose_model.track(
                            fr,
                            persist=True,
                            verbose=False,
                            device=self.device,
                            imgsz=self.native_imgsz
                        )
                        if len(p_res) > 0:
                            kps, bbox, tid = self._select_actor_keypoints(p_res[0])
                            item.skeleton = kps
                            item.actor_bbox = bbox
                            item.track_id = tid
                        else:
                            item.skeleton = np.zeros((17, 3), dtype=np.float32)
                    else:
                        item.skeleton = np.zeros((17, 3), dtype=np.float32)

                if item.skeleton is not None and item.skeleton.shape[0] == 17:
                    skeletons.append(item.skeleton[:, :2])
                else:
                    skeletons.append(np.zeros((17, 2), dtype=np.float32))

            skeletons_arr = np.array(skeletons, dtype=np.float32) # (50, 17, 2)
            cleaned = interpolate_missing_joints(skeletons_arr)
            smoothed = smooth_kinematics(cleaned, sigma=1.0)
            normed = normalize_skeleton_clip(smoothed) # Torso-scale normalization (v1.02)

            # Convert to Tensor (1, 2, 50, 17, 1)
            inp_tensor = torch.from_numpy(normed).permute(2, 0, 1).unsqueeze(-1).unsqueeze(0).to(self.device).float()

            with torch.no_grad():
                logits, feats = self.action_model(inp_tensor, return_features=True)
                prob = torch.sigmoid(logits).item()

                if self.has_ref_data and self.rmd_mu_0 is not None:
                    score, _, _ = compute_relative_mahalanobis_distance(
                        feats, self.rmd_mu_0, self.rmd_mu_c, self.rmd_inv_Sigma_0, self.rmd_inv_Sigma_c
                    )
                    threat_score = score.item()
                    is_violent = threat_score >= self.threat_threshold
                else:
                    threat_score = prob
                    is_violent = prob >= self.threat_threshold

            if is_violent:
                self.is_violent_alert = True
                self.alert_until = time.time() + 3.0
                self.status_text = f"CRITICAL: VIOLENCE DETECTED (SkateFormer RMD={threat_score:.2f}, Prob={prob:.2f})"
                self.status_color = (0, 0, 255)
            else:
                self.status_text = f"THREAT ACTIVE: Action Non-Violent / OOD (RMD={threat_score:.2f}, Prob={prob:.2f})"
                self.status_color = (0, 165, 255)

            time.sleep(stride / 30.0)

    def push_frame(self, frame):
        self.frame_idx += 1
        item = FrameItem(frame, time.time(), self.frame_idx, self.jpeg_quality)
        with self.lock:
            self.buffer.append(item)

    def start(self):
        self.running = True
        self.weapon_thread.start()
        self.action_thread.start()

    def stop(self):
        self.running = False
        self.threat_event.set()
        if self.weapon_thread.is_alive():
            self.weapon_thread.join(timeout=1.0)
        if self.action_thread.is_alive():
            self.action_thread.join(timeout=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Threat Detection Pipeline (SkateFormer v2.20)")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or video file path)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    args = parser.parse_args()

    pipeline = HierarchicalThreatPipelineV220()
    print("[RUN] System initialized successfully.")
