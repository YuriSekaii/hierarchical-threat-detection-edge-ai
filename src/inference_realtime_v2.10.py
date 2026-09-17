"""
Hierarchical Real-Time Threat Detection System - Version 2.10
Dual-Stage Surveillance Architecture with:
1. In-Memory JPEG Buffer Compression (v1.01)
2. Torso-Scale Invariant Kinematic Normalization (v1.02)
3. Native Aspect-Ratio Rectangular Inference (v1.03)
4. Multi-Person Tracking Guard with ByteTrack & Spatial-Temporal Continuity (v1.04)
5. PoseConv3D Volumetric 3D Heatmap CNN Backbone (v2.10)
6. Multi-Person Channel-Wise Heatmap Stacking without Graph Topological Collapse
7. Manifold-Aware Relative Mahalanobis Distance (RMD) Threat Gating (v1.08 / v2.10)

Key Advances in Version 2.10:
- Converts 50-frame skeletal sequences to 3D heatmap volumes (17 x 50 x 56 x 56).
- R(2+1)D factorized spatio-temporal residual architecture running at ~3.5 ms on RTX 3090.
- Overcoming graph topology collapse: multi-actor interactions naturally accumulate via channel-wise maximum.
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.poseconv3d import PoseConv3DModel
from src.poseconv3d_utils import rasterize_keypoints_to_heatmap

# Import v1.02 scale-normalized skeleton utils for preprocessing
import importlib.util
skel_spec = importlib.util.spec_from_file_location(
    "skeleton_utils_v1_02",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "skeleton_utils_v1.02.py"))
)
skel_mod = importlib.util.module_from_spec(skel_spec)
skel_spec.loader.exec_module(skel_mod)
interpolate_missing_joints = skel_mod.interpolate_missing_joints
smooth_kinematics = skel_mod.smooth_kinematics

# Import v1.08 RMD metrics
rmd_spec = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "ood_metrics_v1.08.py"))
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
        self.all_actors_skeletons = []

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


class HierarchicalThreatPipelineV210:
    def __init__(
        self,
        yolo_world_weights="weights/yolov8x-worldv2.pt",
        yolo_pose_weights="weights/yolo26s-pose.pt",
        poseconv3d_weights="weights/poseconv3d_violence_v2.10.pth",
        reference_weights="weights/reference_data_poseconv3d_v2.10.pt",
        threat_threshold=None,
        buffer_size=1200,
        jpeg_quality=85,
        device=None,
        imgsz=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Initializing Hierarchical Pipeline v2.10 (PoseConv3D) on device: {self.device}")

        # 1. Stage 1 Weapon Detector (YOLO-World v2)
        print(f"[INIT] Loading Stage 1 Detector: {yolo_world_weights}")
        self.weapon_model = YOLO(yolo_world_weights)
        self.weapon_model.set_classes(["knife", "machete", "gun", "dagger", "sword", "weapon", "blade"])

        # 2. Stage 2a Pose Extractor (YOLO26s-pose)
        print(f"[INIT] Loading Stage 2a Pose Extractor: {yolo_pose_weights}")
        self.pose_model = YOLO(yolo_pose_weights)

        # 3. Stage 2b Action Recognition Model (PoseConv3D v2.10)
        print(f"[INIT] Loading Stage 2b PoseConv3D Volumetric Backbone: {poseconv3d_weights}")
        self.action_model = PoseConv3DModel(num_classes=1).to(self.device)
        if os.path.exists(poseconv3d_weights):
            self.action_model.load_state_dict(torch.load(poseconv3d_weights, map_location=self.device, weights_only=True))
        self.action_model.eval()

        # 4. Calibrated Manifold Reference Bank
        self.has_ref_data = False
        if os.path.exists(reference_weights):
            print(f"[INIT] Loading v2.10 Calibrated Reference Bank: {reference_weights}")
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
            self.threat_threshold = threat_threshold if threat_threshold is not None else float(ref_data.get("optimal_threshold", -0.9))
            self.classes = ref_data.get("classes", ["Cut-Down", "Stab", "Thrust"])
            self.has_ref_data = True
            print(f"[INIT] Calibrated Parameters Loaded (Classes: {len(self.classes)}, Threshold: {self.threat_threshold:.4f})")
        else:
            self.threat_threshold = threat_threshold if threat_threshold is not None else 0.5
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
        self.status_text = "System Initialized - Standby (PoseConv3D v2.10)"
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
            return None, None, None, []

        boxes = pose_result.boxes.xyxy.cpu().numpy()
        keypoints = pose_result.keypoints.data.cpu().numpy()
        track_ids = pose_result.boxes.id.int().cpu().tolist() if pose_result.boxes.id is not None else None

        all_actors = [keypoints[i][:, :2] for i in range(len(boxes))]
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

        # 3. Proximity to Last Known Actor Bounding Box
        if chosen_idx is None and self.last_known_actor_bbox is not None:
            lx1, ly1, lx2, ly2 = self.last_known_actor_bbox
            l_cx, l_cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            min_dist = float("inf")
            for idx, (bx1, by1, bx2, by2) in enumerate(boxes):
                b_cx, b_cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = np.hypot(b_cx - l_cx, b_cy - l_cy)
                if dist < min_dist:
                    min_dist = dist
                    chosen_idx = idx

        # 4. Fallback: Largest Bounding Box Area
        if chosen_idx is None:
            areas = [(bx2 - bx1) * (by2 - by1) for bx1, by1, bx2, by2 in boxes]
            chosen_idx = int(np.argmax(areas))

        selected_bbox = boxes[chosen_idx]
        selected_kpts = keypoints[chosen_idx][:, :2]
        selected_tid = track_ids[chosen_idx] if track_ids is not None else None

        if selected_tid is not None:
            self.active_threat_actor_id = selected_tid
        self.last_known_actor_bbox = selected_bbox
        self.last_known_keypoints = selected_kpts

        return selected_kpts, selected_bbox, selected_tid, all_actors

    def _weapon_scanner_worker(self):
        while self.running:
            start_time = time.time()
            with self.lock:
                if len(self.buffer) == 0:
                    time.sleep(0.01)
                    continue
                latest_item = self.buffer[-1]
                frame = latest_item.get_frame()

            if frame is not None:
                if self.native_imgsz is None:
                    h_f, w_f = frame.shape[:2]
                    self.native_imgsz = self._determine_rectangular_imgsz(w_f, h_f)

                results = self.weapon_model(
                    frame,
                    imgsz=self.native_imgsz,
                    conf=0.15,
                    verbose=False,
                    device=self.device
                )

                found_weapon = False
                for r in results:
                    if len(r.boxes) > 0:
                        found_weapon = True
                        box = r.boxes.xyxy[0].cpu().numpy()
                        self.active_weapon_box = box
                        break

                if found_weapon:
                    if not self.threat_active:
                        print(f"[SECURITY ALERT] Stage 1 Weapon Detected at Frame {latest_item.index}! Triggering PoseConv3D.")
                        self.threat_active = True
                    self.threat_event.set()
                else:
                    if self.threat_active:
                        if time.time() > self.alert_until:
                            self.threat_active = False
                            self.threat_event.clear()
                            self.active_weapon_box = None
                            self.active_threat_actor_id = None
                            self.last_known_actor_bbox = None
                            self.last_known_keypoints = None
                            print("[SECURITY] Weapon Cleared. Returning to Standby Surveillance Mode.")

            elapsed = time.time() - start_time
            sleep_time = max(0.0, 0.333 - elapsed)
            time.sleep(sleep_time)

    def _action_worker(self):
        while self.running:
            self.threat_event.wait()
            if not self.running:
                break
            self._process_threat_event()

    def _process_threat_event(self):
        with self.lock:
            if len(self.buffer) < 50:
                time.sleep(0.01)
                return
            window = list(self.buffer)[-50:]

        # Step 1: Sequential Pose Tracking via ByteTrack Guard (v1.04+)
        for item in window:
            if item.skeleton is None:
                frm = item.get_frame()
                if frm is not None:
                    res = self.pose_model.track(
                        frm,
                        imgsz=self.native_imgsz,
                        persist=True,
                        conf=0.25,
                        verbose=False,
                        device=self.device
                    )
                    kpts, bbox, t_id, all_acts = self._select_actor_keypoints(res[0])
                    item.skeleton = kpts if kpts is not None else np.zeros((17, 2), dtype=np.float32)
                    item.actor_bbox = bbox
                    item.track_id = t_id
                    item.all_actors_skeletons = all_acts
                else:
                    item.skeleton = np.zeros((17, 2), dtype=np.float32)

        # Step 2: Temporal smoothing
        raw_seq = np.array([item.skeleton for item in window])
        cleaned = interpolate_missing_joints(raw_seq)
        smoothed = smooth_kinematics(cleaned, sigma=1.0)

        # Step 3: Volumetric 3D Heatmap Rasterization (v2.10)
        heatmap_tensor = rasterize_keypoints_to_heatmap(
            smoothed, H=56, W=56, sigma=1.5, margin=0.15, device=self.device
        ).unsqueeze(0) # (1, 17, 50, 56, 56)

        # Step 4: PoseConv3D Volumetric Inference (v2.10)
        with torch.no_grad():
            logits, features = self.action_model(heatmap_tensor, return_features=True)

        # Step 5: Threat Gating
        if self.has_ref_data and self.rmd_mu_0 is not None:
            rmd_score, min_mc, m0 = compute_relative_mahalanobis_distance(
                features, self.rmd_mu_0, self.rmd_mu_c, self.rmd_inv_Sigma_0, self.rmd_inv_Sigma_c
            )
            if isinstance(rmd_score, torch.Tensor):
                rmd_score = rmd_score.item()
            is_violence = (rmd_score >= self.threat_threshold)
            detail_str = f"PoseConv3D Score: {rmd_score:+.2f} (Thresh: {self.threat_threshold:+.2f})"
        else:
            prob = torch.sigmoid(logits).item()
            is_violence = (prob >= self.threat_threshold)
            detail_str = f"PoseConv3D Conf: {prob:.2f} (Thresh: {self.threat_threshold:.2f})"

        actor_tag = f"Actor ID: {self.active_threat_actor_id}" if self.active_threat_actor_id else "Actor Locked"

        if is_violence:
            self.is_violent_alert = True
            self.alert_until = time.time() + 5.0
            self.status_text = f"VIOLENCE DETECTED! ({actor_tag} | {detail_str})"
            self.status_color = (0, 0, 255)
            print(f"[ACTION] !!! VIOLENCE DETECTED ({actor_tag} | {detail_str}) !!!")
        else:
            if not self.is_violent_alert:
                self.status_text = f"WEAPON DETECTED ({actor_tag})"
                self.status_color = (0, 165, 255)
                print(f"[ACTION] Non-violent motion verified ({actor_tag} | {detail_str})")

    def ingest_frame(self, frame):
        self.frame_idx += 1
        item = FrameItem(frame, time.time(), self.frame_idx, self.jpeg_quality)
        with self.lock:
            self.buffer.append(item)

    def start(self):
        self.running = True
        self.weapon_thread.start()
        self.action_thread.start()
        print("[PIPELINE] Surveillance Threads Successfully Started.")

    def stop(self):
        self.running = False
        self.threat_event.set()
        if self.weapon_thread.is_alive():
            self.weapon_thread.join(timeout=1.0)
        if self.action_thread.is_alive():
            self.action_thread.join(timeout=1.0)
        print("[PIPELINE] System Stopped Cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Threat Detection v2.10 (PoseConv3D)")
    parser.add_argument("--source", type=str, default=None, help="Video file or camera index")
    args = parser.parse_args()

    pipeline = HierarchicalThreatPipelineV210()
    pipeline.start()

    if args.source is not None:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pipeline.ingest_frame(frame)
            time.sleep(0.033)
        cap.release()

    pipeline.stop()
