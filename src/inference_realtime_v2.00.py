"""
Hierarchical Real-Time Threat Detection System - Version 2.00
Dual-Stage Surveillance Architecture with:
1. In-Memory JPEG Buffer Compression (v1.01)
2. Torso-Scale Invariant Kinematic Normalization (v1.02)
3. Native Aspect-Ratio Rectangular Inference (v1.03)
4. Multi-Person Tracking Guard with ByteTrack & Spatial-Temporal Continuity (v1.04)
5. Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN) Dynamic Backbone (v2.00)
6. Manifold-Aware Likelihood-Ratio Threat Gating (RMD / Hyperspherical Manifold)

Key Advances in Version 2.00:
- Dynamic Channel-Wise Topology Refinement (CTR-GC): Replaces static anatomical graph with
  sample-conditioned adaptive topology A_k^(c) = A_k + B_k + C_k^(c)(X).
- Multi-Scale Temporal Convolutions (MS-TCN): 4-branch multi-scale temporal receptive fields.
- 55.8% Model Parameter Reduction: 1.33 M params (vs 3.01 M in ST-GCN).
- Arm-Weighted Spatial Attention: 5x priority weighting on combat strike joints (shoulders, elbows, wrists).
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

from models.ctrgcn import CTRGCNModel

# Import v1.02 scale-normalized skeleton utils
import importlib.util
skel_spec = importlib.util.spec_from_file_location(
    "skeleton_utils_v1_02",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "skeleton_utils_v1.02.py"))
)
skel_mod = importlib.util.module_from_spec(skel_spec)
skel_spec.loader.exec_module(skel_mod)
interpolate_missing_joints = skel_mod.interpolate_missing_joints
smooth_kinematics = skel_mod.smooth_kinematics
normalize_skeleton_clip = skel_mod.normalize_skeleton_clip

# Import v1.08 RMD metrics
rmd_spec = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "ood_metrics_v1.08.py"))
)
rmd_mod = importlib.util.module_from_spec(rmd_spec)
rmd_spec.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance
is_threat_rmd = rmd_mod.is_threat_rmd


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
            if success:
                self.jpeg_data = enc_bytes
            else:
                self.jpeg_data = None
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


class HierarchicalThreatPipelineV200:
    def __init__(
        self,
        yolo_world_weights="weights/yolov8x-worldv2.pt",
        yolo_pose_weights="weights/yolo26s-pose.pt",
        ctrgcn_weights="weights/ctrgcn_violence_v2.00.pth",
        reference_weights="weights/reference_data_ctrgcn_v2.00.pt",
        threat_threshold=None,
        buffer_size=1200,
        jpeg_quality=85,
        device=None,
        imgsz=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Initializing Hierarchical Pipeline v2.00 (CTR-GCN) on device: {self.device}")

        # 1. Stage 1 Weapon Detector (YOLO-World v2)
        print(f"[INIT] Loading Stage 1 Detector: {yolo_world_weights}")
        self.weapon_model = YOLO(yolo_world_weights)
        self.weapon_model.set_classes(["knife", "machete", "gun", "dagger", "sword", "weapon", "blade"])

        # 2. Stage 2a Pose Extractor (YOLO26s-pose)
        print(f"[INIT] Loading Stage 2a Pose Extractor: {yolo_pose_weights}")
        self.pose_model = YOLO(yolo_pose_weights)

        # 3. Stage 2b Action Recognition Model (CTR-GCN v2.00)
        print(f"[INIT] Loading Stage 2b CTR-GCN Dynamic Backbone: {ctrgcn_weights}")
        self.action_model = CTRGCNModel(num_classes=1).to(self.device)
        if os.path.exists(ctrgcn_weights):
            self.action_model.load_state_dict(torch.load(ctrgcn_weights, map_location=self.device, weights_only=True))
        self.action_model.eval()

        # 4. Calibrated Manifold Reference Bank
        self.has_ref_data = False
        if os.path.exists(reference_weights):
            print(f"[INIT] Loading v2.00 Calibrated Reference Bank: {reference_weights}")
            ref_data = torch.load(reference_weights, map_location=self.device, weights_only=False)
            self.rmd_mu_0 = ref_data.get("mu_0", None)
            if self.rmd_mu_0 is not None: self.rmd_mu_0 = self.rmd_mu_0.to(self.device)
            self.rmd_mu_c = ref_data.get("mu_c", None)
            if self.rmd_mu_c is not None: self.rmd_mu_c = self.rmd_mu_c.to(self.device)
            self.rmd_inv_Sigma_0 = ref_data.get("inv_Sigma_0", None)
            if self.rmd_inv_Sigma_0 is not None: self.rmd_inv_Sigma_0 = self.rmd_inv_Sigma_0.to(self.device)
            self.rmd_inv_Sigma_c = ref_data.get("inv_Sigma_c", None)
            if self.rmd_inv_Sigma_c is not None: self.rmd_inv_Sigma_c = self.rmd_inv_Sigma_c.to(self.device)
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
        self.status_text = "System Initialized - Standby (CTR-GCN v2.00)"
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

        num_actors = len(boxes)
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

        return selected_kpts, selected_bbox, selected_tid

    def _weapon_scanner_worker(self):
        while self.running:
            start_time = time.time()
            with self.lock:
                latest_item = self.buffer[-1] if len(self.buffer) > 0 else None

            if latest_item is not None:
                frm = latest_item.get_frame()
                if frm is not None:
                    res = self.weapon_model(
                        frm,
                        imgsz=self.native_imgsz,
                        conf=0.20,
                        verbose=False,
                        device=self.device
                    )
                    boxes = res[0].boxes
                    if len(boxes) > 0:
                        confidences = boxes.conf.cpu().numpy()
                        best_idx = np.argmax(confidences)
                        self.active_weapon_box = boxes.xyxy.cpu().numpy()[best_idx]
                        if not self.threat_active:
                            self.threat_active = True
                            self.threat_event.set()
                            print(f"\n[SECURITY ALERT] Weapon Confirmed! Triggering Stage 2 CTR-GCN Pipeline (v2.00)")
                    else:
                        if self.threat_active:
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
                    kpts, bbox, t_id = self._select_actor_keypoints(res[0])
                    item.skeleton = kpts if kpts is not None else np.zeros((17, 2), dtype=np.float32)
                    item.actor_bbox = bbox
                    item.track_id = t_id
                else:
                    item.skeleton = np.zeros((17, 2), dtype=np.float32)

        # Step 2: Temporal smoothing and Scale-Invariant Normalization (v1.02)
        raw_seq = np.array([item.skeleton for item in window])
        cleaned = interpolate_missing_joints(raw_seq)
        smoothed = smooth_kinematics(cleaned, sigma=1.0)
        normalized = normalize_skeleton_clip(smoothed)

        # Step 3: CTR-GCN Dynamic Feature Extraction (v2.00)
        clip_tensor = torch.from_numpy(normalized).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, features = self.action_model(clip_tensor, return_features=True)

        # Step 4: Threat Gating
        if self.has_ref_data and self.rmd_mu_0 is not None:
            rmd_score, min_mc, m0 = compute_relative_mahalanobis_distance(
                features, self.rmd_mu_0, self.rmd_mu_c, self.rmd_inv_Sigma_0, self.rmd_inv_Sigma_c
            )
            if isinstance(rmd_score, torch.Tensor):
                rmd_score = rmd_score.item()
            is_violence = (rmd_score >= self.threat_threshold)
            detail_str = f"CTR-GCN Score: {rmd_score:+.2f} (Thresh: {self.threat_threshold:+.2f})"
        else:
            prob = torch.sigmoid(logits).item()
            is_violence = (prob >= self.threat_threshold)
            detail_str = f"CTR-GCN Conf: {prob:.2f} (Thresh: {self.threat_threshold:.2f})"

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

    def run(self, source=0, headless=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video stream / camera: {source}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.native_imgsz = self._determine_rectangular_imgsz(frame_w, frame_h)
        print(f"[STREAM] Stream Resolution: {frame_w}x{frame_h} -> Native Rectangular Imgsz: {self.native_imgsz}")

        self.running = True
        self.weapon_thread.start()
        self.action_thread.start()

        fps_timer = time.time()
        frames_counted = 0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_idx += 1
                now = time.time()

                item = FrameItem(frame, now, self.frame_idx, jpeg_quality=self.jpeg_quality)
                with self.lock:
                    self.buffer.append(item)

                frames_counted += 1
                if now - fps_timer >= 1.0:
                    self.current_fps = frames_counted / (now - fps_timer)
                    frames_counted = 0
                    fps_timer = now

                if self.is_violent_alert and now > self.alert_until:
                    self.is_violent_alert = False
                    if not self.threat_active:
                        self.status_text = "System Standby (v2.00)"
                        self.status_color = (0, 255, 0)

                if not headless:
                    disp = frame.copy()
                    if self.active_weapon_box is not None:
                        wx1, wy1, wx2, wy2 = map(int, self.active_weapon_box)
                        cv2.rectangle(disp, (wx1, wy1), (wx2, wy2), (0, 0, 255), 2)
                        cv2.putText(disp, "WEAPON", (wx1, max(20, wy1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if self.last_known_actor_bbox is not None:
                        ax1, ay1, ax2, ay2 = map(int, self.last_known_actor_bbox)
                        box_col = (0, 0, 255) if self.is_violent_alert else (255, 100, 0)
                        cv2.rectangle(disp, (ax1, ay1), (ax2, ay2), box_col, 2)
                        lbl = f"ACTOR #{self.active_threat_actor_id}" if self.active_threat_actor_id else "ACTOR"
                        cv2.putText(disp, lbl, (ax1, max(20, ay1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_col, 2)

                    cv2.putText(disp, f"STATUS: {self.status_text}", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.status_color, 2)
                    cv2.putText(disp, f"FPS: {self.current_fps:.1f} | Buffer: {len(self.buffer)} frames",
                                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow("Hierarchical Threat Surveillance v2.00 (CTR-GCN)", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.running = False
            self.threat_event.set()
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Pipeline v2.00 stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Threat Surveillance Pipeline v2.00 (CTR-GCN)")
    parser.add_argument("--source", default=0, help="Webcam index or video path")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    args = parser.parse_args()

    pipeline = HierarchicalThreatPipelineV200()
    pipeline.run(source=args.source, headless=args.headless)
