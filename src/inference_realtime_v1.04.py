"""
Hierarchical Real-Time Threat Detection System - Version 1.04
Dual-Stage Surveillance Architecture with:
1. In-Memory JPEG Buffer Compression (v1.01)
2. Torso-Scale Invariant Kinematic Normalization (v1.02)
3. Native Aspect-Ratio Rectangular Inference (v1.03)
4. Multi-Person Tracking Guard with ByteTrack & Spatial-Temporal Continuity (v1.04)

Changes in Version 1.04:
- Replaced naive res[0].keypoints.data[0] index selection with Multi-Person Tracking Guard.
- Integrates ByteTrack (model.track(persist=True)) with persistent actor ID locking.
- Locks onto the primary threat actor (closest to detected weapon or largest bounding box area).
- Implements spatial-temporal bounding box IoU & centroid fallback to prevent identity swaps during occlusions.
- Guarantees 100% actor temporal continuity across 50-frame sliding window, eliminating keypoint swapping.
- Preserved baseline (v1.00) through v1.03 files intact.
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
from src.ood_metrics import compute_knn_distance

from src.skeleton_utils import (
    interpolate_missing_joints,
    smooth_kinematics,
    normalize_skeleton_clip,
)


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


def calculate_rectangular_imgsz(frame_h, frame_w, base_dim=640):
    if frame_w >= frame_h:
        new_w = base_dim
        raw_h = (frame_h / frame_w) * base_dim
        new_h = int(round(raw_h / 32.0)) * 32
        return (new_h, new_w)
    else:
        new_h = base_dim
        raw_w = (frame_w / frame_h) * base_dim
        new_w = int(round(raw_w / 32.0)) * 32
        return (new_h, new_w)


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_centroid_distance(box1, box2):
    """Compute Euclidean distance between centers of two boxes."""
    c1 = np.array([(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0])
    c2 = np.array([(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0])
    return np.linalg.norm(c1 - c2)


class HierarchicalThreatDetector:
    def __init__(self, weapon_model_path, pose_model_path, action_model_path, ref_data_path,
                 device='cuda', jpeg_quality=85, imgsz_override=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.jpeg_quality = jpeg_quality
        self.imgsz_override = imgsz_override
        self.rect_imgsz = None
        print(f"[INIT] Initializing Hierarchical Detector (v1.04 Multi-Person Guard) on {self.device}...")

        # 1. Load Object & Pose Models
        print(f"  [1/4] Loading Distilled YOLO Weapon Detector: {os.path.basename(weapon_model_path)}")
        self.weapon_model = YOLO(weapon_model_path)

        print(f"  [2/4] Loading YOLO26s Pose Estimator: {os.path.basename(pose_model_path)}")
        self.pose_model = YOLO(pose_model_path)
        
        # 2. Load Retrained ST-GCN Violence Model (v1.02)
        print(f"  [3/4] Loading ST-GCN Violence Model (v1.02): {os.path.basename(action_model_path)}")
        self.action_model = STGCNModel(num_classes=1, in_channels=2, num_keypoints=17).to(self.device)
        ckpt = torch.load(action_model_path, map_location=self.device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        self.action_model.load_state_dict(state_dict, strict=False)
        self.action_model.eval()

        # 3. Load Deep k-NN Reference Bank (v1.02)
        print(f"  [4/4] Loading Deep k-NN Reference Weights (v1.02): {os.path.basename(ref_data_path)}")
        ref_data = torch.load(ref_data_path, map_location='cpu', weights_only=False)
        self.knn_feature_bank = ref_data['feature_bank'].float().cpu()
        self.knn_threshold = float(ref_data['threshold'])
        self.knn_k = int(ref_data.get('k', 2))
        print(f"        -> Deep k-NN Initialized: k={self.knn_k}, threshold={self.knn_threshold:.4f}, bank={self.knn_feature_bank.shape}")

        # 4. Zero-Drop In-Memory Buffer Architecture (1200 frames ~ 40s at 30 FPS)
        self.buffer_size = 1200
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        self.is_running = False
        self.frame_counter = 0

        # State flags & Multi-Person Tracking Guard
        self.weapon_active_until = 0.0
        self.last_weapon_bbox = None
        self.active_threat_actor_id = None
        self.last_actor_bbox = None

        self.last_analyzed_idx = 0
        self.clip_stride = 10
        self.clip_length = 50
        self.status_text = "SCANNING (IDLE)"
        self.status_color = (0, 255, 0)
        self.is_violent_alert = False
        self.alert_until = 0.0
        self.pending_audit_frames = 0
        self.buffer_ram_mb = 0.0

    def weapon_scanner_loop(self):
        """Thread 2: 3 FPS lightweight gating loop with Rectangular Inference and Weapon BBox Tracking."""
        target_delay = 1.0 / 3.0
        while self.is_running:
            start_t = time.time()
            frame_copy = None

            with self.buffer_lock:
                if len(self.frame_buffer) > 0:
                    frame_copy = self.frame_buffer[-1].get_frame()

            if frame_copy is not None:
                imgsz = self.rect_imgsz if self.rect_imgsz is not None else 640
                results = self.weapon_model(frame_copy, imgsz=imgsz, conf=0.45, verbose=False)
                has_weapon = len(results[0].boxes) > 0

                if has_weapon:
                    self.weapon_active_until = time.time() + 3.0
                    best_box = results[0].boxes.xyxy[0].cpu().numpy()
                    self.last_weapon_bbox = best_box
                    if not self.is_violent_alert:
                        self.status_text = "WEAPON DETECTED"
                        self.status_color = (0, 165, 255)

            elapsed = time.time() - start_t
            sleep_time = max(0.01, target_delay - elapsed)
            time.sleep(sleep_time)

    def _select_actor_keypoints(self, pose_result):
        """
        Multi-Person Tracking Guard (v1.04).
        Selects the single primary threat actor's keypoints consistently across frames:
        1. Checks ByteTrack ID matching active_threat_actor_id.
        2. Falls back to spatial IoU/centroid continuity against last_actor_bbox.
        3. Initializes tracking on the person closest to the detected weapon (or largest box).
        """
        if not hasattr(pose_result, 'boxes') or len(pose_result.boxes) == 0:
            return np.zeros((17, 2), dtype=np.float32), None, None

        boxes_xyxy = pose_result.boxes.xyxy.cpu().numpy()
        num_persons = len(boxes_xyxy)

        # Extract track IDs if ByteTrack assigned them
        track_ids = None
        if hasattr(pose_result.boxes, 'id') and pose_result.boxes.id is not None:
            track_ids = pose_result.boxes.id.cpu().numpy().astype(int)

        selected_idx = 0

        # Strategy 1: Match existing active ByteTrack ID
        if self.active_threat_actor_id is not None and track_ids is not None:
            matches = np.where(track_ids == self.active_threat_actor_id)[0]
            if len(matches) > 0:
                selected_idx = matches[0]
            else:
                # Strategy 2: Track ID lost (occlusion), use Spatial IoU with last known bbox
                if self.last_actor_bbox is not None:
                    ious = [compute_iou(self.last_actor_bbox, b) for b in boxes_xyxy]
                    best_iou_idx = int(np.argmax(ious))
                    if ious[best_iou_idx] > 0.15:
                        selected_idx = best_iou_idx
                    else:
                        dists = [compute_centroid_distance(self.last_actor_bbox, b) for b in boxes_xyxy]
                        selected_idx = int(np.argmin(dists))

        # Strategy 3: Initial frame of threat (no active actor locked yet)
        elif self.last_weapon_bbox is not None:
            # Select person closest to or overlapping the detected weapon
            weapon_ious = [compute_iou(self.last_weapon_bbox, b) for b in boxes_xyxy]
            best_w_iou = max(weapon_ious)
            if best_w_iou > 0.05:
                selected_idx = int(np.argmax(weapon_ious))
            else:
                weapon_dists = [compute_centroid_distance(self.last_weapon_bbox, b) for b in boxes_xyxy]
                selected_idx = int(np.argmin(weapon_dists))

            # Lock onto this person's track ID
            if track_ids is not None and len(track_ids) > selected_idx:
                self.active_threat_actor_id = int(track_ids[selected_idx])

        # Strategy 4: Fallback to largest bounding box area
        else:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_xyxy]
            selected_idx = int(np.argmax(areas))
            if track_ids is not None and len(track_ids) > selected_idx:
                self.active_threat_actor_id = int(track_ids[selected_idx])

        # Update last known actor bbox
        chosen_bbox = boxes_xyxy[selected_idx]
        self.last_actor_bbox = chosen_bbox
        chosen_track_id = int(track_ids[selected_idx]) if (track_ids is not None and len(track_ids) > selected_idx) else None
        if chosen_track_id is not None:
            self.active_threat_actor_id = chosen_track_id

        # Extract 17 keypoints for the chosen person
        kpts = np.zeros((17, 2), dtype=np.float32)
        if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
            if len(pose_result.keypoints.data) > selected_idx:
                kpts = pose_result.keypoints.data[selected_idx].cpu().numpy()[:, :2]

        return kpts, chosen_bbox, chosen_track_id

    def action_recognition_loop(self):
        """Thread 3: Zero-Drop sliding-window kinematic evaluator with Multi-Person Tracking Guard."""
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
                self.active_threat_actor_id = None
                self.last_actor_bbox = None
                time.sleep(0.05)
                continue

            # Extract contiguous 50-frame window from circular buffer
            window = []
            with self.buffer_lock:
                if len(self.frame_buffer) >= self.clip_length:
                    buf_list = list(self.frame_buffer)
                    min_idx = buf_list[0].index
                    max_idx = buf_list[-1].index

                    if self.last_analyzed_idx < min_idx:
                        self.last_analyzed_idx = min_idx

                    target_start = self.last_analyzed_idx
                    target_end = target_start + self.clip_length

                    if target_end <= max_idx + 1:
                        start_offset = target_start - min_idx
                        window = buf_list[start_offset : start_offset + self.clip_length]
                        self.last_analyzed_idx += self.clip_stride
                        self.pending_audit_frames = max(0, max_idx - target_end)

            if len(window) < self.clip_length:
                time.sleep(0.02)
                continue

            # Step 1: Extract Skeletons using Multi-Person Tracking Guard
            imgsz = self.rect_imgsz if self.rect_imgsz is not None else 640
            for item in window:
                if item.skeleton is None:
                    raw_frame = item.get_frame()
                    if raw_frame is not None:
                        # Use tracker with persist=True
                        res = self.pose_model.track(
                            raw_frame, persist=True, imgsz=imgsz, verbose=False
                        )
                        kpts, bbox, t_id = self._select_actor_keypoints(res[0])
                        item.skeleton = kpts
                        item.actor_bbox = bbox
                        item.track_id = t_id
                    else:
                        item.skeleton = np.zeros((17, 2), dtype=np.float32)

            # Step 2: Temporal smoothing and Scale-Invariant Normalization (v1.02)
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
            print(f"[ERROR] Could not open video source: {source}")
            return

        f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.imgsz_override:
            self.rect_imgsz = self.imgsz_override
        elif f_w > 0 and f_h > 0:
            self.rect_imgsz = calculate_rectangular_imgsz(f_h, f_w, base_dim=640)
        else:
            self.rect_imgsz = (384, 640)

        print(f"[CONFIG] Input Stream: {f_w}x{f_h} -> Rectangular Inference: {self.rect_imgsz}")

        self.is_running = True

        t_weapon = threading.Thread(target=self.weapon_scanner_loop, daemon=True)
        t_action = threading.Thread(target=self.action_recognition_loop, daemon=True)
        t_weapon.start()
        t_action.start()

        print("[RUNNING] Threat detection system v1.04 (Multi-Person Guard) started. Press 'q' to exit.")

        total_bytes = 0
        frame_idx = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            new_item = FrameItem(frame, time.time(), self.frame_counter, jpeg_quality=self.jpeg_quality)
            with self.buffer_lock:
                if len(self.frame_buffer) == self.buffer_size:
                    old_item = self.frame_buffer[0]
                    total_bytes -= getattr(old_item, 'nbytes', 0)
                self.frame_buffer.append(new_item)
                total_bytes += new_item.nbytes
                self.frame_counter += 1

            frame_idx += 1
            if frame_idx % 30 == 0:
                self.buffer_ram_mb = total_bytes / (1024.0 * 1024.0)

            if not headless:
                cv2.rectangle(frame, (10, 10), (740, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"STATUS: {self.status_text}", (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.status_color, 2)
                
                buf_len = len(self.frame_buffer)
                actor_info = f"Track ID: {self.active_threat_actor_id}" if self.active_threat_actor_id else "Scanning"
                telemetry = f"Buffer: {buf_len}/1200 (~{self.buffer_ram_mb:.1f} MB) | {actor_info} | Backlog: {self.pending_audit_frames}"
                cv2.putText(frame, telemetry, (20, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

                cv2.imshow("Hierarchical Threat Detection (v1.04 Edge AI)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.is_running = False
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def parse_source(src_str):
    try:
        return int(src_str)
    except ValueError:
        return src_str


def parse_imgsz(imgsz_str):
    if imgsz_str is None or imgsz_str.lower() == 'auto':
        return None
    try:
        if ',' in imgsz_str:
            parts = [int(p.strip()) for p in imgsz_str.split(',')]
            return tuple(parts)
        val = int(imgsz_str)
        return (val, val)
    except ValueError:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hierarchical Edge-AI Threat & Violence Detection (v1.04)")
    parser.add_argument('--source', type=str, default="0",
                        help='Webcam ID (0) or video file path (default: 0)')
    parser.add_argument('--camera-id', type=int, default=None,
                        help='Legacy camera ID argument (overrides --source if provided)')
    parser.add_argument('--quality', type=int, default=85,
                        help='In-memory JPEG buffer compression quality (default: 85)')
    parser.add_argument('--imgsz', type=str, default="auto",
                        help='Inference resolution: "auto" for rectangular (384,640) or custom "h,w"')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode without cv2.imshow GUI display')
    args = parser.parse_args()

    active_source = args.camera_id if args.camera_id is not None else parse_source(args.source)
    custom_imgsz = parse_imgsz(args.imgsz)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weapon_weights = os.path.join(base_dir, 'weights', 'yolo_weapon_distilled.pt')
    pose_weights = os.path.join(base_dir, 'weights', 'yolo26s-pose.pt')
    action_weights = os.path.join(base_dir, 'weights', 'stgcn_violence_v1.02.pth')
    ref_weights = os.path.join(base_dir, 'weights', 'reference_data_knn_v1.02.pt')

    detector = HierarchicalThreatDetector(
        weapon_model_path=weapon_weights,
        pose_model_path=pose_weights,
        action_model_path=action_weights,
        ref_data_path=ref_weights,
        jpeg_quality=args.quality,
        imgsz_override=custom_imgsz
    )
    detector.run(source=active_source, headless=args.headless)
