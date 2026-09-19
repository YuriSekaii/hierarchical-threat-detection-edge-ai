"""
src/stage2a_pose_tracker.py

Component 3: Stage 2a Kinematic Pose Estimator & Multi-Person Tracking Guard (TensorRT Accelerated).
Extracts 17 COCO skeletal keypoints via YOLO26s-Pose compiled to native TensorRT .engine binary.
Locks tracking onto the verified threat actor using ByteTrack and spatial-temporal continuity.
Applies torso-scale normalization (L_torso) and 1D Gaussian temporal smoothing.
Strict GPU enforcement (No CPU fallback).
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.skeleton_utils import (
    interpolate_missing_joints,
    smooth_kinematics,
    normalize_skeleton_clip,
)
from src.engine_builder import ensure_engine, get_hardware_precision


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_centroid_distance(box1: List[float], box2: List[float]) -> float:
    """Computes Euclidean distance between centers of two boxes."""
    c1 = np.array([(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0])
    c2 = np.array([(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0])
    return float(np.linalg.norm(c1 - c2))


class Stage2aPoseTracker:
    """
    Manages pose estimation and threat actor tracking across video frames.
    Implements ByteTrack persistent locking with spatial continuity fallback.
    Executes via native TensorRT engine compiled from universal .onnx blueprint.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: Optional[Union[int, Tuple[int, int]]] = 640,
        backend: str = "tensorrt",
    ):
        self.backend = backend.lower()
        if self.backend == "tensorrt":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "[STAGE 2a CRITICAL ERROR] TensorRT Stage 2a Pose Tracker strictly requires an NVIDIA CUDA GPU.\n"
                    "CPU execution is not supported by TensorRT."
                )
        else:
            if torch.cuda.is_available():
                self.device = device or "cuda:0"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = device or "xpu:0"
            else:
                self.device = device or "cpu"

        self.imgsz = imgsz

        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        weights_dir = os.path.join(repo_dir, "weights")

        if self.backend == "pytorch":
            self.weights_path = weights_path or os.path.join(weights_dir, "yolo26s-pose.pt")
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"[STAGE 2a] PyTorch pose weights not found at: {self.weights_path}")
            print(f"[STAGE 2a] Loading YOLO26s-Pose (PyTorch Eager) from {os.path.basename(self.weights_path)} on {self.device}...")
            self.pose_model = YOLO(self.weights_path, task="pose")
            self.pose_model.to(self.device)
            print("          -> Stage 2a Initialized (Tracker=ByteTrack, PyTorch Eager)")
        else:
            onnx_path = os.path.join(weights_dir, "yolo26s-pose.onnx")
            engine_path = weights_path or os.path.join(weights_dir, "yolo26s-pose.engine")

            metadata = {
                "description": "YOLO26s-Pose 17-Keypoint Extractor",
                "author": "Ultralytics",
                "date": "2026-09-19",
                "version": "8.4.9",
                "license": "AGPL-3.0",
                "docs": "https://docs.ultralytics.com",
                "stride": 32,
                "task": "pose",
                "batch": 1,
                "imgsz": [640, 640],
                "names": {0: "person"},
                "kpt_shape": [17, 3],
                "end2end": True,
            }

            # Auto-compile engine from ONNX if missing
            self.engine_path = ensure_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                metadata=metadata,
                workspace_gb=1.0,
            )

            prec, _ = get_hardware_precision()
            print(f"[STAGE 2a] Loading YOLO26s-Pose TensorRT Engine ({prec}) on {self.device}...")
            try:
                self.pose_model = YOLO(self.engine_path, task="pose")
            except Exception as e:
                print(f"[STAGE 2a WARNING] Failed to load engine ({e}). Re-compiling from ONNX blueprint...")
                if os.path.exists(self.engine_path):
                    os.remove(self.engine_path)
                self.engine_path = ensure_engine(
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    metadata=metadata,
                    workspace_gb=1.0,
                )
                self.pose_model = YOLO(self.engine_path, task="pose")

        # Tracking state
        self.active_threat_actor_id: Optional[int] = None
        self.last_actor_bbox: Optional[List[float]] = None
        self.last_weapon_bbox: Optional[List[float]] = None

    def reset_tracking(self):
        """Resets active tracking locks when threat clears."""
        self.active_threat_actor_id = None
        self.last_actor_bbox = None
        self.last_weapon_bbox = None

    def update_weapon_anchor(self, weapon_bbox: Optional[List[float]]):
        """Updates the weapon location used to anchor initial actor tracking."""
        if weapon_bbox is not None:
            self.last_weapon_bbox = weapon_bbox

    def extract_actor_pose(
        self,
        frame: np.ndarray,
        imgsz: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tuple[np.ndarray, Optional[List[float]], Optional[int]]:
        """
        Executes tracking on a single frame and isolates the primary threat actor's keypoints.
        Returns:
            - kpts: np.ndarray of shape (17, 2)
            - actor_bbox: [x1, y1, x2, y2] or None
            - track_id: int or None
        """
        target_imgsz = imgsz or self.imgsz or 640
        res = self.pose_model.track(
            frame,
            persist=True,
            imgsz=target_imgsz,
            verbose=False,
            device=self.device,
        )

        if len(res) == 0 or not hasattr(res[0], "boxes") or len(res[0].boxes) == 0:
            return np.zeros((17, 2), dtype=np.float32), None, None

        boxes_xyxy = res[0].boxes.xyxy.cpu().numpy().tolist()
        track_ids = None
        if hasattr(res[0].boxes, "id") and res[0].boxes.id is not None:
            track_ids = res[0].boxes.id.cpu().numpy().astype(int).tolist()

        selected_idx = 0

        # Strategy 1: Match existing active ByteTrack ID
        if self.active_threat_actor_id is not None and track_ids is not None:
            if self.active_threat_actor_id in track_ids:
                selected_idx = track_ids.index(self.active_threat_actor_id)
            elif self.last_actor_bbox is not None:
                best_iou = 0.0
                best_idx = 0
                for i, b in enumerate(boxes_xyxy):
                    iou = compute_iou(b, self.last_actor_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_iou > 0.20:
                    selected_idx = best_idx
                    if track_ids is not None:
                        self.active_threat_actor_id = track_ids[selected_idx]

        # Strategy 2: Anchor to detected weapon proximity
        elif self.last_weapon_bbox is not None:
            best_dist = float("inf")
            best_idx = 0
            for i, b in enumerate(boxes_xyxy):
                dist = compute_centroid_distance(b, self.last_weapon_bbox)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            selected_idx = best_idx
            if track_ids is not None:
                self.active_threat_actor_id = track_ids[selected_idx]

        # Strategy 3: Largest bounding box fallback
        else:
            max_area = 0.0
            best_idx = 0
            for i, b in enumerate(boxes_xyxy):
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > max_area:
                    max_area = area
                    best_idx = i
            selected_idx = best_idx
            if track_ids is not None:
                self.active_threat_actor_id = track_ids[selected_idx]

        # Update state
        self.last_actor_bbox = boxes_xyxy[selected_idx]
        active_id = track_ids[selected_idx] if track_ids is not None else None

        # Extract keypoints
        if hasattr(res[0], "keypoints") and res[0].keypoints is not None:
            kpts_all = res[0].keypoints.xy.cpu().numpy()
            if len(kpts_all) > selected_idx:
                raw_kpts = kpts_all[selected_idx]  # shape (17, 2)
            else:
                raw_kpts = np.zeros((17, 2), dtype=np.float32)
        else:
            raw_kpts = np.zeros((17, 2), dtype=np.float32)

        return raw_kpts, self.last_actor_bbox, active_id

    def process_window(self, kpts_sequence: List[np.ndarray]) -> torch.Tensor:
        """
        Applies temporal interpolation, Gaussian smoothing, and torso-scale normalization.
        Expects a list of 50 frames of (17, 2) keypoints.
        Returns tensor of shape (1, 3, 50, 17, 1) or (50, 17, 2).
        """
        arr = np.array(kpts_sequence, dtype=np.float32)  # (T, 17, 2)
        arr = interpolate_missing_joints(arr)
        arr = smooth_kinematics(arr, sigma=1.0)
        normed = normalize_skeleton_clip(arr)  # (50, 17, 2)
        return torch.from_numpy(normed).float()
