"""
src/stage1_detector.py

Component 2: Stage 1 Real-Time Handheld Weapon Detector (TensorRT / ONNX Accelerated).
Runs Knowledge-Distilled YOLO26s (stage1_repro846_kdgtclean_p250) with knife class filtering and NMS IoU = 0.50.
Auto-compiles and loads native NVIDIA TensorRT .engine binary compiled from universal .onnx blueprint.
Default operating point: Conf = 0.20, 384x640 input.
Strict GPU enforcement (No CPU fallback).
"""

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torchvision.ops import nms as tv_nms
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.engine_builder import ensure_engine, get_hardware_precision

STAGE1_DEFAULT_STEM = "stage1_repro846_kdgtclean_p250"
STAGE1_DEFAULT_CONF = 0.20
STAGE1_NMS_IOU = 0.50
_WEAPON_CLASS_NAMES = {"knife", "weapon"}


def filter_knife_boxes(res, nms_iou: float = STAGE1_NMS_IOU, normalized: bool = False) -> Tuple[List[List[float]], List[float]]:
    """Keep only weapon/knife class boxes (ignoring phone/hand_grip_knife) and apply NMS."""
    if res.boxes is None or len(res.boxes) == 0:
        return [], []
    names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names or ()))
    keep_ids = {int(k) for k, v in names.items() if str(v).lower() in _WEAPON_CLASS_NAMES}
    cls_t = res.boxes.cls
    if keep_ids:
        mask = torch.zeros_like(cls_t, dtype=torch.bool)
        for cid in keep_ids:
            mask |= (cls_t.long() == cid)
    else:
        mask = torch.ones_like(cls_t, dtype=torch.bool)
    if not mask.any():
        return [], []

    coords = (res.boxes.xyxyn if normalized else res.boxes.xyxy)[mask].float()
    scores = res.boxes.conf[mask].float()
    if nms_iou is not None and coords.shape[0] > 1:
        # Run NMS in pixel space so aspect ratio never skews IoU.
        xyxy_px = res.boxes.xyxy[mask].float()
        keep = tv_nms(xyxy_px, scores, nms_iou)
        coords, scores = coords[keep], scores[keep]
    else:
        order = torch.argsort(scores, descending=True)
        coords, scores = coords[order], scores[order]
    return coords.cpu().numpy().tolist(), scores.cpu().numpy().tolist()


def calculate_rectangular_imgsz(frame_h: int, frame_w: int, base_dim: int = 640) -> Tuple[int, int]:
    """
    Calculates 32-stride divisible dynamic rectangular resolution preserving aspect ratio.
    Saves ~40% compute compared to square letterboxing.
    """
    if frame_w >= frame_h:
        new_w = base_dim
        raw_h = (frame_h / frame_w) * base_dim
        new_h = math.ceil(raw_h / 32.0) * 32  # up, never down: 16:10 -> 416, not 384 (keeps the 640-wide scale)
        return (new_h, new_w)
    else:
        new_h = base_dim
        raw_w = (frame_w / frame_h) * base_dim
        new_w = math.ceil(raw_w / 32.0) * 32
        return (new_h, new_w)


class Stage1WeaponDetector:
    """
    Production Stage 1 Weapon Detector.
    Encapsulates the Distilled YOLO26s student model compiled as a hardware-tailored
    TensorRT engine with One-to-One Hungarian matching for end-to-end NMS-free inference.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = STAGE1_DEFAULT_CONF,
        device: Optional[str] = None,
        imgsz: Optional[Union[int, Tuple[int, int]]] = (384, 640),
        backend: str = "tensorrt",
        nms_iou: float = STAGE1_NMS_IOU,
    ):
        self.backend = backend.lower()
        if self.backend == "tensorrt":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "[STAGE 1 CRITICAL ERROR] TensorRT Stage 1 Weapon Detector strictly requires an NVIDIA CUDA GPU.\n"
                    "CPU execution is not supported by TensorRT."
                )
            self.device = device or "cuda:0"
        else:
            if torch.cuda.is_available():
                self.device = device or "cuda:0"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = device or "xpu:0"
            else:
                self.device = device or "cpu"

        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.nms_iou = nms_iou

        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        weights_dir = os.path.join(repo_dir, "weights")

        if self.backend == "pytorch":
            self.weights_path = weights_path or os.path.join(weights_dir, f"{STAGE1_DEFAULT_STEM}.pt")
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"[STAGE 1] PyTorch weights not found at: {self.weights_path}")
            print(f"[STAGE 1] Loading Distilled YOLO26s (PyTorch Eager) from {os.path.basename(self.weights_path)} on {self.device}...")
            self.model = YOLO(self.weights_path, task="detect")
            self.model.to(self.device)
            if hasattr(self.model, "model") and self.model.model is not None:
                self.model.model.end2end = True
                if hasattr(self.model.model, "model") and len(self.model.model.model) > 0:
                    last_module = self.model.model.model[-1]
                    if hasattr(last_module, "end2end"):
                        last_module.end2end = True
            print(f"          -> Stage 1 Initialized (Conf={self.conf_threshold:.2f}, NMS={self.nms_iou:.2f}, PyTorch Eager)")
        else:
            # Determine engine & onnx paths
            onnx_path = os.path.join(weights_dir, f"{STAGE1_DEFAULT_STEM}.onnx")
            engine_path = weights_path or os.path.join(weights_dir, f"{STAGE1_DEFAULT_STEM}.engine")

            metadata = {
                "description": "Distilled YOLO26s Weapon Detector (kdgtclean)",
                "author": "Ultralytics",
                "date": "2026-11-26",
                "version": "8.4.9",
                "license": "AGPL-3.0",
                "docs": "https://docs.ultralytics.com",
                "stride": 32,
                "task": "detect",
                "batch": 1,
                "imgsz": [384, 640],
                "names": {0: "hand_grip_knife", 1: "knife", 2: "phone"},
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
            try:
                self.model = YOLO(self.engine_path, task="detect")
            except Exception as e:
                print(f"[STAGE 1 WARNING] Failed to load engine ({e}). Re-compiling from ONNX blueprint...")
                if os.path.exists(self.engine_path):
                    os.remove(self.engine_path)
                self.engine_path = ensure_engine(
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    metadata=metadata,
                    workspace_gb=1.0,
                )
                self.model = YOLO(self.engine_path, task="detect")
            print(f"          -> Stage 1 Initialized (Conf={self.conf_threshold:.2f}, NMS={self.nms_iou:.2f}, Engine={os.path.basename(self.engine_path)})")

    def predict(self, frame: np.ndarray, custom_conf: Optional[float] = None, imgsz: Optional[Union[int, Tuple[int, int]]] = None) -> Dict:
        """
        Executes lightweight weapon scanning on a single video frame.

        Returns:
            Dict containing:
                - "has_threat": bool
                - "boxes": list of [x1, y1, x2, y2]
                - "scores": list of float confidence values
                - "best_box": [x1, y1, x2, y2] of top detection (or None)
                - "latency_ms": float
        """
        conf_val = custom_conf if custom_conf is not None else self.conf_threshold
        target_imgsz = imgsz or self.imgsz or (384, 640)

        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=conf_val,
            device=self.device,
            imgsz=target_imgsz,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        boxes_list, scores_list = filter_knife_boxes(results[0], nms_iou=self.nms_iou) if len(results) > 0 else ([], [])
        best_box = boxes_list[0] if boxes_list else None

        return {
            "has_threat": len(boxes_list) > 0,
            "boxes": boxes_list,
            "scores": scores_list,
            "best_box": best_box,
            "latency_ms": latency_ms,
        }
