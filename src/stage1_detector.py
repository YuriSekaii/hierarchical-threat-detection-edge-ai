"""
src/stage1_detector.py

Component 2: Stage 1 Real-Time Handheld Weapon Detector (TensorRT / ONNX Accelerated).
Runs Knowledge-Distilled YOLO26s with One-to-One Hungarian Bipartite Matching (NMS-Free).
Auto-compiles and loads native NVIDIA TensorRT .engine binary compiled from universal .onnx blueprint.
Operating point: Conf = 0.45, 3.27 ms latency on RTX 3090, 86.88% F1.
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

from src.engine_builder import ensure_engine, get_hardware_precision


def calculate_rectangular_imgsz(frame_h: int, frame_w: int, base_dim: int = 640) -> Tuple[int, int]:
    """
    Calculates 32-stride divisible dynamic rectangular resolution preserving aspect ratio.
    Saves ~40% compute compared to square letterboxing.
    """
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


class Stage1WeaponDetector:
    """
    Production Stage 1 Weapon Detector.
    Encapsulates the Distilled YOLO26s student model compiled as a hardware-tailored
    TensorRT engine with One-to-One Hungarian matching for end-to-end NMS-free inference.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.45,
        device: Optional[str] = None,
        imgsz: Optional[Union[int, Tuple[int, int]]] = 640,
        backend: str = "tensorrt",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[STAGE 1 CRITICAL ERROR] Stage 1 Weapon Detector strictly requires an NVIDIA CUDA GPU.\n"
                "CPU execution is disabled to prevent severe latency degradation and dropped attacks."
            )

        self.device = device or "cuda:0"
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.backend = backend.lower()

        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        weights_dir = os.path.join(repo_dir, "weights")

        if self.backend == "pytorch":
            self.weights_path = weights_path or os.path.join(weights_dir, "yolo_weapon_distilled.pt")
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
            print(f"          -> Stage 1 Initialized (Conf={self.conf_threshold:.2f}, NMS-Free=True, PyTorch Eager)")
        else:
            # Determine engine & onnx paths
            onnx_path = os.path.join(weights_dir, "yolo_weapon_distilled.onnx")
            engine_path = weights_path or os.path.join(weights_dir, "yolo_weapon_distilled.engine")

            metadata = {
                "description": "Distilled YOLO26s Weapon Detector",
                "author": "Ultralytics",
                "date": "2026-09-19",
                "version": "8.4.9",
                "license": "AGPL-3.0",
                "docs": "https://docs.ultralytics.com",
                "stride": 32,
                "task": "detect",
                "batch": 1,
                "imgsz": [640, 640],
                "names": {0: "weapon"},
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
            print(f"[STAGE 1] Loading Distilled YOLO26s TensorRT Engine ({prec}) on {self.device}...")
            self.model = YOLO(self.engine_path, task="detect")
            print(f"          -> Stage 1 Initialized (Conf={self.conf_threshold:.2f}, NMS-Free=True, Engine={os.path.basename(self.engine_path)})")

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
        target_imgsz = imgsz or self.imgsz or 640

        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=conf_val,
            device=self.device,
            imgsz=target_imgsz,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        boxes_list = []
        scores_list = []
        best_box = None

        if len(results) > 0 and len(results[0].boxes) > 0:
            res_boxes = results[0].boxes
            boxes_list = res_boxes.xyxy.cpu().numpy().tolist()
            scores_list = res_boxes.conf.cpu().numpy().tolist()
            if len(boxes_list) > 0:
                best_box = boxes_list[0]

        return {
            "has_threat": len(boxes_list) > 0,
            "boxes": boxes_list,
            "scores": scores_list,
            "best_box": best_box,
            "latency_ms": latency_ms,
        }
