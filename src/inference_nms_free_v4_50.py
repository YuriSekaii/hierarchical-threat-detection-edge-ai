"""
src/inference_nms_free_v4_50.py

Version 4.50 Modular Inference Engine:
NMS-Free Real-Time Edge Architecture (One-to-One Bipartite Matching vs. Greedy NMS)

Empirically isolates and evaluates:
1. One-to-One Bipartite Matching Head (NMS-Free End-to-End, Hungarian query assignment)
2. One-to-Many Head with Greedy NMS (Multi-positive anchor assignment + heuristic pairwise IoU suppression)
3. Zero-Shot RT-DETR / YOLOv10 edge baseline wrappers
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO, RTDETR


class NMSFreeWeaponDetector:
    """
    Detector supporting both One-to-One Bipartite Matching (NMS-Free)
    and One-to-Many Greedy NMS inference on YOLO26s distilled weapon weights.
    Uses separate model instances to guarantee pristine computational graph separation.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        default_mode: str = "one2one",
        default_conf: float = 0.45,
        default_iou_nms: float = 0.45,
        device: Optional[str] = None,
    ):
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if weights_path is None:
            weights_path = os.path.join(repo_dir, "weights", "yolo_weapon_distilled.pt")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        self.weights_path = weights_path
        self.default_mode = default_mode.lower()
        self.default_conf = default_conf
        self.default_iou_nms = default_iou_nms

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 1. Initialize One-to-One (NMS-Free) model instance
        self.model_one2one = YOLO(self.weights_path)
        self.model_one2one.to(self.device)
        self.model_one2one.model.end2end = True
        self.model_one2one.model.model[-1].end2end = True

        # 2. Initialize One-to-Many (Greedy NMS) model instance
        self.model_one2many = YOLO(self.weights_path)
        self.model_one2many.to(self.device)
        self.model_one2many.model.end2end = False
        self.model_one2many.model.model[-1].end2end = False

    def predict(
        self,
        source: Union[str, np.ndarray, torch.Tensor],
        conf: Optional[float] = None,
        iou_nms: Optional[float] = None,
        mode: Optional[str] = None,
        max_det: int = 300,
    ) -> Dict:
        """
        Run inference on image using designated mode ("one2one" or "one2many").
        """
        conf_val = conf if conf is not None else self.default_conf
        iou_val = iou_nms if iou_nms is not None else self.default_iou_nms
        mode_val = (mode or self.default_mode).lower()

        if mode_val == "one2one":
            target_model = self.model_one2one
        elif mode_val == "one2many":
            target_model = self.model_one2many
        else:
            raise ValueError(f"Unknown mode '{mode_val}'. Must be 'one2one' or 'one2many'.")

        t0 = time.perf_counter()
        results = target_model(
            source,
            conf=conf_val,
            iou=iou_val,
            max_det=max_det,
            verbose=False,
            device=self.device,
        )
        t_total_ms = (time.perf_counter() - t0) * 1000.0

        boxes_list = []
        norm_boxes_list = []
        scores_list = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            res_boxes = results[0].boxes
            boxes_list = res_boxes.xyxy.cpu().numpy().tolist()
            norm_boxes_list = res_boxes.xyxyn.cpu().numpy().tolist()
            scores_list = res_boxes.conf.cpu().numpy().tolist()

        has_threat = len(boxes_list) > 0
        alert_status = "CRITICAL_WEAPON_CONFIRMED" if has_threat else "NORMAL_CLEARED"

        internal_inf_ms = results[0].speed.get("inference", t_total_ms) if len(results) > 0 else t_total_ms

        return {
            "alert_status": alert_status,
            "threat_detected": has_threat,
            "num_detections": len(boxes_list),
            "boxes": boxes_list,
            "normalized_boxes": norm_boxes_list,
            "scores": scores_list,
            "latency_ms": internal_inf_ms,
            "wall_latency_ms": t_total_ms,
            "mode": mode_val,
            "conf_threshold": conf_val,
            "iou_nms_threshold": iou_val if mode_val == "one2many" else None,
        }


class ZeroShotEdgeDETRDetector:
    """
    Wrapper for zero-shot RT-DETR-L baseline (COCO pretrained).
    """

    def __init__(
        self,
        weights_path: str = "rtdetr-l.pt",
        target_class_id: int = 43,  # COCO knife class
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = RTDETR(weights_path)
        self.model.to(self.device)
        self.target_class_id = target_class_id

    def predict(
        self,
        source: Union[str, np.ndarray, torch.Tensor],
        conf: float = 0.20,
    ) -> Dict:
        t0 = time.perf_counter()
        results = self.model(
            source,
            conf=conf,
            classes=[self.target_class_id],
            verbose=False,
            device=self.device,
        )
        t_total_ms = (time.perf_counter() - t0) * 1000.0

        boxes_list = []
        norm_boxes_list = []
        scores_list = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            res_boxes = results[0].boxes
            boxes_list = res_boxes.xyxy.cpu().numpy().tolist()
            norm_boxes_list = res_boxes.xyxyn.cpu().numpy().tolist()
            scores_list = res_boxes.conf.cpu().numpy().tolist()

        has_threat = len(boxes_list) > 0
        internal_inf_ms = results[0].speed.get("inference", t_total_ms) if len(results) > 0 else t_total_ms

        return {
            "alert_status": "CRITICAL_WEAPON_CONFIRMED" if has_threat else "NORMAL_CLEARED",
            "threat_detected": has_threat,
            "num_detections": len(boxes_list),
            "boxes": boxes_list,
            "normalized_boxes": norm_boxes_list,
            "scores": scores_list,
            "latency_ms": internal_inf_ms,
            "wall_latency_ms": t_total_ms,
            "mode": "rtdetr_hungarian",
            "conf_threshold": conf,
        }
