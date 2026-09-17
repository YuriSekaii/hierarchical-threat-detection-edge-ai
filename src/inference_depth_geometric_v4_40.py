"""
src/inference_depth_geometric_v4_40.py

Version 4.40: Geometry-Informed Monocular Depth Pipeline Modular Inference Engine
Combines 2D object proposal generation with dense monocular depth estimation and
metric surface discontinuity filtering to reject planar clothing seams, zipper lines,
and cast shadows.

Key Innovations:
1. Dense Monocular Depth Estimation:
   - Evaluates frame using Depth Anything V2 Small (24.8M parameters) to produce
     a continuous relative metric depth field d(x, y).
2. Surface Normal & Depth Gradient Discontinuity:
   - Evaluates Sobel spatial derivatives nabla d = (dd/dx, dd/dy) to capture
     surface tilt, boundary steps, and 3D physical relief.
3. Contextual Boundary Step Discontinuity:
   - Measures the depth difference Delta d_step between the candidate object interior
     and its immediate local contextual border. Planar fabric seams and shadows lie
     flat on the textile manifold (Delta d_step approx 0), whereas physical handheld
     weapons protrude in 3D space (Delta d_step >= tau_step).
4. Safety High-Confidence Bypass:
   - Very thin blades viewed edge-on may suffer spatial erosion in monocular depth maps;
     high-confidence proposals (conf >= tau_high = 0.70) bypass geometric step rejection.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEAPON_WEIGHTS = os.path.join(REPO_DIR, "weights", "yolo_weapon_distilled.pt")
DEFAULT_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthGeometricDetector:
    """
    Modular detector engine for Geometry-Informed Monocular Depth Pipeline (Version 4.40).
    Encapsulates Distilled YOLO proposals, Depth Anything V2 depth estimation,
    Sobel surface gradient computation, and boundary step discontinuity filtering.
    """

    def __init__(
        self,
        weapon_weights_path: str = DEFAULT_WEAPON_WEIGHTS,
        depth_model_id: str = DEFAULT_DEPTH_MODEL_ID,
        device: Optional[str] = None,
        weapon_conf_thresh: float = 0.20,
        min_step_thresh: float = 0.003,
        min_grad_thresh: float = 0.020,
        max_var_thresh: float = 0.035,
        high_conf_bypass: float = 0.70,
        context_pad_ratio: float = 0.25,
        use_amp: bool = True,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.weapon_conf_thresh = weapon_conf_thresh
        self.min_step_thresh = min_step_thresh
        self.min_grad_thresh = min_grad_thresh
        self.max_var_thresh = max_var_thresh
        self.high_conf_bypass = high_conf_bypass
        self.context_pad_ratio = context_pad_ratio
        self.use_amp = use_amp and ("cuda" in self.device)

        print(f"[Depth Geometric v4.40] Initializing Pipeline on {self.device}...")

        # 1. Load Weapon Detector
        if not os.path.exists(weapon_weights_path):
            raise FileNotFoundError(f"Weapon weights not found: {weapon_weights_path}")
        self.weapon_model = YOLO(weapon_weights_path)
        print(f"  Stream 1: Loaded Weapon Detector from {os.path.basename(weapon_weights_path)}")

        # 2. Load Depth Anything V2
        print(f"  Stream 2: Loading Depth Anything V2 ({depth_model_id})...")
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(self.device).eval()

        vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if "cuda" in self.device else 0
        print(f"[Depth Geometric v4.40] Initialization complete. Active VRAM footprint: {vram_mb:.2f} MB")

    def _compute_depth_map(self, pil_img: Image.Image, target_h: int, target_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes normalized dense monocular depth map and Sobel gradient magnitude field.
        """
        inputs = self.depth_processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.depth_model(**inputs)
            else:
                outputs = self.depth_model(**inputs)

            depth_logits = outputs.predicted_depth
            # Interpolate to target frame size
            depth_resized = F.interpolate(
                depth_logits.unsqueeze(1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            ).squeeze().cpu().numpy()

        # Min-max normalization to [0.0, 1.0]
        d_min, d_max = float(depth_resized.min()), float(depth_resized.max())
        d_norm = ((depth_resized - d_min) / (d_max - d_min + 1e-6)).astype(np.float32)

        # Compute Sobel gradients
        gx = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx**2 + gy**2)

        return d_norm, gmag

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        weapon_conf_thresh: Optional[float] = None,
        min_step_thresh: Optional[float] = None,
        min_grad_thresh: Optional[float] = None,
        max_var_thresh: Optional[float] = None,
    ) -> Dict:
        """
        Executes end-to-end depth geometric inference on a single surveillance frame.
        """
        t0 = time.perf_counter()

        w_conf = weapon_conf_thresh if weapon_conf_thresh is not None else self.weapon_conf_thresh
        step_thresh = min_step_thresh if min_step_thresh is not None else self.min_step_thresh
        grad_thresh = min_grad_thresh if min_grad_thresh is not None else self.min_grad_thresh
        var_thresh = max_var_thresh if max_var_thresh is not None else self.max_var_thresh

        # Load image
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found at: {image_input}")
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
            img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        h_img, w_img = img_bgr.shape[:2]

        # -------------------------------------------------------------
        # Stream 1: Candidate Weapon Proposals (Distilled YOLO)
        # -------------------------------------------------------------
        weapon_results = self.weapon_model(img_bgr, conf=w_conf, verbose=False)[0]
        raw_proposals = []
        for b in weapon_results.boxes:
            box_coords = b.xyxy[0].cpu().numpy().tolist() # [x1, y1, x2, y2]
            conf = float(b.conf[0])
            raw_proposals.append((box_coords, conf))

        # If no proposals, return empty early
        if not raw_proposals:
            return {
                "boxes": [],
                "scores": [],
                "geometric_metrics": [],
                "raw_proposal_count": 0,
                "suppressed_proposals": [],
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "image_size": (w_img, h_img)
            }

        # -------------------------------------------------------------
        # Stream 2: Monocular Depth & Surface Gradient Field
        # -------------------------------------------------------------
        d_norm, gmag = self._compute_depth_map(pil_img, h_img, w_img)

        # -------------------------------------------------------------
        # Geometric Surface Discontinuity Verification
        # -------------------------------------------------------------
        validated_boxes = []
        validated_scores = []
        validated_metrics = []
        suppressed_details = []

        for p_box, p_conf in raw_proposals:
            # High-confidence bypass
            if p_conf >= self.high_conf_bypass:
                validated_boxes.append(p_box)
                validated_scores.append(p_conf)
                validated_metrics.append({"status": "high_conf_bypass", "step": -1.0, "grad": -1.0, "var": -1.0})
                continue

            x1, y1 = max(0, int(p_box[0])), max(0, int(p_box[1]))
            x2, y2 = min(w_img, int(p_box[2])), min(h_img, int(p_box[3]))

            # Degenerate box guard
            if (x2 - x1) < 4 or (y2 - y1) < 4:
                suppressed_details.append({
                    "box": p_box, "conf": p_conf, "reason": "degenerate_box_dimensions"
                })
                continue

            crop_d = d_norm[y1:y2, x1:x2]
            crop_g = gmag[y1:y2, x1:x2]

            d_mean = float(np.mean(crop_d))
            d_var = float(np.var(crop_d))
            p90_g = float(np.percentile(crop_g, 90))

            # Contextual border step computation
            pad_x = max(6, int((x2 - x1) * self.context_pad_ratio))
            pad_y = max(6, int((y2 - y1) * self.context_pad_ratio))
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

            outer_rings = []
            if cy1 < y1: outer_rings.append(d_norm[cy1:y1, cx1:cx2].ravel())
            if cy2 > y2: outer_rings.append(d_norm[y2:cy2, cx1:cx2].ravel())
            if cx1 < x1: outer_rings.append(d_norm[y1:y2, cx1:x1].ravel())
            if cx2 > x2: outer_rings.append(d_norm[y1:y2, x2:cx2].ravel())

            if outer_rings:
                outer_vals = np.concatenate(outer_rings)
                step_discontinuity = abs(d_mean - float(np.mean(outer_vals)))
            else:
                step_discontinuity = 0.0

            geo_info = {
                "step": step_discontinuity,
                "grad_p90": p90_g,
                "depth_var": d_var,
                "depth_mean": d_mean
            }

            # Geometric Gating Decision:
            # 1. Planar Surface Check: Flat clothing seam / shadow has near-zero step and low gradient
            if step_discontinuity < step_thresh and p90_g < grad_thresh:
                suppressed_details.append({
                    "box": p_box, "conf": p_conf, "reason": "planar_seam_rejection", "metrics": geo_info
                })
                continue

            # 2. Clutter Variance Check: Scattered background clutter / wood piles have excessive depth variance
            if d_var > var_thresh:
                suppressed_details.append({
                    "box": p_box, "conf": p_conf, "reason": "clutter_variance_rejection", "metrics": geo_info
                })
                continue

            # Confirmed 3D entity
            validated_boxes.append(p_box)
            validated_scores.append(p_conf)
            validated_metrics.append({"status": "verified_3d_object", **geo_info})

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "boxes": validated_boxes,
            "scores": validated_scores,
            "geometric_metrics": validated_metrics,
            "raw_proposal_count": len(raw_proposals),
            "suppressed_proposals": suppressed_details,
            "latency_ms": latency_ms,
            "image_size": (w_img, h_img)
        }
