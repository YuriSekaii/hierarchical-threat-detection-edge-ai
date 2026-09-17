"""
src/inference_grounding_dino_v4_20.py

Version 4.20: Grounding DINO 1.5 Modular Inference Engine
Implements open-vocabulary zero-shot threat object detection using multi-scale
deformable attention and cross-modal feature fusion.

Key Innovations:
1. Non-autoregressive parallel evaluation: Eliminates MLLM sequential token generation
   bottleneck (bypassing Sa2VA's 1.9s per-frame latency wall).
2. Competing Negative Prompt Suppression: Injects negative civilian confounder tokens
   ("bare hand", "clenched fist", "clothing seam", "zipper", "shadow") directly into the
   contrastive text embedding space. A candidate detection is rejected if its latent feature
   affinity with negative tokens exceeds or competes with positive weapon tokens:
       S_threat >= box_threshold AND S_threat >= S_neg + margin
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.ops
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_PROMPT = "a knife . a blade . a dagger . a cutter . a bare hand . a clenched fist . a clothing seam . a zipper . a shadow ."

DEFAULT_THREAT_KEYWORDS = ["knife", "blade", "dagger", "cutter"]
DEFAULT_NEGATIVE_KEYWORDS = ["hand", "fist", "seam", "zipper", "shadow"]


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert center-x, center-y, width, height to x1, y1, x2, y2 format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class GroundingDinoDetector:
    """
    Modular detector engine for Grounding DINO (Version 4.20).
    Encapsulates model weights, text prompt parsing, token-to-query matching,
    and competing negative token suppression.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        default_prompt: str = DEFAULT_PROMPT,
        threat_keywords: Optional[List[str]] = None,
        negative_keywords: Optional[List[str]] = None,
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        neg_margin: float = 0.0,
        nms_threshold: Optional[float] = 0.50,
        use_autocast: bool = True,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.default_prompt = default_prompt
        self.threat_keywords = threat_keywords or DEFAULT_THREAT_KEYWORDS
        self.negative_keywords = negative_keywords or DEFAULT_NEGATIVE_KEYWORDS
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.neg_margin = neg_margin
        self.nms_threshold = nms_threshold
        self.use_autocast = use_autocast and ("cuda" in self.device)

        print(f"[GroundingDINO v4.20] Loading processor & model ({model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()

        # Cache token mappings for the default prompt
        self._cached_prompt = None
        self._cached_threat_indices = []
        self._cached_neg_indices = []
        self._cached_token_strs = []
        self._update_prompt_cache(self.default_prompt)

        vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if "cuda" in self.device else 0
        print(f"[GroundingDINO v4.20] Model loaded successfully. VRAM footprint: {vram_mb:.2f} MB")

    def _update_prompt_cache(self, prompt: str):
        """Pre-computes token index boundaries for positive and negative prompt keywords."""
        if prompt == self._cached_prompt:
            return

        tokenizer = self.processor.tokenizer
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding.input_ids[0]
        token_strs = tokenizer.convert_ids_to_tokens(input_ids)

        threat_indices = []
        neg_indices = []

        for idx, tok in enumerate(token_strs):
            cleaned = tok.replace("##", "").lower()
            if any(tw in cleaned for tw in self.threat_keywords):
                threat_indices.append(idx)
            elif any(nw in cleaned for nw in self.negative_keywords):
                neg_indices.append(idx)

        self._cached_prompt = prompt
        self._cached_threat_indices = threat_indices
        self._cached_neg_indices = neg_indices
        self._cached_token_strs = token_strs

    def detect(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        neg_margin: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        enable_negative_suppression: bool = True,
    ) -> Dict:
        """
        Runs Grounding DINO detection on an image.

        Args:
            image: Image file path, PIL Image, or RGB numpy array.
            prompt: Text prompt string. Defaults to self.default_prompt.
            box_threshold: Minimum confidence for box acceptance.
            text_threshold: Minimum token affinity for phrase assignment.
            neg_margin: Margin required for positive score over negative score.
            nms_threshold: IoU threshold for NMS deduplication.
            enable_negative_suppression: If True, suppresses boxes where S_neg >= S_threat.

        Returns:
            Dictionary containing:
                - 'alert_status': 'THREAT_CONFIRMED' or 'CLEARED_SAFE'
                - 'threat_detected': bool
                - 'boxes_xyxy': list of [x1, y1, x2, y2] in pixel coordinates
                - 'boxes_xywhn': list of [cx, cy, w, h] normalized to [0, 1]
                - 'scores': list of confidence floats
                - 'labels': list of winning token label strings
                - 'suppressed_boxes_count': count of boxes rejected by negative competition
                - 'raw_boxes_count': count of boxes prior to negative suppression
                - 'latency_ms': inference time in milliseconds
                - 'image_size': (width, height)
        """
        # Load and convert image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        w_orig, h_orig = pil_img.size
        active_prompt = prompt or self.default_prompt
        self._update_prompt_cache(active_prompt)

        b_thresh = box_threshold if box_threshold is not None else self.box_threshold
        t_thresh = text_threshold if text_threshold is not None else self.text_threshold
        margin = neg_margin if neg_margin is not None else self.neg_margin

        # Preprocess image and text inputs
        inputs = self.processor(images=pil_img, text=active_prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            if self.use_autocast:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            if "cuda" in self.device:
                torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # outputs.logits shape: [1, num_queries, seq_len]
        # outputs.pred_boxes shape: [1, num_queries, 4] (cx, cy, w, h in [0, 1])
        logits = outputs.logits[0]
        pred_boxes = outputs.pred_boxes[0]

        probs = torch.sigmoid(logits)  # [num_queries, seq_len]

        threat_idxs = self._cached_threat_indices
        neg_idxs = self._cached_neg_indices

        if threat_idxs:
            threat_scores, threat_sub_idx = probs[:, threat_idxs].max(dim=-1)
            threat_token_ids = [threat_idxs[i.item()] for i in threat_sub_idx]
        else:
            threat_scores = torch.zeros(logits.shape[0], device=self.device)
            threat_token_ids = [0] * logits.shape[0]

        if neg_idxs:
            neg_scores, _ = probs[:, neg_idxs].max(dim=-1)
        else:
            neg_scores = torch.zeros(logits.shape[0], device=self.device)

        # Candidate selection: queries passing box_threshold
        candidate_mask = (threat_scores >= b_thresh)
        raw_candidates_count = int(candidate_mask.sum().item())

        suppressed_count = 0
        final_boxes_xyxy = []
        final_boxes_xywhn = []
        final_scores = []
        final_labels = []

        if raw_candidates_count > 0:
            candidate_indices = torch.where(candidate_mask)[0]

            for i, q_idx in enumerate(candidate_indices):
                s_threat = threat_scores[q_idx].item()
                s_neg = neg_scores[q_idx].item()

                if enable_negative_suppression and (s_threat < (s_neg + margin)):
                    suppressed_count += 1
                    continue

                # Unnormalize box to pixel coordinates
                cx, cy, w_box, h_box = pred_boxes[q_idx].tolist()
                x1 = max(0.0, (cx - w_box / 2.0) * w_orig)
                y1 = max(0.0, (cy - h_box / 2.0) * h_orig)
                x2 = min(float(w_orig), (cx + w_box / 2.0) * w_orig)
                y2 = min(float(h_orig), (cy + h_box / 2.0) * h_orig)

                # Clamp & sanity check
                if (x2 <= x1) or (y2 <= y1):
                    continue

                tok_id = threat_token_ids[q_idx]
                label_name = self._cached_token_strs[tok_id].replace("##", "")

                final_boxes_xyxy.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                final_boxes_xywhn.append([round(cx, 6), round(cy, 6), round(w_box, 6), round(h_box, 6)])
                final_scores.append(round(s_threat, 4))
                final_labels.append(label_name)

        # Apply NMS deduplication if requested
        nms_iou = nms_threshold if nms_threshold is not None else self.nms_threshold
        if nms_iou is not None and len(final_boxes_xyxy) > 1:
            boxes_t = torch.tensor(final_boxes_xyxy, dtype=torch.float32)
            scores_t = torch.tensor(final_scores, dtype=torch.float32)
            keep_indices = torchvision.ops.nms(boxes_t, scores_t, nms_iou).tolist()
            final_boxes_xyxy = [final_boxes_xyxy[i] for i in keep_indices]
            final_boxes_xywhn = [final_boxes_xywhn[i] for i in keep_indices]
            final_scores = [final_scores[i] for i in keep_indices]
            final_labels = [final_labels[i] for i in keep_indices]

        has_threat = len(final_boxes_xyxy) > 0
        status = "THREAT_CONFIRMED" if has_threat else "CLEARED_SAFE"

        return {
            "alert_status": status,
            "threat_detected": has_threat,
            "boxes_xyxy": final_boxes_xyxy,
            "boxes_xywhn": final_boxes_xywhn,
            "scores": final_scores,
            "labels": final_labels,
            "suppressed_boxes_count": suppressed_count,
            "raw_boxes_count": raw_candidates_count,
            "latency_ms": round(latency_ms, 2),
            "image_size": (w_orig, h_orig),
        }


if __name__ == "__main__":
    detector = GroundingDinoDetector()
    print("Detector initialized successfully.")
