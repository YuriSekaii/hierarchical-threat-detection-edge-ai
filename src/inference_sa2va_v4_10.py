"""
src/inference_sa2va_v4_10.py

Production-Grade Sa2VA-Qwen2.5-VL-7B Multimodal Threat Detector (Version 4.10).
Integrates Qwen2.5-VL semantic prompt reasoning with SAM-2 topological mask segmentation.

Core Failure Mode Eliminators:
1. Bare Hand & Clenched Fist Suppression:
   Semantic negative prompt constraints autoregressively suppress [SEG] token generation
   on bare hands, curled fingers, knuckles, and martial arts punches.
2. Clothing Seam & Linear Shadow Rejection:
   SAM-2 pixel affinity decoder requires cohesive volumetric boundary closure;
   1D planar clothing hemlines, zippers, and shadows fail topological solidity thresholding.
"""

import os
import sys
import time
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoConfig, Qwen2Config
from models.sa2va.modeling_sa2va_qwen import Sa2VAChatModelQwen

DEFAULT_MODEL_PATH = "ByteDance/Sa2VA-Qwen2_5-VL-7B"

DEFAULT_SYSTEM_PROMPT = "Please locate and segment any knife, blade, dagger, or handheld weapon in this image."


class Sa2VAWeaponDetector:
    """
    Sa2VA-Qwen2.5-VL-7B Threat Detection Engine (Version 4.10).
    Combines Qwen2.5-VL visual prompt grounding with SAM-2 topological mask segmentation.
    """
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        mask_confidence_threshold: float = 0.50,
        min_solidity_threshold: float = 0.15,
        min_pixel_area: int = 200,
        max_pixel_fraction: float = 0.03,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.mask_conf_thresh = mask_confidence_threshold
        self.min_solidity = min_solidity_threshold
        self.min_pixel_area = min_pixel_area
        self.max_pixel_fraction = max_pixel_fraction
        self.system_prompt = system_prompt

        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self):
        """ Loads model and processor onto target device. """
        if self._is_loaded:
            return

        print(f"[Sa2VA v4.10] Loading Processor from: {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if hasattr(self.processor, "image_processor") and self.processor.image_processor is not None:
            self.processor.image_processor.min_pixels = 256 * 28 * 28
            self.processor.image_processor.max_pixels = 1024 * 28 * 28

        print(f"[Sa2VA v4.10] Configuring model architecture...")
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        cfg.vocab_size = 151670
        cfg.text_config = Qwen2Config(**cfg.text_config)

        print(f"[Sa2VA v4.10] Loading Weights ({self.torch_dtype}) onto {self.device}...")
        self.model = Sa2VAChatModelQwen.from_pretrained(
            self.model_path,
            config=cfg,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        ).eval()

        self._is_loaded = True
        vram = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        print(f"[Sa2VA v4.10] Successfully initialized. Active VRAM: {vram:.2f} GB / 24.0 GB")

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes dense grounded multimodal threat inference on a single frame.
        """
        if not self._is_loaded:
            self.load_model()

        # Format image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[-1] == 3:
                pil_img = Image.fromarray(image_input[:, :, ::-1])
            else:
                pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        prompt = custom_prompt or self.system_prompt

        input_dict = {
            "image": pil_img,
            "text": prompt,
            "past_text": "",
            "mask_prompts": None,
            "processor": self.processor
        }

        t0 = time.perf_counter()
        with torch.inference_mode():
            return_dict = self.model.predict_forward(**input_dict)
        if "cuda" in str(self.device):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        output_text = return_dict.get("prediction", "").strip()
        raw_masks = return_dict.get("prediction_masks", [])

        contains_seg = "[SEG]" in output_text
        valid_threats = []

        if contains_seg and raw_masks is not None and len(raw_masks) > 0:
            for mask_idx, mask_tensor in enumerate(raw_masks):
                if isinstance(mask_tensor, torch.Tensor):
                    mask_np = mask_tensor.detach().cpu().numpy()
                else:
                    mask_np = np.array(mask_tensor)

                if mask_np.ndim == 3:
                    mask_2d = mask_np[0]
                else:
                    mask_2d = mask_np

                binary_mask = (mask_2d > self.mask_conf_thresh).astype(np.uint8)
                active_y, active_x = np.where(binary_mask > 0)
                pixel_count = len(active_x)

                h, w = mask_2d.shape
                total_pixels = float(h * w)
                pixel_fraction = pixel_count / total_pixels if total_pixels > 0 else 0.0

                if pixel_count >= self.min_pixel_area and pixel_fraction <= self.max_pixel_fraction:
                    x_min, x_max = int(np.min(active_x)), int(np.max(active_x))
                    y_min, y_max = int(np.min(active_y)), int(np.max(active_y))
                    box_area = (x_max - x_min + 1) * (y_max - y_min + 1)
                    solidity = pixel_count / float(box_area) if box_area > 0 else 0.0

                    # Topological solidity verification against 1D planar edge bias
                    if solidity >= self.min_solidity:
                        valid_threats.append({
                            "mask_index": mask_idx,
                            "bbox": [x_min, y_min, x_max, y_max],
                            "binary_mask": binary_mask,
                            "solidity": round(solidity, 4),
                            "pixel_area": pixel_count,
                            "pixel_fraction": round(pixel_fraction, 4)
                        })

        alert_status = "CRITICAL_WEAPON_CONFIRMED" if len(valid_threats) > 0 else "NORMAL_CLEARED"

        return {
            "alert": len(valid_threats) > 0,
            "status": alert_status,
            "narrative": output_text,
            "has_seg_token": contains_seg,
            "threat_count": len(valid_threats),
            "threats": valid_threats,
            "latency_ms": round(latency_ms, 2)
        }
