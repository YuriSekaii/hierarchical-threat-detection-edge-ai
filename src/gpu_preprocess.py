"""
src/gpu_preprocess.py

Portable CUDA letterbox + BGR->RGB + normalize preprocessing and exact inverse
coordinate rescaling for Stage 1 (YOLO26s) and Stage 2a (YOLO26s-pose).
Uses pure PyTorch CUDA ops (no custom C++/CUDA extensions), portable across
RTX 3090, RTX 3050, and Jetson Orin Nano.
"""

from typing import List, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

LetterboxMeta = Tuple[float, float, float, int, int]  # (ratio, pad_x, pad_y, orig_h, orig_w)


def _normalize_imgsz(imgsz: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(imgsz, int):
        return int(imgsz), int(imgsz)
    return int(imgsz[0]), int(imgsz[1])


def letterbox_bgr_to_tensor(
    frame: np.ndarray,
    imgsz: Union[int, Tuple[int, int]] = (384, 640),
    device: Union[str, torch.device] = "cuda:0",
) -> Tuple[torch.Tensor, LetterboxMeta]:
    """
    Upload uint8 BGR HWC frame to GPU, convert BGR->RGB, resize with exact
    Ultralytics LetterBox geometry, pad with 114/255, and normalize to [0, 1] float32.

    Returns:
        (tensor (1, 3, target_h, target_w) float32 on device,
         meta (r, pad_x, pad_y, orig_h, orig_w))
    """
    orig_h, orig_w = frame.shape[:2]
    target_h, target_w = _normalize_imgsz(imgsz)

    r = min(target_h / orig_h, target_w / orig_w)
    new_unpad_w = int(round(orig_w * r))
    new_unpad_h = int(round(orig_h * r))
    dw = (target_w - new_unpad_w) / 2.0
    dh = (target_h - new_unpad_h) / 2.0
    top = max(0, int(round(dh - 0.1)))
    left = max(0, int(round(dw - 0.1)))

    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)

    t = torch.from_numpy(frame).to(device, non_blocking=True)
    t = t.permute(2, 0, 1).unsqueeze(0).flip(1).float()

    if orig_h != new_unpad_h or orig_w != new_unpad_w:
        t = F.interpolate(t, size=(new_unpad_h, new_unpad_w), mode="bilinear", align_corners=False)

    out = torch.full((1, 3, target_h, target_w), 114.0 / 255.0, dtype=torch.float32, device=device)
    out[:, :, top : top + new_unpad_h, left : left + new_unpad_w] = t * (1.0 / 255.0)
    return out, (r, float(left), float(top), orig_h, orig_w)


def unletterbox_boxes(
    boxes_xyxy: Sequence[Sequence[float]],
    meta: LetterboxMeta,
) -> List[List[float]]:
    """Rescale [x1, y1, x2, y2] boxes from letterbox pixel space back to original (orig_h, orig_w)."""
    if not boxes_xyxy:
        return []
    r, pad_x, pad_y, orig_h, orig_w = meta
    inv_r = 1.0 / r
    out: List[List[float]] = []
    for box in boxes_xyxy:
        x1 = min(max((float(box[0]) - pad_x) * inv_r, 0.0), float(orig_w))
        y1 = min(max((float(box[1]) - pad_y) * inv_r, 0.0), float(orig_h))
        x2 = min(max((float(box[2]) - pad_x) * inv_r, 0.0), float(orig_w))
        y2 = min(max((float(box[3]) - pad_y) * inv_r, 0.0), float(orig_h))
        out.append([x1, y1, x2, y2])
    return out


def unletterbox_keypoints(
    kpts_xy: np.ndarray,
    meta: LetterboxMeta,
) -> np.ndarray:
    """
    Rescale (..., 2) keypoints from letterbox pixel space back to original (orig_h, orig_w),
    preserving exact (0, 0) missing COCO keypoints.
    """
    arr = np.asarray(kpts_xy, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()
    r, pad_x, pad_y, orig_h, orig_w = meta
    missing = (arr[..., 0] == 0.0) & (arr[..., 1] == 0.0)
    res = arr.copy()
    inv_r = 1.0 / r
    res[..., 0] = np.clip((res[..., 0] - pad_x) * inv_r, 0.0, float(orig_w))
    res[..., 1] = np.clip((res[..., 1] - pad_y) * inv_r, 0.0, float(orig_h))
    res[missing] = 0.0
    return res
