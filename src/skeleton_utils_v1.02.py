"""
Kinematic Signal Processing and Scale-Invariant Normalization Utilities (v1.02).
Fixes bug where motion vectors were proportional to camera distance by dividing
centered coordinates by torso length ||mid_shoulder - mid_hip|| (with bbox diagonal fallback).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch


def interpolate_missing_joints(keypoints_sequence):
    """
    Linearly interpolates dropped/missing keypoints (0, 0) across the time dimension.
    
    Args:
        keypoints_sequence (np.ndarray): Shape (T, V, C) where T=time, V=joints, C=coords (x, y)
    Returns:
        np.ndarray: Cleaned keypoint array with continuous trajectories.
    """
    cleaned = keypoints_sequence.copy()
    T, V, C = cleaned.shape

    for v in range(V):
        for c in range(C):
            series = cleaned[:, v, c]
            missing = np.where(series == 0)[0]
            valid = np.where(series != 0)[0]

            if len(missing) > 0 and len(valid) > 1:
                series[missing] = np.interp(missing, valid, series[valid])
            elif len(missing) > 0 and len(valid) == 1:
                series[missing] = series[valid[0]]
            cleaned[:, v, c] = series

    return cleaned


def smooth_kinematics(keypoints_sequence, sigma=1.0):
    """
    Applies 1D Gaussian temporal smoothing to suppress high-frequency joint jitter.
    """
    smoothed = keypoints_sequence.copy()
    T, V, C = smoothed.shape

    for v in range(V):
        for c in range(C):
            if np.any(smoothed[:, v, c]):
                smoothed[:, v, c] = gaussian_filter1d(smoothed[:, v, c], sigma=sigma, mode='nearest')
    return smoothed


def normalize_skeleton_clip(keypoints_sequence, eps=1e-6):
    """
    Performs centroid centering AND torso-scale normalization (v1.02).
    
    1. Subtracts frame-level mean centroid across visible joints.
    2. Computes torso length L_torso = ||mid_shoulder - mid_hip|| using COCO indices:
       - Left Shoulder: 5, Right Shoulder: 6
       - Left Hip: 11, Right Hip: 12
    3. Divides coordinates by (L_torso + eps).
    4. Falls back to 0.5 * bbox_diagonal if shoulder or hip keypoints are occluded.
    
    Supports both NumPy ndarrays and PyTorch tensors.
    """
    if isinstance(keypoints_sequence, torch.Tensor):
        return _normalize_skeleton_clip_torch(keypoints_sequence, eps=eps)
    else:
        return _normalize_skeleton_clip_numpy(keypoints_sequence, eps=eps)


def _normalize_skeleton_clip_numpy(keypoints_sequence, eps=1e-6):
    normed = np.zeros_like(keypoints_sequence, dtype=np.float32)
    T = keypoints_sequence.shape[0]

    for t in range(T):
        frame_skel = keypoints_sequence[t]
        valid_mask = np.sum(np.abs(frame_skel), axis=-1) > 1e-6
        if not np.any(valid_mask):
            continue

        valid_pts = frame_skel[valid_mask]
        centroid = np.mean(valid_pts, axis=0)
        centered = frame_skel - centroid

        # Check shoulder keypoints (5: left_shoulder, 6: right_shoulder)
        has_l_sh = valid_mask[5] if len(valid_mask) > 5 else False
        has_r_sh = valid_mask[6] if len(valid_mask) > 6 else False
        # Check hip keypoints (11: left_hip, 12: right_hip)
        has_l_hip = valid_mask[11] if len(valid_mask) > 11 else False
        has_r_hip = valid_mask[12] if len(valid_mask) > 12 else False

        scale = None
        if (has_l_sh or has_r_sh) and (has_l_hip or has_r_hip):
            if has_l_sh and has_r_sh:
                mid_sh = (frame_skel[5] + frame_skel[6]) / 2.0
            elif has_l_sh:
                mid_sh = frame_skel[5]
            else:
                mid_sh = frame_skel[6]

            if has_l_hip and has_r_hip:
                mid_hip = (frame_skel[11] + frame_skel[12]) / 2.0
            elif has_l_hip:
                mid_hip = frame_skel[11]
            else:
                mid_hip = frame_skel[12]

            l_torso = np.linalg.norm(mid_sh - mid_hip)
            if l_torso > 1e-3:
                scale = l_torso

        # Fallback to bounding box diagonal
        if scale is None:
            bbox_min = np.min(valid_pts, axis=0)
            bbox_max = np.max(valid_pts, axis=0)
            bbox_diag = np.linalg.norm(bbox_max - bbox_min)
            if bbox_diag > 1e-3:
                scale = bbox_diag * 0.5
            else:
                scale = 1.0

        normed[t][valid_mask] = centered[valid_mask] / (scale + eps)

    return normed


def _normalize_skeleton_clip_torch(keypoints_sequence, eps=1e-6):
    normed = torch.zeros_like(keypoints_sequence)
    T = keypoints_sequence.shape[0]

    for t in range(T):
        frame_skel = keypoints_sequence[t]
        valid_mask = torch.sum(torch.abs(frame_skel), dim=-1) > 1e-6
        if not torch.any(valid_mask):
            continue

        valid_pts = frame_skel[valid_mask]
        centroid = torch.mean(valid_pts, dim=0)
        centered = frame_skel - centroid

        has_l_sh = bool(valid_mask[5]) if len(valid_mask) > 5 else False
        has_r_sh = bool(valid_mask[6]) if len(valid_mask) > 6 else False
        has_l_hip = bool(valid_mask[11]) if len(valid_mask) > 11 else False
        has_r_hip = bool(valid_mask[12]) if len(valid_mask) > 12 else False

        scale = None
        if (has_l_sh or has_r_sh) and (has_l_hip or has_r_hip):
            if has_l_sh and has_r_sh:
                mid_sh = (frame_skel[5] + frame_skel[6]) / 2.0
            elif has_l_sh:
                mid_sh = frame_skel[5]
            else:
                mid_sh = frame_skel[6]

            if has_l_hip and has_r_hip:
                mid_hip = (frame_skel[11] + frame_skel[12]) / 2.0
            elif has_l_hip:
                mid_hip = frame_skel[11]
            else:
                mid_hip = frame_skel[12]

            l_torso = torch.norm(mid_sh - mid_hip)
            if l_torso > 1e-3:
                scale = l_torso

        if scale is None:
            bbox_min, _ = torch.min(valid_pts, dim=0)
            bbox_max, _ = torch.max(valid_pts, dim=0)
            bbox_diag = torch.norm(bbox_max - bbox_min)
            if bbox_diag > 1e-3:
                scale = bbox_diag * 0.5
            else:
                scale = torch.tensor(1.0, device=frame_skel.device)

        normed[t][valid_mask] = centered[valid_mask] / (scale + eps)

    return normed
