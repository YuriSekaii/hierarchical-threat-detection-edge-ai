"""
Kinematic Signal Processing and Normalization Utilities for Skeleton Sequences.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def normalize_skeleton_clip(keypoints_sequence):
    """
    Performs centroid centering to ensure translation and scale invariance.
    Subtracts the frame-level mean coordinate across visible joints.
    """
    normed = np.zeros_like(keypoints_sequence)
    for t in range(keypoints_sequence.shape[0]):
        frame_skel = keypoints_sequence[t]
        mask = np.sum(np.abs(frame_skel), axis=1) > 1e-6
        pts = frame_skel[mask]
        if pts.shape[0] > 0:
            normed[t][mask] = pts - np.mean(pts, axis=0)
    return normed
