"""
PoseConv3D (PoseC3D) Volumetric Heatmap Rasterization Utilities (Version 2.10).
Converts 2D skeletal keypoint trajectories into compact 3D spatio-temporal heatmap volumes:
(K x T x H x W), supporting torso-scale kinematic normalization, aspect-ratio preservation,
limb-crossing robustness, and seamless multi-person channel-wise maximum aggregation.

Zero dependency on legacy milestone scripts.
"""

import numpy as np
import torch
import torch.nn.functional as F
from src.skeleton_utils import normalize_skeleton_clip


def rasterize_keypoints_to_heatmap(
    keypoints,
    H=56,
    W=56,
    sigma=1.5,
    norm_range=3.0,
    device="cpu"
):
    """
    Vectorized conversion of skeletal keypoint trajectories into a 3D heatmap volume.
    Applies torso-scale normalization so human posture is canonical and scale-invariant,
    preventing walking drift from distorting civilian movements.
    
    Args:
        keypoints: (T, K, 2) or (M, T, K, 2) array/tensor of 2D joint coordinates.
        H, W: Spatial dimensions of each heatmap slice (default 56x56).
        sigma: Standard deviation of Gaussian kernel in pixel units.
        norm_range: Canonical coordinate boundary (default [-3.0, 3.0] -> [0.0, 1.0]).
        device: Target PyTorch device ("cpu" or "cuda").
        
    Returns:
        torch.Tensor of shape (K, T, H, W) in float32.
    """
    if isinstance(keypoints, torch.Tensor):
        kpts_tensor = keypoints.detach().clone().float()
    else:
        kpts_tensor = torch.from_numpy(np.array(keypoints, dtype=np.float32))

    if kpts_tensor.ndim == 3:
        # (T, K, 2) -> (1, T, K, 2)
        kpts_tensor = kpts_tensor.unsqueeze(0)

    M, T, K, _ = kpts_tensor.shape

    actor_heatmaps = []
    y_grid = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
    x_grid = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W)
    s2 = 2.0 * ((sigma / float(W)) ** 2)

    for m in range(M):
        actor_kpts = kpts_tensor[m]  # (T, K, 2)
        valid = torch.sum(torch.abs(actor_kpts), dim=-1) > 1e-4  # (T, K)

        # 1. Torso-scale normalization (removes distance & walking drift)
        normed = normalize_skeleton_clip(actor_kpts)  # (T, K, 2)

        # 2. Canonical mapping: [-norm_range, +norm_range] -> [0.0, 1.0]
        mapped_x = torch.clamp((normed[..., 0] + norm_range) / (2.0 * norm_range), 0.0, 1.0).to(device)
        mapped_y = torch.clamp((normed[..., 1] + norm_range) / (2.0 * norm_range), 0.0, 1.0).to(device)
        vm = valid.float().to(device).permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # (K, T, 1, 1)

        kx = mapped_x.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # (K, T, 1, 1)
        ky = mapped_y.permute(1, 0).unsqueeze(-1).unsqueeze(-1)

        dist_sq = (x_grid - kx) ** 2 + (y_grid - ky) ** 2
        hm = torch.exp(-dist_sq / s2) * vm  # (K, T, H, W)
        actor_heatmaps.append(hm)

    if M > 1:
        aggregated_hm = torch.stack(actor_heatmaps, dim=0).max(dim=0).values
    else:
        aggregated_hm = actor_heatmaps[0]

    return aggregated_hm


def batch_rasterize_gpu(kpts_tensor, H=56, W=56, sigma=1.5, norm_range=3.0):
    """
    High-performance GPU-vectorized batch conversion of skeletal keypoint trajectories
    into 3D heatmap volumes with canonical torso-scale normalization.
    
    Args:
        kpts_tensor: torch.Tensor of shape (B, T, K, 2) on CUDA.
        H, W: Spatial dimensions (default 56x56).
        sigma: Gaussian sigma (default 1.5).
        norm_range: Canonical coordinate boundary (default 3.0).
        
    Returns:
        torch.Tensor of shape (B, K, T, H, W) on CUDA.
    """
    B, T, K, _ = kpts_tensor.shape
    device = kpts_tensor.device

    # 1. Normalize batch of skeletons
    normed_batch = []
    for b in range(B):
        normed_batch.append(normalize_skeleton_clip(kpts_tensor[b]))
    normed = torch.stack(normed_batch, dim=0)  # (B, T, K, 2)

    valid = (torch.sum(torch.abs(kpts_tensor), dim=-1) > 1e-4).to(device)  # (B, T, K)

    # 2. Canonical mapping to [0, 1]
    mapped_x = torch.clamp((normed[..., 0] + norm_range) / (2.0 * norm_range), 0.0, 1.0).to(device)
    mapped_y = torch.clamp((normed[..., 1] + norm_range) / (2.0 * norm_range), 0.0, 1.0).to(device)

    kx = mapped_x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, K, T, 1, 1)
    ky = mapped_y.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    vm = valid.float().permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, K, T, 1, 1)

    y_grid = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, 1, H, 1)
    x_grid = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, 1, W)

    s2 = 2.0 * ((sigma / float(W)) ** 2)
    dist_sq = (x_grid - kx) ** 2 + (y_grid - ky) ** 2
    hm = torch.exp(-dist_sq / s2) * vm  # (B, K, T, H, W)
    return hm
