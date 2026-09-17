"""
models/skeleton_ood.py

Skeleton-OOD: End-to-End Latent Boundary Learning for Skeletal Action Recognition.
Architecture combining:
1. 9-Block Spatio-Temporal Graph Convolutional Network (ST-GCN) backbone with arm-weighted attention (5x on arms).
2. In-training dynamic Activation Shaping (ASH-B / ASH-S) for penultimate feature denoising.
3. Attention-based Feature Fusion Block (SE-Channel Attention + MLP).
4. Unit-Hyperspherical Projection onto S^{D-1} with learnable class centers.
5. Free Energy scoring head for decision boundary gating.

Formulation:
- Xu et al., "Action-OOD: An End-to-End Skeleton-Based Model for Robust Out-of-Distribution Human Action Detection"
  (Neurocomputing 2025 / arXiv:2405.20633)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stgcn import STConvBlock, Graph


def activation_shaping(x, percentile=80, mode='ash_b'):
    """
    Applies dynamic percentile activation shaping to penultimate features.
    
    Args:
        x (torch.Tensor): Feature tensor (B, D).
        percentile (float): Percentile threshold (e.g. 80 keeps top 20% activations).
        mode (str): 'ash_b' (binarization), 'ash_s' (scaled pruning), 'ash_p' (pruning).
    Returns:
        torch.Tensor: Shaped activation tensor (B, D).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    D = x.size(-1)
    k_keep = max(1, int(round(D * (100.0 - percentile) / 100.0)))
    top_vals, _ = torch.topk(x, k_keep, dim=-1)
    threshold = top_vals[:, -1:]
    mask = (x >= threshold)
    
    if mode == 'ash_b':
        return mask.float()
    elif mode == 'ash_p':
        return x * mask.float()
    elif mode == 'ash_s':
        x_pruned = x * mask.float()
        s1 = torch.sum(torch.abs(x), dim=-1, keepdim=True)
        s2 = torch.sum(torch.abs(x_pruned), dim=-1, keepdim=True)
        scale = s1 / (s2 + 1e-8)
        return x_pruned * scale
    else:
        raise ValueError(f"Unknown activation shaping mode: {mode}")


class SqueezeExcitationBlock(nn.Module):
    """ Squeeze-and-Excitation (SE) Channel Attention Layer """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(16, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C)
        weights = self.fc(x)
        return x * weights


class FeatureFusionBlock(nn.Module):
    """
    Attention-based Feature Fusion Block (ASH-SE-MLP)
    Concatenates raw feature F and activated shaped feature ASH(F),
    applies Squeeze-and-Excitation channel attention, and projects back to D dimensions.
    """
    def __init__(self, in_features=256, out_features=256, reduction=16, dropout=0.1):
        super().__init__()
        self.concat_dim = in_features * 2
        self.se = SqueezeExcitationBlock(self.concat_dim, reduction=reduction)
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, out_features),
            nn.LayerNorm(out_features),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features)
        )

    def forward(self, f_raw, f_ash):
        f_cat = torch.cat([f_raw, f_ash], dim=-1) # (B, 512)
        f_se = self.se(f_cat)
        f_fuse = self.mlp(f_se) # (B, 256)
        return f_fuse


class SkeletonOODModel(nn.Module):
    """
    Skeleton-OOD End-to-End Model.
    Integrates the validated 9-block ST-GCN backbone with in-training Activation Shaping,
    SE Feature Fusion, Unit-Hyperspherical Projection (S^{D-1}), and Energy Gating Head.
    """
    def __init__(
        self,
        num_classes=3,
        in_channels=2,
        num_keypoints=17,
        dropout=0.1,
        ash_percentile=80,
        ash_mode='ash_b',
        temperature=1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ash_percentile = ash_percentile
        self.ash_mode = ash_mode
        self.temperature = temperature

        # Graph structure
        self.graph = Graph()
        self.register_buffer('A', self.graph.A)
        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)

        # 9 ST-GCN convolutional blocks (3x64 -> 3x128 -> 3x256)
        self.st_gcn_networks = nn.ModuleList((
            STConvBlock(in_channels, 64, kernel_size, stride=1, residual=False, dropout=dropout),
            STConvBlock(64, 64, kernel_size, stride=1, dropout=dropout),
            STConvBlock(64, 64, kernel_size, stride=1, dropout=dropout),
            STConvBlock(64, 128, kernel_size, stride=2, dropout=dropout),
            STConvBlock(128, 128, kernel_size, stride=1, dropout=dropout),
            STConvBlock(128, 128, kernel_size, stride=1, dropout=dropout),
            STConvBlock(128, 256, kernel_size, stride=2, dropout=dropout),
            STConvBlock(256, 256, kernel_size, stride=1, dropout=dropout),
            STConvBlock(256, 256, kernel_size, stride=1, dropout=dropout),
        ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Joint-weighted spatial attention: arms (5, 6, 7, 8, 9, 10) get 5x weight
        jw = torch.ones(num_keypoints)
        jw[[5, 6, 7, 8, 9, 10]] = 5.0
        jw = jw / jw.sum() * num_keypoints
        self.register_buffer('joint_weights', jw)

        # Attention-based Feature Fusion Block
        self.feature_fusion = FeatureFusionBlock(
            in_features=256,
            out_features=256,
            reduction=16,
            dropout=dropout
        )

        # Classification Head on Hypersphere
        self.classifier = nn.Linear(256, num_classes, bias=False)

        # Learnable Class Centers on S^{D-1}
        self.class_centers = nn.Parameter(torch.randn(num_classes, 256))
        nn.init.orthogonal_(self.class_centers)

    def extract_backbone_features(self, x):
        """ Runs 9-block ST-GCN backbone with arm weighting """
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N * M, C, V, T)
        x = x.view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()

        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        # Apply joint-weighted spatial attention before pooling
        x = x * self.joint_weights.view(1, 1, 1, -1)
        x = self.pool(x)
        x = x.view(N, M, -1)
        features = x.squeeze(1) if M == 1 else x.mean(dim=1)
        return features # (N, 256)

    def forward(self, x, return_features=False, return_energy=False):
        """
        Forward pass with in-training Activation Shaping, Feature Fusion,
        Hyperspherical Normalization, and Energy computation.
        """
        # 1. Extract raw penultimate features
        f_raw = self.extract_backbone_features(x) # (B, 256)

        # 2. Dynamic Activation Shaping (ASH)
        f_ash = activation_shaping(f_raw, percentile=self.ash_percentile, mode=self.ash_mode)

        # 3. Attention-based Feature Fusion
        f_fuse = self.feature_fusion(f_raw, f_ash) # (B, 256)

        # 4. Project onto Unit Hypersphere S^{D-1}
        z = F.normalize(f_fuse, p=2, dim=-1) # (B, 256)

        # 5. Calibrated Logits
        logits = self.classifier(z) # (B, num_classes)

        # 6. Free Energy Score: E(x) = -T * log(sum(exp(logits / T)))
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)

        if return_features and return_energy:
            return logits, z, energy
        if return_features:
            return logits, z
        if return_energy:
            return logits, energy
        return logits

    def compute_energy_from_features(self, z):
        """ Computes energy directly from unit-normalized features z """
        logits = self.classifier(z)
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)
