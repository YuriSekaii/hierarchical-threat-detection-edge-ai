"""
models/ctrgcn_scaled.py

Scaled Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN)
for Dynamic Spatio-Temporal Violent Action Recognition (Task 1 Architectural Scaling).

Supports:
1. Tier S (Inductive Regularization): channels=[32, 64, 128], ~335k parameters (-74.8% vs baseline)
2. Tier M (Baseline): channels=[64, 128, 256], ~1.33M parameters
3. Tier L (High Capacity): channels=[96, 192, 384], ~3.00M parameters
4. Configurable arm attention weight (arm_weight: multiplier for joints 5-10)
5. Configurable feature projection dimension for RMD manifold estimation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stgcn import Graph


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolutional Block (MS-TCN).
    Decomposes temporal convolutions across 4 multi-scale dilated receptive branches.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2], dropout=0.1):
        super().__init__()
        assert out_channels % 4 == 0, f"out_channels ({out_channels}) must be divisible by 4"
        branch_channels = out_channels // 4

        # Branch 1: 1x1 instantaneous projection
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        )

        # Branch 2: 3x1 conv, dilation 1
        pad_d1 = (kernel_size - 1) * dilations[0] // 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(kernel_size, 1),
                      stride=(stride, 1), padding=(pad_d1, 0), dilation=(dilations[0], 1)),
            nn.BatchNorm2d(branch_channels)
        )

        # Branch 3: 3x1 conv, dilation 2
        pad_d2 = (kernel_size - 1) * dilations[1] // 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(kernel_size, 1),
                      stride=(stride, 1), padding=(pad_d2, 0), dilation=(dilations[1], 1)),
            nn.BatchNorm2d(branch_channels)
        )

        # Branch 4: 3x1 MaxPool + 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels)
        )

        # Residual branch
        if (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = self.residual(x)
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
        out = self.dropout(out)
        return self.relu(out + res)


class CTRGraphConv(nn.Module):
    """
    Channel-wise Topology Refinement Graph Convolution (CTR-GC).
    Dynamically refines adjacency matrix per channel group conditioned on motion input.
    """
    def __init__(self, in_channels, out_channels, A, num_sub_adj=3, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sub_adj = num_sub_adj
        self.num_node = A.size(1)
        self.reduction = max(1, reduction)
        self.inter_channels = max(8, in_channels // self.reduction)

        # Physical anatomical graph (buffer)
        self.register_buffer('A', A)  # (num_sub_adj, V, V)

        # Learnable static graph prior
        self.B = nn.Parameter(torch.zeros(num_sub_adj, self.num_node, self.num_node))

        # Channel refinement transformations
        self.conv_down = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_q = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Main convolutions
        self.conv_in = nn.Conv2d(in_channels, out_channels * num_sub_adj, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Initialization
        conv_init(self.conv_down)
        conv_init(self.conv_q)
        conv_init(self.conv_k)
        conv_init(self.conv_in)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # 1. Static + Learnable Adjacency
        A_static = self.A + self.B  # (K, V, W)

        # 2. Dynamic Channel-Wise Topology Refinement
        x_pool = x.mean(dim=2, keepdim=True)  # (N, C, 1, V)
        x_inter = self.conv_down(x_pool)  # (N, C', 1, V)
        q = self.conv_q(x_inter).squeeze(2)  # (N, C', V)
        k = self.conv_k(x_inter).squeeze(2)  # (N, C', V)

        # Pairwise joint channel correlation
        diff = q.unsqueeze(-1) - k.unsqueeze(-2)  # (N, C', V, V)
        dyn_adj = torch.tanh(diff)

        # 3. Graph Convolution
        x_proj = self.conv_in(x)  # (N, K*C_out, T, V)
        x_proj = x_proj.view(N, self.num_sub_adj, self.out_channels, T, V)

        # Static + Learnable Graph aggregation
        out_stat = torch.einsum('nkctv,kvw->nctw', x_proj, A_static)

        # Dynamic Graph modulation
        dyn_adj_mean = dyn_adj.mean(dim=1)  # (N, V, V)
        out_dyn = torch.einsum('nctv,nvw->nctw', out_stat, dyn_adj_mean)

        out = out_stat + self.alpha * out_dyn
        out = self.bn(out)
        return self.relu(out)


class CTRConvBlock(nn.Module):
    """
    Unified Spatio-Temporal CTR-GCN Block.
    Pairs CTR-GC Spatial Convolution with MS-TCN Multi-Scale Temporal Convolution.
    """
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.1, reduction=8):
        super().__init__()
        self.gcn = CTRGraphConv(in_channels, out_channels, A, reduction=reduction)
        self.tcn = MultiScaleTemporalConv(out_channels, out_channels, stride=stride, dropout=dropout)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return x + res


class ScaledCTRGCNModel(nn.Module):
    """
    Parameterized CTR-GCN Model supporting arbitrary channel configurations.
    Default configurations:
    - Tier S: channels=[32, 64, 128], reduction=4 (~335k params)
    - Tier M: channels=[64, 128, 256], reduction=8 (~1.33M params)
    - Tier L: channels=[96, 192, 384], reduction=8 (~3.00M params)
    """
    def __init__(
        self,
        num_classes=1,
        in_channels=2,
        num_keypoints=17,
        channels=[64, 128, 256],
        reduction=8,
        arm_weight=5.0,
        feature_dim=None,
        dropout=0.1
    ):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.arm_weight = arm_weight
        c1, c2, c3 = channels

        self.graph = Graph()
        self.register_buffer('A', self.graph.A)
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)

        # 9 CTR-GCN blocks (3x c1 -> 3x c2 -> 3x c3)
        self.blocks = nn.ModuleList([
            CTRConvBlock(in_channels, c1, self.A, stride=1, residual=False, dropout=dropout, reduction=reduction),
            CTRConvBlock(c1, c1, self.A, stride=1, dropout=dropout, reduction=reduction),
            CTRConvBlock(c1, c1, self.A, stride=1, dropout=dropout, reduction=reduction),
            CTRConvBlock(c1, c2, self.A, stride=2, dropout=dropout, reduction=reduction),
            CTRConvBlock(c2, c2, self.A, stride=1, dropout=dropout, reduction=reduction),
            CTRConvBlock(c2, c2, self.A, stride=1, dropout=dropout, reduction=reduction),
            CTRConvBlock(c2, c3, self.A, stride=2, dropout=dropout, reduction=reduction),
            CTRConvBlock(c3, c3, self.A, stride=1, dropout=dropout, reduction=reduction),
            CTRConvBlock(c3, c3, self.A, stride=1, dropout=dropout, reduction=reduction),
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Optional feature projection
        if feature_dim is not None and feature_dim != c3:
            self.proj = nn.Linear(c3, feature_dim)
            self.final_feature_dim = feature_dim
        else:
            self.proj = None
            self.final_feature_dim = c3

        self.fc = nn.Linear(self.final_feature_dim, num_classes)

        # Joint-weighted spatial attention: arms (5, 6, 7, 8, 9, 10) get arm_weight multiplier
        jw = torch.ones(num_keypoints)
        jw[[5, 6, 7, 8, 9, 10]] = float(arm_weight)
        jw = jw / jw.sum() * num_keypoints
        self.register_buffer('joint_weights', jw)

    def set_arm_weight(self, new_arm_weight):
        """Allows dynamic adjustment of arm weighting multiplier."""
        self.arm_weight = float(new_arm_weight)
        jw = torch.ones(len(self.joint_weights), device=self.joint_weights.device)
        jw[[5, 6, 7, 8, 9, 10]] = float(new_arm_weight)
        jw = jw / jw.sum() * len(self.joint_weights)
        self.joint_weights.copy_(jw)

    def forward(self, x, return_features=False):
        # Input shape: (N, C, T, V, M=1) or (N, C, T, V)
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N * M, C, V, T)
        x = x.view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()

        # Pass through 9 CTR-GCN blocks
        for block in self.blocks:
            x = block(x)

        # Apply arm-weighted spatial attention before pooling
        x = x * self.joint_weights.view(1, 1, 1, -1)
        x = self.pool(x)
        x = x.view(N, M, -1)
        features = x.squeeze(1) if M == 1 else x.mean(dim=1)

        if self.proj is not None:
            features = self.proj(features)

        logits = self.fc(features)

        if return_features:
            return logits, features
        return logits
