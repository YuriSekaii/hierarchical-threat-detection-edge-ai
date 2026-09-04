"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Skeletal Action Recognition.
Architecture based on Yan et al. (AAAI 2018) with spatial configuration partitioning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph:
    """
    Defines the anatomical graph structure based on COCO 17 keypoint connections.
    Uses Spatial Configuration Partitioning (Identity, Inward, Outward adjacency).
    """
    def __init__(self, layout='coco', strategy='spatial'):
        self.num_node = 17  # COCO 17 keypoints
        self.self_link = [(i, i) for i in range(self.num_node)]

        # Anatomical connections for COCO layout
        self.neighbor_link_coco = [
            (0, 1), (0, 2), (1, 3), (2, 4),                     # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),           # Torso-Arms
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (5, 11), (6, 12)                                    # Shoulders to hips
        ]
        self.edge = self.self_link + self.neighbor_link_coco
        self.center = 1  # Neck/head center approximation
        self.A = self.get_adjacency_matrix(strategy)

    def get_adjacency_matrix(self, strategy):
        A = np.zeros((3, self.num_node, self.num_node), dtype=np.float32)
        if strategy == 'spatial':
            # Sub-graph 0: Self-connections (Identity)
            for i in range(self.num_node):
                A[0, i, i] = 1

            # Sub-graph 1: Inward (towards center) & Sub-graph 2: Outward (away from center)
            for i, j in self.neighbor_link_coco:
                A[1, i, j] = 1
                A[2, j, i] = 1

            # Normalize adjacency matrices
            for k in range(3):
                row_sum = A[k].sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1.0
                A[k] = A[k] / row_sum
        return A


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer operating over partitioned adjacency matrices.
    """
    def __init__(self, in_channels, out_channels, num_sub_adj=3):
        super().__init__()
        self.num_sub_adj = num_sub_adj
        self.conv = nn.Conv2d(in_channels, out_channels * num_sub_adj, kernel_size=1)

    def forward(self, x, A):
        # x shape: (N, C, T, V)
        N, C, T, V = x.size()
        x_conv = self.conv(x)  # (N, out_channels * num_sub_adj, T, V)
        x_conv = x_conv.view(N, self.num_sub_adj, -1, T, V)

        # Apply graph propagation along vertex dimension V
        # A shape: (num_sub_adj, V, V)
        out = torch.einsum('nkctv,kvw->nctw', x_conv, A)
        return out.contiguous()


class STConvBlock(nn.Module):
    """
    Spatial-Temporal Convolution Block: Spatial Graph Conv followed by 1D Temporal Conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(9, 1), stride=1, dropout=0.0, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=((kernel_size[0] - 1) // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNModel(nn.Module):
    """
    Complete ST-GCN Network for Biomechanical Action Recognition.
    Extracts deep kinematic representations across skeleton sequences.
    """
    def __init__(self, num_classes=1, in_channels=2, num_keypoints=17, graph_layout='coco', dropout=0.2):
        super().__init__()
        self.graph = Graph(layout=graph_layout)
        self.A = nn.Parameter(torch.from_numpy(self.graph.A).float(), requires_grad=False)

        # Data batch normalization for input (N, C, T, V)
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)

        # Progressive spatial-temporal conv layers
        self.st_blocks = nn.ModuleList([
            STConvBlock(in_channels, 64, residual=False),
            STConvBlock(64, 64),
            STConvBlock(64, 64),
            STConvBlock(64, 128, stride=2),
            STConvBlock(128, 128),
            STConvBlock(128, 128),
            STConvBlock(128, 256, stride=2),
            STConvBlock(256, 256),
        ])

        # Global pooling and linear projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        # x shape: (N, C, T, V, M) -> M=1 single person
        if x.dim() == 5:
            x = x.squeeze(-1)  # (N, C, T, V)

        N, C, T, V = x.size()
        # Reshape for input batchnorm
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        # Forward through ST-GCN blocks
        for block in self.st_blocks:
            x = block(x, self.A)

        # Global average pool over temporal (T) and vertex (V) dimensions
        features = self.pool(x).view(N, -1)  # (N, 256)

        if return_features:
            return None, features

        out = self.fc(features)
        return out, features
