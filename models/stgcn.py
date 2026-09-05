"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Skeletal Action Recognition.
Authentic architecture matching Fold 2 trained weights (stgcn_violence_fold2.pth).
9 ST-GCN convolutional blocks with joint-weighted spatial attention (5x weight on arm joints).
"""

import numpy as np
import torch
import torch.nn as nn


def normalize_adjacency(A):
    D = np.sum(A, axis=1)
    D[D <= 1e-6] = 1.0
    D_inv_sqrt = np.power(D, -0.5)
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    return D_mat_inv_sqrt.dot(A).dot(D_mat_inv_sqrt)


class Graph:
    """
    Defines the anatomical graph structure based on COCO 17 keypoint connections.
    Uses Spatial Configuration Partitioning (3 adjacency matrices: Identity, Inward, Outward).
    """
    def __init__(self, layout='coco', strategy='spatial'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor_link_coco = [
             (0, 1), (0, 2), (1, 3), (2, 4), 
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (5, 11), (6, 12)
        ]
        self.sym_neighbor_link = self.neighbor_link_coco + [(v, u) for u, v in self.neighbor_link_coco]
        self.A = self.get_adjacency_matrix(strategy)

    def get_adjacency_matrix(self, strategy):
        num_sub_adj = 3 
        A = np.zeros((num_sub_adj, self.num_node, self.num_node), dtype=np.float32)
        if strategy == 'spatial':
            for i in range(self.num_node):
                A[0, i, i] = 1
            for i, j in self.sym_neighbor_link:
                 if j < i: A[1, i, j] = 1 
                 elif j > i: A[2, i, j] = 1
        for i in range(num_sub_adj):
            if np.any(A[i]): A[i] = normalize_adjacency(A[i])
        return torch.from_numpy(A)


class STConvBlock(nn.Module):
    """ Spatial-Temporal Graph Convolutional Block """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = nn.Conv2d(in_channels, out_channels * kernel_size[1], kernel_size=1)
        self.num_sub_adj = kernel_size[1]
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
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
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        N, C, T, V = x.size()
        x = self.gcn(x)
        x = x.view(N, self.num_sub_adj, -1, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class STGCNModel(nn.Module):
    """
    9-Block Spatial Temporal Graph Convolutional Network.
    Includes joint-weighted spatial attention (5x on arms) and 256-dim feature pooling.
    """
    def __init__(self, num_classes=1, in_channels=2, num_keypoints=17, dropout=0.1):
        super().__init__()
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
        self.fc = nn.Linear(256, num_classes)

        # Joint-weighted spatial attention: arms (5, 6, 7, 8, 9, 10) get 5x weight
        jw = torch.ones(num_keypoints)
        jw[[5, 6, 7, 8, 9, 10]] = 5.0
        jw = jw / jw.sum() * num_keypoints
        self.register_buffer('joint_weights', jw)

    def forward(self, x, return_features=False):
        # x shape: (N, C, T, V, M=1) or (N, C, T, V)
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
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits