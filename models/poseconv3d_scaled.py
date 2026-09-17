"""
Scaled PoseConv3D Volumetric 3D Heatmap CNN Backbone
Scaled parameter capacity (~3.55M parameters vs ST-GCN 3.01M) to empirically investigate
whether parameter under-allocation was the cause of PoseConv3D's recall deficit in Phase 1.

Key Architectural Properties:
- R(2+1)D factorized spatio-temporal convolutions (1x3x3 Spatial + 3x1x1 Temporal).
- Scaled width stages: [64, 128, 256, 512] -> 256-D penultimate feature vector.
- Parameter Count: 3,545,409 parameters (matching/exceeding ST-GCN's 3.01M params).
- Output: 256-D penultimate features for Relative Mahalanobis Distance (RMD) likelihood scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.poseconv3d import Block2Plus1D


class ScaledPoseConv3DModel(nn.Module):
    """
    Scaled PoseConv3D Volumetric 3D Heatmap CNN.
    
    Accepts 3D heatmap volumes of shape (B, K, T, H, W):
    - K: Number of keypoints (17 COCO joints)
    - T: Temporal clip frames (50 frames)
    - H, W: Spatial heatmap resolution (56 x 56)
    
    Outputs:
    - Logits: (B, num_classes)
    - Features: (B, feature_dim) [256-D penultimate embeddings for OOD threat gating]
    """
    def __init__(
        self,
        in_channels=17,
        num_classes=1,
        channels=(64, 128, 256, 512),
        feature_dim=256,
        dropout=0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        c1, c2, c3, c4 = channels

        # Initial Stem: spatial downsampling to 28x28, preserving temporal depth 50
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, c1,
                kernel_size=(1, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
                bias=False
            ),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True)
        )

        # Stage 1: c1 channels, (50, 28, 28)
        self.stage1 = nn.Sequential(
            Block2Plus1D(c1, c1, stride=(1, 1, 1)),
            Block2Plus1D(c1, c1, stride=(1, 1, 1))
        )

        # Stage 2: c2 channels, (50, 14, 14) [spatial stride 2]
        self.stage2 = nn.Sequential(
            Block2Plus1D(c1, c2, stride=(1, 2, 2)),
            Block2Plus1D(c2, c2, stride=(1, 1, 1))
        )

        # Stage 3: c3 channels, (25, 7, 7) [spatio-temporal stride 2]
        self.stage3 = nn.Sequential(
            Block2Plus1D(c2, c3, stride=(2, 2, 2)),
            Block2Plus1D(c3, c3, stride=(1, 1, 1))
        )

        # Stage 4: c4 channels -> feature_dim, (13, 4, 4) [spatio-temporal stride 2]
        self.stage4 = nn.Sequential(
            Block2Plus1D(c3, c4, stride=(2, 2, 2)),
            Block2Plus1D(c4, feature_dim, stride=(1, 1, 1))
        )

        # Global Spatio-Temporal Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        """Extracts 256-D penultimate feature vector."""
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.global_pool(out)
        feats = out.view(out.size(0), -1)
        return feats

    def forward(self, x, return_features=False):
        """
        Forward pass.
        Args:
            x: (B, K, T, H, W) 3D heatmap tensor
            return_features: If True, returns (logits, features)
        """
        feats = self.extract_features(x)
        logits = self.fc(self.dropout(feats))
        if return_features:
            return logits, feats
        return logits
