"""
SkateFormer: Partitioned Skeletal-Temporal Vision Transformer for Violent Threat Recognition (Version 2.20)
Reference:
"SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition"
(ECCV 2024, KAIST-VICLab)
and "Modernizing Violent Action Detection with Next-Generation Spatio-Temporal Backbones
and Manifold-Aware OOD Detection"

Key Architectural Principles:
1. Coordinate-First Modeling:
   Directly ingests normalized 2D skeleton coordinates (C=2, T=50, V=17), preserving exact
   sub-pixel micro-velocities and avoiding the 1,000x data bloat and spatial downsampling
   blurring of 3D heatmaps (which caused PoseConv3D's failure).
2. Skate-Embedding (Skeletal-Temporal Embedding):
   Combines coordinate linear projection with:
   - Joint-wise spatial positional embedding (E_joint in R^{1 x d x 1 x V})
   - Temporal 1D positional embedding (E_time in R^{1 x d x T x 1})
   - Anatomical limb type bias (Head, Torso, Left Arm, Right Arm, Legs)
3. Skate-MSA (Partitioned Spatio-Temporal Multi-Head Self-Attention):
   Partitions the token attention into 4 physical-temporal regimes:
   - Branch 1 (S_near / T_near): Local temporal window x local anatomical limb groups
   - Branch 2 (S_near / T_dist): Full temporal trajectory attention per joint (T=50)
   - Branch 3 (S_dist / T_near): Cross-joint spatial coordination per frame (V=17)
   - Branch 4 (S_dist / T_dist): Global macro-phase spatio-temporal correlation
4. Arm-Weighted Spatial Attention:
   5x priority weighting on high-velocity knife-strike joints (shoulders, elbows, wrists: joints 5-10)
   before global pooling into the 256-D penultimate latent space.
5. Drop-In Compatibility:
   Provides standard `forward(x, return_features=False)` interface:
   - `return_features=True` returns `(logits, features)` where `features` is in R^{256}.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO-17 Anatomical Group Partitions for S_near:
# 0: Head (0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear)
# 1: Torso (5: left shoulder, 6: right shoulder, 11: left hip, 12: right hip)
# 2: Left Arm (5: left shoulder, 7: left elbow, 9: left wrist)
# 3: Right Arm (6: right shoulder, 8: right elbow, 10: right wrist)
# 4: Legs (13: left knee, 14: right knee, 15: left ankle, 16: right ankle)
ANATOMICAL_PARTS = [
    [0, 1, 2, 3, 4],       # Head
    [5, 6, 11, 12],         # Torso
    [5, 7, 9],              # Left Arm
    [6, 8, 10],             # Right Arm
    [11, 12, 13, 14, 15, 16] # Legs
]


class SkateEmbedding(nn.Module):
    """
    Skeletal-Temporal Embedding (Skate-Embedding).
    Projects raw coordinates and injects spatial joint and temporal position encodings.
    """
    def __init__(self, in_channels=2, embed_dim=128, num_frames=50, num_keypoints=17, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_keypoints = num_keypoints

        # Linear projection of coordinates: (B, C, T, V) -> (B, d, T, V)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim)
        )

        # Learnable spatial joint embedding
        self.joint_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, num_keypoints))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        # Learnable temporal frame positional embedding
        self.time_embed = nn.Parameter(torch.zeros(1, embed_dim, num_frames, 1))
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        # Anatomical part identity embedding (5 parts)
        self.part_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 5))
        nn.init.trunc_normal_(self.part_embed, std=0.02)

        # Map each joint to its primary anatomical part index
        joint_to_part = [0, 0, 0, 0, 0, 1, 1, 2, 3, 2, 3, 1, 1, 4, 4, 4, 4]
        self.register_buffer('joint_to_part', torch.tensor(joint_to_part, dtype=torch.long))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.size()
        feat = self.proj(x) # (B, d, T, V)

        # Add joint and temporal positional encodings
        feat = feat + self.joint_embed[:, :, :, :V]
        feat = feat + self.time_embed[:, :, :T, :]

        # Add anatomical limb type bias
        part_idx = self.joint_to_part[:V] # (V,)
        part_bias = self.part_embed[:, :, :, part_idx] # (1, d, 1, V)
        feat = feat + part_bias

        return self.dropout(feat)


class RelativeTemporalAttention(nn.Module):
    """
    Temporal Multi-Head Self-Attention with 1D Relative Positional Bias.
    Computes trajectory attention across all T frames for each skeletal joint.
    Partition: S_near / T_dist
    """
    def __init__(self, dim, num_heads=4, num_frames=50, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_frames = num_frames

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Relative positional bias table: [-T+1, T-1] -> length 2*T - 1
        self.rel_pos_bias_table = nn.Parameter(torch.zeros(2 * num_frames - 1, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        coords = torch.arange(num_frames)
        rel_coords = coords[None, :] - coords[:, None] # (T, T)
        rel_coords += num_frames - 1
        self.register_buffer('rel_coords', rel_coords)

    def forward(self, x):
        # x: (B * V, T, C)
        BV, T, C = x.size()
        qkv = self.qkv(x).reshape(BV, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (BV, heads, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (BV, heads, T, T)

        # Add relative positional bias
        rel_bias = self.rel_pos_bias_table[self.rel_coords[:T, :T].reshape(-1)].reshape(T, T, -1)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0) # (1, heads, T, T)
        attn = attn + rel_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(BV, T, C)
        out = self.proj(out)
        return self.dropout(out)


class RelativeSpatialAttention(nn.Module):
    """
    Spatial Multi-Head Self-Attention with Spatial Geometric Bias.
    Computes cross-joint coordination across all V joints for each frame.
    Partition: S_dist / T_near
    """
    def __init__(self, dim, num_heads=4, num_keypoints=17, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_keypoints = num_keypoints

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Spatial pairwise relation bias matrix: (heads, V, V)
        self.spatial_bias = nn.Parameter(torch.zeros(num_heads, num_keypoints, num_keypoints))
        nn.init.trunc_normal_(self.spatial_bias, std=0.02)

    def forward(self, x):
        # x: (B * T, V, C)
        BT, V, C = x.size()
        qkv = self.qkv(x).reshape(BT, V, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (BT, heads, V, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (BT, heads, V, V)
        attn = attn + self.spatial_bias[:, :V, :V].unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(BT, V, C)
        out = self.proj(out)
        return self.dropout(out)


class LocalLimbWindowAttention(nn.Module):
    """
    Local Temporal Window x Local Anatomical Limb Group Attention.
    Partition: S_near / T_near (G-Part)
    Captures rapid micro-kinematics within local anatomical groups (e.g. wrist relative to elbow)
    across short temporal intervals (window_size = 10 frames).
    """
    def __init__(self, dim, num_heads=2, window_size=10, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.size()
        W = self.window_size
        pad_t = (W - T % W) % W
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, pad_t))
            T_padded = T + pad_t
        else:
            T_padded = T

        num_windows = T_padded // W
        # Reshape to local windows: (B, C, num_windows, W, V) -> (B * num_windows, W * V, C)
        x_win = x.view(B, C, num_windows, W, V).permute(0, 2, 3, 4, 1).contiguous()
        x_win = x_win.view(B * num_windows, W * V, C)

        L = W * V
        qkv = self.qkv(x_win).reshape(B * num_windows, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * num_windows, L, C)
        out = self.proj(out)
        out = self.dropout(out)

        # Reshape back to (B, C, T_padded, V)
        out = out.view(B, num_windows, W, V, C).permute(0, 4, 1, 2, 3).contiguous()
        out = out.view(B, C, T_padded, V)
        if pad_t > 0:
            out = out[:, :, :T, :]
        return out


class GlobalMacroAttention(nn.Module):
    """
    Global Macro-Phase Spatio-Temporal Attention.
    Partition: S_dist / T_dist
    Attends across temporally strided / downsampled representations to capture
    extended physical confrontation progression.
    """
    def __init__(self, dim, num_heads=2, stride=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.stride = stride

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.size()
        # Query tokens: all (T, V)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, T * V, C)
        q = self.q(x_flat).reshape(B, T * V, self.num_heads, self.head_dim).transpose(1, 2)

        # Key / Value tokens: temporally strided by stride (reduces sequence length for efficiency)
        x_strided = x[:, :, ::self.stride, :] # (B, C, T//stride, V)
        T_s = x_strided.size(2)
        x_s_flat = x_strided.permute(0, 2, 3, 1).contiguous().view(B, T_s * V, C)
        kv = self.kv(x_s_flat).reshape(B, T_s * V, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # (B, heads, T_s * V, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T * V, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()


class SkateMSA(nn.Module):
    """
    Partitioned Skeletal-Temporal Multi-Head Self-Attention Block (Skate-MSA).
    Splits channel capacity into 4 parallel interaction partitions:
    - Branch 1: S_near / T_near (Local Limb Window Attention)
    - Branch 2: S_near / T_dist (Relative Temporal Trajectory Attention)
    - Branch 3: S_dist / T_near (Relative Spatial Coordination Attention)
    - Branch 4: S_dist / T_dist (Global Macro Attention)
    Followed by feature fusion, LayerNorm, and FFN.
    """
    def __init__(self, dim=128, num_heads=8, num_frames=50, num_keypoints=17,
                 window_size=10, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4 for the 4 Skate-MSA partitions"
        branch_dim = dim // 4
        heads_per_branch = max(1, num_heads // 4)

        # 4 Partitioned Branches
        self.branch1_local = LocalLimbWindowAttention(
            dim=branch_dim, num_heads=heads_per_branch, window_size=window_size, dropout=dropout
        )
        self.branch2_temp = RelativeTemporalAttention(
            dim=branch_dim, num_heads=heads_per_branch, num_frames=num_frames, dropout=dropout
        )
        self.branch3_spat = RelativeSpatialAttention(
            dim=branch_dim, num_heads=heads_per_branch, num_keypoints=num_keypoints, dropout=dropout
        )
        self.branch4_macro = GlobalMacroAttention(
            dim=branch_dim, num_heads=heads_per_branch, stride=2, dropout=dropout
        )

        # Branch fusion projection
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)

        # Feed-Forward Network (FFN)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C=dim, T, V)
        B, C, T, V = x.size()
        res = x

        # Split channels into 4 sub-channels
        b_dim = C // 4
        x1 = x[:, 0 * b_dim : 1 * b_dim, :, :]
        x2 = x[:, 1 * b_dim : 2 * b_dim, :, :]
        x3 = x[:, 2 * b_dim : 3 * b_dim, :, :]
        x4 = x[:, 3 * b_dim : 4 * b_dim, :, :]

        # Branch 1: Local Window (B, b_dim, T, V)
        out1 = self.branch1_local(x1)

        # Branch 2: Temporal Trajectory Attention (B*V, T, b_dim)
        x2_flat = x2.permute(0, 3, 2, 1).contiguous().view(B * V, T, b_dim)
        out2 = self.branch2_temp(x2_flat).view(B, V, T, b_dim).permute(0, 3, 2, 1).contiguous()

        # Branch 3: Spatial Coordination Attention (B*T, V, b_dim)
        x3_flat = x3.permute(0, 2, 3, 1).contiguous().view(B * T, V, b_dim)
        out3 = self.branch3_spat(x3_flat).view(B, T, V, b_dim).permute(0, 3, 1, 2).contiguous()

        # Branch 4: Macro Spatio-Temporal Attention (B, b_dim, T, V)
        out4 = self.branch4_macro(x4)

        # Concatenate 4 partitions along channel dimension
        out = torch.cat([out1, out2, out3, out4], dim=1) # (B, dim, T, V)
        out = self.fusion(out)
        x = self.norm1(res + out)

        # FFN Block
        x = self.norm2(x + self.ffn(x))
        return x


class SkateFormerModel(nn.Module):
    """
    SkateFormer Skeletal-Temporal Vision Transformer Model (Version 2.20).
    Stack of 4 Skate-MSA blocks with Skate-Embedding, Arm-Weighted Spatial Attention,
    and 256-D Latent Feature Interface.
    """
    def __init__(
        self,
        num_classes=1,
        in_channels=2,
        num_frames=50,
        num_keypoints=17,
        embed_dim=128,
        num_blocks=4,
        num_heads=8,
        window_size=10,
        mlp_ratio=2.0,
        dropout=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.num_keypoints = num_keypoints
        self.embed_dim = embed_dim

        # Input Normalization & Skate-Embedding
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)
        self.embedding = SkateEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            num_keypoints=num_keypoints,
            dropout=dropout
        )

        # Stack of 4 Skate-MSA Blocks
        self.blocks = nn.ModuleList([
            SkateMSA(
                dim=embed_dim,
                num_heads=num_heads,
                num_frames=num_frames,
                num_keypoints=num_keypoints,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # Arm-Weighted Spatial Attention: 5x weighting on combat joints (shoulders: 5,6; elbows: 7,8; wrists: 9,10)
        jw = torch.ones(num_keypoints)
        jw[[5, 6, 7, 8, 9, 10]] = 5.0
        jw = jw / jw.sum() * num_keypoints
        self.register_buffer('joint_weights', jw)

        # Penultimate Projection to 256-D Latent Feature
        self.feature_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256)
        )

        # Linear Classification Head
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        """
        Args:
            x (torch.Tensor): Input skeleton clip.
               Supported shapes: (B, C, T, V, M=1), (B, C, T, V), or (B, T, V, C).
            return_features (bool): If True, returns (logits, penultimate_features).
        Returns:
            logits (torch.Tensor): Shape (B, num_classes).
            features (torch.Tensor, optional): Shape (B, 256).
        """
        # Ensure 5D tensor (B, C, T, V, M)
        if x.dim() == 4:
            if x.size(1) == self.in_channels:
                x = x.unsqueeze(-1)
            elif x.size(-1) == self.in_channels:
                x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        elif x.dim() == 5 and x.size(-1) != 1 and x.size(1) != self.in_channels:
            x = x.permute(0, 4, 1, 2, 3)

        B, C, T, V, M = x.size()

        # Standardize joint coordinates via data BatchNorm
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(B * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(B * M, C, V, T).permute(0, 1, 3, 2).contiguous() # (B*M, C, T, V)

        # Skate-Embedding
        feat = self.embedding(x) # (B*M, embed_dim, T, V)

        # Forward through Skate-MSA Blocks
        for block in self.blocks:
            feat = block(feat)

        # Apply Arm-Weighted Spatial Attention before pooling
        weights = self.joint_weights.view(1, 1, 1, V)
        feat = feat * weights

        # Spatio-temporal global pooling over (T, V)
        pooled = feat.mean(dim=[-2, -1]) # (B*M, embed_dim)
        pooled = pooled.view(B, M, -1)
        feat_vector = pooled.squeeze(1) if M == 1 else pooled.mean(dim=1) # (B, embed_dim)

        # Penultimate 256-D feature embedding
        penultimate_features = self.feature_proj(feat_vector) # (B, 256)
        logits = self.fc(self.dropout(penultimate_features)) # (B, num_classes)

        if return_features:
            return logits, penultimate_features
        return logits
