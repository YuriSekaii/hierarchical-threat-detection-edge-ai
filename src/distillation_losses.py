"""
distillation_losses.py

Knowledge Distillation Loss Functions for transferring representations from
ST-GCN (Graph Convolutional Network) to PoseConv3D (3D Volumetric CNN).

Implements:
1. FeatureAlignmentLoss: Cosine similarity + Mean Squared Error between penultimate latent vectors.
2. RelationalKDLoss: Distance-wise and Angle-wise Relational Knowledge Distillation (Park et al., CVPR 2019).
3. ArmAttentionDistillationLoss: Joint-saliency alignment matching ST-GCN's 5x arm-weighted spatial attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLoss(nn.Module):
    """
    Method A: Direct Penultimate Feature Alignment.
    Aligns student's 256-D feature representations with teacher's 256-D representations
    via unit hypersphere cosine alignment and normalized Euclidean MSE.
    """
    def __init__(self, alpha_cos=1.0, alpha_mse=1.0):
        super().__init__()
        self.alpha_cos = alpha_cos
        self.alpha_mse = alpha_mse

    def forward(self, f_student, f_teacher):
        # 1. Cosine similarity alignment: 1.0 - cos(f_s, f_t)
        cos_sim = F.cosine_similarity(f_student, f_teacher, dim=-1)
        loss_cos = torch.mean(1.0 - cos_sim)

        # 2. MSE distance alignment
        loss_mse = F.mse_loss(f_student, f_teacher)

        return self.alpha_cos * loss_cos + self.alpha_mse * loss_mse


class RelationalKDLoss(nn.Module):
    """
    Method B: Relational Knowledge Distillation (RKD - Park et al., CVPR 2019).
    Transfers the geometric manifold topology:
    - Distance-wise RKD: Huber loss between normalized pairwise sample distance matrices.
    - Angle-wise RKD: Huber loss between sample triplet angles.
    """
    def __init__(self, w_dist=1.0, w_angle=2.0):
        super().__init__()
        self.w_dist = w_dist
        self.w_angle = w_angle

    def distance_loss(self, f_s, f_t):
        # Pairwise Euclidean distances
        diff_s = f_s.unsqueeze(0) - f_s.unsqueeze(1)
        dist_s = torch.norm(diff_s, p=2, dim=-1)
        mean_s = dist_s.mean() + 1e-7
        norm_dist_s = dist_s / mean_s

        diff_t = f_t.unsqueeze(0) - f_t.unsqueeze(1)
        dist_t = torch.norm(diff_t, p=2, dim=-1)
        mean_t = dist_t.mean() + 1e-7
        norm_dist_t = dist_t / mean_t

        return F.smooth_l1_loss(norm_dist_s, norm_dist_t)

    def angle_loss(self, f_s, f_t):
        # Vector differences: e_ij = f_i - f_j
        diff_s = f_s.unsqueeze(1) - f_s.unsqueeze(0)  # (B, B, D)
        norm_s = F.normalize(diff_s, p=2, dim=-1)
        # Cosine angles between triplets: (B, B, B)
        cos_s = torch.einsum("ijk,ilk->ijl", norm_s, norm_s)

        diff_t = f_t.unsqueeze(1) - f_t.unsqueeze(0)
        norm_t = F.normalize(diff_t, p=2, dim=-1)
        cos_t = torch.einsum("ijk,ilk->ijl", norm_t, norm_t)

        return F.smooth_l1_loss(cos_s, cos_t)

    def forward(self, f_student, f_teacher):
        l_dist = self.distance_loss(f_student, f_teacher)
        if f_student.size(0) >= 3:
            l_angle = self.angle_loss(f_student, f_teacher)
            return self.w_dist * l_dist + self.w_angle * l_angle
        return self.w_dist * l_dist


class ArmAttentionDistillationLoss(nn.Module):
    """
    Method C: Arm Attention Distillation.
    Enforces that the student's representations emphasize arm kinematics (wrists, elbows, shoulders: joints 5-10)
    proportionally matching ST-GCN's 5x spatial attention weighting.
    """
    def __init__(self, num_keypoints=17, arm_indices=(5, 6, 7, 8, 9, 10), arm_multiplier=5.0):
        super().__init__()
        weights = torch.ones(num_keypoints)
        weights[list(arm_indices)] = arm_multiplier
        weights = weights / weights.sum() * num_keypoints
        self.register_buffer("joint_weights", weights)

    def forward(self, student_hm, f_student, f_teacher):
        # Compute joint energy from input heatmaps (B, 17, 50, 56, 56)
        joint_energy = student_hm.mean(dim=(2, 3, 4))  # (B, 17)
        joint_prob = F.softmax(joint_energy, dim=-1)
        target_prob = F.softmax(self.joint_weights.unsqueeze(0).expand_as(joint_prob), dim=-1)
        
        loss_kl = F.kl_div(joint_prob.log(), target_prob, reduction="batchmean")
        
        # Weighted feature guidance: arms get prioritized
        loss_feat = F.mse_loss(f_student, f_teacher)
        return loss_kl + loss_feat
