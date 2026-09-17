"""
Out-of-Distribution (OOD) Gating using Hyperspherical Cosine k-NN (Version 1.05).

Formulation based on:
"Modernizing Violent Action Detection with Next-Generation Spatio-Temporal Backbones
and Manifold-Aware OOD Detection"

Mathematical Foundation:
1. Projects latent feature vectors onto the unit hypersphere S^{D-1} via L2 normalization:
   z_tilde = z / (||z||_2 + eps)
2. Computes Cosine distance to reference bank:
   ||z_tilde_1 - z_tilde_2||_2^2 = 2 * (1 - cos(z_tilde_1, z_tilde_2)) = 2 - 2 * (z_tilde_1^T * z_tilde_2)
   D_cos(q, b) = 1.0 - (q_tilde^T * b_tilde)
3. Eliminates feature norm magnitude distortion and distance concentration in 256D latent space.
4. Accelerates nearest-neighbor retrieval via hardware-accelerated matrix-vector multiplication.
"""

import numpy as np
import torch
import torch.nn.functional as F


def project_hypersphere(tensor, eps=1e-8):
    """
    Projects input feature tensor onto the unit hypersphere S^{D-1}.
    Args:
        tensor (torch.Tensor): Feature tensor of shape (D,) or (B, D).
        eps (float): Epsilon for numerical stability.
    Returns:
        torch.Tensor: L2-normalized feature vector(s) on unit hypersphere.
    """
    return F.normalize(tensor.float(), p=2, dim=-1, eps=eps)


def compute_hyperspherical_cosine_knn_distance(query_embedding, feature_bank, k=2, mode='kth', is_bank_normalized=True):
    """
    Calculates the k-Nearest Neighbor Cosine distance between a query embedding
    and the reference feature bank on the unit hypersphere S^{D-1}.
    
    D_cos = 1.0 - (q_tilde * b_tilde^T)
    
    Args:
        query_embedding (torch.Tensor): Query latent vector, shape (1, D) or (B, D) or (D,).
        feature_bank (torch.Tensor): Reference latent bank, shape (N, D).
        k (int): k value for nearest neighbor search (default: k=2).
        mode (str): 'kth' for distance to the k-th nearest neighbor,
                    'mean' for average distance across top-k neighbors.
        is_bank_normalized (bool): If True, assumes feature_bank is already unit-normalized.
    Returns:
        float or torch.Tensor: Cosine distance metric.
    """
    with torch.no_grad():
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
            
        device = query_embedding.device
        if feature_bank.device != device:
            feature_bank = feature_bank.to(device)
            
        # L2-normalize query onto unit hypersphere
        q_norm = project_hypersphere(query_embedding)
        
        # Ensure reference bank is unit-normalized
        fb_norm = feature_bank if is_bank_normalized else project_hypersphere(feature_bank)
        
        # Fast dense inner products via matrix multiplication (BLAS GEMM)
        # cosine_similarity in [-1.0, 1.0]
        cos_sim = torch.mm(q_norm, fb_norm.t())
        
        # Cosine distance in [0.0, 2.0]
        cos_dist = torch.clamp(1.0 - cos_sim, min=0.0, max=2.0)
        
        effective_k = min(k, fb_norm.size(0))
        
        if mode == 'kth':
            kth_val, _ = cos_dist.kthvalue(effective_k, dim=1)
            if kth_val.numel() == 1:
                return kth_val.item()
            return kth_val
        else:
            topk_vals, _ = torch.topk(cos_dist, k=effective_k, largest=False, dim=1)
            mean_val = topk_vals.mean(dim=1)
            if mean_val.numel() == 1:
                return mean_val.item()
            return mean_val


def compute_legacy_euclidean_knn_distance(query_embedding, feature_bank, k=2, mode='kth'):
    """Legacy unnormalized Euclidean distance (v1.00 - v1.04 baseline for head-to-head comparison)."""
    with torch.no_grad():
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        dists = torch.cdist(query_embedding.float(), feature_bank.float(), p=2)
        effective_k = min(k, feature_bank.size(0))
        if mode == 'kth':
            kth_val, _ = dists.kthvalue(effective_k, dim=1)
            return kth_val.item()
        else:
            topk_vals, _ = torch.topk(dists, k=effective_k, largest=False, dim=1)
            return topk_vals.mean().item()
