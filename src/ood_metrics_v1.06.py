"""
Out-of-Distribution (OOD) Gating using Virtual-Logit Matching (ViM) (Version 1.06).

Formulation based on:
"ViM: Out-Of-Distribution with Virtual-logit Matching" (Wang et al., CVPR 2022)
and "Modernizing Violent Action Detection with Next-Generation Spatio-Temporal Backbones
and Manifold-Aware OOD Detection"

Mathematical Foundation:
1. Decomposes centered latent feature space into an affine principal subspace V (top K eigenvectors)
   and its orthogonal complement (residual null space P_null = I - V * V^T).
2. For query embedding z in R^{D}:
   r(z) = (z - mu) - V * V^T * (z - mu)
   res_norm = ||r(z)||_2
3. Assigns virtual logit v(z) = alpha * res_norm to quantify probability of unseen OOD movement.
4. Synthesizes unified ViM score:
   S_ViM(z) = exp(logit) / (exp(logit) + exp(v(z))) = sigmoid(logit - alpha * res_norm)
   In-distribution actions: low residual norm -> high S_ViM.
   OOD actions: high residual energy in null space -> low S_ViM.
5. Eliminates full reference bank distance searches; operates via lightweight linear projection (K=24).
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_null_space_residual(query_embedding, mu, V_k):
    """
    Computes projection onto residual null space P_null = I - V_k * V_k^T.
    
    Args:
        query_embedding (torch.Tensor): Shape (B, D) or (1, D) or (D,).
        mu (torch.Tensor): Mean vector of training distribution, shape (1, D).
        V_k (torch.Tensor): Principal subspace projection matrix, shape (D, K).
    Returns:
        torch.Tensor: Residual vector r in R^{B x D} and L2 norm in R^{B}.
    """
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
        
    device = query_embedding.device
    if mu.device != device: mu = mu.to(device)
    if V_k.device != device: V_k = V_k.to(device)
    
    diff = query_embedding.float() - mu.float()
    proj = torch.mm(diff, V_k.float()) # (B, K)
    recon = torch.mm(proj, V_k.float().t()) # (B, D)
    residual = diff - recon # (B, D)
    res_norm = torch.norm(residual, p=2, dim=-1)
    
    return residual, res_norm


def compute_vim_score(query_embedding, logits, mu, V_k, alpha):
    """
    Computes the Virtual-Logit Matching (ViM) score for an input feature embedding.
    
    Args:
        query_embedding (torch.Tensor): Penultimate action embedding (B, D).
        logits (torch.Tensor): Raw classifier logit (B, 1) or (B,).
        mu (torch.Tensor): Centroid of training features (1, D).
        V_k (torch.Tensor): Principal subspace matrix (D, K).
        alpha (float): Scaling factor aligning residual scale with logits.
    Returns:
        float or torch.Tensor: ViM score in [0, 1] (higher = In-Distribution Violence, lower = OOD).
    """
    with torch.no_grad():
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
            
        _, res_norm = compute_null_space_residual(query_embedding, mu, V_k)
        
        # Virtual logit
        v = alpha * res_norm
        
        # Unified ViM score via sigmoid(logit - v)
        # S_ViM = exp(logit) / (exp(logit) + exp(v))
        s_vim = torch.sigmoid(logits - v)
        
        if s_vim.numel() == 1:
            return s_vim.item()
        return s_vim
