"""
Out-of-Distribution (OOD) Gating using Relative Mahalanobis Distance (RMD) (Version 1.08).

Formulation based on:
"A Simple Fix to Mahalanobis Distance for Out-of-Distribution Detection" (Ren et al., NeurIPS 2021)
and "Modernizing Violent Action Detection with Next-Generation Spatio-Temporal Backbones
and Manifold-Aware OOD Detection"

Mathematical Foundation:
1. Standard Mahalanobis distance M(z) = (z - mu)^T Sigma^{-1} (z - mu) collapses when:
   - Features have multi-modal class clusters (e.g. Cut-Down, Stab, Thrust), causing a unimodal centroid
     to sit in empty high-dimensional space (dropping recall to 84.6% in v1.00).
   - Non-informative background variations (shared human posture, camera perspective) dominate covariance,
     causing OOD civilian movements to register deceptively close.
2. Relative Mahalanobis Distance (RMD) resolves this via a likelihood ratio between class-conditional
   Gaussians N(mu_c, Sigma_c) and a global background distribution N(mu_0, Sigma_0):
   - Background distance:
     M_0(z) = (z - mu_0)^T Sigma_0^{-1} (z - mu_0)
   - Class-conditional distance:
     M_c(z) = (z - mu_c)^T Sigma_c^{-1} (z - mu_c)
   - Relative score:
     RMD(z) = M_0(z) - min_{c} M_c(z)
3. Under this subtraction, common background kinematics cancel out in (M_0 - M_c), isolating the
   discriminative action-specific manifold energy.
4. Covariance Regularization:
   Uses ridge shrinkage Sigma_reg = Sigma + epsilon * I (epsilon = 0.005) to ensure well-conditioned
   matrix inversion in D=256 dimensional latent space.
"""

import torch
import torch.nn.functional as F


def compute_relative_mahalanobis_distance(query_embedding, mu_0, mu_c, inv_Sigma_0, inv_Sigma_c):
    """
    Computes Relative Mahalanobis Distance (RMD) score for input query embeddings.
    
    Args:
        query_embedding (torch.Tensor): Shape (B, D) or (D,).
        mu_0 (torch.Tensor): Global background mean vector, shape (1, D) or (D,).
        mu_c (torch.Tensor): Class-conditional mean vectors, shape (C, D).
        inv_Sigma_0 (torch.Tensor): Precision matrix of background distribution, shape (D, D).
        inv_Sigma_c (torch.Tensor): Precision matrices of class distributions, shape (C, D, D).
        
    Returns:
        tuple: (rmd_score, min_mc, m0)
            - rmd_score: Tensor of shape (B,) or float (higher = In-Distribution Violence, lower = OOD).
            - min_mc: Tensor of shape (B,) distance to closest violence class.
            - m0: Tensor of shape (B,) distance to global background distribution.
    """
    single_input = False
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)
        single_input = True
        
    device = query_embedding.device
    mu_0 = mu_0.to(device)
    mu_c = mu_c.to(device)
    inv_Sigma_0 = inv_Sigma_0.to(device)
    inv_Sigma_c = inv_Sigma_c.to(device)
    
    if mu_0.dim() == 1:
        mu_0 = mu_0.unsqueeze(0)
        
    # 1. Background Mahalanobis Distance: M_0(z) = (z - mu_0)^T inv_Sigma_0 (z - mu_0)
    diff_0 = query_embedding.float() - mu_0.float() # (B, D)
    m0 = torch.sum((torch.mm(diff_0, inv_Sigma_0.float())) * diff_0, dim=-1) # (B,)
    
    # 2. Class-Conditional Mahalanobis Distances: M_c(z) = (z - mu_c)^T inv_Sigma_c (z - mu_c)
    # query: (B, 1, D), mu_c: (1, C, D) -> diff_c: (B, C, D)
    diff_c = query_embedding.unsqueeze(1).float() - mu_c.unsqueeze(0).float() # (B, C, D)
    
    # mc: (B, C)
    mc = torch.einsum('bcd,cde,bce->bc', diff_c, inv_Sigma_c.float(), diff_c)
    min_mc, _ = mc.min(dim=-1) # (B,)
    
    # 3. Relative Mahalanobis Score: RMD(z) = M_0(z) - min_c M_c(z)
    rmd_score = m0 - min_mc # (B,)
    
    if single_input:
        return rmd_score.item(), min_mc.item(), m0.item()
    return rmd_score, min_mc, m0


def is_threat_rmd(query_embedding, mu_0, mu_c, inv_Sigma_0, inv_Sigma_c, threshold=-0.9289):
    """
    Evaluates whether an action feature embedding represents an active violent threat.
    
    Args:
        query_embedding (torch.Tensor): Feature embedding (D,) or (1, D).
        mu_0, mu_c, inv_Sigma_0, inv_Sigma_c: Calibrated RMD reference parameters.
        threshold (float): Decision threshold tau_RMD (default: -0.9289 for optimal F1=0.9167).
        
    Returns:
        bool: True if classified as In-Distribution Violence, False if rejected as OOD.
        float: Computed RMD score.
    """
    with torch.no_grad():
        score, _, _ = compute_relative_mahalanobis_distance(
            query_embedding, mu_0, mu_c, inv_Sigma_0, inv_Sigma_c
        )
        if isinstance(score, torch.Tensor):
            score = score.item()
        return (score >= threshold), score
