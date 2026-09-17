"""
Out-of-Distribution (OOD) Gating using Deep k-NN & Mahalanobis Feature Distances.
Supports both runtime evaluation modes developed in the project.
"""

import numpy as np
import torch


def compute_knn_distance(query_embedding, feature_bank, k=2, mode='kth'):
    """
    Calculates the k-Nearest Neighbor Euclidean distance between query embedding
    and the reference feature bank.
    
    Args:
        query_embedding (torch.Tensor): Shape (1, D) or (B, D)
        feature_bank (torch.Tensor): Shape (N, D)
        k (int): k value for nearest neighbor search (production default: k=2)
        mode (str): 'kth' for distance to the k-th nearest neighbor (matches Real_Time.py),
                    'mean' for average distance across top-k neighbors.
    Returns:
        float: Distance metric
    """
    with torch.no_grad():
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # cdist computes pair-wise euclidean distance
        dists = torch.cdist(query_embedding.float(), feature_bank.float(), p=2)
        effective_k = min(k, feature_bank.size(0))
        
        if mode == 'kth':
            kth_val, _ = dists.kthvalue(effective_k, dim=1)
            return kth_val.item()
        else:
            topk_vals, _ = torch.topk(dists, k=effective_k, largest=False, dim=1)
            return topk_vals.mean().item()


def compute_mahalanobis_distance(query_embedding, mean_vector, inv_cov_matrix):
    """
    Calculates Mahalanobis distance accounting for feature covariance:
    D_M(x) = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
    
    Args:
        query_embedding (np.ndarray): 1D feature vector (D,)
        mean_vector (np.ndarray): 1D centroid vector (D,)
        inv_cov_matrix (np.ndarray): 2D inverse covariance matrix (D, D)
    Returns:
        float: Mahalanobis distance
    """
    diff = query_embedding - mean_vector
    dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
    return float(dist)


def compute_relative_mahalanobis_distance(query_embedding, mu_0, mu_c, inv_Sigma_0, inv_Sigma_c):
    """
    Computes Relative Mahalanobis Distance (RMD) score for input query embeddings (Ren et al., NeurIPS 2021).
    
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
