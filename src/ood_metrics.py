"""
Out-of-Distribution (OOD) Gating using Deep k-NN & Mahalanobis Feature Distances.
Supports both runtime evaluation modes developed in the project.
"""

import numpy as np
import torch


def compute_knn_distance(query_embedding, feature_bank, k=5, mode='kth'):
    """
    Calculates the k-Nearest Neighbor Euclidean distance between query embedding
    and the reference feature bank.
    
    Args:
        query_embedding (torch.Tensor): Shape (1, D) or (B, D)
        feature_bank (torch.Tensor): Shape (N, D)
        k (int): k value for nearest neighbor search
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
