"""
Out-of-Distribution (OOD) Gating using Mahalanobis & k-NN Feature Distances.
"""

import numpy as np
import torch


def compute_knn_distance(query_embedding, feature_bank, k=5):
    """
    Calculates the k-Nearest Neighbor Euclidean distance between a query
    embedding vector and the reference normal/threat feature bank.

    Args:
        query_embedding (torch.Tensor): Shape (1, D)
        feature_bank (torch.Tensor): Shape (N, D)
        k (int): Number of nearest neighbors
    Returns:
        float: Mean distance across k nearest neighbors.
    """
    with torch.no_grad():
        diff = feature_bank - query_embedding
        dists = torch.norm(diff, dim=1)
        topk_dists, _ = torch.topk(dists, k=min(k, feature_bank.size(0)), largest=False)
        return topk_dists.mean().item()


def compute_mahalanobis_distance(query_embedding, mean_vector, inv_cov_matrix):
    """
    Calculates Mahalanobis distance accounting for feature covariance:
    D_M(x) = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
    """
    diff = query_embedding - mean_vector
    dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
    return float(dist)
