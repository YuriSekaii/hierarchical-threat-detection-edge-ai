"""
Training Script for ST-GCN Action Recognition using Triplet Margin Loss.
Trains embeddings to separate violent kinematic movements from normal/OOD actions.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.stgcn import STGCNModel


class TripletActionDataset(Dataset):
    """
    Yields (Anchor, Positive, Negative) skeleton kinematic clips for Triplet Loss.
    - Anchor: Violent movement clip (e.g. Stab)
    - Positive: Different violent clip of same class
    - Negative: Out-of-Distribution or passive normal movement clip
    """
    def __init__(self, data_list, clip_length=50):
        self.data = data_list
        self.clip_length = clip_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Anchor, Positive, Negative tensors: shape (C=2, T=50, V=17)
        anchor = torch.randn(2, self.clip_length, 17)
        positive = torch.randn(2, self.clip_length, 17)
        negative = torch.randn(2, self.clip_length, 17)
        return anchor, positive, negative


def train_triplet_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for anchor, pos, neg in dataloader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()

        _, feat_a = model(anchor, return_features=True)
        _, feat_p = model(pos, return_features=True)
        _, feat_n = model(neg, return_features=True)

        loss = criterion(feat_a, feat_p, feat_n)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCNModel(num_classes=1, in_channels=2, num_keypoints=17).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"ST-GCN Triplet Training initialized on {device}.")
