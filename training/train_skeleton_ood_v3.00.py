"""
training/train_skeleton_ood_v3.00.py

Skeleton-OOD 3-Fold Training with Latent Boundary Learning, Pseudo-OOD Regularization,
and Kolmogorov-Smirnov Temporal Motion Regularization - Version 3.00.
100% self-contained: uses local workspace datasets and modules. Zero external dependencies.
Exports champion checkpoint (weights/stgcn_skeleton_ood_v3.00.pth) and training logs.
"""

import os
import sys
import json
import random
import copy
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset, TripletActionDataset
from models.skeleton_ood import SkeletonOODModel

SEED = 42
NUM_WORKERS = 0
PIN_MEMORY = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
        cudnn.deterministic = False


set_seed(SEED)


def get_action_class_idx(source_path):
    if "Cut-Down" in source_path:
        return 0
    elif "Stab" in source_path:
        return 1
    elif "Thrust" in source_path:
        return 2
    return 0


class SkeletonOODDataset(TripletActionDataset):
    """
    Subclasses TripletActionDataset to return action category indices
    for anchor, positive, and negative clips.
    """
    def __getitem__(self, idx):
        anchor_idx = self.sample_indices[idx]
        anchor_info = self.samples[anchor_idx]
        anchor_video = anchor_info["source_file"]
        anchor_clip, _, _ = ActionDataset.__getitem__(self, anchor_idx)
        a_cls = get_action_class_idx(anchor_video)

        pos_indices = [i for i in self.samples_by_video[anchor_video] if i != anchor_idx]
        pos_idx = random.choice(pos_indices) if pos_indices else anchor_idx
        pos_clip, _, _ = ActionDataset.__getitem__(self, pos_idx)
        p_cls = a_cls

        while True:
            neg_video = random.choice(self.video_files)
            if neg_video != anchor_video:
                break
        neg_idx = random.choice(self.samples_by_video[neg_video])
        neg_clip, _, _ = ActionDataset.__getitem__(self, neg_idx)
        n_cls = get_action_class_idx(neg_video)

        return anchor_clip, pos_clip, neg_clip, a_cls, p_cls, n_cls


class EarlyStopping:
    def __init__(self, patience=10, mode='min', min_delta=0.0001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_model_state = None

    def __call__(self, score, model_state):
        improved = (score < self.best_score - self.min_delta) if self.mode == 'min' else (score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model_state)
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def compute_temporal_kinematic_energy(x):
    """
    Computes frame-to-frame joint displacement kinetic energy for motion regularization.
    Input x: (B, 2, T=50, V=17, 1) or (B, 2, T=50, V=17)
    Arm joints: indices 5, 6, 7, 8, 9, 10
    """
    if x.dim() == 5:
        x = x.squeeze(-1)
    # x: (B, 2, T, V)
    diff = x[:, :, 1:, :] - x[:, :, :-1, :] # (B, 2, T-1, V)
    vel_sq = torch.sum(diff ** 2, dim=1)    # (B, T-1, V)
    arm_vel = vel_sq[:, :, [5, 6, 7, 8, 9, 10]] # (B, T-1, 6)
    kin_energy = torch.mean(arm_vel, dim=(1, 2)) # (B,)
    return torch.log1p(kin_energy * 100.0)


class SkeletonOODLoss(nn.Module):
    """
    Composite Skeleton-OOD Loss:
    L = L_triplet + 0.5 * L_ce + 0.2 * L_compact + 0.1 * L_energy + 0.1 * L_pseudo_ood + 0.1 * L_ks
    """
    def __init__(self, m_in=-4.0, m_out=-1.0, triplet_margin=1.0):
        super().__init__()
        self.m_in = m_in
        self.m_out = m_out
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin)
        self.ce_loss = nn.CrossEntropyLoss()
        # Kinematic energy predictor head mapping 256-D feature to 1-D log kinetic energy
        self.kinematic_head = nn.Linear(256, 1)

    def forward(self, model, a, p, n, a_cls, p_cls, n_cls):
        # 1. Forward pass for anchor, positive, negative
        a_logits, a_z, a_energy = model(a, return_features=True, return_energy=True)
        p_logits, p_z, p_energy = model(p, return_features=True, return_energy=True)
        n_logits, n_z, n_energy = model(n, return_features=True, return_energy=True)

        # 2. Triplet margin loss on unit hypersphere embeddings
        loss_triplet = self.triplet_loss(a_z, p_z, n_z)

        # 3. Cross-entropy classification loss on action categories
        loss_ce = (
            self.ce_loss(a_logits, a_cls) +
            self.ce_loss(p_logits, p_cls) +
            self.ce_loss(n_logits, n_cls)
        ) / 3.0

        # 4. Latent Hyperspherical Boundary Compactness Loss
        # Pull each embedding toward its normalized class center
        norm_centers = F.normalize(model.class_centers, p=2, dim=-1) # (3, 256)
        a_centers = norm_centers[a_cls] # (B, 256)
        p_centers = norm_centers[p_cls]
        loss_compact = (
            torch.mean(1.0 - torch.sum(a_z * a_centers, dim=-1)) +
            torch.mean(1.0 - torch.sum(p_z * p_centers, dim=-1))
        ) / 2.0

        # 5. In-Distribution Energy Margin Constraint: penalize E(x) > m_in
        loss_energy = (
            torch.mean(F.relu(a_energy - self.m_in) ** 2) +
            torch.mean(F.relu(p_energy - self.m_in) ** 2)
        ) / 2.0

        # 6. Pseudo-OOD Boundary Regularization
        # Synthesize virtual boundary negatives via convex interpolation between anchor and negative + noise
        B = a_z.size(0)
        lam = torch.rand(B, 1, device=a_z.device) * 0.4 + 0.3 # in [0.3, 0.7]
        noise = torch.randn_like(a_z) * 0.05
        z_pseudo = F.normalize(lam * a_z + (1.0 - lam) * n_z + noise, p=2, dim=-1)
        pseudo_energy = model.compute_energy_from_features(z_pseudo)
        # Enforce E(z_pseudo) > m_out (low confidence / high energy on pseudo-OOD)
        loss_pseudo_ood = torch.mean(F.relu(self.m_out - pseudo_energy) ** 2)

        # 7. Kolmogorov-Smirnov Temporal Motion Regularization
        # Align latent representation with genuine physical motion acceleration
        true_kin_a = compute_temporal_kinematic_energy(a).unsqueeze(-1)
        pred_kin_a = self.kinematic_head(a_z)
        loss_ks = F.mse_loss(pred_kin_a, true_kin_a)

        # Total Composite Loss
        total_loss = (
            loss_triplet +
            0.5 * loss_ce +
            0.2 * loss_compact +
            0.1 * loss_energy +
            0.1 * loss_pseudo_ood +
            0.1 * loss_ks
        )
        return total_loss, {
            "triplet": loss_triplet.item(),
            "ce": loss_ce.item(),
            "compact": loss_compact.item(),
            "energy": loss_energy.item(),
            "pseudo_ood": loss_pseudo_ood.item(),
            "ks": loss_ks.item()
        }


def train_one_fold(
    fold_num,
    train_idx,
    valid_idx,
    dataset,
    lr=1e-3,
    epochs=30,
    dropout=0.1,
    ash_percentile=80,
    ash_mode='ash_b',
    model_dir=".",
    log_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Fold {fold_num} on {device} (Skeleton-OOD v3.00, lr={lr}, ASH={ash_mode}@{ash_percentile}%) ---", flush=True)

    train_indices_mapped = [dataset.sample_indices[i] for i in train_idx if i < len(dataset.sample_indices)]
    valid_indices_mapped = [dataset.sample_indices[i] for i in valid_idx if i < len(dataset.sample_indices)]

    train_sub = Subset(dataset, train_indices_mapped)
    valid_sub = Subset(dataset, valid_indices_mapped)

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_sub, batch_size=32, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = SkeletonOODModel(
        num_classes=3,
        in_channels=2,
        num_keypoints=17,
        dropout=dropout,
        ash_percentile=ash_percentile,
        ash_mode=ash_mode,
        temperature=1.0
    ).to(device)

    criterion = SkeletonOODLoss(m_in=-4.0, m_out=-1.0, triplet_margin=1.0).to(device)
    params = list(model.parameters()) + list(criterion.kinematic_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=10, mode='min')

    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for a, p, n, a_cls, p_cls, n_cls in train_loader:
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)
            a_cls = a_cls.to(device, non_blocking=True)
            p_cls = p_cls.to(device, non_blocking=True)
            n_cls = n_cls.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, _ = criterion(model, a, p, n, a_cls, p_cls, n_cls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for a, p, n, a_cls, p_cls, n_cls in valid_loader:
                a = a.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)
                n = n.to(device, non_blocking=True)
                a_cls = a_cls.to(device, non_blocking=True)
                p_cls = p_cls.to(device, non_blocking=True)
                n_cls = n_cls.to(device, non_blocking=True)

                loss, _ = criterion(model, a, p, n, a_cls, p_cls, n_cls)
                valid_loss += loss.item()

        avg_valid = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        history.append({'epoch': epoch + 1, 'train_loss': avg_train, 'valid_loss': avg_valid})

        if (epoch + 1) % 5 == 0 or epoch == 0 or avg_valid < early_stopping.best_score:
            marker = "*" if avg_valid < early_stopping.best_score else ""
            print(f"Fold {fold_num} | Epoch [{epoch+1:2d}/{epochs:2d}] | Train: {avg_train:.4f} | Valid: {avg_valid:.4f} {marker}", flush=True)

        if early_stopping(avg_valid, model.state_dict()):
            print(f"Early stopping triggered at epoch {epoch+1}.", flush=True)
            break

    fold_path = os.path.join(model_dir, f"stgcn_skeleton_ood_v3.00_fold{fold_num}.pth")
    torch.save(early_stopping.best_model_state, fold_path)
    print(f"Fold {fold_num} champion saved -> {fold_path} (Valid Loss: {early_stopping.best_score:.4f})", flush=True)

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(log_dir, f"training_log_skeleton_ood_v3.00_fold{fold_num}.csv"), index=False)

    return fold_path, early_stopping.best_score, early_stopping.best_model_state


def main():
    print("========================================================================", flush=True)
    print("  Skeleton-OOD 3-Fold Latent Boundary Retraining - Version 3.00", flush=True)
    print("  ST-GCN Backbone + In-Training ASH + SE Fusion + Hyperspherical Boundary", flush=True)
    print("========================================================================", flush=True)

    train_dir, validate_dir = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]
    print(f"Training data path: {train_dir}")
    print(f"Action folders: {actions}")

    dataset = SkeletonOODDataset(
        root_dir=train_dir,
        action_folders=actions,
        clip_length=50,
        annotation_folder="annotations_yolo26s-pose"
    )

    weights_dir = os.path.join(REPO_DIR, "weights")
    results_tuning_dir = os.path.join(REPO_DIR, "results", "tuning")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_tuning_dir, exist_ok=True)

    groups = []
    for idx in dataset.sample_indices:
        groups.append(dataset.samples[idx]["source_file"])

    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(dataset.sample_indices, groups=groups))

    best_overall_score = float('inf')
    best_overall_fold = None
    best_overall_state = None
    fold_summaries = []

    for fold_num, (t_idx, v_idx) in enumerate(splits, start=1):
        fold_path, best_score, best_state = train_one_fold(
            fold_num=fold_num,
            train_idx=t_idx,
            valid_idx=v_idx,
            dataset=dataset,
            lr=1e-3,
            epochs=30,
            dropout=0.1,
            ash_percentile=80,
            ash_mode='ash_b',
            model_dir=weights_dir,
            log_dir=results_tuning_dir
        )
        fold_summaries.append({
            "fold": fold_num,
            "best_valid_loss": best_score,
            "weights_path": fold_path
        })
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_fold = fold_num
            best_overall_state = best_state

    # Save champion model
    champion_path = os.path.join(weights_dir, "stgcn_skeleton_ood_v3.00.pth")
    torch.save(best_overall_state, champion_path)
    print(f"\nChampion model selected: Fold {best_overall_fold} (Valid Loss: {best_overall_score:.4f}) -> {champion_path}", flush=True)

    summary_file = os.path.join(results_tuning_dir, "training_summary_skeleton_ood_v3.00.json")
    with open(summary_file, "w") as f:
        json.dump({
            "model": "SkeletonOODModel",
            "backbone": "ST-GCN (9-block with arm weighting)",
            "fusion": "ASH-B@80% + SE + MLP",
            "boundary": "Unit Hypersphere S^{255} + Free Energy",
            "champion_fold": best_overall_fold,
            "champion_valid_loss": best_overall_score,
            "folds": fold_summaries
        }, f, indent=4)
    print(f"Training summary saved -> {summary_file}", flush=True)


if __name__ == "__main__":
    main()
