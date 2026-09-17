"""
PoseConv3D Triplet Retraining with Volumetric 3D Heatmap CNN - Version 2.10 (Refined)
Trains PoseConv3D on canonical torso-normalized spatio-temporal keypoint heatmaps across 3 folds.
100% self-contained: uses local workspace datasets and modules. Zero external dependencies.
"""

import os
import sys
import json
import random
import copy
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import (
    resolve_data_paths,
    PoseConv3DHeatmapDataset,
    PoseConv3DTripletHeatmapDataset,
)
from models.poseconv3d import PoseConv3DModel

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


class EarlyStopping:
    def __init__(self, patience=10, mode="min", min_delta=0.0001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_model_state = None

    def __call__(self, score, model_state):
        improved = (score < self.best_score - self.min_delta) if self.mode == "min" else (score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model_state)
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_fold(
    fold_num,
    train_idx,
    valid_idx,
    dataset,
    lr=1e-3,
    epochs=30,
    base_channels=24,
    dropout=0.2,
    model_dir=".",
    log_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Fold {fold_num} on {device} (PoseConv3D v2.10, lr={lr}, base_channels={base_channels}) ---", flush=True)

    train_indices_mapped = [dataset.sample_indices[i] for i in train_idx if i < len(dataset.sample_indices)]
    valid_indices_mapped = [dataset.sample_indices[i] for i in valid_idx if i < len(dataset.sample_indices)]

    train_sub = Subset(dataset, train_indices_mapped)
    valid_sub = Subset(dataset, valid_indices_mapped)

    train_loader = DataLoader(train_sub, batch_size=16, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_sub, batch_size=16, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = PoseConv3DModel(num_classes=1, base_channels=base_channels, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.TripletMarginLoss(margin=1.0)
    early_stopping = EarlyStopping(patience=10, mode="min")

    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for a, p, n in train_loader:
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            n = n.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, af = model(a, return_features=True)
            _, pf = model(p, return_features=True)
            _, nf = model(n, return_features=True)

            loss = criterion(af, pf, nf)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for a, p, n in valid_loader:
                a = a.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)
                n = n.to(device, non_blocking=True)

                _, af = model(a, return_features=True)
                _, pf = model(p, return_features=True)
                _, nf = model(n, return_features=True)

                loss = criterion(af, pf, nf)
                valid_loss += loss.item()

        avg_valid = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        history.append({"epoch": epoch + 1, "train_loss": avg_train, "valid_loss": avg_valid})

        if (epoch + 1) % 5 == 0 or epoch == 0 or avg_valid < early_stopping.best_score:
            print(f"Fold {fold_num} | Epoch [{epoch+1:2d}/{epochs:2d}] | Train: {avg_train:.4f} | Valid: {avg_valid:.4f} {'*' if avg_valid < early_stopping.best_score else ''}", flush=True)

        if early_stopping(avg_valid, model.state_dict()):
            print(f"Early stopping triggered at epoch {epoch+1}.", flush=True)
            break

    save_path = os.path.join(model_dir, f"poseconv3d_violence_v2.10_fold{fold_num}.pth")
    torch.save(early_stopping.best_model_state, save_path)

    df_hist = pd.DataFrame(history)
    log_csv = os.path.join(log_dir, f"training_log_poseconv3d_v2.10_fold{fold_num}.csv")
    df_hist.to_csv(log_csv, index=False)

    return save_path, early_stopping.best_score, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("   TRAINING: POSECONV3D CANONICAL VOLUMETRIC 3D HEATMAP CNN (Version 2.10 Refined)", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    train_data_dir, validate_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]
    print(f"Train Data Source: {train_data_dir}")
    print(f"Validate Source  : {validate_root}")

    model_output_dir = os.path.join(REPO_DIR, "weights")
    log_output_dir = os.path.join(REPO_DIR, "results")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    full_dataset = PoseConv3DTripletHeatmapDataset(
        train_data_dir, actions, clip_length=50, heatmap_size=56, sigma=1.5
    )

    k_folds = 3
    clip_groups = [s["source_file"] for s in full_dataset.samples]
    unique_files = list(set(clip_groups))
    file_to_id = {f: i for i, f in enumerate(unique_files)}
    groups = [file_to_id[f] for f in clip_groups]

    gkf = GroupKFold(n_splits=k_folds)

    best_valid_loss = float("inf")
    best_fold_idx = -1
    best_model_path = ""
    fold_summaries = []

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(full_dataset.samples, groups=groups)):
        print(f"\n{'#'*25} FOLD {fold+1}/{k_folds} {'#'*25}", flush=True)
        path, val_loss, hist = train_one_fold(
            fold + 1, train_idx, valid_idx, full_dataset,
            lr=1e-3, epochs=30, base_channels=24, dropout=0.2,
            model_dir=model_output_dir, log_dir=log_output_dir
        )
        fold_summaries.append({
            "fold": fold + 1,
            "train_samples": len(train_idx),
            "valid_samples": len(valid_idx),
            "final_train_loss": round(hist[-1]["train_loss"], 4),
            "best_valid_loss": round(val_loss, 4),
            "checkpoint": path
        })

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_fold_idx = fold + 1
            best_model_path = path

    print(f"\n>> 3-FOLD TRAINING COMPLETE!", flush=True)
    print(f">> BEST FOLD: Fold {best_fold_idx} (Valid Loss: {best_valid_loss:.4f})", flush=True)
    print(f">> Champion Model Path: {best_model_path}", flush=True)

    champion_target = os.path.join(model_output_dir, "poseconv3d_violence_v2.10.pth")
    import shutil
    shutil.copyfile(best_model_path, champion_target)
    print(f">> Champion Checkpoint Exported to: {champion_target}", flush=True)

    summary_path = os.path.join(log_output_dir, "training_summary_poseconv3d_v2.10.json")
    with open(summary_path, "w") as f:
        json.dump({
            "architecture": "PoseConv3D (R(2+1)D Torso-Normalized Volumetric Heatmap CNN)",
            "version": "2.10",
            "k_folds": k_folds,
            "best_fold": best_fold_idx,
            "best_valid_loss": best_valid_loss,
            "folds": fold_summaries
        }, f, indent=2)
    print(f">> Summary logged to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
