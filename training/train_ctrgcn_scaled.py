"""
training/train_ctrgcn_scaled.py

Trains Scaled CTR-GCN variants (Tier S and Tier L) across 3 folds on scale-normalized
skeletal action recognition data.
Exports champion checkpoints:
- weights/ctrgcn_scaled_tier_s.pth (~353k parameters)
- weights/ctrgcn_scaled_tier_l.pth (~2.97M parameters)
"""

import os
import sys
import json
import random
import copy
import argparse
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

from models.ctrgcn_scaled import ScaledCTRGCNModel
from src.dataset import resolve_data_paths, TripletActionDataset

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


def train_one_fold(
    fold_num, train_idx, valid_idx, dataset,
    channels, reduction, arm_weight=5.0,
    lr=1e-3, epochs=30, dropout=0.1,
    tier_name="tier_s", model_dir=".", log_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Fold {fold_num} on {device} (CTR-GCN {tier_name}, channels={channels}, lr={lr}) ---", flush=True)

    train_indices_mapped = [dataset.sample_indices[i] for i in train_idx if i < len(dataset.sample_indices)]
    valid_indices_mapped = [dataset.sample_indices[i] for i in valid_idx if i < len(dataset.sample_indices)]

    train_sub = Subset(dataset, train_indices_mapped)
    valid_sub = Subset(dataset, valid_indices_mapped)

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_sub, batch_size=32, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ScaledCTRGCNModel(
        num_classes=1,
        channels=channels,
        reduction=reduction,
        arm_weight=arm_weight,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.TripletMarginLoss(margin=1.0)
    early_stopping = EarlyStopping(patience=10, mode='min')

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
        history.append({'epoch': epoch + 1, 'train_loss': avg_train, 'valid_loss': avg_valid})

        if (epoch + 1) % 5 == 0 or epoch == 0 or avg_valid < early_stopping.best_score:
            print(f"Fold {fold_num} | Epoch [{epoch+1:2d}/{epochs:2d}] | Train: {avg_train:.4f} | Valid: {avg_valid:.4f} {'*' if avg_valid < early_stopping.best_score else ''}", flush=True)

        if early_stopping(avg_valid, model.state_dict()):
            print(f"Early stopping triggered at epoch {epoch+1}.", flush=True)
            break

    save_path = os.path.join(model_dir, f"ctrgcn_{tier_name}_fold{fold_num}.pth")
    torch.save(early_stopping.best_model_state, save_path)

    # Save log CSV
    df_hist = pd.DataFrame(history)
    log_csv = os.path.join(log_dir, f"training_log_ctrgcn_{tier_name}_fold{fold_num}.csv")
    df_hist.to_csv(log_csv, index=False)

    return save_path, early_stopping.best_score, history


def train_tier(tier_name, channels, reduction, arm_weight=5.0, epochs=30, lr=1e-3):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_temp = ScaledCTRGCNModel(channels=channels, reduction=reduction, arm_weight=arm_weight)
    num_params = sum(p.numel() for p in model_temp.parameters())
    print("=" * 80, flush=True)
    print(f"   TRAINING CTR-GCN {tier_name.upper()} ({num_params:,} parameters)", flush=True)
    print(f"   Channels: {channels} | Reduction: {reduction} | Arm Weight: {arm_weight}x | Device: {device}", flush=True)
    print("=" * 80, flush=True)

    train_data_dir, validate_root = resolve_data_paths()
    actions = ['Cut-Down', 'Stab', 'Thrust']

    model_output_dir = os.path.join(REPO_DIR, "weights")
    log_output_dir = os.path.join(REPO_DIR, "results")
    tuning_dir = os.path.join(REPO_DIR, "results", "tuning")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)
    os.makedirs(tuning_dir, exist_ok=True)

    full_dataset = TripletActionDataset(train_data_dir, actions)

    k_folds = 3
    clip_groups = [s['source_file'] for s in full_dataset.samples]
    unique_files = list(set(clip_groups))
    file_to_id = {f: i for i, f in enumerate(unique_files)}
    groups = [file_to_id[f] for f in clip_groups]

    gkf = GroupKFold(n_splits=k_folds)

    best_valid_loss = float('inf')
    best_fold_idx = -1
    best_model_path = ""
    fold_summaries = []

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(full_dataset.samples, groups=groups)):
        print(f"\n{'#'*25} {tier_name.upper()} FOLD {fold+1}/{k_folds} {'#'*25}", flush=True)
        path, val_loss, hist = train_one_fold(
            fold + 1, train_idx, valid_idx, full_dataset,
            channels=channels, reduction=reduction, arm_weight=arm_weight,
            lr=lr, epochs=epochs, dropout=0.1,
            tier_name=tier_name, model_dir=model_output_dir, log_dir=log_output_dir
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

    print(f"\n>> {tier_name.upper()} 3-FOLD TRAINING COMPLETE!", flush=True)
    print(f">> BEST FOLD: Fold {best_fold_idx} (Valid Loss: {best_valid_loss:.4f})", flush=True)
    print(f">> Champion Model Path: {best_model_path}", flush=True)

    # Copy champion model to production checkpoint name
    champion_target = os.path.join(model_output_dir, f"ctrgcn_{tier_name}.pth")
    import shutil
    shutil.copyfile(best_model_path, champion_target)
    print(f">> Champion Checkpoint Saved as: {champion_target}", flush=True)

    # Save fold summary JSON
    summary_path = os.path.join(tuning_dir, f"training_summary_ctrgcn_{tier_name}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "tier": tier_name,
            "parameters": num_params,
            "channels": channels,
            "reduction": reduction,
            "arm_weight": arm_weight,
            "k_folds": k_folds,
            "best_fold": best_fold_idx,
            "best_valid_loss": round(best_valid_loss, 4),
            "folds": fold_summaries
        }, f, indent=2)
    print(f">> Fold training summary saved to: {summary_path}", flush=True)
    return champion_target


def main():
    parser = argparse.ArgumentParser(description="Train Scaled CTR-GCN models")
    parser.add_argument("--tier", type=str, default="all", choices=["all", "tier_s", "tier_l"],
                        help="Which tier to train: 'tier_s', 'tier_l', or 'all'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs per fold")
    args = parser.parse_args()

    if args.tier in ["all", "tier_s"]:
        train_tier(
            tier_name="tier_s",
            channels=[32, 64, 128],
            reduction=4,
            arm_weight=5.0,
            epochs=args.epochs
        )

    if args.tier in ["all", "tier_l"]:
        train_tier(
            tier_name="tier_l",
            channels=[96, 192, 384],
            reduction=8,
            arm_weight=5.0,
            epochs=args.epochs
        )


if __name__ == "__main__":
    main()
