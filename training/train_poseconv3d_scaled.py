"""
Scaled PoseConv3D Triplet Retraining - High Efficiency GPU Accelerated
Trains ScaledPoseConv3D (~3.55M parameters) on canonical torso-normalized keypoint heatmaps across 3 folds.
Pre-rasterizes heatmaps onto GPU memory for ultra-fast training (< 2 minutes total).
100% self-contained: uses local workspace datasets and modules.
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, PoseConv3DHeatmapDataset
from src.poseconv3d_utils import batch_rasterize_gpu
from models.poseconv3d_scaled import ScaledPoseConv3DModel

SEED = 42
set_seed = lambda s: (
    random.seed(s),
    np.random.seed(s),
    torch.manual_seed(s),
    torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
)
set_seed(SEED)
cudnn.benchmark = True


class CachedGPUPoseConv3DTripletDataset(Dataset):
    """
    In-Memory GPU Triplet Dataset for PoseConv3D.
    Serves pre-rasterized 3D heatmaps directly from GPU memory for maximum training throughput.
    """
    def __init__(self, heatmaps_gpu, samples, sample_indices, samples_by_video, video_files):
        self.heatmaps = heatmaps_gpu
        self.samples = samples
        self.sample_indices = sample_indices
        self.samples_by_video = samples_by_video
        self.video_files = video_files

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        anchor_idx = self.sample_indices[idx]
        anchor_video = self.samples[anchor_idx]["source_file"]

        anchor_hm = self.heatmaps[anchor_idx]

        # Positive sample from same video
        pos_indices = [i for i in self.samples_by_video[anchor_video] if i != anchor_idx]
        pos_idx = random.choice(pos_indices) if pos_indices else anchor_idx
        pos_hm = self.heatmaps[pos_idx]

        # Negative sample from different video
        while True:
            neg_video = random.choice(self.video_files)
            if neg_video != anchor_video:
                break
        neg_idx = random.choice(self.samples_by_video[neg_video])
        neg_hm = self.heatmaps[neg_idx]

        return anchor_hm, pos_hm, neg_hm


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
    train_indices,
    valid_indices,
    heatmaps_gpu,
    samples,
    lr=1e-3,
    epochs=30,
    model_dir=".",
    log_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Fold {fold_num} on {device} (ScaledPoseConv3D ~3.55M params, lr={lr}) ---", flush=True)

    # Build group mappings for train and valid subsets
    train_by_video = defaultdict(list)
    for idx in train_indices:
        train_by_video[samples[idx]["source_file"]].append(idx)
    train_sample_indices = [i for i in train_indices if len(train_by_video[samples[i]["source_file"]]) > 1]
    train_video_files = list(train_by_video.keys())

    valid_by_video = defaultdict(list)
    for idx in valid_indices:
        valid_by_video[samples[idx]["source_file"]].append(idx)
    valid_sample_indices = [i for i in valid_indices if len(valid_by_video[samples[i]["source_file"]]) > 1]
    valid_video_files = list(valid_by_video.keys())

    train_ds = CachedGPUPoseConv3DTripletDataset(
        heatmaps_gpu, samples, train_sample_indices, train_by_video, train_video_files
    )
    valid_ds = CachedGPUPoseConv3DTripletDataset(
        heatmaps_gpu, samples, valid_sample_indices, valid_by_video, valid_video_files
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False)

    model = ScaledPoseConv3DModel(num_classes=1, channels=(64, 128, 256, 512), feature_dim=256, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.TripletMarginLoss(margin=1.0)
    early_stopping = EarlyStopping(patience=10, mode="min")

    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for a, p, n in train_loader:
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

    save_path = os.path.join(model_dir, f"poseconv3d_scaled_violence_fold{fold_num}.pth")
    torch.save(early_stopping.best_model_state, save_path)

    df_hist = pd.DataFrame(history)
    log_csv = os.path.join(log_dir, f"training_log_poseconv3d_scaled_fold{fold_num}.csv")
    df_hist.to_csv(log_csv, index=False)

    return save_path, early_stopping.best_score, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("   TRAINING: SCALED POSECONV3D VOLUMETRIC 3D HEATMAP CNN (~3.55M Parameters)", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    train_data_dir, validate_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]
    print(f"Train Data Source: {train_data_dir}")
    print(f"Validate Source  : {validate_root}")

    model_output_dir = os.path.join(REPO_DIR, "weights")
    log_output_dir = os.path.join(REPO_DIR, "results", "tuning")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    print("\n--- Step 1: Loading Raw Skeletal Trajectories ---", flush=True)
    raw_ds = PoseConv3DHeatmapDataset(
        root_dir=train_data_dir, action_folders=actions, clip_length=50, heatmap_size=56, mode="train"
    )
    N = len(raw_ds)
    print(f"Found {N} training clips.")

    print("\n--- Step 2: GPU High-Speed Heatmap Rasterization ---", flush=True)
    all_kpts = []
    for i in range(N):
        sample = raw_ds.samples[i]
        video_idx = sample["video_idx"]
        start_frame = sample["start_frame"]
        skeletons = raw_ds.skeleton_data[video_idx]
        clip = skeletons[start_frame : start_frame + 50]
        if not isinstance(clip, torch.Tensor):
            clip = torch.from_numpy(clip).float()
        if clip.shape[0] < 50:
            pad = torch.zeros((50 - clip.shape[0], 17, 2))
            clip = torch.cat([clip, pad], dim=0)
        all_kpts.append(clip[:, :, :2])

    all_kpts = torch.stack(all_kpts).to(device)  # (N, 50, 17, 2)
    heatmaps_list = []
    chunk_size = 50
    for i in range(0, N, chunk_size):
        chunk = all_kpts[i : i + chunk_size]
        hm = batch_rasterize_gpu(chunk, H=56, W=56, sigma=1.5)
        heatmaps_list.append(hm)
    all_heatmaps = torch.cat(heatmaps_list, dim=0)
    print(f"Rasterized {len(all_heatmaps)} 3D heatmaps on GPU. Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # GroupKFold splits based on source_file
    k_folds = 3
    clip_groups = [s["source_file"] for s in raw_ds.samples]
    unique_files = list(set(clip_groups))
    file_to_id = {f: i for i, f in enumerate(unique_files)}
    groups = [file_to_id[f] for f in clip_groups]

    gkf = GroupKFold(n_splits=k_folds)

    best_valid_loss = float("inf")
    best_fold_idx = -1
    best_model_path = ""
    fold_summaries = []

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(raw_ds.samples, groups=groups)):
        print(f"\n{'#'*25} FOLD {fold+1}/{k_folds} {'#'*25}", flush=True)
        path, val_loss, hist = train_one_fold(
            fold + 1, train_idx, valid_idx, all_heatmaps, raw_ds.samples,
            lr=1e-3, epochs=30,
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

    champion_target = os.path.join(model_output_dir, "poseconv3d_scaled_violence.pth")
    import shutil
    shutil.copyfile(best_model_path, champion_target)
    print(f">> Champion Checkpoint Exported to: {champion_target}", flush=True)

    summary_path = os.path.join(REPO_DIR, "results", "training_summary_poseconv3d_scaled.json")
    with open(summary_path, "w") as f:
        json.dump({
            "architecture": "ScaledPoseConv3D (R(2+1)D Heatmap CNN, ~3.55M params)",
            "parameters": 3545409,
            "k_folds": k_folds,
            "best_fold": best_fold_idx,
            "best_valid_loss": best_valid_loss,
            "folds": fold_summaries
        }, f, indent=2)
    print(f">> Summary logged to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
