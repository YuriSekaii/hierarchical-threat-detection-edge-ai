"""
train_poseconv3d_downscale_sweep.py

Automated Multi-Tier Training Pipeline for Downscaled PoseConv3D Models.
Pre-rasterizes the 297 training clips once onto GPU VRAM (6.34 GB) and sequentially trains:
- Tier 1: base_channels=20, feature_dim=256 (~703k params)
- Tier 2: base_channels=16, feature_dim=256 (~647k params)
- Tier 3: base_channels=12, feature_dim=256 (~598k params)
- Tier 4: base_channels=16, feature_dim=128 (~232k params)
- Tier 5: base_channels=12, feature_dim=96  (~132k params)

Using Triplet Margin Loss, AdamW, and Cosine Annealing with Early Stopping.
100% self-contained: adheres strictly to AgentRule.md.
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
from models.poseconv3d import PoseConv3DModel

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


def train_single_model(
    tier_name,
    bc,
    fd,
    heatmaps_gpu,
    samples,
    lr=1e-3,
    epochs=30,
    model_dir=".",
    log_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_proto = PoseConv3DModel(base_channels=bc, feature_dim=fd, dropout=0.2).to(device)
    params = sum(p.numel() for p in model_proto.parameters())
    print(f"\n{'='*70}", flush=True)
    print(f"   TRAINING {tier_name.upper()}: bc={bc}, fd={fd} -> {params:,} parameters", flush=True)
    print(f"{'='*70}", flush=True)

    # 3-Fold GroupKFold split
    clip_groups = [s["source_file"] for s in samples]
    unique_files = list(set(clip_groups))
    file_to_id = {f: i for i, f in enumerate(unique_files)}
    groups = [file_to_id[f] for f in clip_groups]
    gkf = GroupKFold(n_splits=3)

    best_fold_loss = float("inf")
    best_fold_state = None
    best_fold_idx = -1

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(samples, groups=groups)):
        train_by_video = defaultdict(list)
        for idx in train_idx:
            train_by_video[samples[idx]["source_file"]].append(idx)
        train_sample_indices = [i for i in train_idx if len(train_by_video[samples[i]["source_file"]]) > 1]
        train_video_files = list(train_by_video.keys())

        valid_by_video = defaultdict(list)
        for idx in valid_idx:
            valid_by_video[samples[idx]["source_file"]].append(idx)
        valid_sample_indices = [i for i in valid_idx if len(valid_by_video[samples[i]["source_file"]]) > 1]
        valid_video_files = list(valid_by_video.keys())

        train_ds = CachedGPUPoseConv3DTripletDataset(heatmaps_gpu, samples, train_sample_indices, train_by_video, train_video_files)
        valid_ds = CachedGPUPoseConv3DTripletDataset(heatmaps_gpu, samples, valid_sample_indices, valid_by_video, valid_video_files)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False)

        model = PoseConv3DModel(base_channels=bc, feature_dim=fd, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.TripletMarginLoss(margin=1.0)
        early_stopping = EarlyStopping(patience=10, mode="min")

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

            if early_stopping(avg_valid, model.state_dict()):
                break

        print(f"  {tier_name} Fold {fold+1} | Best Valid Loss: {early_stopping.best_score:.4f}", flush=True)

        if early_stopping.best_score < best_fold_loss:
            best_fold_loss = early_stopping.best_score
            best_fold_state = early_stopping.best_model_state
            best_fold_idx = fold + 1

    save_path = os.path.join(model_dir, f"poseconv3d_downscale_{tier_name}.pth")
    torch.save(best_fold_state, save_path)
    print(f">> {tier_name.upper()} Champion: Fold {best_fold_idx} (Val Loss: {best_fold_loss:.4f}) -> {save_path}", flush=True)

    return {
        "tier": tier_name,
        "base_channels": bc,
        "feature_dim": fd,
        "parameters": params,
        "best_fold": best_fold_idx,
        "best_valid_loss": float(best_fold_loss),
        "checkpoint": save_path
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("   AUTOMATED TRAINING: DOWNSCALED POSECONV3D SWEEP", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    train_data_dir, validate_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    model_output_dir = os.path.join(REPO_DIR, "weights")
    log_output_dir = os.path.join(REPO_DIR, "results", "tuning")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_output_dir, exist_ok=True)

    print("\n--- Step 1: Loading & GPU Pre-Rasterizing 297 Heatmaps ---", flush=True)
    raw_ds = PoseConv3DHeatmapDataset(root_dir=train_data_dir, action_folders=actions, clip_length=50, heatmap_size=56, mode="train")
    N = len(raw_ds)

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

    all_kpts = torch.stack(all_kpts).to(device)
    heatmaps_list = []
    for i in range(0, N, 50):
        chunk = all_kpts[i : i + 50]
        hm = batch_rasterize_gpu(chunk, H=56, W=56, sigma=1.5)
        heatmaps_list.append(hm)
    all_heatmaps = torch.cat(heatmaps_list, dim=0)
    print(f"Rasterized {len(all_heatmaps)} heatmaps on GPU VRAM ({torch.cuda.memory_allocated()/1e9:.2f} GB).", flush=True)

    tiers = [
        ("tier1_703k", 20, 256),
        ("tier2_647k", 16, 256),
        ("tier3_598k", 12, 256),
        ("tier4_232k", 16, 128),
        ("tier5_132k", 12, 96),
    ]

    tier_results = []
    for t_name, bc, fd in tiers:
        res = train_single_model(
            t_name, bc, fd, all_heatmaps, raw_ds.samples,
            lr=1e-3, epochs=30,
            model_dir=model_output_dir, log_dir=log_output_dir
        )
        tier_results.append(res)

    summary_file = os.path.join(REPO_DIR, "results", "training_summary_poseconv3d_downscale_sweep.json")
    with open(summary_file, "w") as f:
        json.dump(tier_results, f, indent=2)
    print(f"\n>> All 5 Downscaled Tiers Trained! Summary logged to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
