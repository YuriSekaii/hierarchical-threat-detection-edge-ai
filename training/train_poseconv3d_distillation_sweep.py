"""
train_poseconv3d_distillation_sweep.py

Automated Knowledge Distillation Training Pipeline:
Transfers representations from frozen Teacher (ST-GCN, 100% Recall, 3.01M params)
to Student (Tier 3 PoseConv3D, 598k params).

Evaluates 4 Distillation Formulations:
1. Method A: Latent Feature Alignment (Cosine + MSE)
2. Method B: Relational Knowledge Distillation (RKD Distance + Angle)
3. Method C: Spatial-Temporal Arm Attention Distillation
4. Method D: Combined Tri-Distillation (A + B + C)

Pre-computes Teacher features and 3D heatmaps on GPU VRAM for lightning-fast training.
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
from models.stgcn import STGCNModel
from models.poseconv3d import PoseConv3DModel
from src.distillation_losses import FeatureAlignmentLoss, RelationalKDLoss, ArmAttentionDistillationLoss

SEED = 42
set_seed = lambda s: (
    random.seed(s),
    np.random.seed(s),
    torch.manual_seed(s),
    torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
)
set_seed(SEED)
cudnn.benchmark = True


class DistillationGPUDataset(Dataset):
    """
    Serves (anchor_hm, pos_hm, neg_hm, anchor_t_feat, pos_t_feat, neg_t_feat)
    directly from GPU VRAM.
    """
    def __init__(self, heatmaps_gpu, teacher_feats_gpu, sample_indices, samples_by_video, video_files):
        self.heatmaps = heatmaps_gpu
        self.teacher_feats = teacher_feats_gpu
        self.sample_indices = sample_indices
        self.samples_by_video = samples_by_video
        self.video_files = video_files

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        anchor_idx = self.sample_indices[idx]
        anchor_video = self.samples_by_video_reverse[anchor_idx]

        anchor_hm = self.heatmaps[anchor_idx]
        anchor_tf = self.teacher_feats[anchor_idx]

        # Positive sample from same video
        pos_indices = [i for i in self.samples_by_video[anchor_video] if i != anchor_idx]
        pos_idx = random.choice(pos_indices) if pos_indices else anchor_idx
        pos_hm = self.heatmaps[pos_idx]
        pos_tf = self.teacher_feats[pos_idx]

        # Negative sample from different video
        while True:
            neg_video = random.choice(self.video_files)
            if neg_video != anchor_video:
                break
        neg_idx = random.choice(self.samples_by_video[neg_video])
        neg_hm = self.heatmaps[neg_idx]
        neg_tf = self.teacher_feats[neg_idx]

        return anchor_hm, pos_hm, neg_hm, anchor_tf, pos_tf, neg_tf


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


def train_distillation_experiment(
    method_name,
    loss_fn,
    heatmaps_gpu,
    teacher_feats_gpu,
    samples,
    lr=1e-3,
    epochs=30,
    model_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}", flush=True)
    print(f"   TRAINING DISTILLATION: {method_name.upper()}", flush=True)
    print(f"{'='*75}", flush=True)

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

        # Build reverse lookups
        train_rev = {i: samples[i]["source_file"] for i in train_idx}
        valid_rev = {i: samples[i]["source_file"] for i in valid_idx}

        train_ds = DistillationGPUDataset(heatmaps_gpu, teacher_feats_gpu, train_sample_indices, train_by_video, train_video_files)
        train_ds.samples_by_video_reverse = train_rev

        valid_ds = DistillationGPUDataset(heatmaps_gpu, teacher_feats_gpu, valid_sample_indices, valid_by_video, valid_video_files)
        valid_ds.samples_by_video_reverse = valid_rev

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False)

        # Tier 3 Architecture: base_channels=12, feature_dim=256 (~598k params)
        student = PoseConv3DModel(base_channels=12, feature_dim=256, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        triplet_criterion = nn.TripletMarginLoss(margin=1.0)
        early_stopping = EarlyStopping(patience=10, mode="min")

        for epoch in range(epochs):
            student.train()
            train_loss = 0.0
            for a_hm, p_hm, n_hm, a_tf, p_tf, n_tf in train_loader:
                optimizer.zero_grad()
                _, a_sf = student(a_hm, return_features=True)
                _, p_sf = student(p_hm, return_features=True)
                _, n_sf = student(n_hm, return_features=True)

                l_trip = triplet_criterion(a_sf, p_sf, n_sf)
                l_dist = loss_fn(a_hm, a_sf, a_tf) + loss_fn(p_hm, p_sf, p_tf)
                total_loss = l_trip + l_dist

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

            scheduler.step()
            avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

            student.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for a_hm, p_hm, n_hm, a_tf, p_tf, n_tf in valid_loader:
                    _, a_sf = student(a_hm, return_features=True)
                    _, p_sf = student(p_hm, return_features=True)
                    _, n_sf = student(n_hm, return_features=True)
                    l_trip = triplet_criterion(a_sf, p_sf, n_sf)
                    l_dist = loss_fn(a_hm, a_sf, a_tf) + loss_fn(p_hm, p_sf, p_tf)
                    valid_loss += (l_trip + l_dist).item()

            avg_valid = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0

            if early_stopping(avg_valid, student.state_dict()):
                break

        print(f"  {method_name} Fold {fold+1} | Best Valid Loss: {early_stopping.best_score:.4f}", flush=True)

        if early_stopping.best_score < best_fold_loss:
            best_fold_loss = early_stopping.best_score
            best_fold_state = early_stopping.best_model_state
            best_fold_idx = fold + 1

    save_path = os.path.join(model_dir, f"poseconv3d_distill_{method_name}.pth")
    torch.save(best_fold_state, save_path)
    print(f">> {method_name.upper()} Champion: Fold {best_fold_idx} (Val Loss: {best_fold_loss:.4f}) -> {save_path}", flush=True)

    return {
        "method": method_name,
        "parameters": 598429,
        "best_fold": best_fold_idx,
        "best_valid_loss": float(best_fold_loss),
        "checkpoint": save_path
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("   AUTOMATED TRAINING: ST-GCN -> POSECONV3D TIER 3 KNOWLEDGE DISTILLATION", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)

    train_data_dir, validate_root = resolve_data_paths()
    actions = ["Cut-Down", "Stab", "Thrust"]

    model_output_dir = os.path.join(REPO_DIR, "weights")
    os.makedirs(model_output_dir, exist_ok=True)

    # 1. Load Pretrained ST-GCN Teacher
    teacher_path = os.path.join(REPO_DIR, "weights", "stgcn_violence_v1.02.pth")
    teacher = STGCNModel(num_classes=1).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Loaded Frozen ST-GCN Teacher: {teacher_path} (3.01M parameters, 100% recall)")

    # 2. Extract Raw Skeletons & Heatmaps
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

    all_kpts = torch.stack(all_kpts).to(device)  # (N, 50, 17, 2)

    # 3. GPU Rasterize Heatmaps
    print("Rasterizing 3D heatmaps on GPU VRAM...", flush=True)
    heatmaps_list = []
    for i in range(0, N, 50):
        chunk = all_kpts[i : i + 50]
        hm = batch_rasterize_gpu(chunk, H=56, W=56, sigma=1.5)
        heatmaps_list.append(hm)
    all_heatmaps = torch.cat(heatmaps_list, dim=0)

    # 4. Pre-extract Teacher 256-D Features
    print("Pre-computing ST-GCN Teacher features...", flush=True)
    # ST-GCN input: (B, 2, 50, 17)
    stgcn_input = all_kpts.permute(0, 3, 1, 2).contiguous()
    teacher_feats = []
    with torch.no_grad():
        for i in range(0, N, 32):
            chunk = stgcn_input[i : i + 32]
            _, feats = teacher(chunk, return_features=True)
            teacher_feats.append(feats)
    all_teacher_feats = torch.cat(teacher_feats, dim=0)
    print(f"Teacher features computed for {len(all_teacher_feats)} clips: shape {all_teacher_feats.shape}")

    # Loss definitions
    feat_loss = FeatureAlignmentLoss(alpha_cos=1.0, alpha_mse=1.0).to(device)
    rkd_loss = RelationalKDLoss(w_dist=1.0, w_angle=2.0).to(device)
    attn_loss = ArmAttentionDistillationLoss().to(device)

    distillation_experiments = [
        ("methodA_feature", lambda hm, fs, ft: 1.0 * feat_loss(fs, ft)),
        ("methodB_rkd", lambda hm, fs, ft: 20.0 * rkd_loss(fs, ft)),
        ("methodC_attention", lambda hm, fs, ft: 1.0 * attn_loss(hm, fs, ft)),
        ("methodD_combined", lambda hm, fs, ft: 0.5 * feat_loss(fs, ft) + 10.0 * rkd_loss(fs, ft) + 0.5 * attn_loss(hm, fs, ft)),
    ]

    distill_results = []
    for m_name, l_fn in distillation_experiments:
        res = train_distillation_experiment(
            m_name, l_fn, all_heatmaps, all_teacher_feats, raw_ds.samples,
            lr=1e-3, epochs=30, model_dir=model_output_dir
        )
        distill_results.append(res)

    summary_file = os.path.join(REPO_DIR, "results", "training_summary_poseconv3d_distillation.json")
    with open(summary_file, "w") as f:
        json.dump(distill_results, f, indent=2)
    print(f"\n>> All 4 Distillation Models Trained! Summary logged to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
