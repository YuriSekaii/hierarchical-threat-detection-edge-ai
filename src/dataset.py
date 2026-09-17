"""
Universal Action Recognition Dataset Loaders & Resolvers
Master Shared Module for Hierarchical Threat Detection & Edge AI.

Features:
- Completely self-contained: resolves local `data/` folder first.
- XML parsing with automatic joint interpolation, Gaussian temporal smoothing,
  and torso-scale invariant kinematic normalization.
- Universal `ActionDataset`, `TripletActionDataset`, and `PoseConv3DDataset`.
- Zero dependency on legacy milestone scripts or external folders.
"""

import os
import sys
import glob
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.skeleton_utils import (
    interpolate_missing_joints,
    smooth_kinematics,
    normalize_skeleton_clip,
)


def resolve_data_paths():
    """
    Resolves the dataset root directories with clean priority:
    1. Local workspace directory: hierarchical-threat-detection-edge-ai/data
    2. Clean backup directory: Clean/Train_Action_Recognition_STGCN_Model/data
    """
    candidates_train = [
        os.path.join(REPO_DIR, "data"),
        os.path.join(REPO_DIR, "..", "Clean", "Train_Action_Recognition_STGCN_Model", "data"),
    ]
    train_dir = None
    for p in candidates_train:
        if os.path.exists(os.path.join(p, "Cut-Down")):
            train_dir = os.path.abspath(p)
            break

    if train_dir is None:
        raise FileNotFoundError(
            f"Could not find action training data folder in {[os.path.abspath(p) for p in candidates_train]}"
        )

    # Resolve validation root
    candidates_val = [
        os.path.join(train_dir, "Validate"),
        os.path.join(train_dir, "Validation"),
        os.path.join(REPO_DIR, "data", "Validate"),
        os.path.join(REPO_DIR, "..", "Clean", "Train_Action_Recognition_STGCN_Model", "data", "Validate"),
    ]
    validate_dir = None
    for p in candidates_val:
        if os.path.exists(p):
            validate_dir = os.path.abspath(p)
            break

    if validate_dir is None:
        validate_dir = os.path.join(train_dir, "Validate")

    return train_dir, validate_dir


class ActionDataset(Dataset):
    """
    Universal Action Recognition Skeleton Dataset.
    Parses CVAT/YOLO XML annotations and produces scale-normalized skeleton clips.
    """
    def __init__(
        self,
        root_dir,
        action_folders,
        clip_length=50,
        num_keypoints=17,
        annotation_folder="annotations_yolo26s-pose",
        mode="train"
    ):
        self.root_dir = root_dir
        self.action_folders = action_folders
        self.clip_length = clip_length
        self.stride = max(1, clip_length // 2)
        self.num_keypoints = num_keypoints
        self.num_coords = 2
        self.annotation_folder = annotation_folder
        self.mode = mode

        self.skeleton_data = []
        self.samples = []

        self._load_data()

    def _load_data(self):
        for action in self.action_folders:
            action_dir = os.path.join(self.root_dir, action) if self.root_dir != "." else action
            annotations_dir = os.path.join(action_dir, self.annotation_folder)

            if not os.path.exists(annotations_dir):
                found_dirs = glob.glob(os.path.join(action_dir, "**", self.annotation_folder), recursive=True)
                if found_dirs:
                    annotations_dir = found_dirs[0]

            if not os.path.exists(annotations_dir):
                xml_files = sorted(glob.glob(os.path.join(action_dir, "*.xml")))
                if not xml_files:
                    xml_files = sorted(glob.glob(os.path.join(action_dir, "**", "*.xml"), recursive=True))
            else:
                xml_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))

            violence_label = 0 if "OOD" in action else 1

            for xml_file in xml_files:
                try:
                    skel_seq = self._parse_xml(xml_file)
                    if skel_seq is None or len(skel_seq) < 10:
                        continue

                    video_idx = len(self.skeleton_data)
                    self.skeleton_data.append(skel_seq)
                    num_frames = len(skel_seq)

                    if num_frames <= self.clip_length:
                        self.samples.append({
                            "video_idx": video_idx,
                            "start_frame": 0,
                            "label": violence_label,
                            "source_file": xml_file
                        })
                    else:
                        for start in range(0, num_frames - self.clip_length + 1, self.stride):
                            self.samples.append({
                                "video_idx": video_idx,
                                "start_frame": start,
                                "label": violence_label,
                                "source_file": xml_file
                            })
                        last_possible = num_frames - self.clip_length
                        if (num_frames - self.clip_length) % self.stride != 0:
                            self.samples.append({
                                "video_idx": video_idx,
                                "start_frame": last_possible,
                                "label": violence_label,
                                "source_file": xml_file
                            })
                except Exception as e:
                    print(f"[DATASET WARNING] Error parsing {xml_file}: {e}")

    def _parse_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            images = root.findall(".//image")
            if not images:
                return None

            num_frames = len(images)
            skeleton_data = np.zeros((num_frames, self.num_keypoints, self.num_coords), dtype=np.float32)
            frames_with_points = 0

            for frame_idx, image in enumerate(images):
                points_elem = image.find("points")
                if points_elem is not None:
                    points_str = points_elem.get("points", "")
                    if points_str:
                        keypoints = points_str.split(";")
                        if len(keypoints) >= self.num_keypoints:
                            frames_with_points += 1
                            for kp_idx in range(self.num_keypoints):
                                try:
                                    kp = keypoints[kp_idx]
                                    x_str, y_str = kp.split(",")
                                    skeleton_data[frame_idx, kp_idx, 0] = float(x_str)
                                    skeleton_data[frame_idx, kp_idx, 1] = float(y_str)
                                except ValueError:
                                    skeleton_data[frame_idx, kp_idx, :] = 0.0

            if frames_with_points == 0:
                return None

            cleaned = interpolate_missing_joints(skeleton_data)
            smoothed = smooth_kinematics(cleaned, sigma=1.0)
            return smoothed
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_idx = sample["video_idx"]
        start_frame = sample["start_frame"]
        label = sample["label"]

        skeletons = self.skeleton_data[video_idx]
        clip = skeletons[start_frame : start_frame + self.clip_length]

        if not isinstance(clip, torch.Tensor):
            clip = torch.from_numpy(clip).float()

        current_len = clip.shape[0]
        if current_len < self.clip_length:
            pad = torch.zeros((self.clip_length - current_len, self.num_keypoints, self.num_coords), dtype=torch.float32)
            clip = torch.cat([clip, pad], dim=0)

        # Scale-invariant normalization (L_torso + centroid centering)
        normed = normalize_skeleton_clip(clip)
        return normed.permute(2, 0, 1).unsqueeze(-1), label, sample["source_file"]


class TripletActionDataset(ActionDataset):
    """
    Universal Triplet Dataset generating (anchor, positive, negative) clips.
    Groups clips by source video file to avoid intra-video leakage.
    """
    def __init__(
        self,
        root_dir,
        action_folders,
        clip_length=50,
        num_keypoints=17,
        annotation_folder="annotations_yolo26s-pose"
    ):
        super().__init__(root_dir, action_folders, clip_length, num_keypoints, annotation_folder, mode="train")

        self.samples_by_video = defaultdict(list)
        for i, sample_info in enumerate(self.samples):
            self.samples_by_video[sample_info["source_file"]].append(i)

        self.video_files = list(self.samples_by_video.keys())
        self.sample_indices = []
        for vid, indices in self.samples_by_video.items():
            if len(indices) > 1:
                self.sample_indices.extend(indices)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        anchor_idx = self.sample_indices[idx]
        anchor_sample_info = self.samples[anchor_idx]
        anchor_video = anchor_sample_info["source_file"]

        anchor_clip, _, _ = super().__getitem__(anchor_idx)

        pos_indices = [i for i in self.samples_by_video[anchor_video] if i != anchor_idx]
        pos_idx = random.choice(pos_indices) if pos_indices else anchor_idx
        pos_clip, _, _ = super().__getitem__(pos_idx)

        while True:
            neg_video = random.choice(self.video_files)
            if neg_video != anchor_video:
                break
        neg_idx = random.choice(self.samples_by_video[neg_video])
        neg_clip, _, _ = super().__getitem__(neg_idx)

        return anchor_clip, pos_clip, neg_clip


class PoseConv3DHeatmapDataset(ActionDataset):
    """
    Universal Dataset for PoseConv3D.
    Produces canonical torso-normalized 3D heatmap volumes (17 x 50 x 56 x 56).
    """
    def __init__(
        self,
        root_dir,
        action_folders,
        clip_length=50,
        heatmap_size=56,
        sigma=1.5,
        annotation_folder="annotations_yolo26s-pose",
        mode="train"
    ):
        super().__init__(root_dir, action_folders, clip_length, annotation_folder=annotation_folder, mode=mode)
        self.H = heatmap_size
        self.W = heatmap_size
        self.sigma = sigma

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_idx = sample["video_idx"]
        start_frame = sample["start_frame"]
        label = sample["label"]

        skeletons = self.skeleton_data[video_idx]
        clip = skeletons[start_frame : start_frame + self.clip_length]
        if not isinstance(clip, torch.Tensor):
            clip = torch.from_numpy(clip).float()

        current_len = clip.shape[0]
        if current_len < self.clip_length:
            pad = torch.zeros((self.clip_length - current_len, self.num_keypoints, self.num_coords), dtype=torch.float32)
            clip = torch.cat([clip, pad], dim=0)

        # 1. Torso-scale normalization (removes distance & walking drift)
        normed_kpts = normalize_skeleton_clip(clip)  # (50, 17, 2), centered around (0,0) in range [-3, 3]

        # 2. Canonical mapping: map [-3.0, 3.0] to [0.0, 1.0] in heatmap coordinate space
        # Body stays perfectly centered, scale is physical human scale!
        mapped_x = torch.clamp((normed_kpts[..., 0] + 3.0) / 6.0, 0.0, 1.0)
        mapped_y = torch.clamp((normed_kpts[..., 1] + 3.0) / 6.0, 0.0, 1.0)

        valid = torch.sum(torch.abs(clip), dim=-1) > 1e-4  # (50, 17)

        kx = mapped_x.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # (17, 50, 1, 1)
        ky = mapped_y.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        vm = valid.float().permute(1, 0).unsqueeze(-1).unsqueeze(-1)

        y_grid = torch.linspace(0.0, 1.0, self.H).view(1, 1, self.H, 1)
        x_grid = torch.linspace(0.0, 1.0, self.W).view(1, 1, 1, self.W)

        s2 = 2.0 * ((self.sigma / float(self.W)) ** 2)
        dist_sq = (x_grid - kx) ** 2 + (y_grid - ky) ** 2
        heatmap = torch.exp(-dist_sq / s2) * vm  # (17, 50, 56, 56)

        return heatmap, label, sample["source_file"]


class PoseConv3DTripletHeatmapDataset(PoseConv3DHeatmapDataset):
    """Universal Triplet Dataset for PoseConv3D yielding (anchor, pos, neg) 3D heatmap volumes."""
    def __init__(
        self,
        root_dir,
        action_folders,
        clip_length=50,
        heatmap_size=56,
        sigma=1.5,
        annotation_folder="annotations_yolo26s-pose"
    ):
        super().__init__(
            root_dir, action_folders, clip_length,
            heatmap_size=heatmap_size, sigma=sigma,
            annotation_folder=annotation_folder, mode="train"
        )
        self.samples_by_video = defaultdict(list)
        for i, sample_info in enumerate(self.samples):
            self.samples_by_video[sample_info["source_file"]].append(i)

        self.video_files = list(self.samples_by_video.keys())
        self.sample_indices = []
        for vid, indices in self.samples_by_video.items():
            if len(indices) > 1:
                self.sample_indices.extend(indices)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        anchor_idx = self.sample_indices[idx]
        anchor_video = self.samples[anchor_idx]["source_file"]

        anchor_hm, _, _ = super().__getitem__(anchor_idx)

        pos_indices = [i for i in self.samples_by_video[anchor_video] if i != anchor_idx]
        pos_idx = random.choice(pos_indices) if pos_indices else anchor_idx
        pos_hm, _, _ = super().__getitem__(pos_idx)

        while True:
            neg_video = random.choice(self.video_files)
            if neg_video != anchor_video:
                break
        neg_idx = random.choice(self.samples_by_video[neg_video])
        neg_hm, _, _ = super().__getitem__(neg_idx)

        return anchor_hm, pos_hm, neg_hm
