"""
ST-GCN Triplet Training with Deep k-NN Manifold Gating - Version 1.02
Incorporates Scale-Invariant Kinematics Normalization (Torso Length L_torso = ||mid_shoulder - mid_hip||).

Version 1.02 Improvements:
- Replaces raw centroid centering with torso-scale normalized coordinates (src.skeleton_utils_v1.02).
- Ensures motion vectors are scale-invariant regardless of distance from the camera.
- Retrains ST-GCN across 3-fold cross-validation with GroupKFold video partitioning.
- Builds calibrated Deep k-NN feature bank and evaluates on the full 148-clip validation suite.
- Exports versioned model checkpoint (weights/stgcn_violence_v1.02.pth) and k-NN weights (weights/reference_data_knn_v1.02.pt).
"""

import os
import sys
import glob
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import xml.etree.ElementTree as ET
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_DIR)

# Dynamically import v1.02 skeleton utils
import importlib.util
skel_spec = importlib.util.spec_from_file_location(
    "skeleton_utils_v1_02",
    os.path.join(REPO_DIR, "src", "skeleton_utils_v1.02.py")
)
skel_mod = importlib.util.module_from_spec(skel_spec)
skel_spec.loader.exec_module(skel_mod)
normalize_skeleton_clip = skel_mod.normalize_skeleton_clip
interpolate_missing_joints = skel_mod.interpolate_missing_joints
smooth_kinematics = skel_mod.smooth_kinematics

from models.stgcn import STGCNModel, Graph, STConvBlock

# =======================================================================
# CONFIG & OPTIMIZATION
# =======================================================================
SEED = 42
NUM_WORKERS = 0       # Set to 0 for Windows multiprocessing safety
PIN_MEMORY = True     # Faster host-to-device transfer
KNN_K = 2             # Number of nearest neighbors (k-th NN distance used as score)
KNN_PERCENTILE = 95   # Percentile of training k-NN distances for threshold

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


# =======================================================================
# 1. DATASET DEFINITIONS (v1.02 Scale-Normalized)
# =======================================================================

class ActionDataset(Dataset):
    def __init__(self, root_dir, action_folders, clip_length=50, num_keypoints=17,
                 annotation_folder="annotations_yolo26s-pose", mode='train'):
        self.root_dir = root_dir
        self.action_folders = action_folders
        self.clip_length = clip_length
        self.stride = max(1, clip_length // 2)
        self.num_keypoints = num_keypoints
        self.num_coords = 2 
        self.annotation_folder = annotation_folder
        self.mode = mode

        self.class_to_idx = {'Violence': 1}
        self.skeleton_data = [] 
        self.samples = []      
        
        print(f"[{mode.upper()}] Loading dataset from: {root_dir}")
        self._load_data()
        print(f"[{mode.upper()}] Total samples loaded: {len(self.samples)}")

    def _load_data(self):
        for action in self.action_folders:
            action_dir = os.path.join(self.root_dir, action) if self.root_dir != "." else action
            annotations_dir = os.path.join(action_dir, self.annotation_folder)
            
            if not os.path.exists(annotations_dir):
                found_dirs = glob.glob(os.path.join(action_dir, "**", self.annotation_folder), recursive=True)
                if found_dirs:
                    annotations_dir = found_dirs[0]
            
            if not os.path.exists(annotations_dir):
                xml_files_direct = glob.glob(os.path.join(action_dir, '*.xml'))
                if xml_files_direct:
                    xml_files = sorted(xml_files_direct)
                else:
                    xml_files = sorted(glob.glob(os.path.join(action_dir, '**', '*.xml'), recursive=True))
            else:
                xml_files = sorted(glob.glob(os.path.join(annotations_dir, '*.xml')))
            
            if not xml_files:
                continue

            for xml_file in xml_files:
                try:
                    skeletons_video = self._parse_xml(xml_file)
                    if skeletons_video is None or skeletons_video.shape[0] < self.clip_length:
                        continue

                    self.skeleton_data.append(torch.tensor(skeletons_video, dtype=torch.float32))
                    video_idx = len(self.skeleton_data) - 1
                    num_frames = skeletons_video.shape[0]
                    violence_label = self.class_to_idx['Violence']
                    
                    for start_frame in range(0, num_frames - self.clip_length + 1, self.stride):
                        self.samples.append({
                            'video_idx': video_idx,
                            'start_frame': start_frame,
                            'label': violence_label,
                            'source_file': xml_file
                        })
                        
                    last_possible_start = num_frames - self.clip_length
                    if last_possible_start >= 0 and (num_frames - self.clip_length) % self.stride != 0:
                        if not any(s['video_idx'] == video_idx and s['start_frame'] == last_possible_start for s in self.samples):
                            self.samples.append({
                                'video_idx': video_idx,
                                'start_frame': last_possible_start,
                                'label': violence_label,
                                'source_file': xml_file
                            })

                except Exception as e:
                    print(f"Error processing {xml_file}: {e}")

    def _parse_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            images = root.findall('.//image')
            if not images: return None
            
            num_frames = len(images)
            skeleton_data = np.zeros((num_frames, self.num_keypoints, self.num_coords), dtype=np.float32)
            frames_with_points = 0
            
            for frame_idx, image in enumerate(images):
                points_elem = image.find('points')
                if points_elem is not None:
                    points_str = points_elem.get('points', '')
                    if points_str:
                        keypoints = points_str.split(';')
                        if len(keypoints) >= self.num_keypoints:
                            frames_with_points += 1
                            for kp_idx in range(self.num_keypoints):
                                try:
                                    kp = keypoints[kp_idx]
                                    x_str, y_str = kp.split(',')
                                    x, y = float(x_str), float(y_str)
                                    skeleton_data[frame_idx, kp_idx, 0] = x
                                    skeleton_data[frame_idx, kp_idx, 1] = y
                                except ValueError:
                                    skeleton_data[frame_idx, kp_idx, :] = 0.0
            
            if frames_with_points == 0: return None
            
            # Interpolation and smoothing
            cleaned = interpolate_missing_joints(skeleton_data)
            smoothed = smooth_kinematics(cleaned, sigma=1.0)
            return smoothed
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_idx = sample['video_idx']
        start_frame = sample['start_frame']
        label = sample['label']
        
        skeletons = self.skeleton_data[video_idx]
        clip = skeletons[start_frame : start_frame + self.clip_length]
        
        current_len = clip.shape[0]
        if current_len < self.clip_length:
            pad = torch.zeros((self.clip_length - current_len, self.num_keypoints, self.num_coords), dtype=torch.float32)
            clip = torch.cat([clip, pad], dim=0)

        # Scale-invariant normalization (Torso length L_torso + bbox fallback)
        normalized_clip = normalize_skeleton_clip(clip)
        return normalized_clip.permute(2, 0, 1).unsqueeze(-1), label, sample['source_file']


class TripletActionDataset(ActionDataset):
    def __init__(self, root_dir, action_folders, clip_length=50, num_keypoints=17, annotation_folder="annotations_yolo26s-pose"):
        super().__init__(root_dir, action_folders, clip_length, num_keypoints, annotation_folder, mode='train')
        
        self.samples_by_video = defaultdict(list)
        for i, sample_info in enumerate(self.samples):
            self.samples_by_video[sample_info['source_file']].append(i)

        self.video_files = list(self.samples_by_video.keys())
        self.sample_indices = []
        for vid, indices in self.samples_by_video.items():
            if len(indices) > 1:
                self.sample_indices.extend(indices)
        
        print(f"Triplet Dataset Ready: {len(self.video_files)} anchor videos.")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        anchor_idx = self.sample_indices[idx]
        anchor_sample_info = self.samples[anchor_idx]
        anchor_video = anchor_sample_info['source_file']

        anchor_clip, _, _ = super().__getitem__(anchor_idx)

        pos_indices = [i for i in self.samples_by_video[anchor_video] if i != anchor_idx]
        if not pos_indices: pos_idx = anchor_idx
        else: pos_idx = random.choice(pos_indices)
        positive_clip, _, _ = super().__getitem__(pos_idx)

        while True:
            neg_video = random.choice(self.video_files)
            if neg_video != anchor_video: break
        neg_idx = random.choice(self.samples_by_video[neg_video])
        negative_clip, _, _ = super().__getitem__(neg_idx)

        return anchor_clip, positive_clip, negative_clip


# =======================================================================
# 2. TRAINING & VALIDATION HELPERS
# =======================================================================

class EarlyStopping:
    def __init__(self, patience=8, mode='min', min_delta=0.0001):
        self.patience = patience; self.mode = mode; self.min_delta = min_delta
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0; self.best_model_state = None
    def __call__(self, score, model_state):
        improved = (score < self.best_score - self.min_delta) if self.mode == 'min' else (score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score; self.best_model_state = copy.deepcopy(model_state); self.counter = 0
        else: self.counter += 1
        return self.counter >= self.patience


def plot_training_history(history, output_dir, fold_num):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, f"training_log_v1.02_fold{fold_num}.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss')
    sns.lineplot(data=df, x='epoch', y='valid_loss', label='Valid Loss')
    plt.title(f'Fold {fold_num} Training Progress (v1.02)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Triplet Margin)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_curve_v1.02_fold{fold_num}.png"))
    plt.close()


def train_one_fold(data_folder, output_prefix, action_folders, fold_num, train_idx, valid_idx,
                   dataset, epochs=30, model_dir=".", log_dir="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Fold {fold_num} on {device} (v1.02) ---")
    
    train_indices_mapped = [dataset.sample_indices[i] for i in train_idx if i < len(dataset.sample_indices)]
    valid_indices_mapped = [dataset.sample_indices[i] for i in valid_idx if i < len(dataset.sample_indices)]
    
    train_sub = Subset(dataset, train_indices_mapped)
    valid_sub = Subset(dataset, valid_indices_mapped)
    
    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(valid_sub, batch_size=32, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    model = STGCNModel(num_classes=1, dropout=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.TripletMarginLoss(margin=1.0)
    early_stopping = EarlyStopping(patience=8, mode='min')
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for a, p, n in train_loader:
            a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
            optimizer.zero_grad()
            _, af = model(a, return_features=True)
            _, pf = model(p, return_features=True)
            _, nf = model(n, return_features=True)
            loss = criterion(af, pf, nf)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for a, p, n in valid_loader:
                a, p, n = a.to(device, non_blocking=True), p.to(device, non_blocking=True), n.to(device, non_blocking=True)
                _, af = model(a, return_features=True)
                _, pf = model(p, return_features=True)
                _, nf = model(n, return_features=True)
                loss = criterion(af, pf, nf)
                valid_loss += loss.item()
        avg_valid = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0
        
        print(f"Epoch {epoch+1:02d}/{epochs}: Train Loss={avg_train:.4f}, Valid Loss={avg_valid:.4f}")
        history.append({'epoch': epoch + 1, 'train_loss': avg_train, 'valid_loss': avg_valid})
        
        if early_stopping(avg_valid, model.state_dict()):
            print("Early stopping triggered.")
            break
            
    save_path = os.path.join(model_dir, f"{output_prefix}_fold{fold_num}.pth")
    torch.save(early_stopping.best_model_state, save_path)
    print(f"Saved best model to {save_path}")
    plot_training_history(history, log_dir, fold_num)
    
    return save_path, early_stopping.best_score


# =======================================================================
# 3. k-NN MANIFOLD & VALIDATION
# =======================================================================

def knn_score(query_features, feature_bank, k):
    effective_k = min(k, feature_bank.shape[0])
    dists = torch.cdist(query_features, feature_bank, p=2)
    kth_dists, _ = dists.kthvalue(effective_k, dim=1)
    return kth_dists


def build_knn_feature_bank(dataset, model, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    features_list = []
    with torch.no_grad():
        for batch in loader:
            a = batch[0]
            _, f = model(a.to(device), return_features=True)
            features_list.append(f.cpu())
    return torch.cat(features_list)


def tune_knn_threshold(feature_bank, k, percentile=95):
    N = feature_bank.shape[0]
    effective_k = min(k + 1, N)
    dists = torch.cdist(feature_bank, feature_bank, p=2)
    kth_dists, _ = dists.kthvalue(effective_k, dim=1)
    train_knn_dists = kth_dists.numpy()
    threshold = float(np.percentile(train_knn_dists, percentile))
    return threshold, train_knn_dists


def validate_category(category_name, folder_paths, model, feature_bank, threshold, k, device):
    print(f"Validating {category_name}...")
    dataset = ActionDataset(root_dir=".", action_folders=folder_paths, clip_length=50, mode='test')
    if len(dataset) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
    
    video_clips = defaultdict(list)
    for i in range(len(dataset)):
        clip, label, src = dataset[i]
        video_clips[src].append(clip)
        
    y_true = []
    y_pred = []
    true_cat_label = 0 if "OOD" in category_name else 1
    
    for src, clips in video_clips.items():
        is_violence = False
        clips_tensor = torch.stack(clips).to(device)
        with torch.no_grad():
            _, feats = model(clips_tensor, return_features=True)
            feats = feats.cpu()
            knn_dists = knn_score(feats, feature_bank, k)
            if torch.any(knn_dists <= threshold):
                is_violence = True
                
        y_true.append(true_cat_label)
        y_pred.append(1 if is_violence else 0)
        
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(set(y_true)) > 1 or len(set(y_pred)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        if true_cat_label == 0:
            if y_pred[0] == 0: tn = len(y_pred)
            else: fp = len(y_pred)
        else:
            if y_pred[0] == 1: tp = len(y_pred)
            else: fn = len(y_pred)
            
    return len(y_true), acc, prec, rec, f1, tp, tn, fp, fn


def resolve_data_paths():
    candidates_train = [
        os.path.join(REPO_DIR, "data"),
        os.path.join(REPO_DIR, "..", "Clean", "Train_Action_Recognition_STGCN_Model", "data"),
        os.path.join(REPO_DIR, "..", "Train_Action_Recognition_STGCN_Model", "data")
    ]
    train_dir = None
    for p in candidates_train:
        if os.path.exists(os.path.join(p, "Cut-Down")):
            train_dir = os.path.abspath(p)
            break
            
    if train_dir is None:
        raise FileNotFoundError("Could not find action training data folder (Cut-Down, Stab, Thrust).")

    validate_dir = os.path.join(train_dir, "Validate")
    if not os.path.exists(validate_dir):
        validate_dir = os.path.join(REPO_DIR, "..", "Clean", "Train_Action_Recognition_STGCN_Model", "data", "Validate")

    return train_dir, validate_dir


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========================================================================")
    print(f"   RETRAINING ST-GCN + DEEP k-NN WITH SCALE NORMALIZATION (v1.02)")
    print(f"=========================================================================")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"OOD Method    : Deep k-NN Feature Gating (k={KNN_K}, Percentile={KNN_PERCENTILE})")
    print(f"Normalization : Torso Length Scale Normalization (L_torso + Bbox Fallback)")
    
    train_data_dir, validate_root = resolve_data_paths()
    print(f"Training Data : {train_data_dir}")
    print(f"Validate Data : {validate_root}")
    
    actions = ["Cut-Down", "Stab", "Thrust"]
    output_prefix = "stgcn_violence_v1.02"
    
    model_output_dir = os.path.join(REPO_DIR, "weights")
    os.makedirs(model_output_dir, exist_ok=True)
    
    log_output_dir = os.path.join(REPO_DIR, "results", "action_recognition_benchmark")
    os.makedirs(log_output_dir, exist_ok=True)
    
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
    
    for fold, (train_idx, valid_idx) in enumerate(gkf.split(full_dataset.samples, groups=groups)):
        print(f"\n{'#'*20} FOLD {fold+1}/{k_folds} {'#'*20}")
        path, val_loss = train_one_fold(
            train_data_dir, output_prefix, actions, fold+1,
            train_idx, valid_idx, full_dataset, epochs=30,
            model_dir=model_output_dir, log_dir=log_output_dir
        )
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_fold_idx = fold + 1
            best_model_path = path

    print(f"\n>> BEST FOLD: Fold {best_fold_idx} (Valid Loss: {best_valid_loss:.4f})")
    print(f">> Champion Model Path: {best_model_path}")
    
    # Copy champion model to production checkpoint name: weights/stgcn_violence_v1.02.pth
    champion_target = os.path.join(model_output_dir, "stgcn_violence_v1.02.pth")
    import shutil
    shutil.copyfile(best_model_path, champion_target)
    print(f">> Champion Checkpoint Saved as: {champion_target}")

    # Build k-NN Feature Bank using champion model
    model = STGCNModel(num_classes=1).to(device)
    model.load_state_dict(torch.load(champion_target, weights_only=True))
    model.eval()
    
    print("\n--- Building k-NN Feature Bank on Training Data (v1.02) ---")
    feature_bank = build_knn_feature_bank(full_dataset, model, device)
    N_bank = feature_bank.shape[0]
    print(f"Feature Bank Shape: {feature_bank.shape}")
    
    effective_k = min(KNN_K, N_bank - 1)
    threshold_val, train_knn_dists = tune_knn_threshold(feature_bank, effective_k, percentile=KNN_PERCENTILE)
    print(f"Calibrated k-NN Threshold (v1.02): {threshold_val:.6f}")
    
    # Save k-NN feature bank and calibrated threshold
    ref_knn_path = os.path.join(model_output_dir, "reference_data_knn_v1.02.pt")
    torch.save({
        'feature_bank': feature_bank,
        'threshold': threshold_val,
        'k': effective_k,
        'normalization': 'torso_scale_v1.02'
    }, ref_knn_path)
    print(f">> Reference Bank & Threshold Saved to: {ref_knn_path}")

    # Run Final Benchmark on 148-Clip Validation Suite
    print("\n" + "="*50)
    print("FINAL BENCHMARK VALIDATION (148 CLIPS)")
    print(f"Method: ST-GCN + Deep k-NN (v1.02) | k={effective_k} | Threshold={threshold_val:.6f}")
    print("="*50)
    
    categories = {
        "OOD": [os.path.join(validate_root, "OOD")],
        "With_Coat": [os.path.join(validate_root, "With_Coat")],
        "Without_Coat": [os.path.join(validate_root, "Without_Coat")] 
    }
    
    metrics = []
    labels = []
    
    print(f"{'Category':<15} | {'N':<3} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'TP':<3} | {'TN':<3} | {'FP':<3} | {'FN':<3}")
    print("-" * 100)
    
    for cat, paths in categories.items():
        n, acc, prec, rec, f1, tp, tn, fp, fn = validate_category(
            cat, paths, model, feature_bank, threshold_val, effective_k, device
        )
        print(f"{cat:<15} | {n:<3d} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {tp:<3d} | {tn:<3d} | {fp:<3d} | {fn:<3d}")
        metrics.append([acc, prec, rec, f1, tp, tn, fp, fn])
        labels.append(cat)
        
    avg_metrics = np.mean([m[:4] for m in metrics], axis=0)
    total_tp = sum([m[4] for m in metrics])
    total_tn = sum([m[5] for m in metrics])
    total_fp = sum([m[6] for m in metrics])
    total_fn = sum([m[7] for m in metrics])
    
    print("-" * 100)
    print(f"{'AVERAGE/TOTAL':<15} | {'-':<3} | {avg_metrics[0]:.4f} | {avg_metrics[1]:.4f} | {avg_metrics[2]:.4f} | {avg_metrics[3]:.4f} | {total_tp:<3d} | {total_tn:<3d} | {total_fp:<3d} | {total_fn:<3d}")
    print("-" * 100)
    
    val_df = pd.DataFrame([m[:4] for m in metrics], columns=['Accuracy', 'Precision', 'Recall', 'F1'], index=labels)
    val_df.loc['AVERAGE'] = avg_metrics
    res_csv = os.path.join(log_output_dir, "validation_results_knn_v1.02.csv")
    val_df.to_csv(res_csv)
    print(f">> Validation results saved to {res_csv}")


if __name__ == "__main__":
    main()
