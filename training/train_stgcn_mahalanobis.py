import os
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

# =======================================================================
# CONFIG & OPTIMIZATION
# =======================================================================
SEED = 42
NUM_WORKERS = 0       # Set to 0 to avoid Windows multiprocessing hang
PIN_MEMORY = True     # Faster host-to-device transfer
PREFETCH_FACTOR = None 

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
# 1. DATASET DEFINITIONS
# =======================================================================

class ActionDataset(Dataset):
    def __init__(self, root_dir, action_folders, clip_length=50, num_keypoints=17, annotation_folder="annotations_yolo26s-pose", mode='train'):
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
            action_dir = os.path.join(self.root_dir, action)
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
            
            if frames_with_points < num_frames and frames_with_points > 0:
                for kp_idx in range(self.num_keypoints):
                    for coord in range(self.num_coords):
                        kp_series = skeleton_data[:, kp_idx, coord]
                        missing = np.where(kp_series == 0)[0]
                        valid = np.where(kp_series != 0)[0]
                        if len(missing) > 0 and len(valid) > 1:
                            kp_series[missing] = np.interp(missing, valid, kp_series[valid])
                        elif len(missing) > 0 and len(valid) == 1:
                            kp_series[missing] = kp_series[valid[0]]
            
            if frames_with_points > 0 and np.any(skeleton_data != 0):
                sigma = 1.0
                for kp_idx in range(self.num_keypoints):
                    for coord in range(self.num_coords):
                        if np.any(skeleton_data[:, kp_idx, coord]):
                             skeleton_data[:, kp_idx, coord] = gaussian_filter1d(skeleton_data[:, kp_idx, coord], sigma=sigma, mode='nearest')
            
            return skeleton_data
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

        normalized_clip = torch.zeros_like(clip)
        for t in range(clip.shape[0]):
            frame_data = clip[t]
            valid_mask = torch.sum(torch.abs(frame_data), dim=1) > 1e-6
            valid_points = frame_data[valid_mask]

            if valid_points.shape[0] > 0:
                mean_coords = torch.mean(valid_points, dim=0)
                normalized_clip[t][valid_mask] = frame_data[valid_mask] - mean_coords

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
# 2. MODEL (Identical)
# =======================================================================
def normalize_adjacency(A):
    D = np.sum(A, axis=1)
    D[D <= 1e-6] = 1.0
    D_inv_sqrt = np.power(D, -0.5)
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    return D_mat_inv_sqrt.dot(A).dot(D_mat_inv_sqrt)

class Graph:
    def __init__(self, layout='coco', strategy='spatial'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor_link_coco = [
             (0, 1), (0, 2), (1, 3), (2, 4), 
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (5, 11), (6, 12)
        ]
        self.sym_neighbor_link = self.neighbor_link_coco + [(v, u) for u, v in self.neighbor_link_coco]
        self.A = self.get_adjacency_matrix(strategy)

    def get_adjacency_matrix(self, strategy):
        num_sub_adj = 3 
        A = np.zeros((num_sub_adj, self.num_node, self.num_node), dtype=np.float32)
        if strategy == 'spatial':
            for i in range(self.num_node):
                A[0, i, i] = 1
            for i, j in self.sym_neighbor_link:
                 if j < i: A[1, i, j] = 1 
                 elif j > i: A[2, i, j] = 1
        for i in range(num_sub_adj):
            if np.any(A[i]): A[i] = normalize_adjacency(A[i])
        return torch.from_numpy(A)

class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = nn.Conv2d(in_channels, out_channels * kernel_size[1], kernel_size=1)
        self.num_sub_adj = kernel_size[1]
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True),
        )
        if not residual: self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1): self.residual = lambda x: x
        else: self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        N, C, T, V = x.size()
        x = self.gcn(x)
        x = x.view(N, self.num_sub_adj, -1, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCNModel(nn.Module):
    def __init__(self, num_classes, in_channels=2, num_keypoints=17, dropout=0.1):
        super().__init__()
        self.graph = Graph()
        self.register_buffer('A', self.graph.A)
        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * num_keypoints)
        self.st_gcn_networks = nn.ModuleList((
            STConvBlock(in_channels, 64, kernel_size, stride=1, residual=False, dropout=dropout),
            STConvBlock(64, 64, kernel_size, stride=1, dropout=dropout),
            STConvBlock(64, 64, kernel_size, stride=1, dropout=dropout),
            STConvBlock(64, 128, kernel_size, stride=2, dropout=dropout),
            STConvBlock(128, 128, kernel_size, stride=1, dropout=dropout),
            STConvBlock(128, 128, kernel_size, stride=1, dropout=dropout),
            STConvBlock(128, 256, kernel_size, stride=2, dropout=dropout),
            STConvBlock(256, 256, kernel_size, stride=1, dropout=dropout),
            STConvBlock(256, 256, kernel_size, stride=1, dropout=dropout),
        ))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Joint-weighted spatial attention: arms (5,6,7,8,9,10) get 5× weight
        jw = torch.ones(num_keypoints)
        jw[[5, 6, 7, 8, 9, 10]] = 5.0
        jw = jw / jw.sum() * num_keypoints  # Normalize to keep scale
        self.register_buffer('joint_weights', jw)

    def forward(self, x, return_features=False):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N * M, C, V, T)
        x = x.view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        for gcn in self.st_gcn_networks: x = gcn(x, self.A)
        # Apply joint-weighted spatial attention before pooling
        # x shape: (N*M, 256, T', V=17)
        x = x * self.joint_weights.view(1, 1, 1, -1)
        x = self.pool(x)
        x = x.view(N, M, -1)
        if M == 1: features = x.squeeze(1)
        else: features = x.mean(dim=1)
        logits = self.fc(features)
        if return_features: return logits, features
        return logits

# =======================================================================
# 3. HELPER FUNCTIONS
# =======================================================================

class EarlyStopping:
    def __init__(self, patience=10, mode='min', min_delta=0.0001):
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
    df.to_csv(os.path.join(output_dir, f"training_log_fold{fold_num}.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss')
    sns.lineplot(data=df, x='epoch', y='valid_loss', label='Valid Loss')
    plt.title(f'Fold {fold_num} Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Triplet)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_curve_fold{fold_num}.png"))
    plt.close()

def train_one_fold(data_folder, output_prefix, action_folders, fold_num, train_idx, valid_idx, dataset, epochs=50, model_dir=".", log_dir="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Fold {fold_num} on {device} ---")
    
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
    early_stopping = EarlyStopping(patience=5, mode='min') # Shorter patience for demo
    
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
        
        avg_train = train_loss/len(train_loader) if len(train_loader) > 0 else 0
        
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
        avg_valid = valid_loss/len(valid_loader) if len(valid_loader) > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train:.4f}, Valid Loss={avg_valid:.4f}")
        
        history.append({'epoch': epoch + 1, 'train_loss': avg_train, 'valid_loss': avg_valid})
        
        if early_stopping(avg_valid, model.state_dict()):
            print("Early stopping triggered.")
            break
            
    save_path = os.path.join(model_dir, f"{output_prefix}_fold{fold_num}.pth")
    torch.save(early_stopping.best_model_state, save_path)
    print(f"Saved best model to {save_path}")
    plot_training_history(history, log_dir, fold_num)
    
    return save_path, early_stopping.best_score

def calc_mahalanobis_param(dataset, model, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    features_list = []
    
    with torch.no_grad():
        for batch in loader:
             a = batch[0]
             _, f = model(a.to(device), return_features=True)
             features_list.append(f.cpu())
    
    features = torch.cat(features_list)
    mean = torch.mean(features, dim=0)
    cov = torch.cov(features.T) # No need to subtract mean, cov handles it.
    reg_cov = cov + torch.eye(features.shape[1]) * 1e-3
    inv_cov = torch.linalg.pinv(reg_cov)
    
    return mean, inv_cov

def tune_ood_threshold(dataset, model, mean, inv_cov, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dists = []
    
    with torch.no_grad():
        for batch in loader:
             a = batch[0]
             _, f = model(a.to(device), return_features=True)
             f = f.cpu()
             diff = f - mean
             dists_sq = torch.einsum('bf,fg,bg->b', diff, inv_cov, diff)
             dists.extend(torch.sqrt(dists_sq).tolist())
    
    dists = np.array(dists)
    threshold = np.percentile(dists, 95)
    return float(threshold)

def validate_category(category_name, folder_paths, model, mean, inv_cov, threshold, device):
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
    
    # OOD -> Label 0 (Negative)
    # Violence -> Label 1 (Positive)
    true_cat_label = 0 if "OOD" in category_name else 1
    
    for src, clips in video_clips.items():
        is_violence = False
        clips_tensor = torch.stack(clips).to(device)
        
        with torch.no_grad():
            _, feats = model(clips_tensor, return_features=True)
            feats = feats.cpu()
            
            diff = feats - mean
            dists_sq = torch.einsum('bf,fg,bg->b', diff, inv_cov, diff)
            dists = torch.sqrt(dists_sq)
            
            if torch.any(dists <= threshold):
                is_violence = True
                
        y_true.append(true_cat_label)
        y_pred.append(1 if is_violence else 0)
        
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    # Confusion Matrix
    # Sklearn confusion_matrix(y_true, y_pred) returns [[TN, FP], [FN, TP]] for binary
    # If all same class, shape might be smaller, so we handle it manually or pad.
    tn, fp, fn, tp = 0, 0, 0, 0
    
    if len(set(y_true)) > 1 or len(set(y_pred)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        # Single class case
        if true_cat_label == 0: # All true OOD
            if y_pred[0] == 0: tn = len(y_pred)
            else: fp = len(y_pred)
        else: # All true Violence
            if y_pred[0] == 1: tp = len(y_pred)
            else: fn = len(y_pred)
            
    return len(y_true), acc, prec, rec, f1, tp, tn, fp, fn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # 1. TRAIN 3 FOLDS
    train_data_dir = "./data/data"
    actions = ["Cut-Down", "Stab", "Thrust"]
    output_prefix = "Train_Violence_STGCN"
    
    # Output config
    model_output_dir = "trained_models"
    os.makedirs(model_output_dir, exist_ok=True)
    
    log_output_dir = "results"
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
        
        path, val_loss = train_one_fold(train_data_dir, output_prefix, actions, fold+1, train_idx, valid_idx, full_dataset, epochs=30, model_dir=model_output_dir, log_dir=log_output_dir)
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            best_fold_idx = fold + 1
            best_model_path = path

    print(f"\n>> BEST FOLD: {best_fold_idx} (Valid Loss: {best_valid_loss:.4f})")
    print(f">> Loading: {best_model_path}")

    # 2. TUNE BEST FOLD
    model = STGCNModel(num_classes=1).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    print("Calculating OOD Parameters on Training Data...")
    mean_val, inv_cov_val = calc_mahalanobis_param(full_dataset, model, device)
    
    print("Tuning Threshold (95% ID retention)...")
    threshold_val = tune_ood_threshold(full_dataset, model, mean_val, inv_cov_val, device)
    print(f"Tunned Threshold: {threshold_val:.4f}")
    
    # 3. FINAL VALIDATION
    print("\n" + "="*40)
    print("FINAL VALIDATION ON 'VALIDATE' FOLDER")
    print("="*40)
    
    validate_root = "./data/Validate"
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
        n, acc, prec, rec, f1, tp, tn, fp, fn = validate_category(cat, paths, model, mean_val, inv_cov_val, threshold_val, device)
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
    val_df.to_csv(os.path.join(log_output_dir, "validation_results.csv"))
    print(f"Validation results saved to {os.path.join(log_output_dir, 'validation_results.csv')}")

if __name__ == "__main__":
    main()
