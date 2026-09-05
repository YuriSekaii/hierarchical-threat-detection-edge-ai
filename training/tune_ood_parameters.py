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

    def forward(self, x, return_features=False):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N * M, C, V, T)
        x = x.view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        for gcn in self.st_gcn_networks: x = gcn(x, self.A)
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




def calc_harmonic_score(id_acc, ood_acc):
    if id_acc + ood_acc == 0: return 0.0
    return 2 * (id_acc * ood_acc) / (id_acc + ood_acc)

def extract_features(dataset, model, device):
    """Extract features for all items in the dataset once."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    features_list = []
    labels_list = []
    src_list = []
    
    with torch.no_grad():
        for batch in loader:
             # batch: (data, label, src_path)
             clips = batch[0].to(device)
             labels = batch[1]
             srcs = batch[2]
             
             _, f = model(clips, return_features=True)
             features_list.append(f.cpu())
             labels_list.extend(labels.tolist())
             src_list.extend(srcs)
    
    features = torch.cat(features_list)
    return features, labels_list, src_list

def calc_maha_params_from_features(train_features, reg_cov_val):
    """Calculate mean and inv_cov from cached training features."""
    mean = torch.mean(train_features, dim=0)
    cov = torch.cov(train_features.T)
    reg_cov = cov + torch.eye(train_features.shape[1]) * reg_cov_val
    inv_cov = torch.linalg.pinv(reg_cov)
    return mean, inv_cov

def calc_threshold_from_features(train_features, mean, inv_cov, percentile_val):
    """Calculate threshold using cached features."""
    diff = train_features - mean
    dists_sq = torch.einsum('bf,fg,bg->b', diff, inv_cov, diff)
    dists = torch.sqrt(dists_sq)
    dists_np = dists.numpy()
    threshold = np.percentile(dists_np, percentile_val)
    return float(threshold)

def validate_from_features(validation_data, mean, inv_cov, threshold):
    """
    Validation using cached features.
    validation_data: dict {category: {'features': tensor, 'srcs': list}}
    """
    total_metrics = [] # List of [acc, prec, rec, f1, tp, tn, fp, fn] per category
    
    for cat, data in validation_data.items():
        feats = data['features']
        srcs = data['srcs']
        
        # Group by video source (logic matches original validate_category)
        video_feats = defaultdict(list)
        for i, src in enumerate(srcs):
            video_feats[src].append(feats[i])
            
        y_true = []
        y_pred = []
        true_cat_label = 0 if "OOD" in cat else 1
        
        # For each video
        for src, clip_feats_list in video_feats.items():
            is_violence = False
            # clip_feats_list is a list of 1D tensors. Stack them.
            clip_feats_tensor = torch.stack(clip_feats_list) # (n_clips, feat_dim)
            
            # Mahalanobis dist
            diff = clip_feats_tensor - mean
            dists_sq = torch.einsum('bf,fg,bg->b', diff, inv_cov, diff)
            dists = torch.sqrt(dists_sq)
            
            if torch.any(dists <= threshold):
                is_violence = True
            
            y_true.append(true_cat_label)
            y_pred.append(1 if is_violence else 0)
            
        # Calc metrics for this category
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(y_true) > 0:
            if len(set(y_true)) > 1 or len(set(y_pred)) > 1:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
            else:
                 if true_cat_label == 0: # All OOD
                     if y_pred[0] == 0: tn = len(y_pred)
                     else: fp = len(y_pred)
                 else: # All Violence
                     if y_pred[0] == 1: tp = len(y_pred)
                     else: fn = len(y_pred)

        acc = accuracy_score(y_true, y_pred) if y_true else 0
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        total_metrics.append([len(y_true), acc, prec, rec, f1, tp, tn, fp, fn])
        
    return total_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Optimization Tuning on: {device}")
    
    # Paths
    train_data_dir = "./data/data"
    actions = ["Cut-Down", "Stab", "Thrust"]
    model_path = "./data/trained_models/Train_Violence_STGCN_fold2.pth"
    log_output_dir = "./data/results"
    validate_root = "./data/Validate"
    
    # 1. Load Data & Model
    print(f">> Loading Triplet Dataset from {train_data_dir}...")
    full_dataset = TripletActionDataset(train_data_dir, actions)
    
    print(f">> Loading Best Model: {model_path}")
    model = STGCNModel(num_classes=1).to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Extract Features (Cache)
    print(">> Caching Training Features...")
    train_features, _, _ = extract_features(full_dataset, model, device)
    
    print(">> Caching Validation Features...")
    val_categories = {
        "OOD": [os.path.join(validate_root, "OOD")],
        "With_Coat": [os.path.join(validate_root, "With_Coat")],
        "Without_Coat": [os.path.join(validate_root, "Without_Coat")] 
    }
    validation_cache = {}
    for cat, paths in val_categories.items():
        print(f"   - {cat}...")
        ds = ActionDataset(root_dir=".", action_folders=paths, clip_length=50, mode='test')
        if len(ds) == 0:
            print(f"WARNING: No data for {cat}")
            validation_cache[cat] = {'features': torch.tensor([]), 'srcs': []}
            continue
        feats, _, srcs = extract_features(ds, model, device)
        validation_cache[cat] = {'features': feats, 'srcs': srcs}

    # Helper tuning function
    def run_tuning_grid(reg_covs, percentiles, save_name):
        results = []
        total_iter = len(reg_covs) * len(percentiles)
        curr = 0
        print(f"\n--- Starting {save_name} ({total_iter} combinations) ---")
        
        for reg in reg_covs:
            # Calc Maha params once per reg_cov
            mean, inv_cov = calc_maha_params_from_features(train_features, reg)
            
            for pct in percentiles:
                curr += 1
                if curr % 5 == 0: print(f"Processing {curr}/{total_iter}...", end='\r')
                
                # Calc Threshold
                thresh = calc_threshold_from_features(train_features, mean, inv_cov, pct)
                
                # Validate using cached features
                # metrics: list of [N, acc, prec, rec, f1, tp, tn, fp, fn] per category
                # Order of val_categories iteration matters. Dict preserves insertion order in Py3.7+
                # We iterate val_categories and match with validation_cache
                
                # Re-construct metrics dict to map back to categories easily
                cat_results = {}
                metrics_list = validate_from_features(validation_cache, mean, inv_cov, thresh)
                
                # Map back: keys of val_categories are same as validation_cache
                keys = list(validation_cache.keys())
                for i, key in enumerate(keys):
                    cat_results[key] = metrics_list[i]
                
                # Aggregate for Harmonic Score
                # ID: With_Coat, Without_Coat
                id_tp = cat_results["With_Coat"][5] + cat_results["Without_Coat"][5]
                id_total_n = cat_results["With_Coat"][0] + cat_results["Without_Coat"][0]
                id_acc = id_tp / id_total_n if id_total_n > 0 else 0
                
                # OOD
                ood_tn = cat_results["OOD"][6] # TN is correct rejection
                ood_total_n = cat_results["OOD"][0]
                ood_acc = ood_tn / ood_total_n if ood_total_n > 0 else 0
                
                harm = calc_harmonic_score(id_acc, ood_acc)
                
                results.append({
                    "maha_reg_covariance": reg,
                    "maha_percentile": pct,
                    "threshold": thresh,
                    "ID_Accuracy": id_acc,
                    "OOD_Accuracy": ood_acc,
                    "harmonic_score": harm
                })
        
        df = pd.DataFrame(results)
        df = df.sort_values(by="harmonic_score", ascending=False)
        out_path = os.path.join(log_output_dir, save_name)
        df.to_csv(out_path, index=False)
        print(f"\nSaved {save_name} to {out_path}")
        return df.iloc[0] # Best row

    # 3. Coarse Search
    coarse_pcts = np.array([85, 90, 92, 94, 95, 96, 98, 99])
    coarse_regs = np.array([0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01])
    
    best_coarse = run_tuning_grid(coarse_regs, coarse_pcts, "ood_tuning_coarse_fold2.csv")
    print(f"Best Coarse: Reg={best_coarse['maha_reg_covariance']}, Pct={best_coarse['maha_percentile']}, Harm={best_coarse['harmonic_score']:.4f}")
    
    # 4. Fine Search (Adaptive)
    # Define range around best coarse
    center_reg = best_coarse['maha_reg_covariance']
    center_pct = best_coarse['maha_percentile']
    
    # Heuristic: +/- significant step
    # Regs: e.g. [0.8x, 1.0x, 1.2x] or fixed steps if center is small
    # Pcts: +/- 0.5, 1.0
    
    fine_regs = sorted(list(set([
        center_reg * 0.8, center_reg * 0.9, center_reg, center_reg * 1.1, center_reg * 1.2,
        0.0008, 0.001, 0.0012, 0.0015 # Include known good region specifically
    ])))
    fine_pcts = np.arange(max(80, center_pct - 2.0), min(100, center_pct + 2.5), 0.5)
    
    best_fine = run_tuning_grid(fine_regs, fine_pcts, "ood_tuning_fine_fold2.csv")
    print(f"Best Fine: Reg={best_fine['maha_reg_covariance']}, Pct={best_fine['maha_percentile']}, Harm={best_fine['harmonic_score']:.4f}")
    
    print("\nOptimization Complete.")

if __name__ == "__main__":
    main()
