"""
Run_Final_Validation_kNN.py

Standalone validation script for the k-NN OOD method.
Loads the trained STGCN model (Fold 2), builds the k-NN feature bank,
validates against OOD/Violence categories, and saves the reference data
needed by Real_Time.py.

Output:
  - results/reference_data_knn.pt  (feature_bank tensor + threshold + k)
  - results/validation_results_knn.csv
"""
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import xml.etree.ElementTree as ET
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import pandas as pd

# Reuse definitions from Train_Violence_STGCN.py
# To ensure exact compatibility, we redefine classes here.

SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

NUM_WORKERS = 0
PIN_MEMORY = True

# --- k-NN OOD Hyperparameters (must match Train_Violence_STGCN_kNN.py) ---
BEST_KNN_K = 2
BEST_KNN_PERCENTILE = 95

# --- CLASS DEFINITIONS (Copy from Train script) ---
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
        x = x * self.joint_weights.view(1, 1, 1, -1)
        x = self.pool(x)
        x = x.view(N, M, -1)
        if M == 1: features = x.squeeze(1)
        else: features = x.mean(dim=1)
        logits = self.fc(features)
        if return_features: return logits, features
        return logits

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
        self._load_data()

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
            
            if not xml_files: continue

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
                except Exception: pass

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
        clip = self.skeleton_data[sample['video_idx']][sample['start_frame'] : sample['start_frame'] + self.clip_length]
        
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

        return normalized_clip.permute(2, 0, 1).unsqueeze(-1), sample['label'], sample['source_file']




# --- k-NN OOD FUNCTIONS ---

def knn_score(query_features, feature_bank, k):
    """Compute k-th nearest neighbor distance for each query."""
    effective_k = min(k, feature_bank.shape[0])
    dists = torch.cdist(query_features, feature_bank, p=2)
    kth_dists, _ = dists.kthvalue(effective_k, dim=1)
    return kth_dists

def build_knn_feature_bank(dataset, model, device):
    """Extract raw feature embeddings from all training clips."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    features_list = []
    with torch.no_grad():
        for batch in loader:
             a = batch[0]
             _, f = model(a.to(device), return_features=True)
             features_list.append(f.cpu())
    return torch.cat(features_list)

def tune_knn_threshold(feature_bank, k, percentile=95):
    """Compute leave-one-out k-NN threshold on training set."""
    N = feature_bank.shape[0]
    effective_k = min(k + 1, N)
    dists = torch.cdist(feature_bank, feature_bank, p=2)
    kth_dists, _ = dists.kthvalue(effective_k, dim=1)
    train_knn_dists = kth_dists.numpy()
    threshold = float(np.percentile(train_knn_dists, percentile))
    return threshold, train_knn_dists

def validate_category(category_name, folder_paths, model, feature_bank, threshold, k, device):
    """Validate a category using k-NN OOD detection."""
    print(f"Validating {category_name}...")
    dataset = ActionDataset(root_dir=".", action_folders=folder_paths, clip_length=50, mode='test')
    
    if len(dataset) == 0:
        return 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0
    
    video_clips = defaultdict(list)
    for i in range(len(dataset)):
        clip, label, src = dataset[i]
        video_clips[src].append(clip)
        
    y_true = []
    y_pred = []
    
    true_cat_label = 0 if ("OOD" in category_name or "False" in category_name) else 1
    
    for src, clips in video_clips.items():
        is_violence = False
        clips_tensor = torch.stack(clips).to(device)
        
        with torch.no_grad():
            _, feats = model(clips_tensor, return_features=True)
            feats = feats.cpu()  # Raw features, no L2-norm
            knn_dists = knn_score(feats, feature_bank, k)
            if torch.any(knn_dists <= threshold):
                is_violence = True
                
        y_true.append(true_cat_label)
        y_pred.append(1 if is_violence else 0)
    
    # Calculate Metrics
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    return len(y_true), tp, tn, fp, fn, accuracy, precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running k-NN Validation on: {device}")
    
    # 1. LOAD BEST MODEL (Fold 2)
    model_path = os.path.join("trained_models", "Train_Violence_STGCN_fold2.pth")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
             
    print(f"Loading Model: {model_path}")
    model = STGCNModel(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. BUILD k-NN FEATURE BANK
    print(f"\n--- 1. BUILDING k-NN FEATURE BANK ---")
    print(f"Using Hyperparameters: k={BEST_KNN_K}, percentile={BEST_KNN_PERCENTILE}")
    
    train_data_dir = "./data/data"
    actions = ["Cut-Down", "Stab", "Thrust"]
    train_dataset = ActionDataset(train_data_dir, actions)
    
    feature_bank = build_knn_feature_bank(train_dataset, model, device)
    threshold_val, train_dists = tune_knn_threshold(feature_bank, BEST_KNN_K, percentile=BEST_KNN_PERCENTILE)
    
    print(f"Feature Bank: {feature_bank.shape[0]} clips, {feature_bank.shape[1]}-dim")
    print(f"k-NN Threshold (k={BEST_KNN_K}, p{BEST_KNN_PERCENTILE}): {threshold_val:.6f}")
    print(f"Train Dist Stats: min={np.min(train_dists):.4f}, mean={np.mean(train_dists):.4f}, max={np.max(train_dists):.4f}")
    
    # Save Reference Data as .pt file (efficient for tensors)
    ref_data = {
        'feature_bank': feature_bank,      # (N, 256) tensor
        'threshold': threshold_val,         # float
        'k': BEST_KNN_K,                   # int
        'percentile': BEST_KNN_PERCENTILE, # int
        'ood_method': 'knn',               # string identifier
    }
    ref_path = os.path.join("results", "reference_data_knn.pt")
    os.makedirs("results", exist_ok=True)
    torch.save(ref_data, ref_path)
    print(f"Saved k-NN reference data to '{ref_path}'.")
    
    # 3. VALIDATE
    print(f"\n--- 2. FINAL VALIDATION (MICRO-AVERAGE) ---")
    validate_root = "./data/Validate"
    false_detected_root = "./data/False Detected Clip"
    categories = {
        "OOD": [os.path.join(validate_root, "OOD")],
        "False_Detected": [false_detected_root],
        "With_Coat": [os.path.join(validate_root, "With_Coat")],
        "Without_Coat": [os.path.join(validate_root, "Without_Coat")] 
    }
    
    results = []
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    
    print(f"{'Category':<15} | {'TP':<4} | {'TN':<4} | {'FP':<4} | {'FN':<4} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 80)
    
    for cat, paths in categories.items():
        n, tp, tn, fp, fn, acc, prec, rec, f1 = validate_category(cat, paths, model, feature_bank, threshold_val, BEST_KNN_K, device)
        total_tp += tp; total_tn += tn; total_fp += fp; total_fn += fn
        
        results.append({
            'Category': cat, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1
        })
        print(f"{cat:<15} | {tp:<4d} | {tn:<4d} | {fp:<4d} | {fn:<4d} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f}")
        
    print("-" * 80)
    
    grand_total = total_tp + total_tn + total_fp + total_fn
    micro_acc = (total_tp + total_tn) / grand_total if grand_total > 0 else 0
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0
    
    print(f"{'MICRO-AVG':<15} | {total_tp:<4d} | {total_tn:<4d} | {total_fp:<4d} | {total_fn:<4d} | {micro_acc:.4f} | {micro_prec:.4f} | {micro_rec:.4f} | {micro_f1:.4f}")
    print("-" * 80)
    
    results.append({
        'Category': 'MICRO-AVERAGE', 'TP': total_tp, 'TN': total_tn, 'FP': total_fp, 'FN': total_fn,
        'Accuracy': micro_acc, 'Precision': micro_prec, 'Recall': micro_rec, 'F1': micro_f1
    })
    
    df = pd.DataFrame(results)
    csv_path = os.path.join("results", "validation_results_knn.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved '{csv_path}'.")

if __name__ == "__main__":
    main()
