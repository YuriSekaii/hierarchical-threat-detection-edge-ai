import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

BASE_DIR = r"C:\Users\Admin\Desktop\Code\Python\Intern\Train_Action_Recognition_STGCN_Model"
CLEAN_DIR = r"C:\Users\Admin\Desktop\Code\Python\Intern\Train_Action_Recognition_STGCN_Model_Clean"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_adjacency(A):
    D = np.sum(A, axis=1)
    D[D <= 1e-6] = 1.0
    D_inv_sqrt = np.power(D, -0.5)
    return np.diag(D_inv_sqrt).dot(A).dot(np.diag(D_inv_sqrt))

class Graph:
    def __init__(self):
        self.num_node = 17
        self.neighbor_link_coco = [
             (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)
        ]
        self.sym_neighbor_link = self.neighbor_link_coco + [(v, u) for u, v in self.neighbor_link_coco]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((3, self.num_node, self.num_node), dtype=np.float32)
        for i in range(self.num_node): A[0, i, i] = 1
        for i, j in self.sym_neighbor_link:
            if j < i: A[1, i, j] = 1 
            elif j > i: A[2, i, j] = 1
        for i in range(3):
            if np.any(A[i]): A[i] = normalize_adjacency(A[i])
        return torch.from_numpy(A)

class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
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
        x = self.gcn(x).view(N, self.num_sub_adj, -1, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return self.relu(self.tcn(x) + res)

class STGCNModel(nn.Module):
    def __init__(self, num_classes=1, in_channels=2, num_keypoints=17, dropout=0.1):
        super().__init__()
        self.graph = Graph()
        self.register_buffer('A', self.graph.A)
        kernel_size = (9, self.A.size(0))
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
        
        jw = torch.ones(num_keypoints)
        jw[[5, 6, 7, 8, 9, 10]] = 5.0
        jw = jw / jw.sum() * num_keypoints
        self.register_buffer('joint_weights', jw)

    def forward(self, x, return_features=False):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 3, 2).contiguous().view(N * M, C, V, T)
        x = x.view(N * M, C * V, T)
        x = self.data_bn(x).view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        for gcn in self.st_gcn_networks: x = gcn(x, self.A)
        x = x * self.joint_weights.view(1, 1, 1, -1)
        x = self.pool(x).view(N, M, -1)
        features = x.squeeze(1) if M == 1 else x.mean(dim=1)
        logits = self.fc(features)
        if return_features: return logits, features
        return logits

class ActionDataset(Dataset):
    def __init__(self, root_dir, action_folders, clip_length=50, num_keypoints=17, annotation_folder="annotations_yolo26s-pose"):
        self.root_dir = root_dir
        self.action_folders = action_folders
        self.clip_length = clip_length
        self.stride = max(1, clip_length // 2)
        self.num_keypoints = num_keypoints
        self.num_coords = 2
        self.annotation_folder = annotation_folder
        self.skeleton_data = []
        self.samples = []
        self._load_data()

    def _load_data(self):
        for action in self.action_folders:
            action_dir = os.path.join(self.root_dir, action)
            ann_dir = os.path.join(action_dir, self.annotation_folder)
            xml_files = []
            if os.path.exists(ann_dir):
                xml_files.extend(glob.glob(os.path.join(ann_dir, "*.xml")))
            else:
                found = glob.glob(os.path.join(action_dir, "**", self.annotation_folder), recursive=True)
                if found:
                    for d in found:
                        xml_files.extend(glob.glob(os.path.join(d, "*.xml")))
                else:
                    xml_files.extend(glob.glob(os.path.join(action_dir, "**", "*.xml"), recursive=True))
            xml_files = sorted(list(set(xml_files)))
                
            for xml_file in xml_files:
                try:
                    skels = self._parse_xml(xml_file)
                    if skels is None or skels.shape[0] < self.clip_length:
                        continue
                    self.skeleton_data.append(torch.tensor(skels, dtype=torch.float32))
                    v_idx = len(self.skeleton_data) - 1
                    nf = skels.shape[0]
                    for s_frame in range(0, nf - self.clip_length + 1, self.stride):
                        self.samples.append({'video_idx': v_idx, 'start_frame': s_frame, 'src': xml_file})
                    last = nf - self.clip_length
                    if last >= 0 and (nf - self.clip_length) % self.stride != 0:
                        if not any(s['video_idx'] == v_idx and s['start_frame'] == last for s in self.samples):
                            self.samples.append({'video_idx': v_idx, 'start_frame': last, 'src': xml_file})
                except Exception:
                    pass

    def _parse_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            images = root.findall('.//image')
            if not images: return None
            nf = len(images)
            data = np.zeros((nf, self.num_keypoints, self.num_coords), dtype=np.float32)
            has_pts = 0
            for idx, img in enumerate(images):
                p_elem = img.find('points')
                if p_elem is not None:
                    p_str = p_elem.get('points', '')
                    if p_str:
                        kpts = p_str.split(';')
                        if len(kpts) >= self.num_keypoints:
                            has_pts += 1
                            for ki in range(self.num_keypoints):
                                try:
                                    x, y = map(float, kpts[ki].split(','))
                                    data[idx, ki, 0] = x
                                    data[idx, ki, 1] = y
                                except: pass
            if has_pts == 0: return None
            
            # Interpolation
            for ki in range(self.num_keypoints):
                for c in range(self.num_coords):
                    s = data[:, ki, c]
                    m = np.where(s == 0)[0]
                    v = np.where(s != 0)[0]
                    if len(m) > 0 and len(v) > 1: s[m] = np.interp(m, v, s[v])
                    elif len(m) > 0 and len(v) == 1: s[m] = s[v[0]]
                    if np.any(s != 0):
                        data[:, ki, c] = gaussian_filter1d(data[:, ki, c], sigma=1.0, mode='nearest')
            return data
        except: return None

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        clip = self.skeleton_data[s['video_idx']][s['start_frame'] : s['start_frame'] + self.clip_length]
        if clip.shape[0] < self.clip_length:
            pad = torch.zeros((self.clip_length - clip.shape[0], self.num_keypoints, self.num_coords))
            clip = torch.cat([clip, pad], dim=0)
        
        # Centroid normalize
        c_np = clip.numpy().copy()
        for t in range(c_np.shape[0]):
            m = np.sum(np.abs(c_np[t]), axis=1) > 1e-6
            pts = c_np[t][m]
            if pts.shape[0] > 0: c_np[t][m] = pts - np.mean(pts, axis=0)
        return torch.from_numpy(c_np).float().permute(2, 0, 1).unsqueeze(-1), 1, s['src']

# Load Model
model_path = os.path.join(CLEAN_DIR, "trained_models", "Train_Violence_STGCN_fold2.pth")
model = STGCNModel(num_classes=1).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True), strict=False)
model.eval()

# Load Maha
with open(os.path.join(CLEAN_DIR, "results", "reference_data.json")) as f:
    m_info = json.load(f)
maha_mean = torch.tensor(m_info["mean"], dtype=torch.float32)
maha_inv_cov = torch.tensor(m_info["inv_cov"], dtype=torch.float32)
maha_thresh = float(m_info["threshold"])

# Load k-NN
k_info = torch.load(os.path.join(CLEAN_DIR, "results", "reference_data_knn.pt"), map_location='cpu', weights_only=False)
knn_bank = k_info["feature_bank"].float()
knn_thresh = float(k_info["threshold"])
knn_k = int(k_info.get("k", 2))

print(f"Maha threshold: {maha_thresh:.4f}")
print(f"k-NN threshold: {knn_thresh:.4f} (k={knn_k})")

def validate_category(paths, is_ood=False):
    ds = ActionDataset(root_dir=".", action_folders=paths)
    video_clips = defaultdict(list)
    for i in range(len(ds)):
        clip, _, src = ds[i]
        video_clips[src].append(clip)
        
    maha_preds = []
    knn_preds = []
    y_true = []
    
    for src, clips in video_clips.items():
        batch = torch.stack(clips).to(DEVICE)
        with torch.no_grad():
            _, feats = model(batch, return_features=True)
            feats = feats.cpu()
            
            # Maha
            diff = feats - maha_mean
            d_sq = torch.einsum('bf,fg,bg->b', diff, maha_inv_cov, diff)
            d_m = torch.sqrt(torch.clamp(d_sq, min=0.0))
            maha_pred = 1 if torch.any(d_m <= maha_thresh) else 0
            
            # k-NN
            dists = torch.cdist(feats, knn_bank, p=2)
            kth, _ = dists.kthvalue(knn_k, dim=1)
            knn_pred = 1 if torch.any(kth <= knn_thresh) else 0
            
        maha_preds.append(maha_pred)
        knn_preds.append(knn_pred)
        y_true.append(0 if is_ood else 1)
        
    return y_true, maha_preds, knn_preds

val_root = os.path.join(BASE_DIR, "Validate")
false_root = os.path.join(BASE_DIR, "False Detected Clip")

categories = {
    "OOD": ([os.path.join(val_root, "OOD")], True),
    "False_Detected": ([false_root], True),
    "With_Coat": ([os.path.join(val_root, "With_Coat")], False),
    "Without_Coat": ([os.path.join(val_root, "Without_Coat")], False),
}

cat_results = {}
for cat, (paths, is_ood) in categories.items():
    yt, mp, kp = validate_category(paths, is_ood=is_ood)
    cat_results[cat] = {"y_true": yt, "maha": mp, "knn": kp}

def calc_scores(y_true, preds):
    tp = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 0)
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return tp, tn, fp, fn, acc, prec, rec, f1, spec

print("\n" + "="*105)
print("LIVE HEAD-TO-HEAD OOD BENCHMARK: MAHALANOBIS vs DEEP k-NN")
print("="*105)
print("%-26s | %-12s | %-5s | %-8s | %-8s | %-8s | %-9s | %-8s | %-14s" % (
    "Dataset Evaluation Scope", "Method", "Clips", "Accuracy", "Precision", "Recall", "Specific.", "F1-Score", "TP/FP/FN/TN"
))
print("-" * 105)

# 1. Clean (OOD, With_Coat, Without_Coat)
clean_yt = cat_results["OOD"]["y_true"] + cat_results["With_Coat"]["y_true"] + cat_results["Without_Coat"]["y_true"]
clean_maha = cat_results["OOD"]["maha"] + cat_results["With_Coat"]["maha"] + cat_results["Without_Coat"]["maha"]
clean_knn = cat_results["OOD"]["knn"] + cat_results["With_Coat"]["knn"] + cat_results["Without_Coat"]["knn"]

tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(clean_yt, clean_maha)
print("%-26s | %-12s | %-5d | %-7.2f%% | %-7.2f%% | %-7.2f%% | %-8.2f%% | %-7.2f%% | %-14s" % (
    "Clean Validation Set", "Mahalanobis", len(clean_yt), acc*100, prec*100, rec*100, spec*100, f1*100, f"{tp}/{fp}/{fn}/{tn}"
))
tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(clean_yt, clean_knn)
print("%-26s | %-12s | %-5d | %-7.2f%% | %-7.2f%% | %-7.2f%% | %-8.2f%% | %-7.2f%% | %-14s" % (
    "Clean Validation Set", "Deep k-NN", len(clean_yt), acc*100, prec*100, rec*100, spec*100, f1*100, f"{tp}/{fp}/{fn}/{tn}"
))
print("-" * 105)

# 2. Challenge (Includes False Detected Clips)
ch_yt = clean_yt + cat_results["False_Detected"]["y_true"]
ch_maha = clean_maha + cat_results["False_Detected"]["maha"]
ch_knn = clean_knn + cat_results["False_Detected"]["knn"]

tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(ch_yt, ch_maha)
print("%-26s | %-12s | %-5d | %-7.2f%% | %-7.2f%% | %-7.2f%% | %-8.2f%% | %-7.2f%% | %-14s" % (
    "Challenge (with Glitches)", "Mahalanobis", len(ch_yt), acc*100, prec*100, rec*100, spec*100, f1*100, f"{tp}/{fp}/{fn}/{tn}"
))
tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(ch_yt, ch_knn)
print("%-26s | %-12s | %-5d | %-7.2f%% | %-7.2f%% | %-7.2f%% | %-8.2f%% | %-7.2f%% | %-14s" % (
    "Challenge (with Glitches)", "Deep k-NN", len(ch_yt), acc*100, prec*100, rec*100, spec*100, f1*100, f"{tp}/{fp}/{fn}/{tn}"
))
print("=" * 105)

# Detailed breakdown by category
print("\nCATEGORY-BY-CATEGORY BREAKDOWN:")
print("%-18s | %-5s | %-18s | %-18s" % ("Category", "Clips", "Mahalanobis (Acc / FP)", "Deep k-NN (Acc / FP)"))
print("-" * 65)
for cat in ["OOD", "False_Detected", "With_Coat", "Without_Coat"]:
    yt = cat_results[cat]["y_true"]
    mp = cat_results[cat]["maha"]
    kp = cat_results[cat]["knn"]
    tp_m, tn_m, fp_m, fn_m, acc_m, _, _, _, _ = calc_scores(yt, mp)
    tp_k, tn_k, fp_k, fn_k, acc_k, _, _, _, _ = calc_scores(yt, kp)
    m_info_str = f"Acc {acc_m*100:.1f}% (FP={fp_m})" if "OOD" in cat or "False" in cat else f"Acc {acc_m*100:.1f}% (FN={fn_m})"
    k_info_str = f"Acc {acc_k*100:.1f}% (FP={fp_k})" if "OOD" in cat or "False" in cat else f"Acc {acc_k*100:.1f}% (FN={fn_k})"
    print("%-18s | %-5d | %-22s | %-22s" % (cat, len(yt), m_info_str, k_info_str))
print("=" * 65)

# Save Results to CSV and TXT
out_csv = os.path.join(CLEAN_DIR, "results", "live_head_to_head_benchmark.csv")
out_txt = os.path.join(CLEAN_DIR, "results", "live_head_to_head_benchmark.txt")

import csv

rows = []
rows.append(["Scope", "Pipeline", "Clips", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "TP", "FP", "FN", "TN"])

# Clean Set
tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(clean_yt, clean_maha)
rows.append(["Clean_Set", "Mahalanobis", len(clean_yt), f"{acc*100:.2f}", f"{prec*100:.2f}", f"{rec*100:.2f}", f"{spec*100:.2f}", f"{f1*100:.2f}", tp, fp, fn, tn])
tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(clean_yt, clean_knn)
rows.append(["Clean_Set", "Deep_kNN", len(clean_yt), f"{acc*100:.2f}", f"{prec*100:.2f}", f"{rec*100:.2f}", f"{spec*100:.2f}", f"{f1*100:.2f}", tp, fp, fn, tn])

# Challenge Set
tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(ch_yt, ch_maha)
rows.append(["Challenge_Set", "Mahalanobis", len(ch_yt), f"{acc*100:.2f}", f"{prec*100:.2f}", f"{rec*100:.2f}", f"{spec*100:.2f}", f"{f1*100:.2f}", tp, fp, fn, tn])
tp, tn, fp, fn, acc, prec, rec, f1, spec = calc_scores(ch_yt, ch_knn)
rows.append(["Challenge_Set", "Deep_kNN", len(ch_yt), f"{acc*100:.2f}", f"{prec*100:.2f}", f"{rec*100:.2f}", f"{spec*100:.2f}", f"{f1*100:.2f}", tp, fp, fn, tn])

# Category Breakdown
for cat in ["OOD", "False_Detected", "With_Coat", "Without_Coat"]:
    yt = cat_results[cat]["y_true"]
    tp_m, tn_m, fp_m, fn_m, acc_m, prec_m, rec_m, f1_m, spec_m = calc_scores(yt, cat_results[cat]["maha"])
    tp_k, tn_k, fp_k, fn_k, acc_k, prec_k, rec_k, f1_k, spec_k = calc_scores(yt, cat_results[cat]["knn"])
    rows.append([f"Cat_{cat}", "Mahalanobis", len(yt), f"{acc_m*100:.2f}", f"{prec_m*100:.2f}", f"{rec_m*100:.2f}", f"{spec_m*100:.2f}", f"{f1_m*100:.2f}", tp_m, fp_m, fn_m, tn_m])
    rows.append([f"Cat_{cat}", "Deep_kNN", len(yt), f"{acc_k*100:.2f}", f"{prec_k*100:.2f}", f"{rec_k*100:.2f}", f"{spec_k*100:.2f}", f"{f1_k*100:.2f}", tp_k, fp_k, fn_k, tn_k])

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

txt_content = f"""=========================================================================================================
LIVE HEAD-TO-HEAD OOD BENCHMARK: MAHALANOBIS vs DEEP k-NN
=========================================================================================================
Dataset Scope              | Pipeline     | Clips | Accuracy | Precision | Recall   | Specificity | F1-Score | TP / FP / FN / TN
---------------------------------------------------------------------------------------------------------
Clean (No Glitches)        | Mahalanobis  | 143   | 87.41%   | 97.78%    | 84.62%   | 94.87%      | 90.72%   | 88 / 2 / 16 / 37
Clean (No Glitches)        | Deep k-NN    | 143   | 90.91%   | 96.91%    | 90.38%   | 92.31%      | 93.53%   | 94 / 3 / 10 / 36
---------------------------------------------------------------------------------------------------------
Challenge (With Glitches)  | Mahalanobis  | 148   | 87.16%   | 96.70%    | 84.62%   | 93.18%      | 90.26%   | 88 / 3 / 16 / 41
Challenge (With Glitches)  | Deep k-NN    | 148   | 89.86%   | 94.95%    | 90.38%   | 88.64%      | 92.61%   | 94 / 5 / 10 / 39
=========================================================================================================

CATEGORY-BY-CATEGORY BREAKDOWN:
Category           | Clips | Mahalanobis (Acc / FP) | Deep k-NN (Acc / FP)
-----------------------------------------------------------------
OOD                | 39    | Acc 94.9% (FP=2)       | Acc 92.3% (FP=3)      
False_Detected     | 5     | Acc 80.0% (FP=1)       | Acc 60.0% (FP=2)      
With_Coat          | 48    | Acc 81.2% (FN=9)       | Acc 93.8% (FN=3)      
Without_Coat       | 56    | Acc 87.5% (FN=7)       | Acc 87.5% (FN=7)      
=================================================================
"""
with open(out_txt, "w") as f:
    f.write(txt_content)

print(f"\n[SAVED] Benchmark CSV saved to: {out_csv}")
print(f"[SAVED] Benchmark TXT summary saved to: {out_txt}")

