# Hierarchical Threat Detection & Edge AI: Master Roadmap & New Chat Handoff Guide

> **CRITICAL DIRECTIVE FOR NEW CONVERSATION AGENT:**  
> Read this file and `VERSIONS.md` carefully before reading or touching any other file. It defines your exact workspace boundaries, forbidden directories, python environment, user rules, and roadmap.

> [!IMPORTANT]
> ### 🛑 MANDATORY DEVELOPER & AGENT DIRECTIVE: CODE REUSE & CONTINUITY
> 1. **READ VERSIONS.md CAREFULLY BEFORE TOUCHING CODE:** Every version solved specific physical failure modes (v1.01 buffer memory, v1.02 torso-scale normalization, v1.04 multi-person tracking guard, v1.08 Relative Mahalanobis Distance). You MUST thoroughly understand past lessons to prevent regressions.
> 2. **DO NOT RECREATE ANYTHING FROM SCRATCH — USE OLD CODE:** Always reuse existing, validated modules (`src/skeleton_utils.py`, `src/dataset.py`, `src/inference_realtime_*.py`). Never write new coordinate normalizers, tracking loops, or data loaders from scratch.
> 3. **ONLY MODIFY THE SPECIFIC NEEDED PART:** When developing a new version, keep all established modules intact and alter ONLY the exact backbone or metric under study.
> 4. **FAITHFULLY REPORT ALL MODIFICATIONS IN VERSIONS.md:** Document every file change, line alteration, and parameter sweep in `VERSIONS.md`, explaining precisely *WHY* that change was made and its empirical outcome.

---

## 0. PROJECT DIRECTORY BOUNDARIES & ENVIRONMENT RULES (READ CAREFULLY)

There are multiple folders on the system, but this project is now **100% self-contained**. You must observe these strict boundaries at all times:

| Directory Path | Role / Permission | Strict Policy |
|---|---|---|
| **`D:\Intern AI Project\hierarchical-threat-detection-edge-ai`** | **PRIMARY PROJECT REPOSITORY (YOUR WORKSPACE)** | **ALL work happens here.** All code (`src/`, `models/`, `training/`), benchmark scripts (`scripts/`), weights (`weights/`), results (`results/`), and docs (`VERSIONS.md`, `ROADMAP_AND_HANDOFF.md`) MUST be created and edited inside this folder only. |
| **`D:\Intern AI Project\hierarchical-threat-detection-edge-ai\data`** | **LOCAL DATASET REPOSITORY** | Contains training clips (`Cut-Down`, `Stab`, `Thrust`) and validation suite (`Validate/With_Coat`, `Without_Coat`, `OOD`). Zero legacy dependencies. |
| **`D:\Intern AI Project\hierarchical-threat-detection-edge-ai\.venv\Scripts\python.exe`** | **LOCAL PYTHON RUNTIME** | Dedicated virtual environment with PyTorch 2.5.1 + CUDA 12.1 support on NVIDIA GeForce RTX 3090. Use this local python binary for all training, benchmarking, and inference scripts. |
| **`D:\Intern AI Project\Messy (Don't Touch)` & `D:\Intern AI Project\Failed Attempt`** | **FORBIDDEN / OBSOLETE** | **NEVER touch, access, import, or reference these folders under any circumstances.** |

---

## 1. Non-Negotiable User Rules & Constraints (STRICT COMPLIANCE REQUIRED)

1. **"ONLY START 1 ITEM AT A TIME. REPORT TO ME BEFORE CONTINUE."**  
   - **NEVER** automatically start the next version or roadmap item without explicit user review and permission.
   - Always report the completed version, show the tuning and comparison tables, and **STOP**.
2. **Conversational Comparison Table: Version 1.04 is the Chat Baseline:**  
   - In all conversational response tables, **do NOT show v1.00–v1.03**. Version 1.04 is the baseline column.
   - However, **in the permanent report [`VERSIONS.md`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/VERSIONS.md), ALWAYS preserve all versions from v1.00 to the latest**.
3. **Show Best Configuration Only in Master Comparison Table:**  
   - In the master comparison table, display *only the single best chosen configuration* (accounting for optimal trade-offs).
4. **Dedicated Hyperparameter Tuning Impact Table is MANDATORY:**  
   - For every parameter sweep (shrinkage $\epsilon$, decision boundary $\tau$, pruning percentile, learning rates, etc.), you **MUST** provide a dedicated impact table showing:
     `Recall`, `Precision`, `Accuracy`, `F1-Score`, `FP (False Alarms / 44)`, `FN (Missed Attacks / 33)`.
5. **Always Report Core Metrics:**  
   - Every benchmark report must prominently display **F1-Score** and **Total Time Speed (GPU & CPU)** side-by-side with per-stage latencies, Precision, Recall, Accuracy, False Alarms, FP, FN.
6. **Code & Weight Persistence (Never Overwrite Prior Versions):**  
   - Create modular versioned files: e.g., `src/inference_realtime_v2.10.py`, `models/poseconv3d.py`, `scripts/benchmark_poseconv3d_v2.10.py`.
   - Save trained checkpoints to `weights/`.
   - Save tuning logs to `results/tuning/`.
   - Save benchmark JSONs to `results/`.
   - Save per-category validation CSVs to `results/action_recognition_benchmark/`.
7. **Theoretical Research Document:**  
   - Full research literature is preserved at: [`Modern OOD Action Recognition Models.md`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/Modern%20OOD%20Action%20Recognition%20Models.md).

---

## 2. Complete Roadmap: Version 1.00 to Final

| Version / Item | Technique / Component | Focus / Type | Status | Key Metrics / Results |
|---|---|---|---|---|
| **v1.00** | Raw Frame Ingestion + ST-GCN + Euclidean k-NN | Baseline Prototype | **COMPLETED** | F1: 0.7937, RAM: 1.05 GB (OOM panics on edge) |
| **v1.01** | In-Memory TurboJPEG Buffer Compression | System / Memory | **COMPLETED** | RAM: 47.30 MB (-95.5% RAM savings), 0% accuracy loss |
| **v1.02** | Torso-Scale Invariant Kinematic Normalization | Preprocessing / GCN Retrain | **COMPLETED** | Slashed missed attacks from 24.2% to 6.1% (31/33 detected) |
| **v1.03** | Native Aspect-Ratio Rectangular Inference | Compute Efficiency | **COMPLETED** | 384x640 resolution (-40% pixel waste), -127.6 ms CPU saving |
| **v1.04** | Multi-Person Tracking Guard (ByteTrack + IoU + Continuity)| Stability / Tracking | **COMPLETED (Chat Baseline)** | 0 actor ID swaps (100% tracking lock), F1: 0.7654 |
| **v1.05 (Item 1)**| Hyperspherical Cosine k-NN Manifold Gating | Manifold OOD Metric | **COMPLETED** | F1: 0.7742, Prec: 82.8%, False alarms reduced to 5 |
| **v1.06 (Item 2)**| Virtual-Logit Matching (ViM Subspace PCA) | Subspace OOD Metric | **COMPLETED** | F1: 0.8052, Recall: 93.9%, Reference cache: only 25 KB |
| **v1.07 (Item 3)**| Activation Shaping (ASH-B) + Free Energy | Zero-Cache OOD Metric | **COMPLETED** | F1: 0.7887, Prec: 73.7%, **0 KB reference cache**, 45.6 $\mu$s CPU |
| **v1.08 (Item 4)**| Relative Mahalanobis Distance (RMD) | Likelihood-Ratio OOD | **COMPLETED (CURRENT ALL-TIME RECORD CHAMPION)**| **F1: 0.9167, 100.0% Recall (0 missed attacks), Prec: 84.6%, 6 FP, 37.8 FPS (GPU)** |
| **v2.00 (Item 5)**| CTR-GCN Dynamic Spatio-Temporal Backbone | Dynamic GCN Backbone | **COMPLETED** | F1: 0.9014, 1.33M params (-55.8%), but 6.3x slower GPU backbone (10.7 ms vs 1.7 ms) |
| **v2.10 (Item 6)**| **PoseConv3D (PoseC3D) Volumetric 3D Heatmap CNN** | **3D CNN Backbone** | **COMPLETED (PARADIGM FAILURE)** | **FAILED AS REPLACEMENT FOR ST-GCN.** 1,000x larger input data (2.66M vs 2.5k floats), 20x slower training (~40 min vs 2 min), but WORSE recall (78.8% vs 100.0% in v1.08, 7 missed attacks due to 56x56 spatial blurring). Excellent CPU speed (23.3 FPS) and precision (96.3%), but unacceptable for zero-miss knife defense. |
| **v2.20 (Item 7)**| **SkateFormer Partitioned Spatial-Temporal Transformer**| **Skeletal-Temporal ViT** | **COMPLETED** | F1: 0.7879, Prec: 78.8%, Rec: 78.8% (7 missed), 7 FP, **447k params (-85.2% compression)**, 29.3 FPS (GPU), 6.4 FPS (CPU). Avoided heatmap bloat; fast 2 min training; but unconstrained self-attention overfits on small datasets vs biological GCNs. |
| **v3.00 (Item 8)**| **Skeleton-OOD End-to-End Latent Boundary Learning** | **End-to-End Kinematic OOD**| **COMPLETED (PARADIGM ANALYSIS)** | ST-GCN + In-Training ASH-SE + S^{255} Boundary. F1: 0.7126 (RMD) / 0.6168 (Energy, 100% Recall). In-training gradient truncation & pseudo-negatives distorted clean metric space vs v1.08 champion. |
| **v3.10 (Final)** | **Dual-Tier Edge & Server Architecture Ensembles** | **Final System Optimization** | **COMPLETED (ALL-TIME RECORD CHAMPION)** | Consolidates edge screening & server multi-model ensemble (ST-GCN + CTR-GCN + PoseConv3D). **F1: 0.9706**, **100.0% Recall (0 missed attacks)**, **Prec: 94.3%**, **ONLY 2 FP** (66.7% false alarm cut vs v1.08), 32.5 FPS Dual-Tier GPU / 26.9 FPS CPU. Offloads 44.2% traffic from server. |

---

## 3. Current Performance Leaderboard (v1.04 Baseline to v3.10)

| Metric / Dimension | Version 1.04 (Baseline) | Version 1.05 (Cosine k-NN) | Version 1.06 (ViM Subspace) | Version 1.07 (Activation Shaping) | Version 1.08 (Relative Mahalanobis) | Version 2.00 (CTR-GCN Dynamic) | Version 2.10 (PoseConv3D Volumetric) | Version 2.20 (SkateFormer Partitioned ViT) | Version 3.00 (Skeleton-OOD Latent Boundary) | Version 3.10 (Dual-Tier Edge & Server Ensembles) |
|---|---|---|---|---|---|---|---|---|---|---|
| **F1-Score (Violence Class)** | **0.7654** | **0.7742** | **0.8052** | **0.7887** | **0.9167 (Former Champion)** | **0.9014** | **0.8667 (RMD) / 0.8710 (Cosine)** | **0.7879 (@ $\tau=-0.4275$)** | **0.7126 (@ $\tau=1.6102$) / 0.6168 (Energy)** | **0.9706 (NEW ALL-TIME RECORD CHAMPION)** |
| **Total Active Threat Time (GPU)** | **26.45 ms** (37.8 FPS) | **26.37 ms** (37.9 FPS) | **26.36 ms** (37.9 FPS) | **26.35 ms** (38.0 FPS) | **26.49 ms** (37.8 FPS) | **35.52 ms** (28.2 FPS) | **27.57 ms (36.3 FPS)** | **34.09 ms (29.3 FPS)** | **30.25 ms (33.1 FPS)** | **30.74 ms (32.5 FPS Dual-Tier) / 29.56 ms (33.8 FPS Edge)** |
| **Total Active Threat Time (CPU)** | **139.34 ms** (7.2 FPS) | **139.25 ms** (7.2 FPS) | **139.21 ms** (7.2 FPS) | **139.20 ms** (7.2 FPS) | **139.34 ms** (7.2 FPS) | **148.47 ms** (6.7 FPS) | **42.92 ms (23.3 FPS, 3.2x CPU Speedup!)** | **156.87 ms (6.4 FPS)** | **38.09 ms (26.3 FPS, 3.7x CPU Speedup!)** | **37.22 ms (26.9 FPS Edge CPU)** |
| **Violence Detection Recall** | **93.9%** (31/33 detected) | 72.7% (24/33 detected) | **93.9%** (31/33 detected) | **84.8%** (28/33 detected) | **100.0% (33/33 detected, FLAWLESS)** | **97.0% (32/33 detected)** | **78.8% (26/33) — FAILED GOAL (7 Missed)** | **78.8% (26/33 detected, 7 missed)** | **93.9% (31/33 detected) / 100.0% (Energy)** | **100.0% (33/33 detected, ZERO MISSED ATTACKS)** |
| **Missed Attacks (False Negatives)** | 2 missed (6.1%) | 9 missed | 2 missed (6.1%) | 5 missed (15.2%) | **0 missed attacks (ZERO FALSE NEGATIVES)** | **Only 1 missed (3.0%)** | **7 missed (21.2%) / 6 missed (Cosine)** | **7 missed (21.2%)** | **2 missed (6.1%) / 0 missed (Energy)** | **0 missed attacks (FLAWLESS ZERO MISSES)** |
| **Violence Detection Precision** | **64.6%** (31/48) | **82.8%** (24/29) | **70.5%** (31/44) | **73.7%** (28/38) | **84.6% (33/39, +20.0% gain)** | **84.2% (32/38, +19.6% gain)** | **96.3% (26/27, PROJECT RECORD)** | **78.8% (26/33, +14.2% gain)** | **57.4% (31/54) / 44.6% (Energy)** | **94.3% (33/35, +9.7% gain over v1.08)** |
| **Overall Classification Accuracy** | **73.6%** | **80.6%** | **79.2%** | **79.2%** | **91.7% (+18.1% gain)** | **90.3% (+16.7% gain)** | **89.6% (+16.0% gain)** | **81.8% (+8.2% gain)** | **67.5% / 46.8% (Energy)** | **97.4% (+5.7% gain over v1.08)** |
| **OOD Civilian Rejection Rate** | **56.4%** (22/39) | **87.2%** (34/39) | **66.7%** (26/39) | **74.4%** (29/39) | **84.6% (33/39, 6 false alarms)** | **84.6% (33/39, 6 false alarms)** | **97.7% (43/44, ONLY 1 FALSE ALARM!)** | **84.1% (37/44, 7 false alarms)** | **47.7% (21/44, 23 false alarms) / 6.8% (Energy)** | **95.5% (42/44, ONLY 2 FALSE ALARMS!)** |
| **Civilian False Alarms (FP / 44)** | 17 false alarms | **5 false alarms** | 13 false alarms | 10 false alarms | **6 false alarms** | **6 false alarms** | **ONLY 1 FALSE ALARM (RMD) / 2 (Cosine)** | **7 false alarms** | **23 false alarms (RMD) / 41 (Energy)** | **ONLY 2 FALSE ALARMS (66.7% cut vs v1.08!)** |
| **Input Data Volume / Clip** | **2,550 numbers** (1x) | **2,550 numbers** (1x) | **2,550 numbers** (1x) | **2,550 numbers** (1x) | **2,550 numbers** (1x) | **2,550 numbers** (1x) | **2,665,600 numbers (1,045x DATA BLOAT)** | **2,550 numbers (1x, NORMALIZED COORDINATES)** | **2,550 numbers (1x, NORMALIZED COORDINATES)** | **2,550 numbers (1x, COORDINATES)** |
| **3-Fold Training Duration** | **~2 minutes** | **~2 minutes** | **~2 minutes** | **~2 minutes** | **~2 minutes** | **~3 minutes** | **~40 minutes (20x SLOWER TRAINING)** | **~2 minutes (FAST RETRAINING)** | **~2 minutes (FAST RETRAINING)** | **Zero Retraining Required (Reuses v1.08 champion, v2.00, v2.10)** |
| **Backbone Architecture** | ST-GCN (Static Graph) | ST-GCN (Static Graph) | ST-GCN (Static Graph) | ST-GCN (Static Graph) | ST-GCN (Static Graph) | **CTR-GCN (Dynamic Channel Refinement)** | **PoseConv3D (R(2+1)D 3D Heatmap CNN)** | **SkateFormer (Partitioned Skeletal-Temporal ViT)** | **ST-GCN + In-Training ASH-SE Fusion** | **Dual-Tier Ensemble: ST-GCN + CTR-GCN + PoseConv3D** |
| **Backbone Parameters** | 3,014,981 params | 3,014,981 params | 3,014,981 params | 3,014,981 params | 3,014,981 params | **1,333,369 params (-55.8%)** | **765,529 params (-74.6% COMPRESSION)** | **447,205 params (-85.2% RECORD COMPRESSION)** | **3,247,172 params (3.01M backbone + 232k fusion)** | **3.01M (Edge) / 5.11M (Server Ensemble Tri-Model)** |
| **Stage 1 Weapon Scan Latency** | 12.06 ms | 12.06 ms | 12.06 ms | 12.06 ms | 12.06 ms | 12.06 ms | 12.06 ms | **12.06 ms** | **12.06 ms** | **12.06 ms** |
| **Stage 2a Pose Latency / Frame** | 12.54 ms | 12.54 ms | 12.54 ms | 12.54 ms | 12.54 ms | 12.54 ms | 12.54 ms | **12.54 ms** | **12.54 ms** | **12.54 ms** |
| **Stage 2b Backbone + OOD Latency** | 1.85 ms | 1.77 ms | 1.76 ms | **1.75 ms** | **1.89 ms** | **10.92 ms** | **2.97 ms (2.74 ms fwd + 0.23 ms gating)** | **9.49 ms (9.29 ms fwd + 0.20 ms gating)** | **5.65 ms (5.44 ms fwd + 0.21 ms gating)** | **4.96 ms (Edge) / 22.68 ms (Server Ensemble)** |
| **OOD Gating Latency (CPU)** | 152.79 $\mu$s | 95.25 $\mu$s | 49.78 $\mu$s | **45.60 $\mu$s** | **142.77 $\mu$s** | **131.08 $\mu$s** | **145.47 $\mu$s** | **146.17 $\mu$s** | **153.25 $\mu$s** | **142.77 $\mu$s (Edge RMD)** |
| **Reference Vector Storage Needed** | 297 vectors (~304 KB) | 297 vectors (~304 KB) | 24 vectors (~25 KB) | **0 vectors (0 KB, zero-cache!)** | 4 covariance matrices (~1.0 MB) | 4 covariance matrices (~1.0 MB) | 4 covariance matrices (~1.3 MB) | 4 covariance matrices (~1.0 MB) | 4 covariance matrices (~1.0 MB) | **1.0 MB (Edge) / 3.3 MB (Server Ensemble)** |
| **Actor Tracking Strategy** | ByteTrack + Guard | ByteTrack + Guard | ByteTrack + Guard | ByteTrack + Guard | ByteTrack + Guard | ByteTrack + Guard | Canonical Torso-Norm + 3D Heatmap Stacking | **ByteTrack + Proximity Guard** | **ByteTrack + Proximity Guard** | **ByteTrack + Proximity Guard** |
| **Buffer RAM (1,200 frames)** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** |

---

---

## 4. Architectural Failure Analysis & Rules for Future Assistants

### A. Why PoseConv3D is Deemed a Failure for Knife/Violence Detection
1. **$1,000\times$ Data Overhead for Inferior Accuracy:**
   - ST-GCN processes flat 2D coordinates: $(3, 50, 17) = \mathbf{2,550}$ numbers per clip.
   - PoseConv3D converts coordinates into volumetric heatmaps: $(17, 50, 56, 56) = \mathbf{2,665,600}$ numbers per clip ($1,045\times$ larger).
   - Despite processing $1,000\times$ more data, **it failed the most critical requirement**: ST-GCN v1.08 achieved **100% recall (0 missed attacks)**, whereas PoseConv3D missed **7 attacks** (Recall = 78.8%).
2. **Spatial Downsampling Destroys Fine Threat Kinematics:**
   - Rasterizing continuous human motion into a low-resolution $56 \times 56$ heatmap causes spatial quantization blurring. Subtle, high-velocity wrist flicks, knife extension speeds, and thrusting trajectories get smeared across adjacent pixels, causing subtle knife attacks to be missed.
3. **Severe Training Time Inflation:**
   - Generating 3D Gaussian heatmaps on-the-fly for triplet loss requires calculating $\sim 8\text{ million}$ floats on CPU per batch item, inflating 3-fold training from **2 minutes (ST-GCN)** to **40 minutes (PoseConv3D)**.
4. **The Torso-Scale Normalization Mistake (Past Lesson Relearned):**
   - **Torso-scale normalization ($L_{\text{torso}}$) was already invented and validated in Version 1.02.**
   - In the first iteration of PoseConv3D, a naive bounding-box rasterizer was written from scratch, forgetting v1.02's normalization. This reintroduced raw walking drift (causing 16 false alarms). While applying v1.02's torso normalization fixed the false alarms (dropping to 1 FP), it exposed that the model simply cannot match ST-GCN's recall.

### B. Strict Code & File Organization Guidelines (Preventing Future Mistakes)
1. **Never Reinvent Preprocessing:**
   - Any new backbone (including Item 7: SkateFormer) **MUST directly import and use `normalize_skeleton_clip` from `src.skeleton_utils`**.
   - Do NOT write custom coordinate normalization or rasterization from scratch.
2. **Universal Dataset Inheritance:**
   - All action models must inherit from `ActionDataset` or `TripletActionDataset` in `src.dataset`.
   - Datasets must always resolve local `data/` first (`Cut-Down`, `Stab`, `Thrust`, `Validate`).
3. **Workspace Self-Containment:**
   - All code, scripts, weights, and results MUST stay inside `D:\Intern AI Project\hierarchical-threat-detection-edge-ai`.
   - **STRICT BOUNDARY:** NEVER import from, read from, or reference `Messy (Don't Touch)` or `Failed Attempt`.
   - Use the local Python environment: `.\.venv\Scripts\python.exe`.
4. **Current Project Champions (Stage 2 Action Recognition):**
   - **All-Time Project Record Champion:** **Version 3.10 (Dual-Tier Edge & Server Architecture Ensemble)**:
     $$\mathbf{F_1 = 0.9706}, \quad \mathbf{100.0\%\text{ Recall (0 Missed Attacks)}}, \quad \mathbf{94.3\%\text{ Precision}}, \quad \mathbf{ONLY\ 2\text{ FP (66.7\% reduction vs v1.08)}}, \quad \mathbf{97.4\%\text{ Accuracy}}$$
   - **Standalone Edge Champion:** **Version 1.08 (ST-GCN + Relative Mahalanobis Distance)**:
     $$F_1 = 0.9167, \quad 100.0\%\text{ Recall (0 Missed Attacks)}, \quad 84.6\%\text{ Precision}, \quad 6\text{ FP}, \quad 37.8\text{ FPS GPU} / 26.9\text{ FPS CPU}$$

---

## 5. Machine Environment, Paths, & File Inventory

- **Python Virtualenv (PyTorch 2.5.1 + CUDA RTX 3090):**  
  `D:\Intern AI Project\hierarchical-threat-detection-edge-ai\.venv\Scripts\python.exe`
- **Data Directories (Local Workspace):**  
  - Training Root: `D:\Intern AI Project\hierarchical-threat-detection-edge-ai\data`  
  - Validation Root: `D:\Intern AI Project\hierarchical-threat-detection-edge-ai\data\Validate`  
  - Action Subfolders: `['Cut-Down', 'Stab', 'Thrust']` (297 training clips)  
  - Validation Suite: 77 video items / 148 clips (`With_Coat`: 16, `Without_Coat`: 17, `OOD`: 44).
- **Core Checkpoints & Weights (NEVER DELETE OR OVERWRITE):**  
  - Stage 1 Weapon Production Checkpoint: `weights/yolo_weapon_distilled.pt` (Distilled YOLO26s, 11.18 ms)  
  - Stage 2a Pose Extraction Checkpoint: `weights/yolo26s-pose.pt`  
  - Stage 2b ST-GCN Retrained Champion: `weights/stgcn_violence_v1.02.pth`  
  - Stage 2b RMD Calibrated Weights: `weights/reference_data_rmd_v1.08.pt`  
  - Stage 2b CTR-GCN Model Checkpoint: `weights/ctrgcn_violence_v2.00.pth`  
  - Stage 2b CTR-GCN Calibrated Weights: `weights/reference_data_ctrgcn_v2.00.pt`  
  - Stage 2b PoseConv3D Model Checkpoint: `weights/poseconv3d_violence_v2.10.pth`  
  - Stage 2b PoseConv3D Calibrated Weights: `weights/reference_data_poseconv3d_v2.10.pt`  
  - Stage 2b SkateFormer Model Checkpoint: `weights/skateformer_violence_v2.20.pth`  
  - Stage 2b SkateFormer Calibrated Weights: `weights/reference_data_skateformer_v2.20.pt`  
  - Stage 2b Skeleton-OOD Model Checkpoint: `weights/stgcn_skeleton_ood_v3.00.pth`  
  - Stage 2b Skeleton-OOD Calibrated Weights: `weights/reference_data_skeleton_ood_v3.00.pt`  
  - Stage 2b Dual-Tier Architecture Module: [`models/ensemble_dual_tier.py`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/models/ensemble_dual_tier.py)  
  - Stage 2b Dual-Tier Calibrated Weights: `weights/reference_data_dual_tier_v3.10.pt`  
- **Documentation & Research Files:**  
  - Reference Research Literature: [`Modern OOD Action Recognition Models.md`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/Modern%20OOD%20Action%20Recognition%20Models.md)  
  - System Changelog & Complete Benchmark History: [`VERSIONS.md`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/VERSIONS.md)  
  - Master Benchmark Ground Truth (715 frames): [`results/ground_truth_benchmark/Comprehensive_Live_Verified_Benchmark.txt`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/Comprehensive_Live_Verified_Benchmark.txt)  
  - All Real-Time Pipelines: `src/inference_realtime_v1.04.py` through `src/inference_realtime_v3.10.py`  
  - All Benchmark Suites: `scripts/benchmark_*.py` through `scripts/benchmark_dual_tier_v3.10.py`  
  - All Tuning Sweeps: `results/tuning/` (including `tuning_dual_tier_v3.10.csv`)

---

## 6. PHASE 2: STAGE 1 (OBJECT / WEAPON DETECTION) OVERHAUL ROADMAP (VERSION 4.00+)

> [!IMPORTANT]
> ### 🎯 PRIMARY MISSION FOR NEW CONVERSATION AGENT:
> **Stage 2 (Biomechanical Action Recognition & OOD Gating) is 100% COMPLETE.**  
> Do **NOT** retrain, modify, or regress any Stage 2 code or weights.  
> Your entire focus in the new conversation is **Stage 1: Threat Object & Weapon Detection**.
> Stage 1 is the primary gatekeeper of the hierarchical system: **if a weapon is missed here, Stage 2 is never triggered!**

### A. Current Stage 1 Baseline Performance (To Beat)
- **Model Checkpoint:** `weights/yolo_weapon_distilled.pt` (Distilled YOLO26s student trained from YOLO26x teacher).
- **Ground Truth Test Suite:** 715 annotated surveillance frames (`results/ground_truth_benchmark/Comprehensive_Live_Verified_Benchmark.txt`).
- **Production Operating Point (`Conf = 0.45, IoU = 0.20`):**
  - **Accuracy:** 91.62%
  - **F1-Score:** 86.88%
  - **Precision:** 89.78% (23 False Positives on civilian items)
  - **Recall:** 84.17% (202/240 weapons detected, **38 MISSED WEAPONS**)
  - **Specificity:** 95.29%
  - **Inference Latency:** 11.18 ms (GTX 1060) / 12.06 ms (RTX 3090) (~85 FPS)
- **High-Sensitivity Operating Point (`Conf = 0.20, IoU = 0.20`):**
  - **Accuracy:** 84.27% | **F1-Score:** 78.85% | **Precision:** 69.18% (98 FP) | **Recall:** 91.67% (220/240, 20 missed) | **Latency:** 13.91 ms

### B. Critical Physical Failure Modes of Current Baseline
1. **Severe Misses on High-Velocity Slashing / Stabbing Motions:**
   - Thin metallic blades suffer motion blur across frames, reducing pixel contrast and dropping below the detection threshold (`Conf < 0.45`).
2. **Hand Occlusion & Grip Confusion:**
   - Knives gripped tightly in hands present minimal exposed blade area. Traditional anchor-free CNN heads struggle to decouple hand features from blade features.
3. **Specular Highlights & Ambient Lighting Reflections:**
   - Polished stainless steel blades produce intense specular glare or disappear against light-colored clothing.
4. **False Alarm Trade-off on Handheld Civilian Items:**
   - Lowering confidence threshold from 0.45 down to 0.20 recovers 18 missed weapons, but explodes civilian false alarms from 23 up to 98 (smartphones, pens, vape devices, metallic keys).
5. **Why the High-Resolution P2 Head Failed Previously (`YOLO26s-P2`):**
   - Naively adding a high-res P2 feature map ($160 \times 160$, stride 4) diluted training gradients across 4 multi-scale heads, causing recall to drop from 84.17% to 73.75% (-25 weapons caught) while spiking latency by +73% (19.34 ms).

---

### C. Proposed Architectural Candidates for Stage 1 Overhaul (Ref: `Stage1_SOTA_Architectures_2026.md`)

> [!NOTE]
> The comprehensive 2026 deep research whitepaper is preserved at:  
> [`Stage1_SOTA_Architectures_2026.md`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/Stage1_SOTA_Architectures_2026.md)  
> Title: *"Architectural Elimination of Hand and Geometric False Positives in Handheld Weapon Detection"*

| Version / Item | Proposed Candidate Architecture | Core Mechanism / Innovation | Anticipated Advantage over Baseline |
|---|---|---|---|
| **v4.00** | **Stage 1 Baseline Standardization** | Standalone benchmark runner for `weights/yolo_weapon_distilled.pt` | Establishes identical ground-truth validation pipeline |
| **v4.10 (Item 1 - PoC Champion)** | **ByteDance Sa2VA-Qwen2.5-VL-7B** (IEEE TPAMI 2026 / SAM-2 + Qwen2.5-VL) | Dense Grounded MLLM via `[SEG]` token + topological mask boundary closure | **Directly eliminates BOTH false alarm failure modes on RTX 3090:**<br>1. Qwen2.5-VL suppresses empty hands/fists via negative semantic prompt.<br>2. SAM-2 pixel affinity rejects 1D clothing seams/shadows lacking 3D volumetric solidity. |
| **v4.20 (Item 2)** | **Grounding DINO 1.5 Pro / Edge** | Deep cross-modal feature fusion + negative token competition | Competing negative prompt strings ("bare hand", "clothing seam") suppress weapon head logits |
| **v4.30 (Item 3)** | **Contact-State HOI Transformer** (DINOv2 Dual-Stream) | Bipartite hand-object relational graph + contact-state classifier | Hard mathematical gating: $S_{\text{contact}} \in \{\text{no-contact}, \text{self-contact}\}$ structurally blocks false alarms on bare hands |
| **v4.40 (Item 4)** | **Geometry-Informed Monocular Depth** (Depth Anything V2 + Detector) | Spatial surface normal / depth gradient discontinuity ($\nabla_n d$) | Physical 3D relief check immediately discards flat 2D clothing seams and shadows ($\nabla_n d \approx 0$) |
| **v4.50 (Item 5)** | **NMS-Free Real-Time Architectures** (RT-DETRv2 / YOLOv10/v12) | One-to-one bipartite matching / consistent dual assignment | Eliminates hand-blade greedy NMS suppression for fast edge deployment |

---

### D. Rules for the New Agent Starting Phase 2:
1. **Always read `ROADMAP_AND_HANDOFF.md`, `VERSIONS.md`, and `Stage1_SOTA_Architectures_2026.md` first.**
2. **Focus strictly on Proof-of-Concept on RTX 3090 (24GB VRAM).** Do NOT compromise detection accuracy for "lightweight" or "mobile" constraints.
3. **Execute only ONE item at a time.** Report findings, tuning tables, and trade-offs to the user and wait for explicit confirmation before proceeding.
4. **Maintain 100% self-containment.** Work strictly within `D:\Intern AI Project\hierarchical-threat-detection-edge-ai`. NEVER touch `Messy (Don't Touch)` or `Failed Attempt`.
5. **Use local Python runtime:** `.\.venv\Scripts\python.exe`.
6. **Preserve Stage 2 Champion Assets:** Under no circumstances should Stage 2 checkpoints (`stgcn_violence_v1.02.pth`, `reference_data_rmd_v1.08.pt`, `weights/reference_data_dual_tier_v3.10.pt`, `models/ensemble_dual_tier.py`) be modified or deleted.



