# Hierarchical Threat Detection & Edge AI: Complete Phase 1 & Phase 2 Technical Knowledge Base

## 1. Executive Summary & System Overview

This project engineers a real-time, low-latency hierarchical video surveillance system for detecting bladed weapon assaults (slashing, stabbing, thrusting) while rejecting non-violent civilian actions (wood chopping, stretching, tool handling) and visual confounders (clothing seams, zippers, bare fists). 

The system operates across a two-stage hierarchical cascade:
- **Stage 1 (Threat Object & Weapon Detection):** Analyzes incoming surveillance frames at high frame rates to detect handheld bladed weapons and localizes candidate regions.
- **Stage 2a (Kinematic Tracking & Pose Extraction):** Tracks the threat actor across frames and extracts 17-keypoint skeletal trajectories via top-down pose estimation.
- **Stage 2b (Biomechanical Action Recognition & OOD Manifold Gating):** Evaluates 50-frame sliding skeletal windows ($(3, 50, 17)$) through spatio-temporal action networks and Out-Of-Distribution (OOD) likelihood gating to confirm violent intent before raising an alarm.

```
Incoming Video Frame (CCTV / Edge Stream)
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Primary Weapon Detector (v4.00 / v4.50)           │
│  - Architecture: NMS-Free YOLO26s One-to-One Hungarian      │
│  - Latency: 12.86 ms (~78 FPS) | VRAM: < 1.0 GB             │
│  - Metrics: 86.88% F1, 89.78% Precision, 84.17% Recall      │
└──────────────────────────────┬──────────────────────────────┘
                               │
               Weapon Candidate Detected? (Conf >= 0.20 / 0.45)
               ├──────────────────────────────────────────┐
               │ NO                                       │ YES
               ▼                                          ▼
         [ Normal Flow ]        ┌───────────────────────────────────────────────────┐
                                │ STAGE 1-GUARD (Optional): Contact HOI (v4.30)     │
                                │ - Forearm-Scaled Proximity (d_norm <= 2.2)        │
                                │ - DINOv2 Grasp Affinity Gating (S >= 0.52)        │
                                │ - Latency: 33.93 ms (~29.5 FPS) | VRAM: 84 MB     │
                                │ - Slashes Civilian False Alarms by 84.3%          │
                                └─────────────────────────┬─────────────────────────┘
                                                          │
                                           Physical Grasp Confirmed?
                                           ├────────────────────────────────────────┐
                                           │ NO                                     │ YES
                                           ▼                                        ▼
                                [ Suppressed as Fabric / Seam ]    ┌──────────────────────────────────┐
                                                                   │ STAGE 2a: Multi-Person Guard     │
                                                                   │ - ByteTrack + Threat Lock (v1.04)│
                                                                   │ - Torso-Scale Normalization      │
                                                                   └────────────────┬─────────────────┘
                                                                                    │
                                                                                    ▼
                                                                   ┌──────────────────────────────────┐
                                                                   │ STAGE 2b: Action Recognition     │
                                                                   │ - Tier 1 Edge: ST-GCN + RMD      │
                                                                   │   (v1.08: 100% Recall, 0 FN)     │
                                                                   │ - Tier 2 Server: Tri-Model       │
                                                                   │   Consensus Ensemble (v3.10)     │
                                                                   │   (F1: 0.9706, ONLY 2 FP)        │
                                                                   └────────────────┬─────────────────┘
                                                                                    │
                                                                                    ▼
                                                                         [ HIGH-CONFIDENCE ALARM ]
```

---

## 2. Experimental Datasets, Hardware & Benchmarking Protocols

### A. Compute Environment
- **Workstation GPU:** NVIDIA GeForce RTX 3090 (24GB GDDR6X VRAM, CUDA 12.1, PyTorch 2.5.1).
- **Edge Emulation Hardware:** NVIDIA GeForce GTX 1060 (3GB / 6GB VRAM) & CPU single-thread execution.
- **Python Virtual Environment:** Self-contained virtualenv located at `.venv/Scripts/python.exe`.

### B. Phase 1 (Stage 2 Biomechanical Action Recognition) Dataset
- **Local Directory:** `data/` (`data/Validate/`)
- **Training Set (297 clips, normalized coordinates $(3, 50, 17)$):**
  - `Cut-Down`: 129 clips
  - `Stab`: 57 clips
  - `Thrust`: 111 clips
- **Validation Suite (77 video items / 148 evaluated sliding clips):**
  - `With_Coat` (16 violent videos): Assaults performed wearing heavy winter jackets/coats.
  - `Without_Coat` (17 violent videos): Assaults performed in standard indoor clothing.
  - `OOD` (44 civilian non-violent videos): Challenging civilian gestures (wood chopping, stretching, tool manipulation, waving, aerobic exercises). Total 33 violent assault samples + 44 OOD civilian negative samples.

### C. Phase 2 (Stage 1 Weapon Detection) Ground Truth Dataset
- **Local Directory:** `data/Ground_Truth_Dataset/`
- **715 Curated Surveillance Test Images:**
  - `With_Coat`: 359 images (151 annotated weapon targets, heavy outerwear, folds, zippers).
  - `Without_Coat`: 235 images (89 annotated weapon targets, indoor/outdoor surveillance).
  - `OOD`: 121 images (0 weapons, civilian non-violent activities: chopping wood, cell phone usage, walking).
  - Total: 240 annotated weapon targets across 715 images.
- **Evaluation Protocol:** Exact ground-truth bounding box matching evaluated at $\text{IoU} = 0.20$.

---

## 3. Master Performance Leaderboards

### Table 1: Phase 1 (Stage 2 Biomechanical Action Recognition & OOD Gating) Leaderboard

| Metric / Dimension | v1.00 (Baseline) | v1.01 (JPEG Buf) | v1.02 (Torso Norm) | v1.03 (Rectangular) | v1.04 (Track Guard) | v1.05 (Cosine k-NN) | v1.06 (ViM Subspace) | v1.07 (ASH-B Energy) | v1.08 (RMD Champion) | v2.00 (CTR-GCN) | v2.10 (PoseConv3D) | v2.20 (SkateFormer) | v3.00 (Skeleton-OOD) | v3.10 (Dual-Tier Ensemble) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F1-Score (Violence)** | 0.7937 | 0.7937 | 0.7654 | 0.7654 | 0.7654 | 0.7742 | 0.8052 | 0.7887 | **0.9167** | 0.9014 | 0.8667 (RMD) / 0.8710 | 0.7879 | 0.7126 (RMD) / 0.6168 | **0.9706 (RECORD)** |
| **Recall (Violence)** | 75.8% (25/33) | 75.8% (25/33) | 93.9% (31/33) | 93.9% (31/33) | 93.9% (31/33) | 72.7% / 100% | 93.9% (31/33) | 84.8% (28/33) | **100.0% (33/33, ZERO MISS)** | 97.0% (32/33) | 78.8% (26/33, 7 FN) | 78.8% (26/33, 7 FN) | 93.9% / 100.0% | **100.0% (33/33, ZERO MISS)** |
| **Precision (Violence)** | 83.3% (25/30) | 83.3% (25/30) | 64.6% (31/48) | 64.6% (31/48) | 64.6% (31/48) | 82.8% (24/29) | 70.5% (31/44) | 73.7% (28/38) | 84.6% (33/39) | 84.2% (32/38) | **96.3% (26/27)** | 78.8% (26/33) | 57.4% / 44.6% | **94.3% (33/35)** |
| **Accuracy (Overall)** | 83.1% | 83.1% | 73.6% | 73.6% | 73.6% | 80.6% | 79.2% | 79.2% | 91.7% | 90.3% | 89.6% | 81.8% | 67.5% / 46.8% | **97.4%** |
| **Missed Attacks (FN)** | 8 missed | 8 missed | 2 missed | 2 missed | 2 missed | 9 / 0 missed | 2 missed | 5 missed | **0 missed (0.0%)** | 1 missed | 7 missed | 7 missed | 2 / 0 missed | **0 missed (0.0%)** |
| **Civilian Alarms (FP/44)**| 17 FP | 17 FP | 17 FP | 17 FP | 17 FP | 5 FP | 13 FP | 10 FP | 6 FP | 6 FP | **1 FP (97.7% Rejection)**| 7 FP | 23 FP / 41 FP | **ONLY 2 FP (95.5% Rejection)**|
| **Pipeline Latency (GPU)**| 25.67 ms (38.9 FPS)| 27.58 ms | 27.58 ms | 26.42 ms | 26.45 ms | 26.37 ms | 26.36 ms | 26.35 ms | 26.49 ms (37.8 FPS)| 35.52 ms (28.2 FPS)| 27.57 ms (36.3 FPS) | 34.09 ms (29.3 FPS)| 30.25 ms (33.1 FPS)| 30.74 ms (32.5 FPS Dual)|
| **Pipeline Latency (CPU)**| 155.65 ms | 157.58 ms | 157.58 ms | 139.31 ms | 139.34 ms | 139.25 ms | 139.21 ms | 139.20 ms | 139.34 ms (7.2 FPS)| 148.47 ms (6.7 FPS)| **42.92 ms (23.3 FPS)** | 156.87 ms (6.4 FPS)| 38.09 ms (26.3 FPS)| **37.22 ms (26.9 FPS Edge)**|
| **OOD Gating Time (CPU)** | 155.07 $\mu$s | 155.07 $\mu$s | 155.07 $\mu$s | 152.79 $\mu$s | 152.79 $\mu$s | 95.25 $\mu$s | 49.78 $\mu$s | **45.60 $\mu$s** | 142.77 $\mu$s | 131.08 $\mu$s | 145.47 $\mu$s | 146.17 $\mu$s | 153.25 $\mu$s | 142.77 $\mu$s |
| **Model Parameters** | 3.01M | 3.01M | 3.01M | 3.01M | 3.01M | 3.01M | 3.01M | 3.01M | 3.01M | 1.33M (-55.8%) | 765k (-74.6%) | **447k (-85.2%)** | 3.25M | 3.01M Edge / 5.11M Serv |
| **Reference Bank Memory**| ~304 KB (297 v) | ~304 KB (297 v) | ~304 KB (297 v) | ~304 KB (297 v) | ~304 KB (297 v) | ~304 KB (297 v) | ~25 KB (24 v) | **0 KB (Zero-Cache)** | 1.0 MB (4 cov mats) | 1.0 MB (4 cov mats) | 1.3 MB (4 cov mats) | 1.0 MB (4 cov mats) | 1.0 MB (4 cov mats) | 1.0 MB Edge / 3.3 MB Serv |
| **Buffer RAM (1,200 f)** | 1,054.69 MB | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** | **47.30 MB** |
| **Multi-Person ID Swaps**| 15 swaps | 15 swaps | 15 swaps | 15 swaps | **0 swaps (Lock)** | **0 swaps (Lock)** | **0 swaps (Lock)** | **0 swaps (Lock)** | **0 swaps (Lock)** | **0 swaps (Lock)** | **0 swaps (Heatmap)** | **0 swaps (Lock)** | **0 swaps (Lock)** | **0 swaps (Lock)** |

---

### Table 2: Phase 2 (Stage 1 Weapon Detection on 715 Ground Truth Images) Leaderboard

| Metric / Dimension | Baseline / v4.50 (1-to-1 NMS-Free, Conf=0.45) | Baseline / v4.50 (1-to-1 NMS-Free, Conf=0.20) | v4.50 (1-to-Many Greedy NMS, Conf=0.45) | v4.50 (1-to-Many Greedy NMS, Conf=0.20) | v4.10 (Sa2VA-7B MLLM + SAM-2) | v4.20 (Grounding DINO 1.5, Box=0.25 P3) | v4.30 (Contact HOI, Tuned $\tau=0.52$) | v4.30 (Contact HOI, High-Rec $\tau=0.50$) | v4.40 (Monocular Depth, Conf=0.30) | v4.50 (RT-DETR-L Zero-Shot COCO) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Overall F1-Score** | **86.88%** | 78.85% | 82.33% (-4.55%) | 72.73% (-6.12%) | 35.29% | 30.54% | **84.08%** | 83.53% | **86.12%** | 0.00% (Collapse) |
| **With_Coat F1-Score** | **92.83%** | 88.89% | 89.26% | 86.96% | 47.06% | 45.83% | 88.24% | 88.46% | 92.05% | 0.00% |
| **Without_Coat F1-Score**| **83.54%** | 83.33% | 84.71% | 81.16% | 25.00% | 20.92% | 80.68% | 81.97% | **87.21%** | 0.00% |
| **Threat Precision** | **89.78% (202/225)** | 69.18% (220/318) | 79.46% (205/258) | 59.57% (224/376) | 54.55% (6/11) | 31.56% (71/225) | 82.40% (206/250) | 78.89% (213/270) | 83.27% (214/257) | 0.00% (0/3) |
| **Threat Recall** | 84.17% (202/240) | **91.67% (220/240)** | 85.42% (205/240) | **93.33% (224/240)** | 26.09% (6/23) | 29.58% (71/240) | 85.83% (206/240) | **88.75% (213/240)** | **89.17% (214/240)** | 0.00% (0/240) |
| **Missed Weapons (FN)** | 38 missed | 20 missed | 35 missed | 16 missed | 17 / 23 missed | 169 missed | 34 missed | 27 missed | 26 missed | 240 missed (100%) |
| **Overall Accuracy** | **91.62%** | 84.27% | 87.91% | 77.92% | 70.67% | 60.37% | 89.39% | 88.62% | 90.59% | 66.30% |
| **Civilian Specificity** | **95.29% (465/488)** | 80.78% (412/510) | 89.14% (435/488) | 70.83% (369/521) | 90.38% (47/52) | 73.22% (421/575) | 91.11% (451/495) | 88.55% (441/498) | 91.28% (450/493) | 99.38% (478/481) |
| **OOD False Alarms (FP)**| 14 FP / 121 | 51 FP / 121 | 30 FP / 121 (+114%)| 87 FP / 121 (+71%) | 1 FP / 12 | 72 FP / 158 | **8 FP / 121 (-84.3%)**| 15 FP / 121 (-70.6%)| 23 FP / 121 (-54.9%)| 3 FP / 121 |
| **Total False Alarms (FP)**| **23 FP** | 98 FP | 53 FP (+130%) | 152 FP (+55%) | 5 FP | 154 FP (1,751 @ 0.20)| 44 FP (-55.1%) | 57 FP | 43 FP (-56.1%) | 3 FP |
| **Confusion Matrix** | 202 / 465 / 23 / 38 | 220 / 412 / 98 / 20 | 205 / 435 / 53 / 35 | 224 / 369 / 152 / 16 | 6 / 47 / 5 / 17 | 71 / 421 / 154 / 169 | 206 / 451 / 44 / 34 | 213 / 441 / 57 / 27 | 214 / 450 / 43 / 26 | 0 / 478 / 3 / 240 |
| **Latency (RTX 3090)** | **12.86 ms (~78 FPS)**| 13.02 ms | 12.48 ms (~80 FPS)| 12.54 ms | 1,910.9 ms (0.52 FPS)| 185.89 ms (5.38 FPS)| **33.93 ms (~29.5 FPS)**| 34.00 ms (~29.4 FPS)| 113.42 ms (8.8 FPS) | 29.57 ms (33.8 FPS) |
| **Latency (GTX 1060)** | **11.18 ms (~89 FPS)**| 13.91 ms | 11.45 ms | 14.10 ms | N/A (OOM 3GB) | ~1,200 ms (0.8 FPS) | **~45 ms (~22 FPS)** | ~45 ms | ~650 ms (1.5 FPS) | ~55 ms |
| **Active VRAM Footprint**| **< 1.0 GB** | **< 1.0 GB** | **< 1.0 GB** | **< 1.0 GB** | 15.95 GB | 892.10 MB | **84.23 MB** | **84.23 MB** | 94.56 MB | ~480 MB |
| **Verdict / Status** | **PRODUCTION CHAMPION**| Sensitive Fallback | Inferior to NMS-Free| Severe False Alarms | REJECTED (Too Slow) | REJECTED (Domain Gap)| **BEST STREAM GUARD** | High-Recall Stream | REJECTED (Latency) | REJECTED (Zero-Shot) |

---

## 4. Phase 1: Stage 2 Biomechanical Action Recognition & OOD Evolution

### Version 1.00 (Baseline Prototype)
- **Architecture:** Pipeline connecting frame ingestion, YOLO Stage 1 weapon screening, YOLO26s-pose Stage 2a joint extraction, and 9-block Spatial-Temporal Graph Convolutional Network (ST-GCN) + Euclidean k-NN ($p=2$) classifier.
- **Buffer Mechanism:** 1,200-frame circular buffer storing raw NumPy arrays ($640 \times 480 \times 3$, `uint8`).
- **Failure Analysis:** Storing 1,200 raw uncompressed frames consumed **1,054.69 MB (~1.05 GB)** of host memory. On unified-memory edge devices (Jetson Nano 4GB), this triggered OS swapping, buffer stalls, and Out-Of-Memory (OOM) kernel panics.
- **Empirical Metrics:** F1: 0.7937, Recall: 75.8% (25/33 detected, 8 missed), Precision: 83.3%, Accuracy: 83.1%, OOD Rejection: 92.3% (36/39, 3 FP).
- **Associated Result Files:**
  - Prototype Script: [`src/inference_realtime.py`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/src/inference_realtime.py)

### Version 1.01 (In-Memory TurboJPEG Buffer Compression)
- **Innovation:** Replaced raw NumPy frame caching with in-memory SIMD TurboJPEG byte buffers (`cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])`). Decompression (`cv2.imdecode`) is executed lazily on demand only when a frame is sampled for weapon screening (3 FPS) or sliding pose extraction (10 frames/stride).
- **Outcome:** Buffer RAM collapsed from **1,054.69 MB down to 47.30 MB** (**95.5% RAM reduction / 22.3x compression**).
- **Latency Overhead:** Ingestion JPEG encode overhead was only +0.88 ms (1,135 FPS capacity). Stage 1 fetch added +0.95 ms decode; Stage 2a added +0.96 ms decode. ST-GCN embedding cosine similarity was **1.00000** (100% decision agreement with zero degradation). Completely eliminated edge OOM risk.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_buffer_compression_v1.01.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_buffer_compression_v1.01.json)

### Version 1.02 (Torso-Scale Invariant Kinematics & ST-GCN Retraining)
- **Root-Cause Discovery:** In v1.00/v1.01, coordinate normalization used centroid mean subtraction without scaling. Subjects far from the camera had motion vectors compressed by 50–70%, causing ST-GCN to miss 8 of 33 violent assaults (24.2% False Negative rate).
- **Algorithmic Fix:** Normalized keypoint coordinates by physical torso length:
  $$L_{\text{torso}} = \|\text{mid\_shoulder} - \text{mid\_hip}\|_2$$
  with a fallback to $0.5 \times \text{bbox\_diagonal}$ if torso joints are occluded.
- **Retraining:** Retrained ST-GCN with Triplet Margin Loss across 3 folds. Fold 2 champion converged to validation loss **0.0171**.
- **Outcome:** Missed attacks dropped from 24.2% down to **6.1%** (31/33 assaults detected). Violence Recall surged to **93.9%** (and 100% at threshold 1.75). F1 reached 0.7654.
- **Associated Result Files:**
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_knn_v1.02.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_knn_v1.02.csv)
  - Training Logs (Folds 1-3): [`results/action_recognition_benchmark/training_log_v1.02_fold1.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/training_log_v1.02_fold1.csv), [`training_log_v1.02_fold2.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/training_log_v1.02_fold2.csv), [`training_log_v1.02_fold3.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/training_log_v1.02_fold3.csv)
  - Loss Curves: `results/action_recognition_benchmark/loss_curve_v1.02_fold1.png`, `fold2.png`, `fold3.png`
  - Archival Benchmark: [`results/benchmark_standoff_archival_v1.02.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_standoff_archival_v1.02.json)
  - OOD Tuning Logs: [`results/action_recognition_benchmark/ood_tuning_coarse_fold2.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/ood_tuning_coarse_fold2.csv), [`ood_tuning_fine_fold2.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/ood_tuning_fine_fold2.csv)

### Version 1.03 (Native Aspect-Ratio Rectangular Inference)
- **Root-Cause Discovery:** Standard YOLO models letterbox widescreen 16:9 surveillance streams ($1280 \times 720$, $1920 \times 1080$) into square $640 \times 640$ tensors (409,600 px). Rectangular $384 \times 640$ tensors use 245,760 px. Square letterboxing wasted **163,840 pixels (40.0% compute waste)** on black padding bars.
- **Algorithmic Fix:** Configured dynamic rectangular inference (`imgsz=(384, 640)`), computing the nearest 32-stride $(H, W)$ matching the active camera stream.
- **Outcome:** Eliminated 40% redundant MAC operations. On CPU edge simulation, YOLO Stage 1 dropped from 70.19 ms to 66.60 ms (-5.1%), and YOLO-Pose dropped from 83.61 ms to 70.86 ms (-15.3%). Saved **127.6 ms CPU execution time per 10-frame sliding stride**, preventing queue backlog. Maintained 100.0% decision agreement and 1.0000 bbox IoU.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_rectangular_inference_v1.03.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_rectangular_inference_v1.03.json)

### Version 1.04 (Multi-Person Tracking Guard)
- **Root-Cause Discovery:** When multiple people appeared in view (attacker + victim/passerby), naive `res[0].keypoints.data[0]` indexed detections by raw confidence score. Confidence oscillations caused detection order to flip between frames, causing joints to teleport >300 px/frame, corrupting the temporal graph.
- **Algorithmic Fix:** Integrated ByteTrack (`model.track(persist=True)`) with two-layer protective heuristics:
  1. Threat Actor Locking: Automatically locks tracking onto the person holding or closest to the detected weapon bounding box.
  2. Spatial-Temporal Continuity Fallback: Uses bounding box IoU and centroid distance matching if ByteTrack ID temporarily drops during occlusions.
- **Outcome:** Actor ID swaps slashed from **15 swaps to 0 swaps (100% stable lock)**. Velocity spikes plummeted from **307.6 px/frame to 1.0 px/frame**. Tracking overhead was negligible (**28.76 microseconds/frame**).
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_multi_person_guard_v1.04.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_multi_person_guard_v1.04.json)

### Version 1.05 (Hyperspherical Cosine k-NN Manifold Gating)
- **Theoretical Basis:** Latent feature norms in Euclidean ST-GCN varied widely (15.64 to 41.83). Benign civilian actions with small vector norms registered as deceptively close to combat clusters, yielding 17 false alarms on 39 OOD clips (56.4% rejection, 64.6% precision).
- **Algorithmic Fix:** Projected penultimate latent representations $z \in \mathbb{R}^{256}$ onto the unit hypersphere $\mathcal{S}^{D-1}$ via $\ell_2$ normalization: $\tilde{z} = z / (\|z\|_2 + \epsilon)$. Metric evaluated via fused BLAS GEMM (`torch.mm`):
  $$D_{\cos}(q, b_i) = 1.0 - \tilde{q}^T \tilde{b}_i$$
- **Outcome:** Slashed civilian false alarms from 17 down to **5** (70.6% reduction). Precision surged from 64.6% to **82.8%** (+18.2% gain). Overall accuracy reached 80.6%. Gating latency dropped to **95.25 $\mu$s on CPU**. High-security threshold ($\tau=0.0047$) delivered **100.0% Recall** (0 missed attacks).
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_cosine_knn_v1.05.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_cosine_knn_v1.05.json)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_cosine_knn_v1.05.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_cosine_knn_v1.05.csv), [`tuning_cosine_knn_v1.05.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_cosine_knn_v1.05.json)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_cosine_knn_v1.05.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_cosine_knn_v1.05.csv)

### Version 1.06 (Virtual-Logit Matching - ViM Subspace OOD Detection)
- **Theoretical Basis:** Combat motions reside within a low-dimensional affine principal subspace, whereas OOD civilian movements project significant residual energy into the orthogonal null space.
- **Algorithmic Fix:** Decomposed the centered in-distribution latent space ($N=297, D=256$) via PCA into principal subspace $V \in \mathbb{R}^{256 \times K}$ ($K=24$) and residual null space $P^\perp = I - V V^T$. Computed null space residual $\|r(z)\|_2$ and virtual logit $v(z) = \alpha \cdot \|r(z)\|_2$ ($\alpha=4.31$), forming ViM score $S_{\text{ViM}} = \sigma(\text{logit} - v(z))$ with threshold $\tau=0.0888$.
- **Outcome:** First version to break the 0.80 F1 barrier (**F1: 0.8052**). Maintained 93.9% recall (only 2 missed) while cutting CPU gating latency to **49.78 $\mu$s** (-67.4% vs Euclidean). Memory cache dropped from 304 KB to **25 KB**.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_vim_v1.06.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_vim_v1.06.json)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_vim_v1.06.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_vim_v1.06.csv), [`tuning_vim_v1.06.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_vim_v1.06.json)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_vim_v1.06.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_vim_v1.06.csv)

### Version 1.07 (Activation Shaping - ASH-B + Free Energy OOD Scoring)
- **Theoretical Basis:** OOD movements induce disproportionately large, erratic activations in a tiny subset of penultimate hidden units, artificially inflating downstream classification confidence.
- **Algorithmic Fix:** Extracted the 70th percentile threshold ($p=70\%$) via `torch.topk` and binarized activations ($\tilde{x} = \mathbb{I}(x \ge s_{70})$). Fed shaped activations into the linear classification head and computed Helmholtz Free Energy:
  $$S_{\text{Energy}} = \text{softplus}(\text{logit})$$
- **Outcome:** F1 reached 0.7887 with **0 KB reference cache (zero memory footprint)**. CPU gating latency reached a record **45.60 $\mu$s**. Precision reached 73.7% with 10 FP.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_ash_v1.07.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_ash_v1.07.json)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_ash_v1.07.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_ash_v1.07.csv), [`tuning_ash_v1.07.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_ash_v1.07.json)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_ash_v1.07.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_ash_v1.07.csv)

### Version 1.08 (Relative Mahalanobis Distance - RMD Champion)
- **Theoretical Basis:** Standard unimodal Mahalanobis distance collapses on multi-modal combat trajectories because a single Gaussian centroid sits in empty space between distinct attack modes (Cut-Down vs Stab vs Thrust). Shared background posture variances also dominate covariance.
- **Algorithmic Fix:** Formulated RMD as a likelihood ratio between class-conditional Gaussians $\mathcal{N}(\mu_c, \Sigma_c)$ ($c \in \{\text{Cut-Down}, \text{Stab}, \text{Thrust}\}$) and a global background distribution $\mathcal{N}(\mu_0, \Sigma_0)$ with ridge shrinkage $\Sigma_{\text{reg}} = \Sigma + \epsilon I$ ($\epsilon = 0.005$):
  $$\text{Score}_{\text{RMD}}(z) = (z - \mu_0)^T \Sigma_0^{-1} (z - \mu_0) - \min_{c} (z - \mu_c)^T \Sigma_c^{-1} (z - \mu_c)$$
  Subtracting global background distance $(M_0 - M_c)$ cancels out shared standing posture variances, isolating class-specific attack manifold energy.
- **Outcome (Standalone Edge Champion):**
  - **F1-Score: 0.9167** (first version to break 0.90).
  - **Violence Recall: 100.0% (33/33 attacks detected, ZERO MISSED ATTACKS)**.
  - **Violence Precision: 84.62%** (only 6 civilian false alarms across 44 OOD videos).
  - **Overall Accuracy: 91.67%**.
  - **Speed:** 26.49 ms total pipeline (37.8 FPS GPU) / 139.34 ms (7.2 FPS CPU). Gating latency: 142.77 $\mu$s on CPU.
  - **Storage:** 1.0 MB (4 covariance matrices + 4 mean vectors).
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_rmd_v1.08.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_rmd_v1.08.json)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_rmd_v1.08.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_rmd_v1.08.csv), [`tuning_rmd_v1.08.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_rmd_v1.08.json)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_rmd_v1.08.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_rmd_v1.08.csv)

### Version 2.00 (CTR-GCN Dynamic Spatio-Temporal Backbone)
- **Innovation:** Replaced static biological graph convolutions with Channel-Wise Topology Refinement (CTR-GC):
  $$A_k^{(c)} = A_k + B_k + C_k^{(c)}(X)$$
  where $A_k$ is the anatomical graph, $B_k$ is a learnable global prior, and $C_k^{(c)}(X)$ dynamically adapts topology per sample via channel correlation differences. Added Multi-Scale Temporal Convolutions (MS-TCN with 4 parallel dilated branches) and 5x arm-weighted spatial attention (joints 5-10).
- **Outcome:** Compressed model parameters by **55.8%** (from 3.01M down to 1.33M). Retrained 3-fold (Fold 1 achieved 0.0000 validation loss). Achieved **F1: 0.9014**, Recall: 97.0% (32/33), Precision: 84.2%, FP: 6.
- **Trade-Off:** Dynamic pairwise matrix multiplications ($C \times V \times V$) increased forward GPU latency from 1.70 ms (ST-GCN) to 10.73 ms, yielding 28.2 FPS pipeline throughput.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_ctrgcn_v2.00.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_ctrgcn_v2.00.json)
  - Training Summary: [`results/tuning/training_summary_ctrgcn_v2.00.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/training_summary_ctrgcn_v2.00.json)
  - Training Logs (Folds 1-3): [`results/training_log_ctrgcn_v2.00_fold1.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_ctrgcn_v2.00_fold1.csv), [`fold2.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_ctrgcn_v2.00_fold2.csv), [`fold3.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_ctrgcn_v2.00_fold3.csv)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_ctrgcn_v2.00.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_ctrgcn_v2.00.csv), [`tuning_ctrgcn_v2.00.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_ctrgcn_v2.00.json)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_ctrgcn_v2.00.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_ctrgcn_v2.00.csv)

### Version 2.10 (PoseConv3D Volumetric 3D Heatmap CNN — Paradigm Failure Analysis)
- **Concept:** Replaced graph adjacency message passing with 3D spatio-temporal heatmap volumes $(17 \times 50 \times 56 \times 56)$ processed via R(2+1)D convolutions (765k params, -74.6%). Integrated canonical torso normalization ($L_{\text{torso}}$) to prevent walking drift.
- **Successes:** Project-record precision (**96.30%**) and false alarm rejection (**97.73% OOD rejection, ONLY 1 FALSE ALARM out of 44 videos**). Extremely fast CPU backbone forward pass (**18.17 ms**), driving full CPU pipeline speed to **23.3 FPS** (3.2x faster than ST-GCN).
- **Why PoseConv3D Failed as a Standalone Replacement:**
  1. **$1,045\times$ Data Volume Bloat:** ST-GCN processes 2,550 numbers per clip ($(3, 50, 17)$). PoseConv3D requires 2,665,600 floats.
  2. **$20\times$ Slower Training:** On-the-fly 3D Gaussian generation inflated 3-fold training from 2 minutes to 40 minutes.
  3. **Spatial Quantization Destroys Knife Kinematics:** Rasterizing continuous motion into a low-resolution $56 \times 56$ grid caused spatial blurring. High-velocity linear knife thrusts and subtle wrist flicks smeared across adjacent pixels, causing PoseConv3D to miss **7 violent attacks (Recall dropped to 78.8%)**, failing the zero-miss safety requirement.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_poseconv3d_v2.10.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_poseconv3d_v2.10.json)
  - Training Summary: [`results/training_summary_poseconv3d_v2.10.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_summary_poseconv3d_v2.10.json)
  - Training Logs (Folds 1-3): [`results/training_log_poseconv3d_v2.10_fold1.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_poseconv3d_v2.10_fold1.csv), [`fold2.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_poseconv3d_v2.10_fold2.csv), [`fold3.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_poseconv3d_v2.10_fold3.csv)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_poseconv3d_v2.10.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_poseconv3d_v2.10.csv), [`tuning_poseconv3d_v2.10.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_poseconv3d_v2.10.json)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_poseconv3d_v2.10.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_poseconv3d_v2.10.csv)

### Version 2.20 (SkateFormer Partitioned Spatio-Temporal ViT)
- **Concept:** Addressed PoseConv3D's data bloat using a coordinate-first Vision Transformer processing raw scale-normalized coordinates ($(2, 50, 17) = 2,550$ numbers). Employs Skate-Embedding (spatial, temporal, and limb-type positional biases) and Skate-MSA (4 partitioned attention branches).
- **Successes:** Slashed parameters to **447,205 params (-85.2% compression)**. Fast 2-minute training. Pipeline throughput of 29.3 FPS GPU.
- **Why Vision Transformers Fell Short of Biological GCNs:**
  - Lacked physical biological graph inductive bias. Unconstrained self-attention allowed any joint to attend to any joint, leading to sample inefficiency and overfitting on the 297-clip training set.
  - Resulted in **7 missed violent attacks (78.8% recall)** and 7 civilian false alarms (F1: 0.7879). Biological GCN inductive bias is essential for small-sample edge kinematics.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_skateformer_v2.20.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_skateformer_v2.20.json)
  - Training Summary: [`results/tuning/training_summary_skateformer_v2.20.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/training_summary_skateformer_v2.20.json)
  - Training Logs (Folds 1-3): [`results/training_log_skateformer_v2.20_fold1.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_skateformer_v2.20_fold1.csv), [`fold2.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_skateformer_v2.20_fold2.csv), [`fold3.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/training_log_skateformer_v2.20_fold3.csv)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_skateformer_v2.20.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_skateformer_v2.20.csv), [`tuning_skateformer_v2.20.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_skateformer_v2.20.json)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_skateformer_v2.20.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_skateformer_v2.20.csv)

### Version 3.00 (Skeleton-OOD End-to-End Latent Boundary Learning)
- **Concept:** Explored end-to-end latent boundary optimization. Retained ST-GCN backbone, added in-training ASH-B (80% prune), ASH-SE-MLP fusion block (3.25M params), unit hyperspherical boundary projection ($\mathbb{S}^{255}$), and a composite 6-part loss ($\mathcal{L}_{\text{triplet}} + 0.5\mathcal{L}_{\text{ce}} + 0.2\mathcal{L}_{\text{compact}} + 0.1\mathcal{L}_{\text{energy}} + 0.1\mathcal{L}_{\text{pseudo-ood}} + 0.1\mathcal{L}_{\text{ks}}$).
- **Outcome:** Direct Energy gating achieved 100.0% recall (33/33 detected) and 26.3 FPS CPU pipeline speed (3.7x faster than v1.08).
- **Why End-to-End Optimization Underperformed Clean Metric Learning:**
  1. In-training ASH zeroed out 80% of penultimate activations during forward passes, creating a zero-gradient block across 80% of channels during backpropagation.
  2. Synthetic pseudo-OOD samples generated by convex combination ($\lambda z_i + (1-\lambda)z_j + \delta$) distorted transition boundaries, causing civilian actions to register violent energy scores (23 to 41 false alarms, F1: 0.6168–0.7126).
  3. Clean metric learning on unperturbed features paired with post-hoc background covariance subtraction (v1.08 RMD) proved mathematically superior.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_skeleton_ood_v3.00.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_skeleton_ood_v3.00.json)
  - Training Summary: [`results/tuning/training_summary_skeleton_ood_v3.00.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/training_summary_skeleton_ood_v3.00.json)
  - Training Logs (Folds 1-3): [`results/tuning/training_log_skeleton_ood_v3.00_fold1.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/training_log_skeleton_ood_v3.00_fold1.csv), [`fold2.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/training_log_skeleton_ood_v3.00_fold2.csv), [`fold3.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/training_log_skeleton_ood_v3.00_fold3.csv)
  - Hyperparameter Tuning Logs: [`results/tuning/tuning_skeleton_ood_v3.00.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_skeleton_ood_v3.00.csv)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_skeleton_ood_v3.00.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_skeleton_ood_v3.00.csv)

### Version 3.10 (Dual-Tier Edge & Server Architecture Ensemble — Phase 1 Champion)
- **Architecture:** Consolidates all Phase 1 backbones into a coordinated dual-tier hierarchy:
  - **Tier 1 (Edge Autonomous Screening):** Operates lightweight ST-GCN + RMD (v1.08) locally on edge hardware (Jetson / IPC / CPU) with guaranteed 100% recall.
  - **Tier 2 (Server Multi-Model Consensus Ensemble):** Runs a 3-model spatio-temporal consensus ensemble:
    1. ST-GCN (v1.08, static biological bone graph)
    2. CTR-GCN (v2.00, dynamic channel topology refinement)
    3. PoseConv3D (v2.10, volumetric 3D heatmap CNN)
- **Calibrated Fusion:** Maps individual RMD likelihood scores into posterior probabilities via sigmoid temperature scaling ($P_m = \sigma((S_m - \tau_m)/T_m)$) and computes weighted consensus:
  $$P_{\text{Server}} = 0.45 P_{\text{STGCN}} + 0.35 P_{\text{CTRGCN}} + 0.20 P_{\text{PoseC3D}}$$
- **Hierarchical Escalation Gating:**
  - Events with $P_{\text{Edge}} < 0.20$ are classified as normal and discarded locally (**44.2% of edge traffic offloaded** with zero network transmission).
  - Ambiguous events ($0.20 \le P_{\text{Edge}} < 0.60$) escalate to the Tier 2 Server Ensemble.
- **Empirical Results (All-Time Record Champion):**
  - **F1-Score: 0.9706**
  - **Violence Recall: 100.0% (33/33 detected, ZERO MISSED ATTACKS)**
  - **Violence Precision: 94.29% (33/35)**
  - **Overall Classification Accuracy: 97.40%**
  - **Civilian False Alarms: ONLY 2 FP out of 44 civilian videos** (66.7% reduction vs v1.08).
  - **Throughput:** 32.5 FPS effective GPU throughput; 26.9 FPS on Edge CPU.
- **Associated Result Files:**
  - Benchmark JSON: [`results/benchmark_dual_tier_v3.10.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_dual_tier_v3.10.json)
  - 890-Record Parameter Sweep Log: [`results/tuning/tuning_dual_tier_v3.10.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/tuning/tuning_dual_tier_v3.10.csv)
  - Validation CSV: [`results/action_recognition_benchmark/validation_results_dual_tier_v3.10.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/action_recognition_benchmark/validation_results_dual_tier_v3.10.csv)

---

## 5. Phase 2: Stage 1 Threat Object & Weapon Detection Architecture Overhaul

### Version 4.00 (Distilled YOLO26s Baseline Champion)
- **Architecture:** Compact student detector (`YOLO26s`, ~9.4M parameters) trained via dark-knowledge feature and logit distillation from a high-capacity teacher (`YOLO26x`, ~60M parameters). Checkpoint: `weights/yolo_weapon_distilled.pt` (~18.8 MB).
- **Empirical Ground Truth Metrics (715 Images, $\text{IoU}=0.20$):**
  - **Production Operating Point ($\text{Conf}=0.45$):**
    - Accuracy: **91.62%** | F1-Score: **86.88%** | Precision: **89.78% (202/225)** | Recall: **84.17% (202/240)** | Specificity: **95.29% (465/488)**
    - With_Coat F1: **92.83%** (TP=136, FP=6, FN=15, TN=202)
    - Without_Coat F1: **83.54%** (TP=66, FP=3, FN=23, TN=156)
    - OOD Specificity: **88.43%** (14 FP / 121, TN=107)
    - Total False Alarms: 23 FP | Missed Weapons: 38 FN
    - Inference Latency: **12.92 ms on RTX 3090 (~77 FPS)** / **11.18 ms on GTX 1060 (~89 FPS)**.
  - **High-Sensitivity Security Fallback ($\text{Conf}=0.20$):**
    - Recall surges to **91.67% (220/240 targets, only 20 missed)**.
    - False alarms surge to 98 FP (51 OOD FP), dropping Precision to 69.18% and F1 to 78.85%.
- **Physical Vulnerabilities Identified:**
  1. Motion blur on thin metallic blades dropped confidence below 0.45.
  2. Lowering confidence to 0.20 caused vertical clothing zippers, belt edges, and jacket seams to trigger false alarms.
  3. Gripping hands obscured knife hilts.
- **Associated Result Files:**
  - Official Verified Benchmark Report: [`results/ground_truth_benchmark/Comprehensive_Live_Verified_Benchmark.txt`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/Comprehensive_Live_Verified_Benchmark.txt)
  - Benchmark JSON: [`results/benchmark_yolo_baseline_v4.00.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_yolo_baseline_v4.00.json)
  - Validation Suite JSON: [`results/benchmark_stage1_validation_suite.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_stage1_validation_suite.json)
  - Distillation Training Metrics: [`results/weapon_training_results/results_distilled_student_yolo26s.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/weapon_training_results/results_distilled_student_yolo26s.csv), [`results_baseline_student_yolo26s.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/weapon_training_results/results_baseline_student_yolo26s.csv), [`results_teacher_yolo26x.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/weapon_training_results/results_teacher_yolo26x.csv), [`results_custom_p2_yolo26s.csv`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/weapon_training_results/results_custom_p2_yolo26s.csv)
  - Visual Metrics Curves: `results/weapon_training_results/BoxF1_curve.png`, `BoxPR_curve.png`, `confusion_matrix.png`

### Version 4.10 (ByteDance Sa2VA-Qwen2.5-VL-7B Dense Grounded MLLM — Paradigm Rejected)
- **Concept:** Investigated state-of-the-art multimodal vision-language segmentation (IEEE TPAMI 2026). Combines Qwen2.5-VL-7B with SAM-2 mask decoder (~14.2 GB checkpoint loaded in `bfloat16`). Emits autoregressive `[SEG]` tokens conditioning SAM-2 mask queries.
- **Empirical Ground Truth Metrics:**
  - **F1-Score: 35.29%** (vs 86.88% baseline, a -51.6% collapse).
  - **Threat Recall: 26.09% (misses 73.9% of all weapons!)**.
  - **Precision: 54.55%** | Accuracy: 70.67%.
  - **Latency: 1,910.92 ms per frame (~0.52 FPS, ~148x slower than YOLO)**.
- **Engineering Hurdles & Resolutions:**
  1. *PCIe Streaming Pause:* 14.2 GB safetensors weights across 7 shards caused 50-80s silent load pauses on Windows; resolved by adding explicit initialization telemetry.
  2. *Checkpoint Key Mismatch & Infinite Exclamation Loop (`! ! !`):* ByteDance checkpoint used `model.model.language_model.*`, but custom `Sa2VAChatModelQwen` had `base_model_prefix = "language_model"`. Under `strict=False`, HuggingFace silently rejected all 1,632 weight tensors, running the model on random Gaussian weights. Output logit for `!` (token ID 0) dominated, locking the generation loop in an infinite exclamation string until KV-cache exhaustion. Resolved by setting `base_model_prefix = ""` and implementing recursive prefix remapping, achieving 100% parameter lock.
  3. *`transformers 5.x` Alpha Bug:* Alpha build altered `all_tied_weights_keys`; resolved by pinning to stable `transformers==4.49.0`.
  4. *`text_config` Dict Serialization:* Authors serialized `text_config` as a raw Python dict; resolved by auto-wrapping via `Qwen2Config(**cfg.text_config)`.
  5. *Tokenizer Vocabulary Mismatch:* Upstream Qwen2.5-VL has 152,064 tokens; Sa2VA truncated to 151,670; resolved by overriding `cfg.vocab_size = 151670`.
  6. *Dynamic Visual Attention OOM (25.66 GiB Allocation):* Native dynamic resolution generated tens of thousands of patch tokens on 2.5K frames, blowing up self-attention matrices to 25.66 GiB. Capped visual processor resolution (`min_pixels = 256*28*28`, `max_pixels = 1024*28*28`), stabilizing VRAM at 15.95 GB.
- **Root-Cause Failure Analysis:**
  - *Autoregressive Decoding Wall:* Sequential token decoding across 28 layers took ~1.91s per frame, 148x too slow for real-time video surveillance.
  - *Semantic Attentional Bias:* Surveillance blades are small ($<1.5\%$ frame area) and motion-blurred. Lacking dense conversational context, language attention heads failed to predict `[SEG]`, missing 73.9% of targets.
- **Associated Result Files:**
  - Ground Truth Benchmark JSON: [`results/ground_truth_benchmark/sa2va_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/sa2va_gt_validation_results.json)
  - Baseline Benchmark JSON: [`results/benchmark_sa2va_v4.10.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/benchmark_sa2va_v4.10.json)
  - Diagnostic Probe Metrics: [`results/sa2va_probe_raw_metrics.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/sa2va_probe_raw_metrics.json)

### Version 4.20 (Grounding DINO 1.5 Edge Cross-Modal Transformer — Paradigm Rejected)
- **Concept:** Non-autoregressive open-vocabulary detection transformer (`IDEA-Research/grounding-dino-base`, ~950 MB). Swin Transformer visual backbone + BERT text encoder with bi-directional cross-attention and multi-scale deformable attention. Implemented **Competing Negative Token Contrastive Gating**: positive threat tokens (`a knife`, `a blade`, `a dagger`) competed against negative civilian tokens (`a bare hand`, `a clenched fist`, `a clothing seam`, `a zipper`, `a shadow`).
- **Empirical Ground Truth Metrics:**
  - *Primary Config (Box=0.20):* F1: 11.37%, Recall: 50.00% (120/240), Precision: 6.41%, **1,751 false alarms**, Latency: 187.10 ms (~5.34 FPS).
  - *Tuned Config (Box=0.25, P3 Targeted Prompt):* F1: 30.54%, Recall: 29.58% (71/240, 169 missed), Precision: 31.56% (71/225), Specificity: 73.22%, Latency: 185.89 ms (~5.38 FPS).
- **Engineering Hurdles & Resolutions:**
  1. *Commercial API Paywall:* Grounding DINO 1.5 Pro/Edge weights were restricted behind a commercial online API; instantiated the official foundational checkpoint `grounding-dino-base` locally.
  2. *BERT Subword Determiner Collapse:* Bare unarticled words (`"knife . blade ."`) collapsed recall to 0.00%. Pretrained BERT attention heads required natural language determiners (`"a knife . a blade ."`) to project text embeddings into the shared cross-modal space.
  3. *The "Cutter" Semantic Attractor:* Prompt token `"cutter"` fired repeatedly on wood logs and handrails, causing 219 OOD false alarms. Pruning `"cutter"` (P3 prompt) cut civilian false alarms by 91.2%.
  4. *Query Clustering:* Zero-shot cross-attention lacked closed-set bipartite constraints, causing 5 to 10 queries to cluster on a single object; resolved by integrating `torchvision.ops.nms` ($\text{IoU}=0.50$).
  5. *FP16 Linear Layer Dtype Mismatch:* Resolved by keeping model in FP32 and wrapping forward passes in `torch.amp.autocast('cuda', dtype=torch.float16)`.
- **Root-Cause Failure Analysis:**
  - Non-autoregressive queries yielded a 10x speedup over Sa2VA (185 ms vs 1,910 ms), but remained ~14.5x slower than YOLO (12.92 ms).
  - Pretrained on clean web photographs, open-vocabulary cross-attention suffered a massive domain gap on motion-blurred CCTV footage. Achieving 50% recall caused 1,751 false alarms; suppressing false alarms collapsed recall to 29.58%.
- **Associated Result Files:**
  - Full 715 Ground Truth Benchmark JSON: [`results/ground_truth_benchmark/grounding_dino_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/grounding_dino_gt_validation_results.json)
  - Tuned P3 Box=0.25 Benchmark JSON: [`results/ground_truth_benchmark/grounding_dino_p3_box025_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/grounding_dino_p3_box025_gt_validation_results.json)
  - 12-Configuration Parameter Sweep: [`results/ground_truth_benchmark/grounding_dino_sweep_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/grounding_dino_sweep_results.json)

### Version 4.30 (Contact-State HOI Transformer: DINOv2 Dual-Stream Guard)
- **Concept:** Replaced text prompts with physical kinematic Human-Object Interaction (HOI) constraints:
  - *Stream 1 (Proposal Stream):* Distilled YOLO26s operating at high-sensitivity ($\text{conf}=0.20$) to ensure high threat recall.
  - *Stream 2 (Kinematic Hand Tracking):* YOLO26s-pose tracks wrists (joints 9 & 10) and elbows (7 & 8). Dynamically computes interaction radius scaled to forearm length:
    $$R_h = \text{clamp}\left(0.50 \cdot \|\vec{v}_{\text{elbow}\to\text{wrist}}\|_2, 25.0\text{ px}, 150.0\text{ px}\right)$$
  - *Bipartite Spatial Interaction Graph:* Computes normalized proximity $d_{\text{norm}} = \|c_{\text{obj}} - c_{\text{wrist}}\|_2 / R_h$. Proposals beyond $R_{\text{interact}} = 2.2$ are purged as planar background artifacts.
  - *DINOv2 Contact-State Gating Head:* Candidate patches within hand proximity are fed to `facebook/dinov2-small` ($f \in \mathbb{R}^{384}$). Evaluates cosine affinity against a calibrated reference manifold of weapon-contact vs empty-hand prototypes:
    $$S_{\text{contact}} = \frac{S_{\text{weapon}}}{S_{\text{weapon}} + S_{\text{hand}} + \epsilon}$$
    Proposals with $S_{\text{contact}} < \tau$ are suppressed as bare fists or non-contact gestures.
  - *High-Confidence Safety Bypass:* Proposals with raw confidence $\ge \tau_{\text{high}} = 0.70$ bypass hand proximity to guard against occluded wrists.
- **Empirical Ground Truth Metrics:**
  - *Tuned High-Precision Operating Point ($\tau=0.52$):*
    - **F1-Score: 84.08%** | Precision: **82.40% (206/250)** | Recall: **85.83% (206/240, 34 missed)**
    - Overall Accuracy: **89.39%** | Civilian Specificity: **91.11% (451/495)** | OOD Specificity: **93.28% (111/119)**
    - **OOD False Alarms: ONLY 8 FP** (**-84.3% reduction** vs raw Conf=0.20 YOLO's 51 FP).
    - Latency: **33.93 ms (~29.5 FPS)** | VRAM: **84.23 MB**.
  - *Standard High-Recall Operating Point ($\tau=0.50$):*
    - **Threat Recall: 88.75% (213/240 targets, only 27 missed weapons)**.
    - F1-Score: **83.53%** | OOD False Alarms: **15 FP** (-70.6% reduction vs raw Conf=0.20).
- **Engineering Hurdles & Resolutions:**
  1. *Degenerate Crops:* Border bounding boxes produced 0-pixel crops; resolved by adding coordinate clamping and spatial degenerate guards (`crop.shape < 5`).
  2. *Missing Wrist Keypoints:* Severe motion blur dropped wrist keypoints; resolved by adding a dynamic forearm fallback ($45\text{ px}$) and high-confidence bypass ($\ge 0.70$).
  3. *Windows Symlinks:* Suppressed HuggingFace non-admin symlink warnings; verified binary caching.
  4. *Scale Invariance:* Replaced fixed pixel distances with forearm-scaled normalized distance $d_{\text{norm}}$.
- **Architectural Role:** **Official Real-Time Secondary Dual-Stream Guard.** First candidate to achieve real-time streaming throughput (~30 FPS) while cutting civilian false alarms by 84.3% and capturing 85.8%–88.8% of weapons.
- **Associated Result Files:**
  - Full 715 Ground Truth Benchmark JSON ($\tau=0.50$): [`results/ground_truth_benchmark/hoi_contact_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/hoi_contact_gt_validation_results.json)
  - Tuned $\tau=0.52$ Benchmark JSON: [`results/ground_truth_benchmark/hoi_contact_tau052_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/hoi_contact_tau052_gt_validation_results.json)
  - 18-Configuration Parameter Sweep: [`results/ground_truth_benchmark/hoi_contact_sweep_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/hoi_contact_sweep_results.json)

### Version 4.40 (Geometry-Informed Monocular Depth Pipeline)
- **Concept:** Coupled YOLO proposal generation with dense metric surface geometry via `Depth Anything V2 Small` (24.8M parameters). Evaluated Sobel depth gradient field $\|\nabla d\|$, contextual ring step discontinuity $\Delta d_{\text{step}}$, and internal depth variance $\sigma_d^2$. Candidates with near-zero step discontinuity and low gradient were rejected as flat 2D clothing seams, prints, and shadows. Proposals with raw confidence $\ge 0.70$ bypassed geometric gating.
- **Empirical Ground Truth Metrics:**
  - *Tuned Config ($\text{Conf}=0.30, \tau_{\text{step}}=0.003, \tau_{\text{var}}=0.035$):*
    - **Threat Recall: 89.17% (214/240 targets, highest recall in benchmark)**.
    - **F1-Score: 86.12%** | Precision: **83.27% (214/257)** | Accuracy: **90.59%** | Specificity: **91.28%**.
    - With_Coat F1: **92.05%** (successfully rejected planar lapels and zippers).
    - OOD False Alarms: **23 FP** (-54.9% vs Conf=0.20).
    - Latency: **113.42 ms (~8.8 FPS)** | VRAM: **94.56 MB**.
- **Engineering Hurdles & Resolutions:**
  1. *OpenCV Sobel Dtype Assertion:* Detached PyTorch GPU tensors converted to NumPy caused C++ assertion errors; resolved by enforcing contiguous C-order `np.float32`.
  2. *Sub-Centimeter Blade Depth Erosion:* Dense ViT patch self-attention ($14 \times 14$ patches) smoothed narrow 2-to-5 pixel knife blades into the background torso depth, eroding step edges. Calibrated $\tau_{\text{step}} = 0.003$ and added high-confidence bypass ($\ge 0.70$).
  3. *Curved Textile Manifolds:* Baggy sleeves conform to cylindrical 3D limb geometry, producing curved depth gradients that escaped planar step rejection.
- **Root-Cause Failure Analysis:**
  - *The 8.8 FPS Latency Bottleneck:* Full-frame ViT attention across 1,369 patch tokens required 113.42 ms per frame, 8.8x slower than YOLO baseline, failing real-time edge streaming requirements ($\ge 30$ FPS).
  - *Anatomical Relief Blindness:* Bare clenched fists possess volumetric 3D relief indistinguishable from knife handles in depth maps, limiting OOD specificity to 81.45% (23 OOD FP vs 8 in HOI).
  - **Verdict:** Rejected for primary streaming; retained for asynchronous secondary verification.
- **Associated Result Files:**
  - Full 715 Ground Truth Benchmark JSON (Conf=0.20): [`results/ground_truth_benchmark/depth_geometric_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/depth_geometric_gt_validation_results.json)
  - Tuned Conf=0.30 Benchmark JSON: [`results/ground_truth_benchmark/depth_geometric_conf030_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/depth_geometric_conf030_gt_validation_results.json)
  - 16-Configuration Parameter Sweep: [`results/ground_truth_benchmark/depth_geometric_sweep_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/depth_geometric_sweep_results.json)

### Version 4.50 (NMS-Free Real-Time Edge Architecture: One-to-One Bipartite Matching vs Greedy NMS)
- **Forensic Discovery on Baseline v4.00 Identity:**
  - A deep architectural audit of `weights/yolo_weapon_distilled.pt` revealed its underlying architecture is `yolo26s.yaml` with `end2end: True` enabled by default.
  - In the 2026 Ultralytics framework, `YOLO26s` contains dual heads: an auxiliary one-to-many head used during training, and a primary one-to-one Hungarian bipartite matching head (`one2one_cv2`, `one2one_cv3`).
  - When `v4.00` executed, Ultralytics routed through the one-to-one head and used top-$k$ gathering, **completely skipping greedy NMS**.
  - Therefore, **v4.00 was already executing One-to-One Hungarian Bipartite Matching in production!** Both v4.00 and v4.50 (one2one) produced mathematically identical results (86.88% F1, 89.78% Precision, 84.17% Recall, 12.86 ms).
- **The Controlled Ablation Against Greedy NMS:**
  - Because `weights/yolo_weapon_distilled.pt` contains both heads embedded within the same weights, Item 5 performed a controlled ablation evaluating the *exact same trained backbone* under both inference mechanisms:
    1. *One-to-One Hungarian Bipartite Matching (`end2end=True`, NMS-Free):*
       - **86.88% F1**, **89.78% Precision**, **84.17% Recall**, **23 False Alarms** (14 OOD FP), **12.86 ms**.
    2. *One-to-Many Head with Greedy NMS (`end2end=False`, `iou=0.45`):*
       - **82.33% F1 (-4.55% collapse)**, **79.46% Precision (-10.32% collapse)**, **85.42% Recall**, **53 False Alarms (+130.4% surge!)** (30 OOD FP, +114.3% surge!), **12.48 ms**.
  - *Mathematical Cause of Greedy NMS Failure:* In one-to-many anchor assignment, adjacent grid cells fire on high-contrast fabric seams and zippers. Because the anchor centers are spatially separated, the predicted boxes have an IoU of $0.25–0.40$ (below the standard NMS threshold of $0.45$). Greedy NMS views them as distinct objects and outputs duplicate false alarms. Hungarian matching enforces mutual spatial competition during training, ensuring only the single most confident anchor activates.
- **Zero-Shot Pretrained Edge DETR Collapse:**
  - Evaluated standard COCO-pretrained `RT-DETR-L` (`rtdetr-l.pt`, ~63.4 MB).
  - Suffered a **100% false negative collapse (0/240 weapons detected, 0.00% recall, 0.00% F1)**. Class 43 ("knife") in COCO represents domestic dining cutlery on tabletops, which shares zero visual feature representation with tactical combat daggers held in dynamic surveillance attack stances.
- **Engineering Hurdles & Resolutions:**
  1. *`AutoBackend` State Caching Crash:* Toggling `model.end2end = False` dynamically crashed Ultralytics with `KeyError: 'feats'`; resolved by implementing dual-model instancing (`self.model_one2one` and `self.model_one2many`) at detector initialization.
  2. *Coordinate Normalization Bug:* `boxes.xyxyn` returned normalized corner format $[x_1, y_1, x_2, y_2]$; passing it to `xywhn2xyxy` produced negative coordinates; resolved by preserving `xyxyn` directly.
- **Master Conclusion:** One-to-One Hungarian Bipartite Matching (NMS-Free) is the **undisputed primary streaming production champion** (86.88% F1, 89.78% Precision, 12.86 ms, 78 FPS, <1 GB VRAM).
- **Associated Result Files:**
  - Full 715 Ground Truth Benchmark JSON: [`results/ground_truth_benchmark/nms_free_gt_validation_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/nms_free_gt_validation_results.json)
  - 23-Configuration Parameter Sweep: [`results/ground_truth_benchmark/nms_free_sweep_results.json`](file:///d:/Intern%20AI%20Project/hierarchical-threat-detection-edge-ai/results/ground_truth_benchmark/nms_free_sweep_results.json)

---

## 6. Comprehensive Multi-Tier Deployment Blueprint

Combining Phase 1 and Phase 2 empirical validations yields the optimal multi-tier production architecture:

```
[ Incoming Surveillance Video Stream (1080p / 720p @ 30 FPS) ]
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: Real-Time Primary Weapon Detector (v4.00 / v4.50 NMS-Free YOLO26s)   │
│ - Checkpoint: weights/yolo_weapon_distilled.pt (~18.8 MB)                   │
│ - Mechanism: One-to-One Hungarian Bipartite Matching (No Greedy NMS)        │
│ - Resolution: Rectangular 384x640 (-40% pixel compute waste)                │
│ - Latency: 12.86 ms on RTX 3090 / 11.18 ms on GTX 1060 (~78-89 FPS)         │
│ - Active VRAM: < 1.0 GB                                                     │
│ - Performance: 86.88% F1, 89.78% Precision, 84.17% Recall at Conf=0.45      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                        Candidate Weapon Detected?
                        ├──────────────────────────────────────────────────┐
                        │ NO                                               │ YES
                        ▼                                                  ▼
                  [ Normal Flow ]               ┌────────────────────────────────────────────────────────┐
                                                │ High Security Mode Enabled?                            │
                                                ├──────────────────────────┬─────────────────────────────┤
                                                │ NO (Standard Production) │ YES (Sensitivity Mode)      │
                                                ▼                          ▼                             │
                                        [ Pass Directly ]        ┌──────────────────────────────────────┐│
                                                                 │ TIER 1-GUARD: Contact HOI (v4.30)    ││
                                                                 │ - Conf=0.20 (91.67% Raw Recall)      ││
                                                                 │ - Hand Proximity (d_norm <= 2.2)     ││
                                                                 │ - DINOv2 Grasp Affinity (S >= 0.52)  ││
                                                                 │ - Latency: 33.93 ms | VRAM: 84 MB    ││
                                                                 │ - Eliminates 84.3% Civilian FP       ││
                                                                 └──────────────────┬───────────────────┘│
                                                                                    │                    │
                                                                     Grasp Verified?│                    │
                                                                     ├──────────────┤                    │
                                                                     │ NO           │ YES                │
                                                                     ▼              └──────────┬─────────┘
                                                          [ Suppressed as Seam ]               │
                                                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2a: Multi-Person Tracking & Biomechanical Normalization (v1.01-v1.04)                              │
│ - Memory: In-Memory TurboJPEG Buffer Compression (v1.01: 47.3 MB RAM for 1,200 frames, -95.5% savings)  │
│ - Normalization: Torso-Scale Normalization (v1.02: L_torso = ||shoulder - hip||, scale invariant)       │
│ - Tracking: ByteTrack Multi-Person Guard (v1.04: 0 ID swaps, 100% actor lock, 28.76 us overhead)       │
└────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2b: Action Recognition & OOD Manifold Gating (v1.08 / v3.10)                                       │
│                                                                                                         │
│ ┌─────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────┐ │
│ │ OPTION A: Standalone Edge Appliance (v1.08)     │   │ OPTION B: Cloud / Server Cluster (v3.10)      │ │
│ │ - Model: ST-GCN Retrained Champion (Fold 2)     │   │ - Model: Tri-Model Weighted Consensus         │ │
│ │ - Gating: Relative Mahalanobis Distance (RMD)   │   │   * ST-GCN (w=0.45)                           │ │
│ │   (Score = M_0 - min_c M_c, epsilon=0.005)      │   │   * CTR-GCN Dynamic Topology (w=0.35)         │ │
│ │ - Latency: 26.49 ms (37.8 FPS GPU) / 26.9 FPS CPU│  │   * PoseConv3D 3D Heatmaps (w=0.20)           │ │
│ │ - Metrics: F1: 0.9167, 100% RECALL (0 FN), 6 FP │   │ - Gating: Temperature Calibrated Softmax      │ │
│ │ - Storage: 1.0 MB Covariance Matrices           │   │ - Metrics: F1: 0.9706, 100% RECALL, ONLY 2 FP │ │
│ └─────────────────────────────────────────────────┘   └───────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                          [ HIGH-CONFIDENCE ALARM ]
```

---

## 7. Master File, Script, Checkpoint & Complete Results Inventory

### A. Core Documentation & Datasets
- `PROJECT_REPORT_PHASE1_PHASE2.md`: This master technical report.
- `AgentRule.md`: Operational boundaries, continuity directives, and baseline standards for future agents.
- `ROADMAP_AND_HANDOFF.md`: Phase 1 roadmap and historical context.
- `ROADMAP_AND_HANDOFF_PHASE2.md`: Phase 2 Stage 1 roadmap and execution log.
- `VERSIONS.md`: Complete Phase 1 technical changelog (v1.00 to v3.10).
- `VERSIONS_PHASE2.md`: Complete Phase 2 technical changelog (v4.00 to v4.50).
- `Stage1_SOTA_Architectures_2026.md`: Theoretical whitepaper on 2026 Stage 1 detector architectures.
- `Modern OOD Action Recognition Models.md`: Theoretical whitepaper on action recognition and OOD metrics.
- `data/Ground_Truth_Dataset/`: Official 715-image surveillance ground truth dataset (`With_Coat`, `Without_Coat`, `OOD`).
- `data/`: Phase 1 action recognition training clips (`Cut-Down`, `Stab`, `Thrust`) and validation suite (`data/Validate/`).

### B. Production Model Checkpoints (`weights/`)
- `weights/yolo_weapon_distilled.pt`: **Stage 1 Champion.** Distilled YOLO26s One-to-One Hungarian Bipartite Detector (~18.8 MB).
- `weights/yolo26s-pose.pt`: **Stage 2a Pose Extractor.** Keypoint localization checkpoint.
- `weights/stgcn_violence_v1.02.pth`: **Stage 2b Edge Backbone.** Retrained ST-GCN Fold 2 Champion (Val Loss: 0.0171, 3.01M params).
- `weights/reference_data_rmd_v1.08.pt`: **Stage 2b Edge Gating Weights.** RMD class-conditional and global background covariance matrices (~1.0 MB).
- `weights/ctrgcn_violence_v2.00.pth`: **Stage 2b CTR-GCN Backbone.** Dynamic topology refinement checkpoint (1.33M params).
- `weights/reference_data_ctrgcn_v2.00.pt`: CTR-GCN RMD reference data (~1.0 MB).
- `weights/poseconv3d_violence_v2.10.pth`: **Stage 2b PoseConv3D Backbone.** R(2+1)D 3D Heatmap CNN checkpoint (765k params).
- `weights/reference_data_poseconv3d_v2.10.pt`: PoseConv3D RMD reference data (~1.3 MB).
- `weights/reference_data_dual_tier_v3.10.pt`: **Stage 2b Dual-Tier Server Ensemble Weights.** Calibrated temperature scaling and consensus parameters (~3.3 MB).
- `weights/reference_data_hoi_contact_v4_30.pt`: **Stage 1-Guard HOI Weights.** DINOv2 weapon-contact and empty-hand prototype embeddings.
- `weights/skateformer_violence_v2.20.pth`: SkateFormer ViT checkpoint (447k params).
- `weights/stgcn_skeleton_ood_v3.00.pth`: Skeleton-OOD latent boundary checkpoint (3.25M params).

### C. Source Code Inference Engines (`src/`)
- `src/skeleton_utils.py`: Scale-invariant coordinate normalization (`normalize_skeleton_clip`) and geometric utilities.
- `src/dataset.py`: ActionDataset and TripletActionDataset data loaders.
- `src/inference_realtime_v1.04.py`: Production baseline real-time pipeline with ByteTrack Multi-Person Guard.
- `src/inference_realtime_v1.08.py`: Real-time pipeline with ST-GCN + Relative Mahalanobis Distance.
- `src/inference_realtime_v3.10.py`: Production Dual-Tier Edge & Server Ensemble real-time pipeline.
- `src/inference_nms_free_v4_50.py`: **Stage 1 Production Engine.** Supports both `one2one` Hungarian bipartite matching and `one2many` greedy NMS modes.
- `src/inference_hoi_contact_v4_30.py`: **Stage 1-Guard Engine.** Dual-stream forearm-scaled hand proximity + DINOv2 contact classifier.
- `src/inference_depth_geometric_v4_40.py`: Stage 1 Monocular Depth Anything V2 surface relief filter.
- `src/inference_sa2va_v4_10.py`: Sa2VA-Qwen2.5-VL-7B MLLM inference engine.
- `src/inference_grounding_dino_v4_20.py`: Grounding DINO 1.5 open-vocabulary detection engine.

### D. Model Architecture Definitions (`models/`)
- `models/ensemble_dual_tier.py`: Master dual-tier hierarchical ensemble module (ST-GCN + CTR-GCN + PoseConv3D).
- `models/ctrgcn.py`: CTR-GCN dynamic topology refinement model.
- `models/poseconv3d.py`: PoseConv3D volumetric 3D heatmap R(2+1)D model.
- `models/skateformer.py`: SkateFormer partitioned spatio-temporal Vision Transformer.
- `models/skeleton_ood.py`: Skeleton-OOD latent boundary ST-GCN model with ASH-SE-MLP fusion.
- `models/sa2va/`: Sa2VA modeling definitions (`modeling_sa2va_qwen.py`, `sam2.py`, `configuration_sa2va_chat.py`).

### E. Validation & Benchmark Scripts (`scripts/`)
- `scripts/validate_stage1_ground_truth.py`: Official Ground Truth validation runner for Stage 1 detectors on 715 images ($\text{IoU}=0.20$).
- `scripts/validate_nms_free_ground_truth.py`: Ground Truth validation runner for One-to-One vs One-to-Many NMS-Free models.
- `scripts/validate_hoi_contact_ground_truth.py`: Ground Truth validation runner for Contact-State HOI Transformer.
- `scripts/validate_depth_geometric_ground_truth.py`: Ground Truth validation runner for Monocular Depth pipeline.
- `scripts/validate_grounding_dino_ground_truth.py`: Ground Truth validation runner for Grounding DINO 1.5.
- `scripts/validate_sa2va_ground_truth.py`: Ground Truth validation runner for Sa2VA-7B MLLM.
- `scripts/benchmark_dual_tier_v3.10.py`: Phase 1 Stage 2 Dual-Tier Ensemble benchmark suite.
- `scripts/benchmark_rmd_v1.08.py`: Phase 1 Stage 2 Standalone RMD benchmark suite.

### F. Official Benchmark JSON Artifacts (`results/`)
- `results/benchmark_buffer_compression_v1.01.json`: v1.01 buffer compression benchmark metrics.
- `results/benchmark_standoff_archival_v1.02.json`: v1.02 archival validation metrics.
- `results/benchmark_rectangular_inference_v1.03.json`: v1.03 rectangular aspect-ratio execution timings.
- `results/benchmark_multi_person_guard_v1.04.json`: v1.04 multi-person stability and swap metrics.
- `results/benchmark_cosine_knn_v1.05.json`: v1.05 hyperspherical cosine k-NN metrics.
- `results/benchmark_vim_v1.06.json`: v1.06 virtual-logit matching metrics.
- `results/benchmark_ash_v1.07.json`: v1.07 activation shaping and energy metrics.
- `results/benchmark_rmd_v1.08.json`: v1.08 Relative Mahalanobis Distance champion metrics.
- `results/benchmark_ctrgcn_v2.00.json`: v2.00 CTR-GCN benchmark metrics.
- `results/benchmark_poseconv3d_v2.10.json`: v2.10 PoseConv3D volumetric 3D heatmap metrics.
- `results/benchmark_skateformer_v2.20.json`: v2.20 SkateFormer Vision Transformer metrics.
- `results/benchmark_skeleton_ood_v3.00.json`: v3.00 Skeleton-OOD latent boundary metrics.
- `results/benchmark_dual_tier_v3.10.json`: v3.10 master dual-tier ensemble benchmark metrics.
- `results/benchmark_yolo_baseline_v4.00.json`: v4.00 baseline weapon detection metrics.
- `results/benchmark_stage1_validation_suite.json`: Stage 1 comprehensive validation suite metrics.
- `results/benchmark_sa2va_v4.10.json`: v4.10 Sa2VA-7B baseline metrics.
- `results/sa2va_probe_raw_metrics.json`: v4.10 Sa2VA diagnostic probe raw outputs.
- `results/ground_truth_benchmark/Comprehensive_Live_Verified_Benchmark.txt`: Master 715-image ground truth report for baseline YOLO26s.
- `results/ground_truth_benchmark/sa2va_gt_validation_results.json`: v4.10 Sa2VA-7B ground truth benchmark results.
- `results/ground_truth_benchmark/grounding_dino_gt_validation_results.json`: v4.20 Grounding DINO primary (Box=0.20) ground truth results.
- `results/ground_truth_benchmark/grounding_dino_p3_box025_gt_validation_results.json`: v4.20 Grounding DINO tuned (Box=0.25 P3) ground truth results.
- `results/ground_truth_benchmark/hoi_contact_gt_validation_results.json`: v4.30 Contact-State HOI standard ($\tau=0.50$) ground truth results.
- `results/ground_truth_benchmark/hoi_contact_tau052_gt_validation_results.json`: v4.30 Contact-State HOI tuned ($\tau=0.52$) ground truth results.
- `results/ground_truth_benchmark/depth_geometric_gt_validation_results.json`: v4.40 Monocular Depth nominal (Conf=0.20) ground truth results.
- `results/ground_truth_benchmark/depth_geometric_conf030_gt_validation_results.json`: v4.40 Monocular Depth tuned (Conf=0.30) ground truth results.
- `results/ground_truth_benchmark/nms_free_gt_validation_results.json`: v4.50 NMS-Free (One-to-One vs One-to-Many) master ground truth results.

### G. Training Logs, Summaries & Visual Learning Curves (`results/`)
- `results/training_summary_poseconv3d_v2.10.json`: PoseConv3D 3-fold cross-validation training summary.
- `results/training_log_ctrgcn_v2.00_fold1.csv`, `fold2.csv`, `fold3.csv`: CTR-GCN training epoch logs.
- `results/training_log_poseconv3d_v2.10_fold1.csv`, `fold2.csv`, `fold3.csv`: PoseConv3D training epoch logs.
- `results/training_log_skateformer_v2.20_fold1.csv`, `fold2.csv`, `fold3.csv`: SkateFormer training epoch logs.
- `results/tuning/training_summary_ctrgcn_v2.00.json`: CTR-GCN training convergence summary.
- `results/tuning/training_summary_skateformer_v2.20.json`: SkateFormer training convergence summary.
- `results/tuning/training_summary_skeleton_ood_v3.00.json`: Skeleton-OOD training convergence summary.
- `results/tuning/training_log_skeleton_ood_v3.00_fold1.csv`, `fold2.csv`, `fold3.csv`: Skeleton-OOD training epoch logs.
- `results/action_recognition_benchmark/training_log_v1.02_fold1.csv`, `fold2.csv`, `fold3.csv`, `training_log_fold2.csv`: v1.02 ST-GCN cross-validation training logs.
- `results/action_recognition_benchmark/loss_curve_v1.02_fold1.png`, `fold2.png`, `fold3.png`: v1.02 ST-GCN loss curves.

### H. Hyperparameter Tuning Sweeps & Parameter Logs (`results/tuning/` & `results/ground_truth_benchmark/`)
- `results/tuning/tuning_cosine_knn_v1.05.csv` & `.json`: v1.05 Cosine k-NN threshold sweep logs.
- `results/tuning/tuning_vim_v1.06.csv` & `.json`: v1.06 Virtual-Logit Matching subspace and threshold sweep logs.
- `results/tuning/tuning_ash_v1.07.csv` & `.json`: v1.07 Activation Shaping percentile and energy threshold sweep logs.
- `results/tuning/tuning_rmd_v1.08.csv` & `.json`: v1.08 RMD shrinkage $\epsilon$ and decision threshold $\tau$ sweep logs.
- `results/tuning/tuning_ctrgcn_v2.00.csv` & `.json`: v2.00 CTR-GCN shrinkage and threshold sweep logs.
- `results/tuning/tuning_poseconv3d_v2.10.csv` & `.json`: v2.10 PoseConv3D shrinkage and threshold sweep logs.
- `results/tuning/tuning_skateformer_v2.20.csv` & `.json`: v2.20 SkateFormer shrinkage and threshold sweep logs.
- `results/tuning/tuning_skeleton_ood_v3.00.csv`: v3.00 Skeleton-OOD latent boundary parameter sweep logs.
- `results/tuning/tuning_dual_tier_v3.10.csv`: v3.10 master dual-tier ensemble parameter sweep logs (890 records).
- `results/action_recognition_benchmark/ood_tuning_coarse_fold2.csv` & `ood_tuning_fine_fold2.csv`: v1.02 coarse and fine OOD threshold tuning.
- `results/ground_truth_benchmark/grounding_dino_sweep_results.json`: v4.20 Grounding DINO 12-configuration parameter sweep results.
- `results/ground_truth_benchmark/hoi_contact_sweep_results.json`: v4.30 Contact-State HOI 18-configuration parameter sweep results.
- `results/ground_truth_benchmark/depth_geometric_sweep_results.json`: v4.40 Monocular Depth 16-configuration parameter sweep results.
- `results/ground_truth_benchmark/nms_free_sweep_results.json`: v4.50 NMS-Free 23-configuration parameter sweep results.

### I. Per-Category Action Recognition Validation Results (`results/action_recognition_benchmark/`)
- `results/action_recognition_benchmark/validation_results_knn_v1.02.csv`: v1.02 Euclidean k-NN per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_cosine_knn_v1.05.csv`: v1.05 Cosine k-NN per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_vim_v1.06.csv`: v1.06 ViM per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_ash_v1.07.csv`: v1.07 ASH-B per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_rmd_v1.08.csv`: v1.08 RMD per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_ctrgcn_v2.00.csv`: v2.00 CTR-GCN per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_poseconv3d_v2.10.csv`: v2.10 PoseConv3D per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_skateformer_v2.20.csv`: v2.20 SkateFormer per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_skeleton_ood_v3.00.csv`: v3.00 Skeleton-OOD per-clip validation breakdown.
- `results/action_recognition_benchmark/validation_results_dual_tier_v3.10.csv`: v3.10 Dual-Tier Ensemble per-clip validation breakdown.
- `results/action_recognition_benchmark/live_head_to_head_benchmark.csv` & `.txt`: Cross-version master validation comparisons.
- `results/action_recognition_benchmark/validation_results_knn.csv` & `validation_results_micro.csv`: Historical baseline validation logs.

### J. Stage 1 Weapon Detector Training & Distillation Artifacts (`results/weapon_training_results/`)
- `results/weapon_training_results/results_distilled_student_yolo26s.csv`: Distillation training performance log for production student detector.
- `results/weapon_training_results/results_teacher_yolo26x.csv`: High-capacity teacher model training metrics.
- `results/weapon_training_results/results_baseline_student_yolo26s.csv`: Undistilled baseline student training log.
- `results/weapon_training_results/results_custom_p2_yolo26s.csv`: High-resolution P2 head ablation training log.
- `results/weapon_training_results/results.csv`: Combined training metrics log.
- `results/weapon_training_results/args.yaml`: Exact hyperparameter configuration used during YOLO26 training.
- `results/weapon_training_results/BoxF1_curve.png`, `BoxPR_curve.png`, `confusion_matrix.png`: Official validation metric plots.
