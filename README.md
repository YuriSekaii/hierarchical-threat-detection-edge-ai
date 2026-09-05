# Hierarchical Edge-AI Threat & Violence Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-green.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, resource-efficient dual-stage surveillance system engineered for real-time edge deployment. Combines **YOLO Knowledge Distillation** with **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** and **Out-of-Distribution (OOD) Manifold Gating** to detect violent behavior on live video streams with minimal compute overhead.

---

## 🏛️ System Architecture

Traditional surveillance pipelines run continuous deep neural networks on every frame, resulting in high thermal dissipation and compute exhaustion on edge devices. 

This project solves this by decoupling detection into a **Hierarchical Gating Architecture**:

![Inference Pipeline](assets/inference_pipeline.png)

### 1. Stage 1: Continuous Lightweight Scanning (Always-On Loop)
* **Ingestion:** Ingests live video at 30 FPS into a rolling 60-frame thread-safe buffer.
* **Low-Power Gating:** Subsamples **1 frame per 10 (effective rate: 3 FPS)** for weapon detection using a compact, distilled YOLO student model.
* **Compute Savings:** Reduces continuous idle inference workloads by **~90%**, reserving GPU/NPU compute until a weapon is verified (`Conf > 0.45`).

### 2. Stage 2: On-Demand Biomechanical Action Analysis (Triggered Engine)
* **Kinematic Extraction:** Upon Stage 1 trigger, the rolling 60-frame video buffer is queried to extract a 50-frame kinematic analysis window. YOLO-Pose extracts 17 COCO skeletal joint coordinates.
* **Kinematic Preprocessing:** Implements **1D Gaussian temporal smoothing** ($\sigma = 1.0$) and linear interpolation to resolve dropped joints, followed by **centroid normalization** to achieve scale and position invariance.
* **ST-GCN Feature Extraction:** Feeds normalized skeleton graphs through a **9-block Spatial-Temporal Graph Convolutional Network** with joint-weighted spatial attention (5× weight on arm joints).
* **OOD Distance Gating:** Measures deep feature embeddings against a baseline threat manifold using **Deep k-NN distance**, filtering out passive actions (e.g., calmly holding an everyday object) from aggressive striking motions.

---

## 📊 Model Compression & Knowledge Distillation Benchmark

To detect small weapons reliably on edge devices without the computational overhead and latency of heavy models (`YOLO26x`), Knowledge Distillation was utilized to distill feature representations into a lightweight, real-time `YOLO26s` student.

### 1. Ground Truth Head-to-Head Surveillance Benchmark

Evaluated across 715 annotated test frames containing challenging edge cases (OOD civilian items, concealed weapons under coats, and direct attacks).

#### A. Calibrated Production Operating Point (`Conf = 0.45`, `IoU = 0.20`)
*Source: `results/ground_truth_benchmark/Detailed_GT_Validation_2026-01-27_02-59-15.txt` & `Comprehensive_Live_Verified_Benchmark.txt`*

| Metric | Teacher (`YOLO26x`) | Distilled Student (`YOLO26s`) [Production] | Best Custom P2 Head (Attempt 21) | Production Decision Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 82.33% | **91.62%** | 88.23% | **Student leads (+9.29% vs. Teacher, +3.39% vs. P2)** |
| **F1-Score** | 74.35% | **86.88%** | 80.64% | **Student leads (+12.53% vs. Teacher, +6.24% vs. P2)** |
| **Precision** | 71.10% | **89.78%** | 88.94% | **Student leads (+18.68% vs. Teacher, +0.84% vs. P2)** |
| **Weapon Recall (Threat Safety)** | 77.92% (187/240) | **84.17% (202/240)** | 73.75% (177/240) | **Student detects 25 more weapons than P2** |
| **Specificity (OOD)** | 84.49% | **95.29%** | 95.44% | **High non-threat civilian suppression (>95%)** |
| **Inference Latency** | 18.83 ms (~53 FPS) | **11.18 ms (~89 FPS)** | 19.34 ms (~51 FPS) | **Student is 42% faster than P2 and 41% faster than Teacher** |
| **Deployment Verdict** | Too Heavy for Edge | **Deployed Champion Model** | **Rolled Back (Degraded Recall)** | P2 missed 25 weapons and suffered +73% higher latency. |

#### B. High-Sensitivity Operating Point (`Conf = 0.20`, `IoU = 0.20`)
*Evaluates maximum sensitivity when screening for potential threats under heavy occlusion:*

| Model Architecture | Accuracy | Precision | Recall | Specificity | F1-Score | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Distilled Student (`YOLO26s`)** | 84.27% | 69.18% | **91.67% (220/240)** | 80.78% | 78.85% | **13.91 ms (~72 FPS)** |
| **Best Custom P2 Head (Attempt 21)** | 86.87% | 76.78% | 85.42% (205/240) | 87.58% | 80.87% | 19.01 ms (~53 FPS) |
| **Teacher (`YOLO26x`)** | 77.43% | 59.50% | 88.75% (213/240) | 72.22% | 71.24% | 77.70 ms (~13 FPS) |

---

### 2. Ultralytics Training Set Progression (`weapon_data.yaml`)
Validation metrics over 250 training epochs comparing undistilled baseline, distilled student, and Attempt 21:

| Model Architecture | Scale | mAP@50 | mAP@50-95 | Precision | Recall | Role |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Teacher (`YOLO26x`)** | Heavy (`x`) | **90.67%** | **39.61%** | **91.44%** | 84.03% | Guidance Teacher |
| **Baseline Student (`YOLO26s`)** | Compact (`s`) | 86.11% | 34.20% | 79.53% | 84.88% | Undistilled Student |
| **Distilled Student (`YOLO26s`)** | Compact (`s`) | **90.43%** | **37.77%** | **89.67%** | 80.67% | **Production Model (+4.32% mAP50)** |
| **Best P2 Head (Attempt 21)** | Compact (`s`) | 90.06% | 36.20% | 82.78% | 86.56% | Experimental Ablation |

> **Ablation Insight & Custom P2 Deprecation:**
> A multi-scale P2 high-resolution detection head (stride 4) was engineered across 23 iterations (Attempts 1–23) with surgical neck freezes, feature adapters, and mosaic disabling to capture fine blade contours. While **Attempt 21 (High Fidelity Extended)** raised precision slightly on uncalibrated confidences, at the production threshold (`Conf = 0.45`) it dropped weapon Recall from **84.17% down to 73.75%** (causing 25 additional missed weapons). Because failing to detect an active weapon poses severe security risks and P2 added +73% latency overhead, the custom P2 architecture was deprecated, and the **Distilled YOLO26s Student** was deployed.

---

## 📈 ST-GCN Action Recognition & Convergence

The ST-GCN model was trained on 17-node skeleton trajectories across 3 action categories (`Cut-Down`, `Stab`, `Thrust`) using 3-fold cross-validation with Triplet Margin Loss:

![ST-GCN Loss Curve](assets/stgcn_loss_curve.png)

* **Cross-Validation Result:** Fold 2 converged with optimal validation loss ($L_{val} = 0.0177$) at epoch 19.
* **OOD Discrimination:** Distinguishes between:
  * 🟢 **Normal Scanning:** No weapon present.
  * 🟡 **Passive Threat ("WEAPON SEEN, SAFE"):** Weapon visible, but kinematic trajectories deviate from violent attack patterns.
  * 🔴 **Active Threat ("VIOLENCE DETECTED"):** Kinematic velocity, acceleration, and joint angle vectors match violent attack manifold $\rightarrow$ Alarm triggered.

---

## 📁 Repository Structure

```
assets/
├── inference_pipeline.png          # 2-stage hierarchical architecture diagram
└── stgcn_loss_curve.png            # Cross-validation training curves
weights/
├── yolo_weapon_distilled.pt        # Distilled YOLO26s weapon detection student (Teacher: YOLO26x)
├── yolo26s-pose.pt                 # 17-keypoint skeletal pose estimation
├── stgcn_violence_fold2.pth        # Trained ST-GCN action model (Fold 2, 9-block, joint-weighted)
├── reference_data_knn.pt           # k-NN reference feature embeddings (k=2, tau=3.4794)
└── reference_data_mahalanobis.json # Matched Mahalanobis centroid, inv-cov & threshold (tau=8.2194)
models/
├── stgcn.py                        # PyTorch ST-GCN implementation (9 blocks, joint-weighted attention)
└── yolo_p2_custom.yaml             # Experimental P2-layer configuration (Ablation study)
src/
├── inference_realtime.py           # Multi-threaded live camera inference engine
├── eval_live_head_to_head.py       # Standalone live head-to-head benchmark runner (All 148 clips)
├── evaluate_knn_benchmark.py       # Comprehensive k-NN OOD evaluation & confusion matrix
├── extract_skeletons.py            # Automated YOLO-Pose extraction to XML annotations
├── skeleton_utils.py               # Gaussian filter, interpolation & normalization
└── ood_metrics.py                  # Deep k-NN and Mahalanobis distance calculation
training/
├── train_stgcn_knn.py              # Triplet Loss + Deep k-NN OOD training pipeline
├── train_stgcn_mahalanobis.py      # Baseline: Triplet Loss + Mahalanobis Distance OOD
├── train_yolo_distill.py           # Knowledge Distillation pipeline (Teacher YOLO26x -> Student YOLO26s)
└── tune_ood_parameters.py          # Adaptive OOD threshold calibration
BENCHMARK_EXPLANATION.md            # Detailed mathematical breakdown & empirical benchmark documentation
requirements.txt                    # Project dependencies
.gitignore                          # Optimized to exclude heavy weights/datasets
LICENSE                             # MIT License
README.md                           # Technical report & documentation
```

---

## 🎯 Out-of-Distribution (OOD) Threat Verification Benchmark

To prevent false alarms from ordinary bodily movements while wielding everyday objects, multiple anomaly detection paradigms were trained and benchmarked on skeletal graph representations.

Comprehensive evaluation across the full 148-clip validation suite (39 OOD clips, 5 webcam glitch clips, 48 With-Coat clips, and 56 Without-Coat clips across all 3 action trajectories: `Cut_Down`, `Stab`, `Thrust`):

| Evaluation Scope | Paradigm / Method | Precision | Recall | Specificity | F1-Score | Accuracy | Characteristics & Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Comprehensive Test (148 Clips)** | **Deep k-NN (Champion)** | **94.95%** | **90.38%** | **88.64%** | **92.61%** | **89.86%** | **Non-Parametric Manifold Gating.** Adapts across multi-modal attack trajectories (`Cut_Down`, `Stab`, `Thrust`), achieving 90.38% threat recall (94/104 attacks caught). |
| **Comprehensive Test (148 Clips)** | **Mahalanobis Distance** | **96.70%** | **84.62%** | **93.18%** | **90.26%** | **87.16%** | **Parametric Covariance Distance.** Lowest false alarm rate (3 vs. 5), but drops 6 attacks under diverse attire. |
| Single-Action Baseline (52 Clips) | Mahalanobis Distance | 96.97% | 96.97% | 94.74% | 96.97% | 96.15% | Historical baseline on unimodal `Cut-Down` clips only (`validation_results_micro.csv`). |
| Exploratory Baselines | Deep SVDD / Autoencoder | — | — | — | — | — | Prone to boundary collapse or high false alarms on fast skeletal articulations. |

> Detailed mathematical derivation, dataset breakdown, and script documentation are provided in [`BENCHMARK_EXPLANATION.md`](BENCHMARK_EXPLANATION.md). The evaluation runner is available at [`src/eval_live_head_to_head.py`](src/eval_live_head_to_head.py).

The **Deep k-NN Feature Gating on ST-GCN embeddings** is deployed as the active runtime model (`weights/stgcn_violence_fold2.pth` + `weights/reference_data_knn.pt`), with the **Mahalanobis Distance reference** (`weights/reference_data_mahalanobis.json`) preserved as an alternative low-footprint baseline.

---

## ⚙️ Installation & Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/YuriSekaii/hierarchical-threat-detection-edge-ai.git
cd hierarchical-threat-detection-edge-ai
pip install -r requirements.txt
```

### 2. Run Live Real-Time Detection
```bash
# Connect webcam and run real-time inference
python src/inference_realtime.py
```

### 3. Run Benchmark Head-to-Head Verification
```bash
# Evaluate Mahalanobis vs. Deep k-NN across validation sets
python src/eval_live_head_to_head.py
```

---

## 🔒 Dataset & Privacy Notice
The experimental dataset was recorded in a controlled laboratory environment for **Proof-of-Concept (PoC) feasibility validation**. 

* **Privacy & Governance:** Due to human subject privacy protections and institutional data governance regulations, raw video recordings and facial imagery are withheld from the public repository.
* **Reproducibility:** Pre-trained model weights are provided directly in the [`weights/`](weights/) directory to enable full out-of-the-box pipeline evaluation and live webcam inference.

---

## 💡 Engineering Insights & Failure Mode Analysis

During extensive offline and live deployment evaluations, key machine learning challenges were identified and addressed:

1. **Temporal Video Data Leakage:**
   * *Problem:* Randomly splitting frames from the same video sequence resulted in artificially inflated validation scores (91%+) due to identical room backgrounds and actor appearance.
   * *Solution:* Enforced strict **video-level / scene-level dataset splitting**, testing exclusively on unseen video sequences.

2. **Decoupling Distillation Loss:**
   * *Problem:* Standard logit distillation applying BCE loss across all channels corrupts bounding box coordinate regression.
   * *Solution:* Separated coordinates (channels 0–3, optimized via Smooth L1) from classification logits (channel 4, optimized via BCEWithLogitsLoss).

3. **Hand-Weapon Visual Correlation:**
   * *Problem:* In small datasets, models falsely learn that clenched fists indicate weapons.
   * *Solution:* Hard-negative mining by adding empty-hand gestures and everyday handheld items (pens, phones) into training annotations.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
