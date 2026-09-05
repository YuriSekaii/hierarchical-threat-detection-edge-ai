# Hierarchical Edge-AI Threat & Violence Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-green.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, resource-efficient dual-stage surveillance system engineered for real-time edge deployment. Combines **YOLO Knowledge Distillation** with **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** and **Out-of-Distribution (OOD) Gating** to detect violent behavior on live video streams with minimal compute overhead.

---

## 東 System Architecture

Traditional surveillance pipelines run continuous deep neural networks on every frame, resulting in high thermal dissipation and compute exhaustion on edge devices. 

This project solves this by decoupling detection into a **Hierarchical Gating Architecture**:

![Inference Pipeline](assets/inference_pipeline.png)

### 1. Stage 1: Continuous Lightweight Scanning (Always-On Loop)
* **Ingestion:** Ingests live video at 30 FPS into a rolling 60-frame thread-safe buffer.
* **Low-Power Gating:** Subsamples **1 frame per 10 (effective rate: 3 FPS)** for weapon detection using a compact, distilled YOLO student model.
* **Compute Savings:** Reduces continuous idle inference workloads by **~90%**, reserving GPU/NPU compute until a weapon is verified ($Conf > 0.5$).

### 2. Stage 2: On-Demand Biomechanical Action Analysis (Triggered Engine)
* **Kinematic Extraction:** Upon Stage 1 trigger, a 60-frame analysis window is activated. YOLO-Pose extracts 17 COCO skeletal joint coordinates.
* **Kinematic Preprocessing:** Implements **1D Gaussian temporal smoothing** ($\sigma=1.0$) and linear interpolation to resolve dropped joints, followed by **centroid normalization** to achieve scale and position invariance.
* **ST-GCN Feature Extraction:** Feeds normalized skeleton graphs through an 8-block Spatial-Temporal Graph Convolutional Network.
* **OOD Distance Gating:** Measures deep feature embeddings against a baseline threat manifold using **Deep k-NN distance**, filtering out passive actions (e.g., calmly holding a knife) from aggressive striking motions.

---

## 溌 Model Compression & Knowledge Distillation Benchmark

To detect small weapons reliably on edge devices without the computational overhead and latency of heavy models (`YOLO26x`), Knowledge Distillation was utilized to distill feature representations into a lightweight, real-time `YOLO26s` student.

### 1. Ground Truth Head-to-Head Surveillance Benchmark
Evaluated across 728 test frames (OOD everyday objects, concealed weapons under coats, and direct attacks) comparing the heavy Teacher, the production Distilled Student, and the **Best Custom P2 Head trained across 23 iterations (Attempt 21)**:

| Metric | Teacher (`YOLO26x`) | Distilled Student (`YOLO26s`) [Production] | Best Custom P2 Head (Attempt 21) | Production Decision Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Weapon Recall (Threat Safety)** | 77.92% | **91.67%** (220/240 detected) | 85.42% (205/240 detected) | **Deployed Student:** 15 fewer missed weapons than P2. |
| **Accuracy** | 82.33% | **84.27%** | 86.87% | Balanced across challenging occlusions & coats. |
| **Precision** | 71.10% | **69.18%** | 76.78% | P2 raised precision at the expense of vital Recall. |
| **Specificity (OOD)** | 84.49% | **80.78%** | 87.58% | Effective non-threat suppression. |
| **F1-Score** | 74.35% | **78.85%** | 80.87% | Solid trade-off for security applications. |
| **Inference Latency** | 18.83 ms (~53 FPS) | **12.07 ms (~83 FPS)** | 12.14 ms | Real-time edge throughput. |
| **Deployment Verdict** | Too Heavy for Edge | **Deployed Champion Model** | **Rolled Back (Degraded Recall)** | In threat detection, missing 15 weapons is unacceptable. |

### 2. Ultralytics Training Set Progression (`weapon_data.yaml`)
Validation metrics over 250 training epochs comparing undistilled baseline, distilled student, and Attempt 21:

| Model Architecture | Scale | mAP@50 | mAP@50-95 | Precision | Recall | Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Teacher (`YOLO26x`)** | Heavy (`x`) | **90.67%** | **39.61%** | **91.44%** | 84.03% | Guidance Teacher |
| **Baseline Student (`YOLO26s`)** | Compact (`s`) | 86.11% | 34.20% | 79.53% | 84.88% | Undistilled Student |
| **Distilled Student (`YOLO26s`)** | Compact (`s`) | **90.43%** | **37.77%** | **89.67%** | 80.67% | **Production Model (+4.32% mAP50)** |
| **Best P2 Head (Attempt 21)** | Compact (`s`) | 90.06% | 36.20% | 82.78% | 86.56% | Experimental Ablation |

> **Ablation Insight & Custom P2 Deprecation:**
> A multi-scale P2 high-resolution detection head (stride 4) was engineered across 23 iterations (Attempts 1窶・3) with surgical neck freezes, feature adapters, and mosaic disabling to capture fine blade contours. While **Attempt 21 (High Fidelity Extended)** successfully raised precision to 76.78%, it dropped weapon Recall from **91.67% down to 85.42%** (causing 15 additional missed weapons). Because failing to detect an active weapon poses severe security risks, the custom P2 architecture was deprecated, and the **Distilled YOLO26s Student** was deployed.

---

## 嶋 ST-GCN Action Recognition & Convergence

The ST-GCN model was trained on 17-node skeleton trajectories across 3 action categories (`Cut-Down`, `Stab`, `Thrust`) using 3-fold cross-validation with Triplet Margin Loss:

![ST-GCN Loss Curve](assets/stgcn_loss_curve.png)

* **Cross-Validation Result:** Fold 2 converged with optimal validation loss ($L_{val} = 0.0177$) at epoch 19.
* **OOD Discrimination:** Distinguishes between:
  * 泙 **Normal Scanning:** No weapon present.
  * 泛 **Passive Threat ("WEAPON SEEN, SAFE"):** Weapon visible, but kinematic trajectories deviate from violent attack patterns.
  * 閥 **Active Threat ("VIOLENCE DETECTED"):** Kinematic velocity, acceleration, and joint angle vectors match violent attack manifold $
ightarrow$ Alarm triggered.

---

## 唐 Repository Structure

```
笏懌楳笏 assets/
笏・  笏懌楳笏 inference_pipeline.png    # 2-stage hierarchical architecture diagram
笏・  笏披楳笏 stgcn_loss_curve.png      # Cross-validation training curves
笏懌楳笏 weights/
笏・  笏懌楳笏 yolo_weapon_distilled.pt  # Distilled YOLO26s weapon detection student (Teacher: YOLO26x)
笏・  笏懌楳笏 yolo26s-pose.pt           # 17-keypoint skeletal pose estimation
笏・  笏懌楳笏 stgcn_violence_fold2.pth  # Trained ST-GCN action recognition model
笏・  笏懌楳笏 reference_data_mahalanobis.json # Mahalanobis mean, inv-cov & threshold
笏・  笏披楳笏 reference_data_knn.pt     # k-NN reference feature embeddings
笏懌楳笏 models/
笏・  笏懌楳笏 stgcn.py                  # PyTorch ST-GCN implementation (Spatial + Temporal GCN)
笏・  笏披楳笏 yolo_p2_custom.yaml       # Experimental P2-layer configuration (Ablation study)
笏懌楳笏 src/
笏・  笏懌楳笏 inference_realtime.py     # Multi-threaded live camera inference engine
笏・  笏懌楳笏 extract_skeletons.py      # Automated YOLO-Pose extraction to XML annotations
笏・  笏懌楳笏 evaluate_knn_benchmark.py # Comprehensive k-NN OOD evaluation & confusion matrix
笏・  笏懌楳笏 skeleton_utils.py         # Gaussian filter, interpolation & normalization
笏・  笏披楳笏 ood_metrics.py            # Deep k-NN and Mahalanobis distance calculation
笏懌楳笏 training/
笏・  笏懌楳笏 train_stgcn_knn.py        # Triplet Loss + Deep k-NN OOD training
笏・  笏懌楳笏 train_stgcn_mahalanobis.py# Baseline: Triplet Loss + Mahalanobis Distance OOD
笏・  笏懌楳笏 train_yolo_distill.py     # Knowledge Distillation pipeline (Teacher YOLO26x -> Student YOLO26s)
笏・  笏披楳笏 tune_ood_parameters.py    # Adaptive OOD threshold calibration
笏懌楳笏 requirements.txt              # Project dependencies
笏懌楳笏 .gitignore                    # Optimized to exclude heavy weights/datasets
笏懌楳笏 LICENSE                       # MIT License
笏披楳笏 README.md                     # Technical report & documentation
```

---

## 溌 Out-of-Distribution (OOD) Threat Verification Benchmark

To prevent false alarms from ordinary bodily movements while wielding everyday objects, multiple anomaly detection paradigms were trained and benchmarked on skeletal graph representations:

| Evaluation Level | Paradigm / Method | Precision | Recall | F1-Score | Accuracy | Characteristics & Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Video-Level Pipeline** | **Mahalanobis Distance** | **1.00 (100.0%)** | **0.86 (86.0%)** | 窶・| **0.87 (87.0%)** | **Zero False Alarms (12/12 non-violent videos rejected).** Calibrated with 91.86% harmonic OOD score. |
| **Clip-Level Micro-Average** | **Mahalanobis Distance** | **96.97%** | **96.97%** | **96.97%** | **96.15%** | Evaluated on clean validation clips (`validation_results_micro.csv`). |
| **Clip-Level Challenge Set** | **Deep k-NN (Champion)** | **83.33%** | **75.76%** | **79.37%** | **83.12%** | **Non-Parametric Feature Gating.** Superior on challenging edge-cases including false-detection clips. |
| Exploratory Baselines | Deep SVDD / Autoencoder | 窶・| 窶・| 窶・| 窶・| Prone to boundary collapse or high false alarms on complex skeletal articulations. |

The **Deep k-NN Feature Gating on ST-GCN embeddings** was deployed as the active runtime model (`weights/stgcn_violence_fold2.pth` + `weights/reference_data_knn.pt`), with the **Mahalanobis Distance reference** (`weights/reference_data_mahalanobis.json`) preserved as the parametric baseline.

## 笞呻ｸ・Installation & Quickstart

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

---


---

## 投 Dataset & Privacy Notice
The experimental dataset was recorded in a controlled laboratory environment for **Proof-of-Concept (PoC) feasibility validation**. 

* **Privacy & Governance:** Due to human subject privacy protections and institutional data governance regulations, raw video recordings and facial imagery are withheld from the public repository.
* **Reproducibility:** Pre-trained model weights are provided directly in the [`weights/`](weights/) directory to enable full out-of-the-box pipeline evaluation and live webcam inference.

## 剥 Engineering Insights & Failure Mode Analysis

During extensive offline and live deployment evaluations, key machine learning challenges were identified and addressed:

1. **Temporal Video Data Leakage:**
   * *Problem:* Randomly splitting frames from the same video sequence resulted in artificially inflated validation scores (91%+) due to identical room backgrounds and actor appearance.
   * *Solution:* Enforced strict **video-level / scene-level dataset splitting**, testing exclusively on unseen video sequences.

2. **Decoupling Distillation Loss:**
   * *Problem:* Standard logit distillation applying BCE loss across all channels corrupts bounding box coordinate regression.
   * *Solution:* Separated coordinates (channels 0窶・, optimized via Smooth L1) from classification logits (channel 4, optimized via BCEWithLogitsLoss).

3. **Hand-Weapon Visual Correlation:**
   * *Problem:* In small datasets, models falsely learn that clenched fists indicate weapons.
   * *Solution:* Hard-negative mining by adding empty-hand gestures and everyday handheld items (pens, phones) into training annotations.

---

## 糖 License
This project is licensed under the [MIT License](LICENSE).

