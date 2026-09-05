# Hierarchical Edge-AI Threat & Violence Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-green.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, resource-efficient dual-stage surveillance system engineered for real-time edge deployment. Combines **YOLO Knowledge Distillation** with **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** and **Out-of-Distribution (OOD) Gating** to detect violent behavior on live video streams with minimal compute overhead.

---

## 📌 System Architecture

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

## 🔬 Model Compression & Knowledge Distillation (Distilled Student)

To achieve fast, responsive weapon detection on edge hardware without the computational latency of heavy models (`YOLO26x`), Knowledge Distillation was employed to transfer feature representations from the heavy teacher into a compact `YOLO26s` student:

| Metric | Teacher (YOLO26x) | Distilled Student (`YOLO26s`) [Production] | Experimental Custom P2 (Attempt 21) |
| :--- | :---: | :---: | :---: |
| **Recall (Threat Safety)** | 91.67% | **91.67%** (Best) | 85.42% (Misses weapons) |
| **Accuracy** | 84.27% | **84.27%** | 86.87% |
| **Specificity** | 80.78% | **80.78%** | 87.58% |
| **F1-Score** | 78.85% | **78.85%** | 80.87% |
| **Avg Inference Time** | 23.94 ms | **18.16 ms** (Edge-ready) | 12.14 ms |
| **Production Decision** | Excluded (Heavy compute) | **Deployed Model** | **Rolled Back (Degraded Recall)** |

> **Ablation Insight & Decision:** Extensive engineering efforts were conducted to explore a custom P2 high-resolution detection head (Attempts 1–23) for small blade contours. However, empirical ground-truth testing demonstrated that the custom P2 head degraded weapon Recall from 91.67% down to 85.42% (causing 15 additional missed weapons). Because missing an active weapon is unacceptable in threat detection, the custom P2 modification was deprecated, and the robust **Distilled YOLO26s Student** was deployed for real-time edge gating.

---

## 📈 ST-GCN Action Recognition & Convergence

The ST-GCN model was trained on 17-node skeleton trajectories across 3 action categories (`Cut-Down`, `Stab`, `Thrust`) using 3-fold cross-validation with Triplet Margin Loss:

![ST-GCN Loss Curve](assets/stgcn_loss_curve.png)

* **Cross-Validation Result:** Fold 2 converged with optimal validation loss ($L_{val} = 0.0177$) at epoch 19.
* **OOD Discrimination:** Distinguishes between:
  * 🟢 **Normal Scanning:** No weapon present.
  * 🟠 **Passive Threat ("WEAPON SEEN, SAFE"):** Weapon visible, but kinematic trajectories deviate from violent attack patterns.
  * 🔴 **Active Threat ("VIOLENCE DETECTED"):** Kinematic velocity, acceleration, and joint angle vectors match violent attack manifold $
ightarrow$ Alarm triggered.

---

## 📂 Repository Structure

```
├── assets/
│   ├── inference_pipeline.png    # 2-stage hierarchical architecture diagram
│   └── stgcn_loss_curve.png      # Cross-validation training curves
├── weights/
│   ├── yolo_weapon_distilled.pt  # Distilled YOLO26s weapon detection student (Teacher: YOLO26x)
│   ├── yolo26s-pose.pt           # 17-keypoint skeletal pose estimation
│   ├── stgcn_violence_fold2.pth  # Trained ST-GCN action recognition model
│   ├── reference_data_mahalanobis.json # Mahalanobis mean, inv-cov & threshold
│   └── reference_data_knn.pt     # k-NN reference feature embeddings
├── models/
│   ├── stgcn.py                  # PyTorch ST-GCN implementation (Spatial + Temporal GCN)
│   └── yolo_p2_custom.yaml       # Experimental P2-layer configuration (Ablation study)
├── src/
│   ├── inference_realtime.py     # Multi-threaded live camera inference engine
│   ├── extract_skeletons.py      # Automated YOLO-Pose extraction to XML annotations
│   ├── evaluate_knn_benchmark.py # Comprehensive k-NN OOD evaluation & confusion matrix
│   ├── skeleton_utils.py         # Gaussian filter, interpolation & normalization
│   └── ood_metrics.py            # Deep k-NN and Mahalanobis distance calculation
├── training/
│   ├── train_stgcn_knn.py        # Triplet Loss + Deep k-NN OOD training
│   ├── train_stgcn_mahalanobis.py# Baseline: Triplet Loss + Mahalanobis Distance OOD
│   ├── train_yolo_distill.py     # Knowledge Distillation pipeline (Teacher YOLO26x -> Student YOLO26s)
│   └── tune_ood_parameters.py    # Adaptive OOD threshold calibration
├── requirements.txt              # Project dependencies
├── .gitignore                    # Optimized to exclude heavy weights/datasets
├── LICENSE                       # MIT License
└── README.md                     # Technical report & documentation
```

---

## 🔬 Out-of-Distribution (OOD) Threat Verification Benchmark

To prevent false alarms from ordinary bodily movements while wielding everyday objects, multiple anomaly detection paradigms were trained and benchmarked on skeletal graph representations:

| Method | Paradigm | Strategy | Verdict & Characteristics |
| :--- | :--- | :--- | :--- |
| **Deep k-NN (Champion)** | **Metric Distance** | **Non-Parametric Feature Gating** | **Best Generalization.** Preserves feature magnitude; cleanly discriminates knife attacks from gestures. |
| **Mahalanobis Distance** | Parametric Gaussian | Covariance Matrix Estimation | Strong baseline, but sensitive to non-Gaussian kinematic cluster shapes. |
| **Deep SVDD** | Hypersphere Boundary | One-Class Support Vector | Good boundary enclosure; prone to representation collapse without negative anchors. |
| **Joint Autoencoder** | Reconstruction Error | Spatial-Temporal AE | High reconstruction error on complex multi-joint motion; higher false alarm rate. |
| **Binary Classification** | Supervised Cross-Entropy | Standard Linear Classifier | Overfits to background features and actor silhouettes; poor zero-shot OOD generalization. |

The **Deep k-NN Feature Gating on ST-GCN embeddings** was selected as the champion model (`Train_Violence_STGCN_fold2.pth` + `reference_data_knn.pt`) for real-time threat gating against passive actions.

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

---


---

## 📊 Dataset & Privacy Notice
The experimental dataset was recorded in a controlled laboratory environment for **Proof-of-Concept (PoC) feasibility validation**. 

* **Privacy & Governance:** Due to human subject privacy protections and institutional data governance regulations, raw video recordings and facial imagery are withheld from the public repository.
* **Reproducibility:** Pre-trained model weights are provided directly in the [`weights/`](weights/) directory to enable full out-of-the-box pipeline evaluation and live webcam inference.

## 🔍 Engineering Insights & Failure Mode Analysis

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

## 📜 License
This project is licensed under the [MIT License](LICENSE).
