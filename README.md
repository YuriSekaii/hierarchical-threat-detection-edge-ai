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

## 🔬 Model Compression & Knowledge Distillation Benchmark

To detect small weapons reliably on edge devices without the computational overhead and latency of heavy models (`YOLO26x`), Knowledge Distillation was utilized to distill feature representations into a lightweight, real-time `YOLO26s` student.

### 1. Ground Truth Surveillance Benchmark (Operational Threshold $Conf=0.45$)
Evaluated on the 728-frame multi-scenario ground-truth test set (OOD everyday objects, concealed weapons under coats, and direct unobstructed attacks):

| Metric | Teacher (`YOLO26x`) | Distilled Student (`YOLO26s`) [Production] | Delta vs. Teacher | Custom P2 Head Student |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 82.33% | **91.62%** | **+9.29%** | 79.48% |
| **F1-Score** | 74.35% | **86.88%** | **+12.53%** | 60.48% |
| **Precision** | 71.10% | **89.78%** | **+18.68%** | 83.21% |
| **Recall (Threat Safety)** | 77.92% (187/240) | **84.17% (202/240)** | **+6.25%** | 47.50% (114/240) |
| **Specificity (OOD)** | 84.49% (414/490) | **95.29% (465/488)** | **+10.80%** | 95.27% |
| **False Positives** | 76 | **23 (3.3x reduction)** | **-53 FP** | 23 |
| **False Negatives** | 53 | **38 (15 fewer misses)** | **-15 FN** | 126 |
| **Inference Latency** | 18.83 ms (~53 FPS) | **12.07 ms (~83 FPS)** | **1.56x Faster** | 13.62 ms |
| **Deployment Verdict** | Heavy for Edge | **Deployed Champion** | — | Deprecated (Recall Collapse) |

### 2. Ultralytics Training Set Progression (`weapon_data.yaml`)
Validation metrics over 250 training epochs comparing undistilled baseline against distilled student:

| Model Architecture | Scale | mAP@50 | mAP@50-95 | Precision | Recall | Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Teacher (`YOLO26x`)** | Heavy (`x`) | **90.67%** | **39.61%** | **91.44%** | 84.03% | Guidance Teacher |
| **Baseline Student (`YOLO26s`)** | Compact (`s`) | 86.11% | 34.20% | 79.53% | 84.88% | Undistilled Student |
| **Distilled Student (`YOLO26s`)** | Compact (`s`) | **90.43%** | **37.77%** | **89.67%** | 80.67% | **Production Model (+4.32% mAP50)** |

> **Ablation Insight & Custom P2 Deprecation:**
> A multi-scale P2 high-resolution detection head (stride 4) was engineered and tested across 23 iterations (Attempts 1–23) with surgical neck freezes, feature adapters, and mosaic disabling to improve tiny blade edge detection. However, empirical benchmarking revealed that adding P2 induced gradient instability on compact datasets:
> - On the standard operational benchmark ($Conf=0.45$), initial P2 distillation collapsed weapon Recall to **47.50%** (missing 126 weapons).
> - In the extended surgical fine-tuning attempt (Attempt 21 at $Conf=0.20$), P2 achieved **85.42% Recall** (35 false negatives), significantly underperforming the baseline Distilled Student's **91.67% Recall** (20 false negatives).
> - Because missing 15 active threat events in a physical security pipeline is unacceptable, the custom P2 modification was deprecated, and the **Distilled YOLO26s Student** was deployed.

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

| Method | Accuracy | Precision | Recall | F1-Score | Strategy / Characteristics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Deep k-NN (Champion)** | **83.12%** | **83.33%** | **75.76%** | **79.37%** | **Non-Parametric Feature Gating.** Preserves manifold geometry; **+29.3% Recall** and **+24.6% F1** over Mahalanobis. |
| **Mahalanobis Distance** | 78.07% | 66.67% | 46.45% | 54.75% | Parametric Gaussian. Degrades on non-Gaussian complex biomechanical clusters. |
| **Deep SVDD** | — | — | — | — | Hypersphere boundary. Prone to representation collapse without negative anchors. |
| **Joint Autoencoder** | — | — | — | — | Reconstruction error. High error on complex multi-joint motion; higher false alarm rate. |
| **Binary Classification** | — | — | — | — | Supervised Cross-Entropy. Overfits to background features and actor silhouettes. |

The **Deep k-NN Feature Gating on ST-GCN embeddings** was selected as the champion model (`weights/stgcn_violence_fold2.pth` + `weights/reference_data_knn.pt`) for real-time threat gating against passive actions.

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
