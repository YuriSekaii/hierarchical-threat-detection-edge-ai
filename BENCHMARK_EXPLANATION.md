# Head-to-Head Benchmark Methodology & Technical Explanation
## Mahalanobis Distance vs. Deep k-NN for ST-GCN Violence Recognition

This document provides a comprehensive technical breakdown of the live head-to-head evaluation comparing **Parametric Mahalanobis Distance** and **Non-Parametric Deep k-NN** on top of the **ST-GCN Fold 2** action recognition backbone.

---

### 1. Benchmark Execution Script
- **Script Location:** `src/eval_live_head_to_head.py`
- **Output Files:**
  - CSV Summary: `results/live_head_to_head_benchmark.csv`
  - Text Report: `results/live_head_to_head_benchmark.txt`
- **Backbone Model Weights:** `weights/stgcn_violence_fold2.pth` (12,142,947 bytes)
- **Execution Command:**
```powershell
$env:PYTHONPATH = "C:\Users\Admin\Desktop\Code\Python\Intern\Weapon_Detection_SingleGPU\.venv\Lib\site-packages"
C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe src/eval_live_head_to_head.py
```

---

### 2. Dataset Composition (148 Total Clips)

The evaluation was conducted on every single validation XML annotation file across all available subsets:

| Dataset Category | Action Subsets Included | Number of Clips | True Class Label | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Out-Of-Distribution (`OOD`)** | Everyday civilian activities (walking, gesturing, stretching) | **39** | Negative | Evaluates civilian false alarm rejection |
| **False Detected (`False_Detected`)** | Webcam sensor glitches, occluded detections, broken skeleton trees | **5** | Negative | Evaluates robustness against camera/tracking glitches |
| **With Coat (`With_Coat`)** | `Cut_Down` (16), `Stab` (14), `Thrust` (18) | **48** | Positive | Evaluates violence detection under heavy clothing / obscured joints |
| **Without Coat (`Without_Coat`)** | `Cut_Down` (17), `Stab` (18), `Thrust` (21) | **56** | Positive | Evaluates violence detection under standard clothing |
| **Total Validation Set** | | **148** | | |

---

### 3. Feature Extraction & Preprocessing Pipeline

1. **Skeleton Parsing (`CVAT XML`):**
   - Extracts 17 COCO 2D keypoints per frame: $(x_i, y_i) \in \mathbb{R}^2$ for $i \in \{0, \dots, 16\}$.
   - Handled missing keypoints via 1D linear interpolation across time.
   - Temporal smoothing via 1D Gaussian filter ($\sigma = 1.0$).
   - Centroid normalization: $\mathbf{p}_{t, i} \leftarrow \mathbf{p}_{t, i} - \frac{1}{K} \sum_{k \in \text{valid}} \mathbf{p}_{t, k}$.
2. **Temporal Windowing:**
   - Evaluated as 50-frame temporal clips ($T=50, V=17, C=2$) with stride $S=25$.
3. **ST-GCN Backbone with Joint-Weighted Attention:**
   - 9 Spatio-Temporal Graph Convolutional blocks ($64 \rightarrow 128 \rightarrow 256$ channels).
   - Normalized spatial adjacency matrix $\mathbf{A} \in \mathbb{R}^{3 \times 17 \times 17}$ based on COCO skeleton topology.
   - **Joint-Weighted Spatial Attention:** Upper limbs (shoulders, elbows, wrists: indices 5, 6, 7, 8, 9, 10) are weighted by 5x, normalized across all 17 keypoints.
   - Adaptive average pooling produces a 256-dimensional feature vector $\mathbf{z} \in \mathbb{R}^{256}$ per clip.

---

### 4. Mathematical Comparison of the Two OOD Pipelines

#### Pipeline A: Parametric Mahalanobis Distance
- **Training Data:** 297 violence action clips from `Train_Action_Recognition_STGCN_Model/data`.
- **Centroid:** Sample mean vector $\boldsymbol{\mu} \in \mathbb{R}^{256}$.
- **Covariance Matrix with Shrinkage Regularization:**
  $$\boldsymbol{\Sigma}_{\text{reg}} = \boldsymbol{\Sigma} + \lambda \mathbf{I}_{256}, \quad \text{where } \lambda = 0.0008$$
- **Precision Matrix:** $\boldsymbol{\Sigma}^{-1} = \text{pinv}(\boldsymbol{\Sigma}_{\text{reg}})$.
- **Distance Metric:**
  $$D_M(\mathbf{z}) = \sqrt{(\mathbf{z} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu})}$$
- **Threshold Tuning:** 95.5th percentile of training clip distances: $\tau_M = 8.2194$.
- **Clip Verdict:** Positive (Violence) if $\min_{\text{clips}} D_M(\mathbf{z}) \le \tau_M$.

#### Pipeline B: Non-Parametric Deep k-NN
- **Training Bank:** All $N=297$ normalized 256-dim feature vectors stored as a tensor bank $\mathcal{B} \in \mathbb{R}^{297 \times 256}$.
- **Distance Metric:** Euclidean distance to the $k$-th nearest training neighbor ($k=2$):
  $$D_{\text{kNN}}(\mathbf{z}) = \text{k-th-min}_{\mathbf{z}_j \in \mathcal{B}} \|\mathbf{z} - \mathbf{z}_j\|_2$$
- **Threshold Tuning:** 95th percentile of leave-one-out training distances: $\tau_{\text{kNN}} = 3.4794$.
- **Clip Verdict:** Positive (Violence) if $\min_{\text{clips}} D_{\text{kNN}}(\mathbf{z}) \le \tau_{\text{kNN}}$.

---

### 5. Empirical Results (Live Run)

| Dataset Scope | Pipeline | Clips | Accuracy | Precision | Recall | Specificity | F1-Score | Confusion Matrix |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clean (No Glitches)** | Mahalanobis | 143 | 87.41% | **97.78%** | 84.62% | **94.87%** | 90.72% | 88 TP, 2 FP, 16 FN, 37 TN |
| **Clean (No Glitches)** | Deep k-NN | 143 | **90.91%** | 96.91% | **90.38%** | 92.31% | **93.53%** | 94 TP, 3 FP, 10 FN, 36 TN |
| **Challenge (With Glitches)** | Mahalanobis | 148 | 87.16% | **96.70%** | 84.62% | **93.18%** | 90.26% | 88 TP, 3 FP, 16 FN, 41 TN |
| **Challenge (With Glitches)** | Deep k-NN | 148 | **89.86%** | 94.95% | **90.38%** | 88.64% | **92.61%** | 94 TP, 5 FP, 10 FN, 39 TN |

#### Category Breakdown:

| Category | Clips | Mahalanobis Accuracy & Errors | Deep k-NN Accuracy & Errors |
| :--- | :---: | :--- | :--- |
| **Out-Of-Distribution (`OOD`)** | 39 | **94.9%** (37 TN, 2 FP) | **92.3%** (36 TN, 3 FP) |
| **Webcam Glitches (`False_Detected`)** | 5 | **80.0%** (4 TN, 1 FP) | **60.0%** (3 TN, 2 FP) |
| **With Coat (Concealed Attacks)** | 48 | **81.2%** (39 TP, 9 FN) | **93.8%** (45 TP, 3 FN) |
| **Without Coat (Open Attacks)** | 56 | **87.5%** (49 TP, 7 FN) | **87.5%** (49 TP, 7 FN) |

---

### 6. Why Historical Benchmarks Showed Confusing Numbers

A common source of confusion in the project's historical files was why Mahalanobis had a recorded score of **96.15%** in `results/validation_results_micro.csv`, while k-NN recorded **83.12%** in `results/validation_results_knn.csv`.

1. **Single Action vs. Multi-Action Distribution:**
   - The historical 96.15% score evaluated **only 1 action category: `Cut-Down`** (16 with coat, 17 without coat, 19 OOD = 52 clips total).
   - Because `Cut-Down` has a single unimodal distribution, a single Mahalanobis centroid $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ fit the data tightly.
   - However, the actual dataset contains **3 distinct action modalities**: `Cut-Down` (overhead vertical swing), `Stab` (short linear puncture), and `Thrust` (extended lunge).
   - When evaluating across all 3 actions (148 clips), a single unimodal Gaussian distribution cannot capture the multimodal geometry. This causes Mahalanobis recall to drop from 96% to 84.62% (missing 16 attacks).
2. **Why Deep k-NN Wins on the Full Dataset:**
   - Deep k-NN is non-parametric. It does not force the 3 distinct actions into one centroid.
   - It maintains memory of all 3 action manifolds and checks distance to the closest known attack.
   - As a result, Deep k-NN detects **94 out of 104 attacks (90.38% recall)** compared to Mahalanobis's **88 out of 104 (84.62% recall)**, making Deep k-NN superior for life-safety threat detection.
3. **The "Parametric Collapse" Misunderstanding:**
   - In the public repo README, it was claimed that Mahalanobis "collapsed on webcam glitches".
   - Our audit revealed this was caused by an un-updated reference file (`reference_data.json` had not been recomputed when the joint-weighted Fold 2 model was trained on March 11).
   - When recomputed properly, Mahalanobis actually had **fewer false alarms** on glitches than k-NN (1 vs 2).
   - However, because k-NN catches 6 more real attacks (+5.76% recall) and achieves higher overall accuracy (**89.86% vs 87.16%**), Deep k-NN remains the scientifically validated choice for production.
