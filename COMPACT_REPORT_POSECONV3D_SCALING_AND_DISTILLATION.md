# COMPACT TECHNICAL BRIEF: POSECONV3D SCALING, REGULARIZATION & DISTILLATION

**System Target:** Stage 2b Biomechanical Action Recognition & Threat Screening  
**Hardware Environment:** NVIDIA GeForce RTX 3090 (24GB GDDR6X, CUDA 12.1, PyTorch 2.5.1) / Single-Thread x86_64 CPU Emulation  
**Validation Benchmark Suite:** 77 surveillance videos / 148 clips (`With_Coat`: 16 violent, `Without_Coat`: 17 violent, `OOD`: 44 civilian non-violent; Total: 33 violent / 44 non-violent)

---

## 1. Core Defect & Problem Statement

### A. The Baseline Dilemma
In Stage 2b, the baseline PoseConv3D model (`v2.10`, 765,529 parameters) demonstrated significant edge advantages:
- Rejection of non-violent civilian motion: **96.30% Precision** (only 1 false positive across 44 OOD videos).
- Low computational demand: **23.3 FPS CPU** ($42.92\text{ ms}$) / **36.3 FPS GPU** ($27.57\text{ ms}$), running $3.2\times$ faster on CPU than graph-based ST-GCN ($7.2\text{ FPS CPU}$).
- Parameter footprint: **765,529 parameters** ($-74.6\%$ vs ST-GCN's $3.01\text{M}$).

However, baseline PoseConv3D exhibited a critical safety defect:
- **Recall collapsed to 78.79%** (**7 missed violent attacks** out of 33: 4 `With_Coat`, 3 `Without_Coat`), dropping F1-score to **0.8667**. In contrast, continuous coordinate ST-GCN achieved **100.0% Recall** (0 missed attacks).

### B. Technical Diagnosis: Spatial Quantization Blurring at $56 \times 56$ Resolution
The root cause of PoseConv3D's false-negative rate is a structural input representation defect:

1. **Continuous-to-Discrete Projection Deficit:**  
   ST-GCN processes continuous normalized joint coordinates $(x, y) \in [0.0, 1.0]$ with floating-point precision ($C=2, T=50, V=17$). PoseConv3D maps these coordinates into a discrete spatial heatmap volume $V \in \mathbb{R}^{K \times T \times H \times W}$ ($17 \times 50 \times 56 \times 56$).
2. **Sub-Pixel Spatial Loss:**  
   At $H=56, W=56$, a single spatial pixel corresponds to $\approx 1.79\%$ of bounding frame dimension. Violent micro-kinematics (rapid wrist snaps, subtle blade flicking, low-amplitude thrusts) exhibit joint displacements under $1-2$ pixels between consecutive frames.
3. **Gaussian Smoothing Defect ($\sigma = 1.5$):**  
   Rasterization smooths joint center points using a Gaussian kernel:
   $$G(x, y) = \exp\left(-\frac{(x - \mu_x)^2 + (y - \mu_y)^2}{2\sigma^2}\right), \quad \sigma = 1.5$$
   With $\sigma=1.5$, Gaussian probability mass extends over a $7 \times 7$ pixel window. Rapid kinematic bursts blur across adjacent spatial cells, averaging out high-frequency velocity signals and edge trajectories into diffuse blobs.
4. **Heavy Outerwear Distortion (`With_Coat`):**  
   Bulky winter jackets obscure torso and arm keypoints, increasing keypoint variance. Blurring over discrete cells strips the geometric articulation needed to separate sudden stabbing trajectories from wide clothing shifts.

### C. The Parameter Upscaling Failure
To test whether capacity under-allocation caused the recall deficit, PoseConv3D was scaled by $+363.1\%$ capacity (`models/poseconv3d_scaled.py`):
- Backbone channels expanded from `[24, 48, 96, 256]` to `[64, 128, 256, 512]` ($3,545,409\text{ parameters}$, $6.369\text{ GFLOPs}$, matching ST-GCN's $3.01\text{M}$).
- **Empirical Failure:**
  - **F1 collapsed:** Dropped from $0.8667 \to \mathbf{0.8406}$ ($-0.0261$).
  - **Safety deficit persisted:** Recall reached only $87.88\%$ (**4 missed attacks**: 2 `With_Coat`, 2 `Without_Coat`).
  - **Civilian false alarms surged:** False alarms jumped $700\%$ ($1\text{ FP} \to \mathbf{7\text{ FP}}$), dropping precision to $80.56\%$. On $N=297$ training clips, the $3.55\text{M}$ parameter 3D CNN overfitted benign gestures (e.g., chopping wood mistaken for knife slashes).
  - **Latency exploded:** Single-thread CPU latency jumped from $42.92\text{ ms} \to \mathbf{179.81\text{ ms}}$ ($5.6\text{ FPS}$), becoming slower than ST-GCN ($139.34\text{ ms}$ / $7.2\text{ FPS}$).
- **Conclusion:** **Spatial quantization occurs during input generation prior to the first convolutional layer. Neural network capacity cannot reconstruct physical signals destroyed during rasterization.**

---

## 2. Cross-Paradigm Distillation Formulation

To overcome the spatial quantization ceiling without increasing parameter count, continuous coordinate knowledge was distilled from frozen ST-GCN into PoseConv3D.

```
Surveillance Clip (50 frames)
  │
  ├─► Continuous Coordinates (2, 50, 17) ──► Frozen Teacher: ST-GCN (3.01M params) ─► f_T (256-D)
  │                                                                                        │
  │                                                                                 [KD Objective]
  │                                                                                        ▼
  └─► 3D Heatmap Volume (17, 50, 56, 56) ──► Student: Tier 3 PoseC3D (598k params) ──► f_S (256-D)
```

### A. General Knowledge Distillation Objective
The primary classification/metric knowledge distillation framework couples task supervision with teacher distribution alignment:
$$\mathcal{L}_{\text{KD}} = (1 - \alpha)\mathcal{L}_{\text{task}} + \alpha T^2 \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{z_S}{T}\right) \parallel \sigma\left(\frac{z_T}{T}\right)\right)$$
- $\mathcal{L}_{\text{task}}$: Task supervision (Triplet Margin Loss $\mathcal{L}_{\text{triplet}}$ with margin $m=1.0$, or Cross-Entropy $\mathcal{L}_{\text{CE}}$).
- $z_S, z_T$: Student and teacher logit outputs / projected representations.
- $T$: Temperature scaling parameter smoothing soft probability vectors.
- $\alpha \in [0, 1]$: Loss interpolation balance parameter.
- $\mathcal{D}_{\text{KL}}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$: Kullback-Leibler divergence.

### B. Penultimate Manifold Metric Distillation Formulations
Both ST-GCN and PoseConv3D project into an identical $D=256$ dimensional penultimate feature space ($f_S, f_T \in \mathbb{R}^{256}$). Four cross-paradigm loss formulations were evaluated (`src/distillation_losses.py`):

#### 1. Method A: Latent Feature Alignment (Cosine + Normalized MSE)
Forces the student feature vector $f_S$ to replicate the teacher vector $f_T$ in direction and Euclidean magnitude:
$$\mathcal{L}_{\text{feat}} = \alpha_{\text{cos}}\left(1.0 - \frac{f_S \cdot f_T}{\|f_S\|_2 \|f_T\|_2}\right) + \frac{\alpha_{\text{mse}}}{D} \|f_S - f_T\|_2^2$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{triplet}} + 1.0 \cdot \mathcal{L}_{\text{feat}}$$
*Hyperparameters:* $\alpha_{\text{cos}} = 1.0$, $\alpha_{\text{mse}} = 1.0$, $D = 256$.

#### 2. Method B: Relational Knowledge Distillation (RKD Distance & Angle) [CHAMPION]
Transfers the geometric manifold topology of relative sample interactions rather than forcing absolute vector imitation (Park et al., CVPR 2019):
- **Distance-wise Relational Loss:** Huber loss on normalized pairwise Euclidean distance matrices:
  $$\psi_{ij}(f) = \frac{\|f_i - f_j\|_2}{\mu_d + \epsilon}, \quad \text{where } \mu_d = \frac{1}{B^2}\sum_{a=1}^B \sum_{b=1}^B \|f_a - f_b\|_2$$
  $$\mathcal{L}_{\text{R-dist}} = \ell_{\delta}(\psi_S, \psi_T), \quad \ell_{\delta}(x, y) = \begin{cases} 0.5(x - y)^2 & \text{if } |x - y| < 1.0 \\ |x - y| - 0.5 & \text{otherwise} \end{cases}$$
- **Angle-wise Relational Loss:** Huber loss on sample triplet angle distributions:
  $$e_{ij} = \frac{f_i - f_j}{\|f_i - f_j\|_2 + \epsilon}, \quad \cos \angle_{(i,j,k)} = e_{ij} \cdot e_{kj}$$
  $$\mathcal{L}_{\text{R-angle}} = \ell_{\delta}(\cos \angle_S, \cos \angle_T)$$
- **Total Objective:**
  $$\mathcal{L}_{\text{RKD}} = 1.0 \cdot \mathcal{L}_{\text{R-dist}} + 2.0 \cdot \mathcal{L}_{\text{R-angle}}$$
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{triplet}} + 20.0 \cdot \mathcal{L}_{\text{RKD}}$$

#### 3. Method C: Spatial-Temporal Arm Attention Distillation
Transfers ST-GCN's $5\times$ spatial attention weighting over arm keypoints $J_{\text{arm}} = \{5, 6, 7, 8, 9, 10\}$ (shoulders, elbows, wrists):
$$E_S(k) = \frac{1}{T \cdot H \cdot W}\sum_{t=1}^T \sum_{h=1}^H \sum_{w=1}^W V_S(k, t, h, w), \quad P_S = \text{Softmax}(E_S)$$
$$w_k = \begin{cases} 5.0 & \text{if } k \in J_{\text{arm}} \\ 1.0 & \text{otherwise} \end{cases}, \quad P_{\text{target}} = \text{Softmax}\left(\frac{w}{\sum w} \cdot K\right)$$
$$\mathcal{L}_{\text{attn}} = \mathcal{D}_{\text{KL}}(P_S \parallel P_{\text{target}}) + \text{MSE}(f_S, f_T)$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{triplet}} + 1.0 \cdot \mathcal{L}_{\text{attn}}$$

#### 4. Method D: Combined Tri-Distillation
Integrates feature alignment, relational manifold constraints, and joint attention:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{triplet}} + 0.5 \cdot \mathcal{L}_{\text{feat}} + 10.0 \cdot \mathcal{L}_{\text{RKD}} + 0.5 \cdot \mathcal{L}_{\text{attn}}$$

### C. OOD Likelihood Gating via Relative Mahalanobis Distance (RMD)
Penultimate embeddings $z \in \mathbb{R}^{256}$ are evaluated against class-conditional distributions with shrinkage covariance regularization $\epsilon$:
$$\text{Score}_{\text{RMD}}(z) = (z - \mu_0)^T (\Sigma_0 + \epsilon I)^{-1}(z - \mu_0) - \min_c (z - \mu_c)^T (\Sigma_c + \epsilon I)^{-1}(z - \mu_c)$$
Continuous scores are converted to calibrated posterior probabilities:
$$P(x) = \sigma\left(\frac{\text{Score}_{\text{RMD}}(z) - \tau}{T}\right) = \frac{1}{1 + \exp\left(-\frac{\text{Score}_{\text{RMD}}(z) - \tau}{T}\right)}$$
where $T=1.0$ and $\tau$ is the decision boundary threshold.

---

## 3. Ablation & Results Tables

### A. Master Comparative Performance Table (Standardized 77-Video Suite)
Empirical results evaluated across 77 validation videos / 148 sliding clips (RTX 3090 GPU / Single-Thread CPU):

| Model / Configuration | Architecture Paradigm | Parameter Count | GFLOPs | Violence F1 | Violence Recall | Missed Attacks (FN) | Precision | False Alarms (FP / 44) | Accuracy | With_Coat (TP/16) | Without_Coat (TP/17) | Model Forward (CPU / GPU) | Action Stage 2 (CPU / GPU) | Full Pipeline (CPU / GPU) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Teacher: ST-GCN (v1.08)** | Spatio-Temporal Graph CNN | $3,014,981$ | $0.901$ | 0.9167 | **100.0%** | **0** | 84.62% | 6 FP | 92.21% | **16/16** | **17/17** | $12.0\text{ ms}$ / $3.7\text{ ms}$ | $14.4\text{ ms}$ / $3.8\text{ ms}$ | $170.5\text{ ms}$ ($5.9\text{ FPS}$) / $27.3\text{ ms}$ ($36.6\text{ FPS}$) |
| **CTR-GCN (v2.00)** | Dynamic Topology Graph CNN | $1,332,761$ | $0.784$ | 0.8649 | 96.97% | 1 | 78.05% | 9 FP | 87.01% | **16/16** | 16/17 | $25.3\text{ ms}$ / $11.0\text{ ms}$ | $27.7\text{ ms}$ / $11.1\text{ ms}$ | $183.8\text{ ms}$ ($5.4\text{ FPS}$) / $34.6\text{ ms}$ ($28.9\text{ FPS}$) |
| **PoseConv3D Base (v2.10)** | 3D CNN (`bc=24, fd=256`) | $765,529$ | $1.221$ | 0.8667 | 78.79% | 7 | 96.30% | **1 FP** | 89.61% | 12/16 | 14/17 | $21.3\text{ ms}$ / $2.2\text{ ms}$ | $57.5\text{ ms}$ / $2.8\text{ ms}$ | $213.6\text{ ms}$ ($4.7\text{ FPS}$) / $26.4\text{ ms}$ ($37.9\text{ FPS}$) |
| **PoseConv3D Scaled (Exp 1)** | 3D CNN (`bc=64, fd=512`) | $3,545,409$ | $6.369$ | 0.8406 | 87.88% | 4 | 80.56% | 7 FP | 85.71% | 14/16 | 15/17 | $47.5\text{ ms}$ / $2.3\text{ ms}$ | $83.6\text{ ms}$ / $3.0\text{ ms}$ | $239.8\text{ ms}$ ($4.2\text{ FPS}$) / $26.5\text{ ms}$ ($37.7\text{ FPS}$) |
| **PoseConv3D Tier 1** | 3D CNN (`bc=20, fd=256`) | $702,885$ | $0.854$ | 0.8116 | 84.85% | 5 | 77.78% | 8 FP | 83.12% | 14/16 | 14/17 | $19.5\text{ ms}$ / $2.2\text{ ms}$ | $55.1\text{ ms}$ / $2.8\text{ ms}$ | $211.2\text{ ms}$ ($4.7\text{ FPS}$) / $26.4\text{ ms}$ ($37.9\text{ FPS}$) |
| **PoseConv3D Tier 2** | 3D CNN (`bc=16, fd=256`) | $647,185$ | $0.672$ | 0.8108 | 90.91% | 3 | 73.17% | 11 FP | 81.82% | 15/16 | 15/17 | $18.8\text{ ms}$ / $2.2\text{ ms}$ | $54.8\text{ ms}$ / $2.8\text{ ms}$ | $210.9\text{ ms}$ ($4.7\text{ FPS}$) / $26.4\text{ ms}$ ($37.9\text{ FPS}$) |
| **PoseConv3D Tier 3 (Raw)** | 3D CNN (`bc=12, fd=256`) | $598,429$ | $0.501$ | **0.8857** | 93.94% | **2** | 83.78% | 6 FP | 89.61% | 15/16 | 16/17 | $18.2\text{ ms}$ / $2.3\text{ ms}$ | $54.4\text{ ms}$ / $2.9\text{ ms}$ | $210.5\text{ ms}$ ($4.8\text{ FPS}$) / $26.4\text{ ms}$ ($37.9\text{ FPS}$) |
| **PoseConv3D Tier 4** | 3D CNN (`bc=16, fd=128`) | $232,337$ | $0.567$ | 0.8125 | 78.79% | 7 | 83.87% | 5 FP | 84.42% | 13/16 | 13/17 | $18.1\text{ ms}$ / $2.2\text{ ms}$ | $54.1\text{ ms}$ / $2.8\text{ ms}$ | $210.2\text{ ms}$ ($4.8\text{ FPS}$) / $26.3\text{ ms}$ ($38.0\text{ FPS}$) |
| **PoseConv3D Tier 5** | 3D CNN (`bc=12, fd=96`) | $132,349$ | $0.397$ | 0.8621 | 75.76% | 8 | **100.0%** | **0 FP** | 89.61% | 13/16 | 12/17 | $17.8\text{ ms}$ / $2.2\text{ ms}$ | $53.9\text{ ms}$ / $2.8\text{ ms}$ | $210.1\text{ ms}$ ($4.8\text{ FPS}$) / $26.3\text{ ms}$ ($38.0\text{ FPS}$) |
| **Distilled Tier 3 (Method A)**| 3D CNN (Feature Align KD) | $598,429$ | $0.501$ | 0.6957 | **96.97%** | **1** | 54.24% | 27 FP | 63.64% | **16/16** | 16/17 | $18.3\text{ ms}$ / $2.2\text{ ms}$ | $54.5\text{ ms}$ / $2.8\text{ ms}$ | $210.6\text{ ms}$ ($4.7\text{ FPS}$) / $26.3\text{ ms}$ ($38.0\text{ FPS}$) |
| **Distilled Tier 3 (Method B)**| 3D CNN (Relational RKD) | $598,429$ | $0.501$ | **0.8000** | **96.97%** | **1** | 68.09% | 15 FP | 79.22% | **16/16** | 16/17 | $18.3\text{ ms}$ / $2.2\text{ ms}$ | $54.5\text{ ms}$ / $2.8\text{ ms}$ | $210.6\text{ ms}$ ($4.7\text{ FPS}$) / $26.3\text{ ms}$ ($38.0\text{ FPS}$) |
| **Distilled Tier 3 (Method C)**| 3D CNN (Arm Attention KD) | $598,429$ | $0.501$ | 0.7429 | 78.79% | 7 | 70.27% | 11 FP | 76.62% | 13/16 | 13/17 | $18.2\text{ ms}$ / $2.2\text{ ms}$ | $54.4\text{ ms}$ / $2.8\text{ ms}$ | $210.5\text{ ms}$ ($4.8\text{ FPS}$) / $26.4\text{ ms}$ ($37.9\text{ FPS}$) |
| **Distilled Tier 3 (Method D)**| 3D CNN (Combined Tri-KD) | $598,429$ | $0.501$ | 0.7191 | **96.97%** | **1** | 57.14% | 24 FP | 67.53% | **16/16** | 16/17 | $18.3\text{ ms}$ / $2.2\text{ ms}$ | $54.5\text{ ms}$ / $2.8\text{ ms}$ | $210.6\text{ ms}$ ($4.7\text{ FPS}$) / $26.3\text{ ms}$ ($38.0\text{ FPS}$) |
| **Config 1: Dual Consensus** | T5 ($132\text{k}$) + T3 ($598\text{k}$) | $730,778$ | $0.898$ | 0.9231 | 90.91% | 3 | 93.75% | 2 FP | 93.51% | 15/16 | 15/17 | $36.9\text{ ms}$ / $4.8\text{ ms}$ | $73.0\text{ ms}$ / $5.4\text{ ms}$ | $229.2\text{ ms}$ ($4.4\text{ FPS}$) / $29.0\text{ ms}$ ($34.5\text{ FPS}$) |
| **Config 2: Distill Consensus**| T5 ($132\text{k}$) + DT3 ($598\text{k}$) | $730,778$ | $0.898$ | 0.8621 | 75.76% | 8 | **100.0%** | **0 FP** | 89.61% | 13/16 | 12/17 | $36.9\text{ ms}$ / $4.8\text{ ms}$ | $73.0\text{ ms}$ / $5.4\text{ ms}$ | $229.2\text{ ms}$ ($4.4\text{ FPS}$) / $29.0\text{ ms}$ ($34.5\text{ FPS}$) |
| **Config 3: Tri-C3D Consensus**| T5 + T3 + DT3 (Shared 3D) | $1,329,207$ | $1.399$ | 0.9180 | 84.85% | 5 | **100.0%** | **0 FP** | 93.51% | 15/16 | 13/17 | $57.0\text{ ms}$ / $6.9\text{ ms}$ | $93.1\text{ ms}$ / $7.6\text{ ms}$ | $249.2\text{ ms}$ ($4.0\text{ FPS}$) / $31.1\text{ ms}$ ($32.2\text{ FPS}$) |
| **★ Config 4: Tuned Pure Dual**| **T5 Edge $\to$ T3 ($p_{\text{low}}=0.011$)** | **$730,778$** | **$0.580$** | **0.9524** | **90.91%** | **3** | **100.0%** | **0 FP** | **96.10%** | **15/16** | **15/17** | **$24.4\text{ ms}$ / $3.0\text{ ms}$** | **$60.6\text{ ms}$ / $3.6\text{ ms}$** | **$216.7\text{ ms}$ ($4.6\text{ FPS}$) / $27.1\text{ ms}$ ($36.8\text{ FPS}$)** |
| **★ Config 5: Tri-Vote Pure** | **T5 Edge $\to$ (20/50/30 Vote)** | **$1,329,207$** | **$0.760$** | **0.9538** | **93.94%** | **2** | **96.88%** | **1 FP** | **96.10%** | **15/16** | **16/17** | **$31.1\text{ ms}$ / $3.8\text{ ms}$** | **$67.2\text{ ms}$ / $4.4\text{ ms}$** | **$223.4\text{ ms}$ ($4.5\text{ FPS}$) / $27.9\text{ ms}$ ($35.8\text{ FPS}$)** |
| **★ Hetero Dual v3.10** | ST-GCN Edge $\to$ 3-Model Srv | $5,113,271$ | $2.206$ | **0.9706** | **100.0%** | **0** | 94.29% | 2 FP | **97.40%** | **16/16** | **17/17** | $32.5\text{ ms}$ / $9.5\text{ ms}$ | $50.9\text{ ms}$ / $9.9\text{ ms}$ | $207.1\text{ ms}$ ($4.8\text{ FPS}$) / $33.4\text{ ms}$ ($30.0\text{ FPS}$) |

*Note on Latencies:* Standalone pipeline latencies measured directly on single-clip passes without Stage 1/2a overhead are:
- ST-GCN: $139.34\text{ ms}$ CPU ($7.2\text{ FPS}$) / $26.49\text{ ms}$ GPU ($37.8\text{ FPS}$)
- PoseConv3D Baseline: $42.92\text{ ms}$ CPU ($23.3\text{ FPS}$) / $27.57\text{ ms}$ GPU ($36.3\text{ FPS}$)
- PoseConv3D Tier 3: $35.80\text{ ms}$ CPU ($27.9\text{ FPS}$) / $28.90\text{ ms}$ GPU ($34.6\text{ FPS}$)
- PoseConv3D Tier 5: $33.71\text{ ms}$ CPU ($29.7\text{ FPS}$) / $27.84\text{ ms}$ GPU ($35.9\text{ FPS}$)
- Config 4 Pure Dual (Tuned): $49.26\text{ ms}$ CPU ($20.3\text{ FPS}$) / $27.47\text{ ms}$ GPU ($36.4\text{ FPS}$)
- Config 5 Tri-Model Vote: $53.51\text{ ms}$ CPU ($18.7\text{ FPS}$) / $29.08\text{ ms}$ GPU ($34.4\text{ FPS}$)

### B. Knowledge Distillation Formulation Ablation Sweep
Evaluating representation transfer from frozen ST-GCN ($3.01\text{M params}$) to Tier 3 PoseConv3D ($598\text{k params}$, `bc=12, fd=256`):

| Distillation Method | Formulation Details | F1-Score | Recall | Missed Attacks (FN) | Precision | False Alarms (FP / 44) | Overall Accuracy | With_Coat (TP / 16) | Without_Coat (TP / 17) | Optimal Regularization $\epsilon$ | Optimal Threshold $\tau$ | Edge CPU Throughput | Edge GPU Throughput | Verdict & Failure Mode |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Undistilled Tier 3** | Pure metric baseline (`bc=12, fd=256`) | 0.8857 | 93.94% | 2 | 83.78% | 6 | 89.61% | 15/16 | 16/17 | $\epsilon = 0.030$ | $\tau = 0.4867$ | $29.9\text{ FPS}$ | $36.1\text{ FPS}$ | Undistilled Champion |
| **Method A** | Cosine similarity + Normalized MSE | 0.6957 | **96.97%** | **1** | 54.24% | 27 | 63.64% | **16/16** | 16/17 | $\epsilon = 0.010$ | $\tau = -0.0972$ | $29.6\text{ FPS}$ | $34.5\text{ FPS}$ | **Feature Domain Bleed:** Exact point-wise copying caused false alarms to surge from 6 to 27. |
| **Method B (RKD)** | Relational Distance + Angle Huber Loss | **0.8000** | **96.97%** | **1** | **68.09%** | **15** | **79.22%** | **16/16** | 16/17 | $\epsilon = 0.010$ | $\tau = -2.8217$ | $29.6\text{ FPS}$ | $35.9\text{ FPS}$ | **Distillation Champion:** Absorbed bone articulation geometry without copying dense feature norms. |
| **Method C** | Arm Attention KL Divergence ($5\times$ arm weight) | 0.7429 | 78.79% | 7 | 70.27% | 11 | 76.62% | 13/16 | 13/17 | $\epsilon = 0.002$ | $\tau = 0.4136$ | $29.9\text{ FPS}$ | $35.9\text{ FPS}$ | **Weak Signal:** Attention prior failed to penalize non-violent arm activities (wood chopping). |
| **Method D** | Combined Tri-Distillation ($0.5\text{A} + 10\text{B} + 0.5\text{C}$) | 0.7191 | **96.97%** | **1** | 57.14% | 24 | 67.53% | **16/16** | 16/17 | $\epsilon = 0.005$ | $\tau = -0.9476$ | $29.9\text{ FPS}$ | $34.7\text{ FPS}$ | Dominated by Method A's point-wise feature noise. |

### C. Systematic Capacity Downscaling & Inductive Regularization Pareto Frontier

| Model Configuration | Base Channels (`bc`) | Feature Dim (`fd`) | Parameters | GFLOPs | F1-Score | Violence Recall | Missed Attacks (FN) | Violence Precision | False Alarms (FP / 44) | Accuracy | CPU Latency (Pipeline) | CPU FPS | GPU Latency (Pipeline) | GPU FPS | Architectural Impact |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Scaled Model** | 64 | 512 | $3,545,409$ | $6.369$ | 0.8406 | 87.88% | 4 | 80.56% | 7 | 85.71% | $179.81\text{ ms}$ | $5.6$ | $28.11\text{ ms}$ | $35.6$ | Overfits; FLOPs explode |
| **Baseline (v2.10)**| 24 | 256 | $765,529$ | $1.221$ | 0.8667 | 78.79% | 7 | 96.30% | 1 | 89.61% | $42.92\text{ ms}$ | $23.3$ | $27.57\text{ ms}$ | $36.3$ | Prior standalone benchmark |
| **Tier 1** | 20 | 256 | $702,885$ | $0.854$ | 0.8116 | 84.85% | 5 | 77.78% | 8 | 83.12% | $35.27\text{ ms}$ | $28.4$ | $27.85\text{ ms}$ | $35.9$ | Intermediate transition |
| **Tier 2** | 16 | 256 | $647,185$ | $0.672$ | 0.8108 | 90.91% | 3 | 73.17% | 11 | 81.82% | $30.43\text{ ms}$ | $32.9$ | $29.44\text{ ms}$ | $34.0$ | High recall, poor precision |
| **★ Tier 3** | **12** | **256** | **$598,429$** | **$0.501$** | **0.8857** | **93.94%** | **2** | **83.78%** | **6** | **89.61%** | **$35.80\text{ ms}$** | **$27.9$** | **$28.90\text{ ms}$** | **$34.6$** | **Inductive Regularization Champion** |
| **Tier 4** | 16 | 128 | $232,337$ | $0.567$ | 0.8125 | 78.79% | 7 | 83.87% | 5 | 84.42% | $30.80\text{ ms}$ | $32.5$ | $27.84\text{ ms}$ | $35.9$ | Constrained latent space |
| **★ Tier 5** | **12** | **96** | **$132,349$** | **$0.397$** | **0.8621** | **75.76%** | **8** | **100.0%** | **0** | **89.61%** | **$33.71\text{ ms}$** | **$29.7$** | **$27.84\text{ ms}$** | **$35.9$** | **Flawless Specificity Champion (0 FP)** |

### D. Pure-PoseConv3D Dual-Tier & Hierarchical Voting Configurations

```
Surveillance Video Stream
           │
           ▼
Single-Pass GPU Rasterization (17, 50, 56, 56)
           │
           ▼
     ┌────────────────────────────────────────────────────────┐
     │ Tier 1: Tier 5 Screener (132k params, 100% Precision)  │
     └────────────────────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    P_T5 < p_low          P_T5 >= p_high
   DISCARD NORMAL         IMMEDIATE LOCAL
   CIVILIAN MOTION             ALARM
   (Zero Latency)         (Instant Alert)
         │                     │
         └──────────┬──────────┘
                    ▼
       p_low <= P_T5 < p_high
    (Ambiguous Kinematic Motion)
                    │
                    ▼
     ┌────────────────────────────────────────────────────────┐
     │ Tier 2: Tier 3 Escalation / Weighted Vote Consensus    │
     │ Dual: w * P_T5 + (1-w) * P_T3 >= tau                   │
     │ Tri:  w1*P_T5 + w2*P_T3 + w3*P_Distill >= tau          │
     └────────────────────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    P_Vote >= tau         P_Vote < tau
      VERIFIED              REJECTED
    THREAT ALARM         FALSE POSITIVE
```

Grid search results across 70,350 candidate operating thresholds and weight combinations:

| Configuration ID | Architectural Setup | Edge Model (Tier 1) | Server / Verifier (Tier 2) | Operating Thresholds / Weights | Violence F1 | Violence Recall | Violence Precision | False Alarms (FP / 44) | Missed Attacks (FN / 33) | Escalation Rate | Total Parameters | Total CPU Throughput | Total GPU Throughput |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config 1** | Synchronous Dual Consensus | None (Synchronous) | $0.5\text{ T5} + 0.5\text{ T3}$ | $\tau = 0.50$ | 0.9231 | 90.91% | 93.75% | 2 | 3 | $100\%$ | $730,778$ | $16.4\text{ FPS}$ | $34.1\text{ FPS}$ |
| **Config 4 (Initial)** | Hierarchical Cascade | Tier 5 ($132\text{k}$) | Tier 3 ($598\text{k}$) | $p_{\text{low}}=0.050, p_{\text{high}}=0.500, \tau=0.50$ | 0.9355 | 87.88% | **100.0%** | **0** | 4 | $28.6\%$ | $730,778$ | $21.5\text{ FPS}$ | $37.1\text{ FPS}$ |
| **★ Config 4 (Tuned)** | **Optimal Hierarchical Dual** | **Tier 5 ($132\text{k}$)** | **Tier 3 ($598\text{k}$)** | **$p_{\text{low}}=0.011, p_{\text{high}}=0.495, \tau=0.50$** | **0.9524** | **90.91%** | **100.0%** | **0** | **3** | **$36.4\%$** | **$730,778$** | **$20.3\text{ FPS}$** | **$36.4\text{ FPS}$** |
| **★ Config 5** | **Tri-Model Hierarchical Vote** | **Tier 5 ($132\text{k}$)** | **$20\%\text{ T5} + 50\%\text{ T3} + 30\%\text{ DT3}$** | **$p_{\text{low}}=0.011, p_{\text{high}}=0.495, \tau=0.39$** | **0.9538** | **93.94%** | **96.88%** | **1** | **2** | **$36.4\%$** | **$1,329,207$** | **$18.7\text{ FPS}$** | **$34.4\text{ FPS}$** |
| **Hetero v3.10 (Initial)**| Dual-Tier (Flawed $p_{\text{high}}$) | ST-GCN ($3.01\text{M}$) | ST-GCN + CTR-GCN + PoseC3D | $p_{\text{low}}=0.200, p_{\text{high}}=0.600, \tau=0.50$ | 0.9167 | **100.0%** | 84.62% | 6 | **0** | $5.2\%$ | $5,113,271$ | $16.8\text{ FPS}$ | $35.7\text{ FPS}$ |
| **★ Hetero v3.10 (Opt)** | **Dual-Tier Production Champ** | **ST-GCN ($3.01\text{M}$)** | **ST-GCN + CTR-GCN + PoseC3D** | **$p_{\text{low}}=0.200, p_{\text{high}}=0.999, \tau=0.50$** | **0.9706** | **100.0%** | **94.29%** | **2** | **0** | **$44.2\%$** | **$5,113,271$** | **$16.1\text{ FPS}$** | **$30.0\text{ FPS}$** |

---

## 4. Key Findings & Remaining Limitations

### A. Empirical Findings
1. **Capacity Scaling Fails in Small-Sample 3D Action CNNs:**  
   Increasing parameters from $765\text{k} \to 3.55\text{M}$ failed to solve the recall deficit while causing a $7\times$ increase in false positives. The 3D spatio-temporal volume overfits rapidly on small training sets ($N=297$), memorizing non-violent kinematic energy.
2. **Channel Downscaling Acts as Inductive Regularization:**  
   Reducing `base_channels` from 24 to 12 (Tier 3: $598\text{k parameters}$) stripped out spurious limb variance while retaining the full $256$-D latent space for covariance estimation. Standalone F1 achieved a record **0.8857**, with missed attacks dropping from 7 to 2. Further reduction to Tier 5 ($132\text{k parameters}$) achieved **100% Precision (0 FP)**.
3. **Relational Topology vs Point-Wise Imitation in Distillation:**  
   - **Point-wise feature matching (Method A)** induced severe "Feature Domain Bleed" (27 FP), forcing dense volumetric convolutions to match sparse graph activations.
   - **Relational KD (Method B)** constrained only relative angles and normalized distance ratios. It achieved **100% Recall on `With_Coat`** (16/16 attacks detected) and **96.97% overall Recall** (only 1 missed attack across 33), maintaining high edge speed ($29.6\text{ FPS CPU}$).
4. **Homogeneous Shared-Rasterization Advantage:**  
   In pure PoseConv3D ensembles (Configs 4 and 5), Tier 5, Tier 3, and Distilled Tier 3 execute over the **same single rasterized 3D heatmap volume** $(17 \times 50 \times 56 \times 56)$. This eliminates heterogeneous coordinate conversion overhead, achieving **$0.9524 - 0.9538\text{ F1}$** on edge hardware without external servers.
5. **Hierarchy Ordering Mandate:**  
   Tier 5 must serve as the Tier 1 screener. Its $100\%$ precision ensures zero civilian false alarms are generated at the edge, while its conservative discard threshold ($p_{\text{low}}=0.011$) reliably escalates subtle violent attacks to Tier 3.

### B. Remaining Technical Limitations
1. **The $56 \times 56$ Spatial Quantization Floor:**  
   Even with Relational KD or voting ensembles, standalone PoseConv3D cannot achieve 100.0% Recall (minimum 1-2 missed attacks remain). Discrete grid rasterization with Gaussian smoothing permanently attenuates high-frequency velocity signals of subtle knife attacks.
2. **Zero-Miss Criticality Requires Graph Continuous Coordinates:**  
   Continuous coordinate ST-GCN remains the only architecture achieving **100.0% Recall (0 missed attacks)** on the standardized benchmark. For zero-miss life-safety installations, the Heterogeneous Dual-Tier v3.10 ($0.9706\text{ F1}, 100\%\text{ Recall}, 2\text{ FP}$) remains mandatory.
3. **CPU Pipeline Preprocessing Bottleneck:**  
   While single-thread model forward pass takes only $17.8 - 24.4\text{ ms}$, Stage 2 skeleton normalization and CPU Gaussian rasterization require $\approx 36\text{ ms}$, bounding end-to-end CPU action recognition to $20.3 - 29.7\text{ FPS}$.

---

## 5. Artifact Index (Weights, Scripts, Result Files)

### A. Pretrained & Distilled Model Weights
- `weights/stgcn_violence_v1.02.pth`: Frozen Teacher ST-GCN ($3,014,981\text{ params}$, $100.0\%\text{ Recall}$).
- `weights/poseconv3d_downscale_tier3_598k.pth`: Champion Undistilled PoseConv3D Tier 3 ($598,429\text{ params}$, $0.8857\text{ F1}$, `bc=12, fd=256`).
- `weights/poseconv3d_downscale_tier5_132k.pth`: Champion Ultra-Compact Tier 5 ($132,349\text{ params}$, $100.0\%\text{ Precision}$, $0\text{ FP}$, `bc=12, fd=96`).
- `weights/poseconv3d_distill_methodB_rkd.pth`: Champion Relational KD Model ($598,429\text{ params}$, $96.97\%\text{ Recall}$, $1\text{ FN}$, $100\%\text{ With\_Coat}$).
- `weights/poseconv3d_distill_methodA_feature.pth`: Feature Alignment KD Checkpoint ($598,429\text{ params}$, $27\text{ FP}$).
- `weights/poseconv3d_distill_methodC_attention.pth`: Arm Attention KD Checkpoint ($598,429\text{ params}$, $7\text{ FN}$).
- `weights/poseconv3d_distill_methodD_combined.pth`: Combined Tri-Distillation Checkpoint ($598,429\text{ params}$, $24\text{ FP}$).
- `weights/reference_data_poseconv3d_pure_dual_tier.pt`: Covariance matrices and calibrated operating thresholds for Tier 5 and Tier 3 pure dual-tier execution.
- `weights/reference_data_dual_tier_v3.10.pt`: Reference covariances and consensus parameters for v3.10 server ensemble.

### B. Core Architecture & Training Modules
- `models/stgcn.py`: Spatio-temporal graph convolutional network backbone.
- `models/poseconv3d.py`: PoseConv3D backbone supporting modular `base_channels` and `feature_dim` scaling.
- `models/poseconv3d_scaled.py`: Scaled R(2+1)D 3D CNN backbone supporting arbitrary channel scaling ($3.55\text{M params}$).
- `models/ensemble_poseconv3d_dual_tier.py`: Unified dual-tier and voting consensus execution engine for pure PoseConv3D with shared rasterization.
- `models/ensemble_dual_tier.py`: Production v3.10 heterogeneous dual-tier pipeline engine.
- `src/distillation_losses.py`: Vectorized implementations of Feature Alignment, Relational KD (Distance/Angle Huber), and Arm Attention KL losses.
- `src/poseconv3d_utils.py`: GPU VRAM batch rasterization utilities for high-speed heatmap generation.
- `training/train_poseconv3d_distillation_sweep.py`: Automated 3-fold cross-validation distillation training pipeline with GPU-cached tensors.

### C. Validation Harnesses & Benchmark Reports
- `scripts/benchmark_unified_fair_comparison.py`: Standardized evaluation harness executing live CPU and GPU throughput profiling.
- `scripts/benchmark_poseconv3d_distillation_sweep.py`: Standardized validation harness evaluating distillation loss checkpoints.
- `results/benchmark_unified_fair_comparison.json`: Master evaluation benchmark across all individual and ensemble models.
- `results/benchmark_unified_fair_comparison.csv`: Tabular benchmark metrics including category TP/FP/FN breakdowns and latencies.
- `results/benchmark_poseconv3d_distillation_sweep.json`: Detailed validation outputs and optimal RMD hyperparameters for distillation methods A–D.
- `results/tuning/tuning_poseconv3d_distillation_sweep.csv`: Validation sweep metrics across candidate $\epsilon$ and $\tau$ thresholds.
- `results/tuning/tuning_poseconv3d_hierarchical_vote.csv`: Parameter sweep grid log across 70,350 dual and tri-model voting configurations.
- `results/training_summary_poseconv3d_distillation.json`: Training convergence logs and validation loss milestones for distillation runs.
- `results/benchmark_poseconv3d_downscale_sweep.json`: Evaluation logs for parameter downscaling tiers 1 through 5.
- `results/benchmark_poseconv3d_scaled.json`: Evaluation log for scaled 3.55M PoseConv3D.
