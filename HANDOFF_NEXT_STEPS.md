# Project Handoff: Next-Generation F1-Maximization & Multi-Model Super-Ensemble

**Workspace Directory:** `D:\Intern AI Project\hierarchical-threat-detection-edge-ai`  
**Python Runtime:** `.\.venv\Scripts\python.exe` (PyTorch 2.5.1 + CUDA 12.1, NVIDIA GeForce RTX 3090 24GB)  
**Date / Timestamp:** September 2026

---

## 1. Executive Context & Paradigm Shift

### A. The Research Pivot: From Speed Optimization to Pure F1-Score Maximization
Throughout Phase 1 and the PoseConv3D exploration, extensive live benchmarking (with zero hardcoded numbers) established key empirical realities:
1. **End-to-End Pipeline Latency Bottleneck:**  
   The full video surveillance pipeline is dominated by Stage 1 Weapon Detection (`YOLO26s` distilled) and Stage 2a Pose Extraction (`YOLO26s-pose`), taking **$23.53\text{ ms}$ on GPU** and **$156.13\text{ ms}$ on CPU** per frame.
2. **Skeletal Action Recognition Latency Impact:**  
   In comparison, the Stage 2b action models (ST-GCN at $3.7\text{ ms}$ GPU / $12.0\text{ ms}$ CPU; PoseConv3D Tier 5 at $2.2\text{ ms}$ GPU / $17.8\text{ ms}$ CPU; CTR-GCN at $11.0\text{ ms}$ GPU / $25.3\text{ ms}$ CPU) constitute only a minor fraction of the total system runtime.
3. **The Phase 1 CPU Anomaly Resolved:**  
   The historical belief that PoseConv3D was $3.2\times$ faster on CPU was traced to an apples-to-oranges addition in `scripts/benchmark_poseconv3d_v2.10.py` (which accidentally added GPU Stage 1 & 2a times to CPU PoseConv3D). In reality, 3D heatmap rasterization takes $36.14\text{ ms}$ on CPU, making GCN graph representations faster on CPU ($2.43\text{ ms}$). On GPU, PoseConv3D is faster ($2.16\text{ ms}$ vs $3.68\text{ ms}$).

> [!IMPORTANT]
> **Directive for the Fresh Session:**  
> Since model forward latency variance does not materially bottleneck the system, **we shift 100% of our engineering focus to maximizing the Violence Detection F1-Score**. Our mission is to push past the previous milestone champion (**$0.9706\text{ F1}$**) by scaling models, exploring novel architectures, and constructing multi-model super-ensembles.

---

## 2. Current State of the System & Benchmark Baselines

All evaluations use the standardized 77-video / 148-clip validation suite (`With_Coat`: 16 violent videos, `Without_Coat`: 17 violent videos, `OOD`: 44 civilian non-violent videos; Total: 33 violent assault items + 44 civilian negative items).

### Master Baseline Summary (Standardized Live Evaluation)

| Architecture / System | Parameters | Violence F1 | Recall (TP / 33) | Precision | False Alarms (FP / 44) | Status / Role |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **ST-GCN Baseline (v1.08)** | $3.01\text{M}$ | **0.9167** | **100.0% (33/33)** | 84.62% | 6 FP | Safety baseline; zero missed attacks, moderate false alarms. |
| **CTR-GCN (v2.00)** | $1.33\text{M}$ | **0.8649** | 96.97% (32/33) | 78.05% | 9 FP | Channel-refined dynamic graph; high recall (1 FN), higher FP. |
| **PoseConv3D Tier 5 (132k)** | $132\text{k}$ | **0.8621** | 75.76% (25/33) | **100.0%** | **0 FP (Zero)** | Precision anchor; zero false alarms on 44 civilian gestures. |
| **PoseConv3D Tier 3 (598k)** | $598\text{k}$ | **0.8857** | 93.94% (31/33) | 83.78% | 6 FP | Standalone PoseConv3D record; catches 31/33 attacks. |
| **Distilled Tier 3 (Method B)** | $598\text{k}$ | **0.8000** | 96.97% (32/33) | 68.09% | 15 FP | Relational KD; $100\%$ recall on `With_Coat`, catches 32/33 attacks. |
| **SkateFormer (v2.20)** | $\approx 1.8\text{M}$ | Evaluated in P2 | High Recall | Variable | — | Skeletal Transformer backbone; available in `weights/skateformer_violence_v2.20.pth`. |
| **Version 3.10 Dual-Tier Champion** | $5.11\text{M}$ | **0.9706** | **100.0% (33/33)** | **94.29%** | **2 FP** | **All-time project record:** ST-GCN Edge $\to$ Tri-Model Consensus (45% ST-GCN + 35% CTR-GCN + 20% PoseC3D). |
| **Config 5 Pure-PoseC3D Vote** | $1.33\text{M}$ | **0.9538** | 93.94% (31/33) | 96.88% | 1 FP | Pure PoseConv3D record: Tier 5 Edge $\to$ Tri-PoseConv3D consensus. |

---

## 3. Core Objectives & Hypotheses for the Next Session

### Objective 1: Scale Parameter & Architecture Tuning for CTR-GCN
* **Problem:** Standalone CTR-GCN (`models/ctrgcn.py`) currently achieves high recall ($96.97\%$, only $1\text{ FN}$), but has $9\text{ False Alarms}$ ($78.05\%\text{ Precision}$), pulling its F1 down to $0.8649$.
* **Hypothesis 1.1 (Capacity / Channel Width Scaling):**  
  CTR-GCN currently uses `[64, 128, 256]` across 9 blocks ($1.33\text{M}$ params).  
  - What happens if we scale down to `[32, 64, 128]` (analogous to the inductive regularization success of PoseConv3D Tier 3)? Will narrower channel widths suppress civilian false alarms?
  - What happens if we scale up to `[96, 192, 384]` or deeper layers?
* **Hypothesis 1.2 (Spatial Attention & Temporal Kernel Tuning):**  
  CTR-GCN applies an arm-weighting factor of $5.0\times$ to joints 5-10. Is $5.0\times$ optimal, or does it over-sensitize upper-body civilian motions (e.g. stretching, tool swinging)?
* **Hypothesis 1.3 (RMD Manifold & OOD Gating Tuning):**  
  Tune the ridge shrinkage $\epsilon$ and decision threshold $\tau$ for CTR-GCN features to filter out civilian outliers.

### Objective 2: Super-Ensembles Beyond 3 Models (4-Model & 5-Model Combinations)
* **Problem:** Version 3.10 used a 3-model weighted consensus (ST-GCN + CTR-GCN + PoseConv3D) to achieve $0.9706\text{ F1}$.
* **Hypothesis 2.1 (Incorporating SkateFormer):**  
  We have a trained Skeletal Transformer model: `weights/skateformer_violence_v2.20.pth`. As a self-attention transformer, its inductive bias is distinct from both GCNs (local topological message-passing) and 3D CNNs (Euclidean grid convolutions). Adding SkateFormer as a 4th voter could provide complementary error correction.
* **Hypothesis 2.2 (Incorporating PoseConv3D Tier 5 as a Precision Veto):**  
  PoseConv3D Tier 5 has **$100.0\%\text{ Precision}$ (0 FP)**. Using it as a precision regularizer or high-confidence veto in the Tier 2 server consensus can eliminate the remaining 2 false alarms in v3.10, aiming for **$1.000\text{ F1}$ (33/33 attacks detected, 0 civilian false alarms)**.
* **Hypothesis 2.3 (Systematic Fusion Algorithms):**  
  Evaluate:
  - Multi-model weighted soft voting: $\sum w_i P_i \ge \tau$
  - Multi-tier cascaded gating: Edge screener $\to$ Server GCN committee $\to$ 3D CNN / Transformer arbiter
  - Confidence-weighted consensus: Weighting each model's vote by its distance from its calibrated decision boundary.

### Objective 3: Exploring Additional or Alternative SOTA Backbones
* Explore adding modern SOTA skeletal architectures (e.g., HD-GCN, InfoGCN, or TD-GCN) if existing model ensembles plateau.

---

## 4. Key Repository Files & Reference Map

### Production Models & Checkpoints
- **Stage 1 Weapon Detector:** [`weights/yolo_weapon_distilled.pt`](weights/yolo_weapon_distilled.pt) (`YOLO26s` distilled, NMS-free).
- **Stage 2a Pose Extractor:** [`weights/yolo26s-pose.pt`](weights/yolo26s-pose.pt) (`YOLO26s-pose`, 17 COCO keypoints).
- **ST-GCN Weights:** [`weights/stgcn_violence_v1.02.pth`](weights/stgcn_violence_v1.02.pth) & [`weights/reference_data_rmd_v1.08.pt`](weights/reference_data_rmd_v1.08.pt).
- **CTR-GCN Weights:** [`weights/ctrgcn_violence_v2.00.pth`](weights/ctrgcn_violence_v2.00.pth) & [`weights/reference_data_ctrgcn_v2.00.pt`](weights/reference_data_ctrgcn_v2.00.pt).
- **SkateFormer Weights:** [`weights/skateformer_violence_v2.20.pth`](weights/skateformer_violence_v2.20.pth) & [`weights/reference_data_skateformer_v2.20.pt`](weights/reference_data_skateformer_v2.20.pt).
- **PoseConv3D Tier 3 (598k):** [`weights/poseconv3d_downscale_tier3_598k.pth`](weights/poseconv3d_downscale_tier3_598k.pth).
- **PoseConv3D Tier 5 (132k, 0 FP):** [`weights/poseconv3d_downscale_tier5_132k.pth`](weights/poseconv3d_downscale_tier5_132k.pth).
- **PoseConv3D Distilled Method B (RKD):** [`weights/poseconv3d_distill_methodB_rkd.pth`](weights/poseconv3d_distill_methodB_rkd.pth).
- **Dual-Tier Production Weights (v3.10):** [`weights/reference_data_dual_tier_v3.10.pt`](weights/reference_data_dual_tier_v3.10.pt).

### Architecture Definitions
- **CTR-GCN:** [`models/ctrgcn.py`](models/ctrgcn.py)
- **ST-GCN:** [`models/stgcn.py`](models/stgcn.py)
- **SkateFormer:** [`models/skateformer.py`](models/skateformer.py)
- **PoseConv3D Scaled / Downscaled:** [`models/poseconv3d_scaled.py`](models/poseconv3d_scaled.py)
- **Dual-Tier Ensemble Engine (v3.10):** [`models/ensemble_dual_tier.py`](models/ensemble_dual_tier.py)
- **Pure PoseConv3D Dual-Tier Engine:** [`models/ensemble_poseconv3d_dual_tier.py`](models/ensemble_poseconv3d_dual_tier.py)

### Evaluation & Training Harnesses
- **Unified Fair Benchmark:** [`scripts/benchmark_unified_fair_comparison.py`](scripts/benchmark_unified_fair_comparison.py) (Live timers, zero hardcoding).
- **CTR-GCN Training Script:** [`training/train_ctrgcn_v2.00.py`](training/train_ctrgcn_v2.00.py)
- **CTR-GCN Benchmark Script:** [`scripts/benchmark_ctrgcn_v2.00.py`](scripts/benchmark_ctrgcn_v2.00.py)
- **SkateFormer Benchmark Script:** [`scripts/benchmark_skateformer_v2.20.py`](scripts/benchmark_skateformer_v2.20.py)

---

## 5. Strict Operational Directives for the New Agent (from `AgentRule.md`)

1. **Working Directory:** All work must occur inside `D:\Intern AI Project\hierarchical-threat-detection-edge-ai`.
2. **Forbidden Directories:** NEVER touch, scan, or import from `Messy (Don't Touch)`, `Failed Attempt`, or `Clean`.
3. **Python Runtime:** Always execute scripts with `.\.venv\Scripts\python.exe`.
4. **Zero Hardcoded Numbers:** All latencies, F1-scores, and performance numbers must be computed dynamically via running scripts. Never hardcode constant latencies or benchmark outputs.
5. **Standardized 77-Video Validation Suite:** All action recognition models and ensembles must be evaluated on the 77-video validation set (`With_Coat`: 16 violent, `Without_Coat`: 17 violent, `OOD`: 44 non-violent).
6. **Preserve Established Milestones:** Do not overwrite or modify frozen milestone scripts or production weights. Use modular versioning (`_vX.XX`).

---

## 6. Recommended Immediate Steps for the Fresh Conversation

1. **Step 1:** Verify the environment and check the baseline performance of SkateFormer (`weights/skateformer_violence_v2.20.pth`) on the 77-video validation suite using the unified evaluation protocol.
2. **Step 2:** Analyze CTR-GCN architecture (`models/ctrgcn.py`) and formulate a scaling / parameter sweep experiment (scaling channel widths `[32, 64, 128]` vs `[64, 128, 256]` vs `[96, 192, 384]`, and tuning joint attention weights).
3. **Step 3:** Implement an expanded Multi-Model Consensus / Hierarchical Evaluator combining ST-GCN, CTR-GCN, SkateFormer, PoseConv3D Tier 3, and PoseConv3D Tier 5.
4. **Step 4:** Run hyperparameter optimization on model voting weights and decision thresholds to search for the ultimate configuration that beats $0.9706\text{ F1}$.
