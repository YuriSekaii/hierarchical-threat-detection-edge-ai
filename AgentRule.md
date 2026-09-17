# Agent Directives & Operational Rules: Hierarchical Threat Detection & Edge AI

> **CRITICAL DIRECTIVE FOR ALL FUTURE AI AGENTS:**  
> Read this file alongside [`PROJECT_REPORT_PHASE1_PHASE2.md`](PROJECT_REPORT_PHASE1_PHASE2.md) before executing any task or modifying any code.

---

## 1. Final Production Models

Use **v3.1** and **v4.00** as the final model:
- **Stage 1 (Threat Object & Weapon Detection):** **Version 4.00** (`weights/yolo_weapon_distilled.pt`).
  - Compact distilled student detector (`YOLO26s`) executing native One-to-One Hungarian Bipartite Matching (`end2end=True`).
  - Production operating point: $\text{Conf} = 0.45, \text{IoU} = 0.20$ (86.88% F1, 89.78% Precision, 84.17% Recall, 12.86 ms).
- **Stage 2 (Biomechanical Action Recognition & OOD Gating):** **Version 3.10** (`models/ensemble_dual_tier.py`, `weights/reference_data_dual_tier_v3.10.pt`).
  - Dual-Tier Ensemble (Tier 1 Edge: ST-GCN + RMD; Tier 2 Server: ST-GCN + CTR-GCN + PoseConv3D consensus).
  - All-time record champion: 0.9706 F1, 100% Recall (0 missed attacks), 94.29% Precision, only 2 false alarms.

---

## 2. Workspace Boundaries & Environment Rules

| Component | Path | Strict Policy |
|---|---|---|
| **Primary Project Workspace** | `D:\Intern AI Project\hierarchical-threat-detection-edge-ai` | **ALL work happens here.** All code (`src/`, `models/`), benchmark scripts (`scripts/`), weights (`weights/`), tuning artifacts (`results/`), and docs MUST be created and edited inside this folder only. |
| **Official Ground Truth Dataset** | `data/Ground_Truth_Dataset/` | Contains 715 authentic surveillance test images (`With_Coat`: 359, `Without_Coat`: 235, `OOD`: 121) with 240 annotated weapons. All Stage 1 models must evaluate here at $\text{IoU}=0.20$. |
| **Local Action Dataset** | `data/` & `data/Validate/` | Contains 297 training clips (`Cut-Down`, `Stab`, `Thrust`) and 77 validation videos / 148 clips. |
| **Python Runtime** | `.\.venv\Scripts\python.exe` | Dedicated virtualenv (PyTorch 2.5.1 + CUDA 12.1, NVIDIA GeForce RTX 3090 24GB). **Always use this python binary.** |
| **Forbidden Folders** | `D:\Intern AI Project\Messy (Don't Touch)`<br>`D:\Intern AI Project\Failed Attempt`<br>`D:\Intern AI Project\Clean` | **STRICT BOUNDARY: NEVER touch, access, import, scan, or reference these directories under any circumstances.** |

---

## 3. Code Reuse & Continuity Directives

1. **Do NOT Recreate Anything from Scratch — Reuse Existing Code:**
   - Preprocessing: Always import and use `normalize_skeleton_clip` from `src.skeleton_utils`. Never write new coordinate normalizers or bounding-box rasters from scratch.
   - Datasets: All action recognition models must inherit from `ActionDataset` or `TripletActionDataset` in `src.dataset`.
   - Evaluation: Always reuse validated benchmark harnesses (e.g. `scripts/validate_stage1_ground_truth.py`, `scripts/validate_nms_free_ground_truth.py`).
2. **Only Modify the Specific Needed Part:**
   - When building a new version or running an ablation, keep all established modules intact and alter ONLY the exact backbone, prompt logic, or metric under study.
3. **Modular File Versioning & Persistence:**
   - Never overwrite prior working versions.
   - Follow strict naming conventions:
     - Inference engines: `src/inference_<paradigm>_vX.XX.py`
     - Architectures: `models/<architecture>.py`
     - Evaluation scripts: `scripts/validate_<paradigm>_ground_truth.py`
     - Weights: Save checkpoints to `weights/`
     - Tuning logs: Save parameter sweeps to `results/tuning/`
     - Benchmark reports: Save JSON outputs to `results/ground_truth_benchmark/`

---

## 4. Benchmarking & Reporting Standards

1. **Mandatory Reporting Metrics:**
   - **Stage 1 (Weapon Detection):** Every report must report:
     - `Overall F1-Score`, `With_Coat F1`, `Without_Coat F1`
     - `Threat Recall` (with TP / 240 count)
     - `Threat Precision` (with TP / (TP + FP) count)
     - `Overall Accuracy`
     - `Civilian Specificity (OOD)`
     - `False Alarms (OOD FP / 121 and Total FP)`
     - `Missed Weapons (FN / 240)`
     - `Inference Latency (ms and FPS)` on RTX 3090 and GTX 1060
     - `Active VRAM Footprint`
   - **Stage 2 (Action Recognition):** Every report must report:
     - `F1-Score`
     - `Total Pipeline Latency (GPU & CPU)` side-by-side
     - `Violence Detection Recall` (TP / 33)
     - `Violence Detection Precision`
     - `Overall Classification Accuracy`
     - `Civilian False Alarms (FP / 44)`
     - `Missed Attacks (FN / 33)`
     - `OOD Gating Latency (CPU $\mu$s)`
2. **Mandatory Hyperparameter Tuning Table:**
   - For every parameter sweep (shrinkage $\epsilon$, decision threshold $\tau$, proposal confidence, interaction radius, etc.), you **MUST** provide a dedicated impact table showing:
     `Recall`, `Precision`, `Accuracy`, `F1-Score`, `FP`, `FN`.
3. **Ground Truth Validation Protocol:**
   - Stage 1 models must evaluate against `data/Ground_Truth_Dataset/` across all 3 categories (`With_Coat`, `Without_Coat`, `OOD`) under exact bounding box matching $\text{IoU} = 0.20$.
