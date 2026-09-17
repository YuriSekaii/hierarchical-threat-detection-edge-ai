# Project Archive Notice

The complete engineering logs, historical iterations (v1.00 through v3.00), exploratory model weights, cross-validation folds, and intermediate research reports have been archived externally at:

```text
D:\Intern AI Project\HTD_Archive_Reports_and_Experiments\
```

### Archived Components Overview

1. **`reports/`**: 17 technical investigation reports, roadmap handoffs, and changelogs (`PROJECT_REPORT_*.md`, `COMPACT_REPORT_*.md`, `VERSIONS.md`, `VERSIONS_PHASE2.md`, etc.).
2. **`legacy_src/`**: Historical runtime pipelines (`inference_realtime_v1.01.py` through `v3.00.py`, SA2VA multimodal scripts, and legacy OOD metrics).
3. **`legacy_scripts/`**: 34 parameter sweep scripts, diagnostic frame extractors, and legacy ablation benchmarks.
4. **`legacy_models/`**: Duplicate/exploratory architecture definitions (now consolidated into canonical models).
5. **`training_pipelines/`**: Full cross-validation and training suites (`training/`).
6. **`exploratory_weights/`**: Multi-fold weights (`*_fold1/2/3.pth`), unquantized models, and retired calibration files.
7. **`evidence_and_data/`**: Ground-truth test annotations and visual diagnostic frames.

### Canonical Production Repository

This GitHub repository contains the consolidated production architecture:
- **Champion 2 Progressive Staircase Cascade (v5.10)** (`src/inference_staircase_cascade_v5.10.py`)
- **Stage 1 Hungarian NMS-Free Detector** (`src/inference_nms_free_v4_50.py`)
- **Stage 1-Guard Contact HOI Engine** (`src/inference_hoi_contact_v4_30.py`)
- **Backbone Implementations**: ST-GCN, CTR-GCN, PoseConv3D, SkateFormer, Super-Ensemble, and Dual-Tier Consensus (`models/`)
- **Production Weights**: Official distilled and calibrated checkpoints (`weights/`)
- **Master Documentation**: Complete technical synthesis in [`README.md`](README.md) and [`FULL_FINAL_REPORT.md`](FULL_FINAL_REPORT.md).
