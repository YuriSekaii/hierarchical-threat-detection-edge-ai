# Modification Report: Ampere Edge Speed Optimizations (Steps 1–3)

**Target hardware:** Jetson Orin Nano (`sm_87`, TRT 10.7–10.16) and Edge PC (RTX 3050).
**Lab benchmark environment:** Dual NVIDIA GeForce RTX 3090, Windows 11, `.venv` (`torch 2.5.1+cu121`, `tensorrt 11.3.0.99`, `ultralytics 8.4.9`).
**Baselines preserved:**
- Stage 1: `weights/stage1_repro846_kdgtclean_p250.{pt,onnx,engine}` (YOLO26s NMS-free, `384x640`, `conf=0.20`, `nms_iou=0.50`)
- Stage 2a: `weights/yolo26s-pose.{pt,onnx,engine}` (`384x640`, ByteTrack)
- Stage 2b: `PoseConv3D`, `ST-GCN` (kept in FP16, untouched)

---

## Step 1 — Portable GPU Letterbox + Normalize

### 1. Question
Can replacing Ultralytics CPU `LetterBox` + BGR->RGB + HWC->CHW + `/255` with a portable PyTorch CUDA preprocessor (`src/gpu_preprocess.py`) accelerate Stage 1 and Stage 2a at zero quality drop, and what is the end-to-end bottleneck when passing CUDA tensors to Ultralytics `8.4.9`?

### 2. What was implemented & run
- Added [`src/gpu_preprocess.py`](../src/gpu_preprocess.py): `letterbox_bgr_to_tensor`, `unletterbox_boxes`, `unletterbox_keypoints` (preserving `(0, 0)` missing COCO keypoints).
- Wired `gpu_preprocess: bool = True` toggle into [`src/stage1_detector.py`](../src/stage1_detector.py) and [`src/stage2a_pose_tracker.py`](../src/stage2a_pose_tracker.py).
- Added unit tests in [`tests/test_gpu_preprocess.py`](../tests/test_gpu_preprocess.py) and parity/latency benchmark [`scripts/bench_step1_gpu_preprocess.py`](../scripts/bench_step1_gpu_preprocess.py).

Commands run:
```powershell
.\.venv\Scripts\python.exe tests\test_gpu_preprocess.py
.\.venv\Scripts\python.exe scripts\validate_stage1_ground_truth.py
.\.venv\Scripts\python.exe scripts\bench_step1_gpu_preprocess.py
```

### 3. Results

#### 3a. Stage 1 Ground Truth Parity (`data/Ground_Truth_Dataset`, 715 images, `conf=0.20`, `nms_iou=0.50`, `IoU=0.20`, `384x640`)

| Category | Mode | TP | FP | FN | TN | Precision | Recall | F1 | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| OOD | CPU | 0 | 52 | 0 | 72 | 0.00% | 0.00% | 0.00% | 58.06% | 58.06% |
| OOD | GPU | 0 | 51 | 0 | 73 | 0.00% | 0.00% | 0.00% | 58.87% | 58.87% |
| With_Coat | CPU | 131 | 55 | 20 | 153 | 70.43% | 86.75% | 77.74% | 73.56% | 79.11% |
| With_Coat | GPU | 131 | 55 | 20 | 153 | 70.43% | 86.75% | 77.74% | 73.56% | 79.11% |
| Without_Coat | CPU | 84 | 59 | 5 | 102 | 58.74% | 94.38% | 72.41% | 63.35% | 74.40% |
| Without_Coat | GPU | 84 | 57 | 5 | 104 | 59.57% | 94.38% | 73.04% | 64.60% | 75.20% |
| **Overall** | **CPU** | **215** | **166** | **25** | **327** | **56.43%** | **89.58%** | **69.24%** | **66.33%** | **73.94%** |
| **Overall** | **GPU** | **215** | **163** | **25** | **330** | **56.88%** | **89.58%** | **69.58%** | **66.94%** | **74.35%** |

- **Coordinate drift (376 matched boxes):** median `0.2604 px` original (`0.0625 px` at `384x640`), max `6.6406 px` original (`1.5937 px` at `384x640`).
- **Borderline flips:** 7 images out of 715 had detection-count differences, all near `conf=0.20` (`0.2012`–`0.2194`). Zero TPs or FNs changed.

#### 3b. Stage 1 Lab Holdout + Confuser Parity (`crop_4_3` 1920x1440, every 3rd frame, `conf=0.20`)

| Mode | Nov 14 Knife Frame Mean / Min | Nov 14 Knife Win Mean / Min | Take 002 (Oct 31) Frame Share | Phone 050 FA Share | Remote 056 FA Share |
|---|---|---|---|---|---|
| CPU | 37.19% / 3.61% | 51.22% / 3.96% | 77.28% | 3.61% | 5.28% |
| GPU | 37.17% / 3.61% | 51.22% / 3.96% | 77.18% | 3.61% | 5.28% |

#### 3c. Stage 2a Pose + ByteTrack Parity (150 consecutive 4:3 frames from `2026-10-31/raw/002_checkerboard_142846.avi`)

| Mode | Frames Detected | Unique Track IDs | ID Switches | BBox IoU (Med / Min) | BBox Drift px (Med / Max) | Kpt L2 Drift px (Med / Max) |
|---|---|---|---|---|---|---|
| CPU | 150 / 150 | 1 | 0 | — | — | — |
| GPU | 150 / 150 | 1 | 0 | 0.99929 / 0.99678 | 0.251 / 0.839 | 0.117 / 2.578 |

#### 3d. Latency & FPS on RTX 3090 (`cuda:0`, 30 warmup + 200 timed iterations)

| Resolution | Prep CPU (ms / FPS) | Prep GPU (ms / FPS) | Stage 1 CPU (ms / FPS) | Stage 1 GPU (ms / FPS) | Stage 2a CPU (ms / FPS) | Stage 2a GPU (ms / FPS) |
|---|---|---|---|---|---|---|
| `1920x1440_slice` (non-contig) | 1.463 / 683.4 | 4.511 / 221.7 | 4.456 / 224.4 | 8.191 / 122.1 | 4.801 / 208.3 | 8.408 / 118.9 |
| `1280x720` | 1.089 / 918.1 | 0.970 / 1030.5 | 3.682 / 271.6 | 3.901 / 256.4 | 3.773 / 265.1 | 4.034 / 247.9 |
| `3840x2160` | 1.457 / 686.5 | 5.267 / 189.9 | 4.113 / 243.2 | 8.587 / 116.5 | 4.401 / 227.2 | 8.815 / 113.4 |

**Ultralytics CUDA-tensor micro-overhead (`1920x1440`):**
- `AutoBackend` forward (`384x640`): `1.794 ms` (`557.5 FPS`)
- `LoadTensor._single_check` (`im.max()` scalar sync): `0.136 ms`
- `ops.convert_torch2numpy_batch` (`384x640` D2H copy in `postprocess`): `0.376 ms`

### 4. Step 1 Findings
1. **Quality & tracking parity is exact:** F1 improved slightly (`69.24%` -> `69.58%`, -3 FPs, identical 215 TPs and 25 FNs), lab holdout & confuser FA shares match within 0.1%, and ByteTrack has 0 ID switches with `0.117 px` median keypoint drift.
2. **Host-to-device copy of full-res frames dominates on desktop PCIe:** On Windows RTX 3090, uploading a full `1920x1440x3` (8.3 MB, non-contiguous `crop_4_3` slice requiring `np.ascontiguousarray`) or `3840x2160x3` (24.9 MB) pageable numpy array via `torch.from_numpy(...).to("cuda:0")` takes `4.51–5.27 ms`, which is **3× slower** than CPU `cv2.resize` (`1.46 ms`, which only uploads `384x480x3` = 0.55 MB). Even at `1280x720` where `prep_gpu` (`0.970 ms`) beats `prep_cpu` (`1.089 ms`), Ultralytics' `LoadTensor._single_check` + `convert_torch2numpy_batch` adds `0.512 ms`, leaving end-to-end `stage1_gpu` (`3.901 ms`) slightly behind `stage1_cpu` (`3.682 ms`).
   - **Auto-default policy applied:** `default_gpu_preprocess(device)` returns `True` automatically only on Jetson/Tegra unified-memory devices (`/etc/nv_tegra_release` or `"orin"`/`"tegra"` in GPU name) and `False` by default on discrete PCIe GPUs (RTX 3050/3090), while remaining explicitly overridable via `gpu_preprocess=True|False`.
3. **What was not checked:** Actual Jetson Orin Nano (`sm_87`, unified LPDDR5 memory + slower ARM Cortex-A78AE CPU) and RTX 3050 Edge PC timings.

### 5. Files
- [`src/gpu_preprocess.py`](../src/gpu_preprocess.py)
- [`src/stage1_detector.py`](../src/stage1_detector.py)
- [`src/stage2a_pose_tracker.py`](../src/stage2a_pose_tracker.py)
- [`tests/test_gpu_preprocess.py`](../tests/test_gpu_preprocess.py)
- [`scripts/bench_step1_gpu_preprocess.py`](../scripts/bench_step1_gpu_preprocess.py)
- [`results/amp_edge_speed/step1_gpu_preprocess.json`](../results/amp_edge_speed/step1_gpu_preprocess.json)

---

## Step 2a — Mixed-Precision INT8 PTQ Gate (Entropy Calibration + Residual-Add Patch + Autocast)

### 1. Question
Does Mixed-Precision INT8 PTQ (entropy/histogram calibration on 600 Stage 1 training frames, residual-add Q/DQ patching in `Bottleneck`/`SPPF`, FP16 retention on stem/C2PSA/PSABlock/Detect+Pose heads via `modelopt.onnx.autocast`) compile cleanly into TensorRT INT8+FP16 kernels without falling back to FP32, and does it preserve small-knife recall and false-alarm rates?

### 2. What was implemented & run
- Installed `nvidia-modelopt[onnx]==0.33.0` in `.venv` (keeping `torch 2.5.1+cu121`, `tensorrt 11.3.0.99`, `ultralytics 8.4.9` intact; `0.48.0` would have overwritten `torch` with CPU `2.15.0`).
- Added [`scripts/quantize_stage1_ptq.py`](../scripts/quantize_stage1_ptq.py):
  - Stratified calibration loader from `data/stage1_negatives/lab_gt_clean/train.txt` (600 frames Stage 1 / 500 frames Stage 2a; hard assertions blocking `Ground_Truth` and `Take 002`).
  - `HistogramCalibrator(method='entropy')` on 80 activation quantizers + `MaxCalibrator` per-channel weights on 65 weight quantizers.
  - Patched 14 residual shortcut `TensorQuantizer`s into `Bottleneck(add=True)` and `SPPF(add=True)`.
  - Locked `*model.0.*`, `*model.10.*`, `*model.22.m.0.1.*`, `*model.23.*` to FP16, exported explicit Q/DQ ONNX (`Q=147, DQ=147`, zero `QLinearConv`), ran `modelopt.onnx.autocast` (`data_max=2048` so `/model.0` stem converts to FP16: `1163/1180` nodes = `98.56%` FP16), and compiled `STRONGLY_TYPED` TensorRT engines.
- Enabled `ProfilingVerbosity.DETAILED` in [`src/engine_builder.py`](../src/engine_builder.py).
- Added [`scripts/bench_step2_int8.py`](../scripts/bench_step2_int8.py) for GT confidence sweep, lab holdout + confusers, CUDA-event pure GPU latency, and Stage 2a pose parity.

Commands run:
```powershell
.\.venv\Scripts\python.exe -u scripts\quantize_stage1_ptq.py
.\.venv\Scripts\python.exe -u scripts\quantize_stage1_ptq.py --pose
.\.venv\Scripts\python.exe -u scripts\bench_step2_int8.py --pose-int8-engine weights\yolo26s-pose_int8ptq.engine
```

### 3. Results

#### 3a. TensorRT Engine Size & `DETAILED` Layer Precision Profile

| Engine | Size (MB) | Total Layers | Conv Inputs (`Int8` / `Half` / `Float`) | Fused Q-Conv | Reformat Nodes |
|---|---|---|---|---|---|
| Stage 1 FP16 baseline (`stage1_repro846_kdgtclean_p250.engine`) | 62.0 | 271 | FP16 baseline | 0 | 85 (names-only) |
| Stage 1 INT8 PTQ (`stage1_repro846_kdgtclean_p250_int8ptq.engine`) | **39.42** | 343 | **65 / 37 / 0** | 74 | 98 |
| Stage 2a FP16 baseline (`yolo26s-pose.engine`) | 70.0 | 274 | FP16 baseline | — | — |
| Stage 2a INT8 PTQ (`yolo26s-pose_int8ptq.engine`) | **47.14** | 382 | **65 / 43 / 0** | 80 | 116 |

#### 3b. Stage 1 Latency on RTX 3090 (`cuda:0`, batch=1, 50 warmup + 500 CUDA-event iters)

| Model | Input Dtype | Pure GPU Mean / Median / p95 (ms) | Pure GPU FPS | E2E `1920x1440` Mean (ms / FPS) |
|---|---|---|---|---|
| Stage 1 FP16 baseline | `float32` | 1.796 / 1.721 / 2.172 | 556.7 | 3.715 / 269.2 |
| Stage 1 INT8 PTQ | `float16` | 2.228 / 2.134 / 2.715 | 448.8 | 4.032 / 248.0 |
| Stage 2a FP16 baseline | `float32` | 1.795 / 1.747 / 2.051 | 557.1 | 4.152 / 240.9 |
| Stage 2a INT8 PTQ | `float16` | 2.345 / 2.314 / 2.670 | 426.5 | 4.611 / 216.9 |

*Note:* On RTX 3090 (10,496 CUDA cores), batch=1 YOLO26s is kernel-launch and Q/DQ reformat bound (`0.81x` pure GPU here, consistent with Ultralytics' 0.89x–1.19x desktop GPU study); on Jetson Orin Nano (`sm_87`, 32 Tensor Cores), the 65 INT8 Convs are compute-bound (~1.20–1.35x expected).

#### 3c. Stage 1 GT Confidence Threshold Sweep (`data/Ground_Truth_Dataset`, 715 images, `IoU=0.20`, `nms_iou=0.50`)

| Model | Conf | Precision | Recall | F1 | OOD FP | With_Coat R | Without_Coat R | TP / FP / FN / TN |
|---|---|---|---|---|---|---|---|---|
| **FP16 Baseline** | **0.20** | **56.43%** | **89.58%** | **69.24%** | **52** | **86.75%** | **94.38%** | 215 / 166 / 25 / 327 |
| FP16 Baseline | 0.25 | 58.08% | 88.33% | 70.08% | 45 | 86.09% | 92.13% | 212 / 153 / 28 / 338 |
| FP16 Baseline | 0.30 | 60.81% | 87.92% | 71.89% | 37 | 85.43% | 92.13% | 211 / 136 / 29 / 353 |
| INT8 PTQ | 0.15 | 51.64% | 91.67% | 66.07% | 41 | 90.07% | 94.38% | 220 / 206 / 20 / 295 |
| INT8 PTQ | 0.18 | 54.21% | 91.25% | 68.01% | 32 | 89.40% | 94.38% | 219 / 185 / 21 / 314 |
| **INT8 PTQ** | **0.20** | **56.15%** | **91.25%** | **69.52%** | **27** | **89.40%** | **94.38%** | **219 / 171 / 21 / 325** |
| **INT8 PTQ** | **0.22** | **58.40%** | **91.25%** | **71.22%** | **24** | **89.40%** | **94.38%** | **219 / 156 / 21 / 335** |
| INT8 PTQ | 0.25 | 60.56% | 90.83% | 72.67% | 20 | 88.74% | 94.38% | 218 / 142 / 22 / 345 |
| **INT8 PTQ (best F1 @ base R)** | **0.30** | **62.68%** | **89.58%** | **73.76%** | **16** | **88.08%** | **92.13%** | **215 / 128 / 25 / 358** |

#### 3d. Stage 1 Lab Holdout + Confuser Parity (`crop_4_3` 1920x1440, every 3rd frame)

| Model @ Conf | Nov 14 Knife Frame Mean / Min | Nov 14 Knife Win Mean / Min | Take 002 (Oct 31) | Phone 050 FA | Remote 056 FA |
|---|---|---|---|---|---|
| FP16 Baseline @ 0.20 | 37.19% / 3.61% | 51.22% / 3.96% | 77.28% | 3.61% | 5.28% |
| INT8 PTQ @ 0.20 | **37.60%** / 3.09% | **51.56%** / 1.58% | 76.13% | **3.09%** | 5.28% |
| INT8 PTQ @ 0.30 | 36.18% / 1.55% | 50.37% / 0.00% | 73.51% | 1.55% | 5.28% |

#### 3e. Stage 2a Pose + ByteTrack Parity (150 frames, `2026-10-31/raw/002_checkerboard_142846.avi`)

| Model | Detected | Unique IDs / Switches | BBox IoU (Med) | BBox Drift px (Med) | Kpt L2 Drift px (Med / Max) |
|---|---|---|---|---|---|
| INT8 PTQ vs FP16 | 150 / 150 | 1 / 0 | 0.94874 | 16.782 | **14.973 / 267.991** |

### 4. Step 2a Findings
1. **Stage 1 INT8 PTQ passes GT accuracy & clean TRT kernel fusion:** Zero FP32 Conv fallbacks (`65` INT8 Convs + `37` FP16 Convs), and at `conf=0.20` INT8 PTQ beats FP16 on GT Recall (`91.25%` vs `89.58%`), F1 (`69.52%` vs `69.24%`), OOD FP (`27` vs `52`), and Nov 14 Lab mean (`37.60%` vs `37.19%`). Re-sweeping to `conf=0.22` keeps `91.25%` recall while raising F1 to `71.22%` (`24` OOD FPs).
2. **Where Stage 1 PTQ still trails slightly:** Worst-take lab window (`1.58%` vs `3.96%` at `conf=0.20`) and Take 002 (`76.13%` vs `77.28%`) — targeted for **Step 2b (INT8 QAD + OSA + CrossKD)**, which also serves as the QAD pipeline for Step 3 pruning.
3. **Stage 2a Pose PTQ is NOT safe without QAD:** Despite locking `Pose26` + RLE head to FP16, quantizing the backbone/neck caused `14.97 px` median (`267.99 px` max) keypoint drift. Keep Stage 2a in FP16 (unless QAD is run on pose separately).
4. **What was not checked:** Jetson Orin Nano (`sm_87`) and Edge PC (RTX 3050) actual INT8 latencies.

### 5. Files
- [`scripts/quantize_stage1_ptq.py`](../scripts/quantize_stage1_ptq.py)
- [`scripts/bench_step2_int8.py`](../scripts/bench_step2_int8.py)
- [`src/engine_builder.py`](../src/engine_builder.py)
- [`results/amp_edge_speed/step2a_int8_ptq.json`](../results/amp_edge_speed/step2a_int8_ptq.json)
