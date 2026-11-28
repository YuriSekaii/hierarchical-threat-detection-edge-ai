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
3. **What was not checked:** Actual Jetson Orin Nano (`sm_87`, unified LPDDR5 memory + slower ARM Cortex-A78AE CPU) and RTX 3050 Edge PC timings.

### 5. Files
- [`src/gpu_preprocess.py`](../src/gpu_preprocess.py)
- [`src/stage1_detector.py`](../src/stage1_detector.py)
- [`src/stage2a_pose_tracker.py`](../src/stage2a_pose_tracker.py)
- [`tests/test_gpu_preprocess.py`](../tests/test_gpu_preprocess.py)
- [`scripts/bench_step1_gpu_preprocess.py`](../scripts/bench_step1_gpu_preprocess.py)
- [`results/amp_edge_speed/step1_gpu_preprocess.json`](../results/amp_edge_speed/step1_gpu_preprocess.json)
