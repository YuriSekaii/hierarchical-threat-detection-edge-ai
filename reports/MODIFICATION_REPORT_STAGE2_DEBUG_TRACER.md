# Modification Report — Stage 2 Debug Tracer (`--debug`)

**Date:** 2026-10-22 | **Base commit:** `3aee715` (v6.20) | **Scope:** diagnostics only, no production logic changed

## 1. Motivation

In live use, Stage 2 (pose → staircase cascade) never raises a violence alarm after Stage 1 detects a knife ("the system is blind").
The goal is to see what the pipeline actually does after a knife trigger, and which frames Stage 2 analyzes, without editing the production modules.

## 2. Touched Files

| File | Change |
|---|---|
| `src/debug_stage2_tracer.py` | **NEW.** `DebugHierarchicalPipeline` (subclass of `ProductionHierarchicalPipeline`) plus `DebugTraceWriter`. |
| `src/inference_production_pipeline.py` | **+7 / −1 lines.** New `--debug` CLI flag; when it is set, the entry point builds `DebugHierarchicalPipeline` instead of `ProductionHierarchicalPipeline`. Without the flag, behaviour is unchanged. |
| `archive/inference_production_pipeline_v6.20_baseline.py` | **NEW.** Pre-hook copy of the entry point (rule 3.1). |
| `results/debug_stage2/.gitignore` | Created automatically on the first debug run, so session dumps stay out of git. |

No changes to `stage1_detector.py`, `stage2a_pose_tracker.py`, `stage2b_staircase.py`, `pipeline_buffer.py` or `skeleton_utils.py`.

## 3. Architecture

- The subclass overrides only `weapon_scanner_worker`, `action_auditing_worker` and `_populate_window_skeletons`. Each override is an instrumented copy of the v6.20 logic: same thresholds, cursor rules, calls and status updates.
- `pose_model.track` is wrapped **on that instance only**, so the log can show how many people YOLO found and which actor-locking strategy was taken (S1 ByteTrack ID, S1b IoU re-lock, S2 nearest-to-weapon, S3 largest box).
- Image and array writes run on a separate writer thread through a bounded queue, so disk I/O does not slow the pipeline. Dropped jobs are counted in `summary.json`.
- There are two deliberate, clearly labelled differences from production:
  1. A worker exception is logged as `WORKER_EXCEPTION` with its traceback, and the loop keeps running. In production the same exception kills the daemon thread, and the HUD keeps showing a stale status.
  2. **PROBE** (diagnostic only, never used for the decision): every clip is also scored by all three tiers on (a) the exact production input and (b) a training-style heatmap built like `src/dataset.py` (raw pixels → interpolate/smooth → one normalization → rasterize).

### Usage
```
run_webcam_tensorrt.bat --debug
run_webcam_pytorch.bat  --debug
.\.venv\Scripts\python.exe src\inference_production_pipeline.py --source <video.mp4> --backend tensorrt --debug
```
Optional environment switches: `HTD_DEBUG_SAVE_FRAMES=0`, `HTD_DEBUG_PROBE=0`, `HTD_DEBUG_NEAR_MISS=0`, `HTD_DEBUG_NEAR_MISS_CONF=0.10`, `HTD_DEBUG_DIR=<path>`.

### Output: `results/debug_stage2/session_<YYYYmmdd_HHMMSS>/`
| Item | Content |
|---|---|
| `trace.log` | Timeline: `S1_SCAN`, `KNIFE_TRIGGER`, `WEAPON_LOST`/`WEAPON_REACQUIRED`, `STAGE2_START`/`STAGE2_STOP`, `CURSOR_SNAP`, `CLIP_BEGIN`, `S2A_COLD_START`, `S2A_POSE` (per frame), `ACTOR_SWITCH`, `CLIP_RESULT`, `CLIP_WARNING`, `STATUS`, `WORKER_EXCEPTION` |
| `events.jsonl` | The same events as JSON, one per line |
| `stage1_detections/` | Every Stage 1 weapon frame, annotated. The arming frame is prefixed `TRIGGER_` |
| `frames/` | Every frame Stage 2 analyzed: skeleton, actor box, weapon box and source tag (E = extracted, I = interpolated, C = cached, N = no image) |
| `clips/clip_NNNN_fA-B/` | `contact_sheet.jpg` (all 50 frames), `heatmap_compare.png` (production vs training-style PoseConv3D input), `skeleton_raw.npy`, `coords_normed.npy`, `clip.json` |
| `summary.json` | Episodes, clips, tier and decision histograms, frame ages, pose time, heatmap body size, mean and max p1, exceptions |

## 4. Verification

The environment was a cloud CPU container (PyTorch 2.14.1+cpu, Ultralytics 8.4.183, `--backend pytorch`). The real repo weights were used.

| Check | Command / harness | Result |
|---|---|---|
| Compile | `python -m py_compile src/debug_stage2_tracer.py src/inference_production_pipeline.py` | OK |
| Default path unchanged | `inference_production_pipeline.py --dry-run --backend pytorch` (no `--debug`) | `VERIFIED FUNCTIONAL`, same output as before the hook |
| Debug dry run | `... --dry-run --backend pytorch --debug` | Session created, `STAGE2_START` logged, 0 exceptions |
| End-to-end (30 s video from `zidane.jpg`, Stage 1 forced to report a knife for t = 1–18 s, one injected exception) | scratch harness (not in repo) | 36 detections, 1 `KNIFE_TRIGGER`, `WEAPON_LOST` (sub-threshold 0.30 logged), `WORKER_EXCEPTION` logged and loop survived, `STAGE2_STOP`, 50 frames + 1 clip dump, `events.jsonl` parses (150 events), 0 dropped jobs |

CPU timing in this test (about 440 ms per pose frame) is **not** representative of the RTX/TensorRT target.

## 5. First Finding (diagnosed, NOT fixed)

On the first clip analyzed, the tracer raised:

```
CLIP_WARNING #1: production heatmap body is 6.0 cells tall vs 19.0 for training-style input
 -> skeleton collapsed in the PoseConv3D input
```

| Metric | Production input | Training-style input |
|---|---|---|
| Mean body height in the 56×56 heatmap | **6.0 cells** (one Gaussian blob) | 19.0 cells |
| Mean body width | **6.0 cells** | 30.5 cells |
| Tier 1 p1 | 6.6e-20 | 3.7e-3 (non-violent scene, so a low value is expected) |

**Likely cause: double normalization.**
1. `Stage2aPoseTracker.preprocess_sliding_window` already returns torso-normalized coordinates, with values of about ±2.
2. `rasterize_heatmap_from_coords` (`stage2b_staircase.py`) calls `normalize_skeleton_clip` on them again.
3. In that second pass the torso length is about 1 (below the 10 px guard in `skeleton_utils.py`) and the bbox diagonal is about 3 (also below 10), so the scale falls back to **100**.
4. All 17 joints shrink to about ±0.02 and land on the centre cell. PoseConv3D Tier 1 sees a dot and discards every clip at Tier 1 (p1 < 0.20), so Tiers 2 and 3 are never reached.

Training (`PoseConv3DHeatmapDataset`) normalizes raw pixel skeletons only once.

**Status:** confirmed and fixed.
- Confirmed on the Edge PC (`session_20261022_172102`, knife scenario, 44 clips): production p1/p2 were 0.000 on every clip and the body was 6.0 cells tall. The training-style input reached p1 up to 0.986 and p2 up to 0.996.
- Fix: `rasterize_heatmap_from_coords` takes `normalize=False` for coordinates that are already normalized, and `evaluate_clip` / `evaluate_video` pass it. ST-GCN input is unchanged.
- Verified offline on 3 recorded clips: the fixed production heatmap equals the training-style heatmap (body about 30 cells tall, max abs difference about 3e-6).
