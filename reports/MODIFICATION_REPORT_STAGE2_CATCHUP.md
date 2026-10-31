# Modification Report: Stage 2 Catch-Up Mode (no more 3 s frame skips)

## 1. Problem

Live session 2026-10-31 11:12:47: Stage 2 jumped ahead twice (f383 -> f469, f759 -> f842). About 3 s of video each time was never analyzed. An attack inside a gap could not raise an alarm.

## 2. Cause

Stage 2 runs on 50-frame clips and steps 10 frames per clip. At 30 fps, it must finish a step in **333 ms** to keep up.

| Cost per step (RTX 3050, TensorRT) | Time |
|---|---|
| Pose on the 10 new frames (about 31-34 ms each) | about 300-340 ms |
| Cascade (production) | about 5-20 ms |
| Debug probe (all tiers, twice) and logging, debug mode only | about 60 ms |

The pipeline was slower than the camera by 1-2 frames per step, so the backlog kept growing. Once it passed `max_audit_lag` (80 frames), the cursor snapped to live and the frames in between were dropped. Each snap then forced a cold start (pose on 26 frames, about 0.8-1.6 s), which immediately built a new backlog.

Two smaller effects made it worse:
- The first pose call of a run took 0.26-0.47 s (engine/CUDA start-up), because the pose model had never run before the first knife detection.
- Above about 54 frames of backlog, frames leave the 64-frame uncompressed ring and have to be JPEG-decoded, which slows pose down further.

## 3. Fix

| File | Change |
|---|---|
| `src/inference_production_pipeline.py` | `catchup_lag = 15`: when Stage 2 is more than 15 frames (0.5 s) behind live, pose runs on every other new frame and the frames in between are interpolated (the same rule cold start already used). When it is caught up, pose runs on every frame again. |
| `src/inference_production_pipeline.py` | `_warmup_pose()`: three pose runs on a blank frame before the workers start (`predict`, so ByteTrack state stays clean). |
| `src/debug_stage2_tracer.py` | Same rule; new `S2A_CATCH_UP` event, `clips_catchup` in `summary.json`, and a `Lag:` line in the printed summary (snaps, frames never analyzed, catch-up clips). |

`max_audit_lag` is unchanged and stays as a safety net.

## 4. Verification (Edge PC, RTX 3050, TensorRT)

Test input: 12 held-out videos (Thrust, Stab, Cut-Down, without coat) joined into one 43.5 s video, scaled so the 4:3 crop is 1920x1440 like the live camera. Files are fed at 30 fps in real time. Debug mode, `--headless`.

| | Before (1045590) | After |
|---|---|---|
| Cursor snaps | 2 | **0** |
| Frames never analyzed | 163 (5.4 s) | **0** |
| Backlog behind live: median / p90 / max | 25 / 49 / 73 frames | **14 / 17 / 24 frames** |
| Clips analyzed | 103 | 122 |
| Clips with alarm | 23 | 26 |
| First pose call | 257 ms | 45 ms |
| Clips in catch-up mode | - | 29 of 122 |

Production mode (no `--debug`) on the same video: 32 alarms, clean shutdown, no errors.

## 5. Trade-off

In catch-up mode, 4 of the 10 new frames of a clip get interpolated poses instead of measured ones. Training skeletons are already Gaussian-smoothed (sigma = 1 frame), so a one-frame midpoint is a similar amount of smoothing. On the test video, the alarm count did not drop (23 -> 26).
