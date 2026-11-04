# Modification Report: Stage 2 Retrained on Simulated Skeletons (sim v2)

**Status: checkpoint, not finished.** The new models are in the repo as an option. The live models stay the default. The new set has never run end to end on the Edge PC or the live camera.

## 1. Result

Lab take 002 replayed like the live pipeline: 50-frame windows, stride 10. Stage 1 and the weapon gates are not part of this test.

| | Live models | sim v2 |
|---|---|---|
| Lab attacks with at least one alarm | 1 of 31 | **14 of 31** |
| Attack windows alarmed | 0.5 % | 10.4 % |
| Normal windows alarmed | 2 of 435 | 5 of 435 |
| False-alarm events (5 s alert hold) | 2 in 2.4 min (0.83/min) | 2 in 2.4 min (0.83/min) |

This is a large gain on the live camera, but **sim v2 is not ready to replace the live models**:

| Check | Live models | sim v2 |
|---|---|---|
| Known live thrusts (session 111247 clips 34-38, 114942 clips 37-39) | T1 0.000 | T3 0.21-0.47, **no alarm** (bound 0.55) |
| Live clips alarmed (280, mostly walking) | 0 in the four live sessions, 5 of 7 in e2e | 7 (s111247 clips 22, 25, 26; s114942 clips 50, 52, 53, 54) |
| Validate Thrust, with / without coat | 32 % / 55 % | **0 % / 0 %** |
| Validate Stab, with / without coat | 85 % / 79 % | **0 % / 0 %** |
| Validate Cut-Down, with / without coat | 88 % / 88 % | 31 % / 35 % |
| Validate OOD, false alarms (24 clips) | 0 % | 17 % |

- Clip 52 of session 114942 was reviewed by eye. It shows walking with the knife held low and no strike, so it looks like a false alarm. The other six alarmed clips were not reviewed.
- Sim v2 trades the training camera for the live camera. It has lost most detection on the training-camera videos.

### How far to trust the numbers

- **Take 002 is not a clean test any more.** No model was trained on it and no threshold value was computed from it. But it was used to choose the recipe (simulation-only), to decide that the PoseConv3D tiers should be pass-through, and to choose the 0.98 quantile. So 14 of 31 is somewhat optimistic. A new recording is needed for an unbiased number.
- The sample is small: 31 attacks and 2.4 minutes of normal activity, with labels made by eye from event strips. The two false-alarm counts cannot separate the models.
- Each model was trained once, with one seed. Differences of a few AUC points may be noise.
- Simulated scores near 100 % say little. Simulated Thrust reaches only 63 %.

## 2. What was tried

Window AUC and attacks caught are on the lab takes (both takes, attack vs normal windows). "Caught" means at least one alarmed window, at a threshold chosen to flag the stated share of normal windows. These thresholds are for comparison only.

| Model | Lab window AUC | Caught at 1 % | Caught at 2 % |
|---|---|---|---|
| Live T1 / T2 / T3 | 0.52 / 0.72 / 0.48 | 2 / 1 / 0 | 2 / 5 / 1 |
| v1 T1 / T2 / T3: real + perturbed + simulated attacks, simulated normals only | 0.36 / 0.67 / 0.56 | 2 / 2 / 3 | 4 / 2 / 3 |
| ST-GCN, simulation only, first normals | 0.79 | 4 | 6 |
| ST-GCN, mix with real negatives and noise randomization, first normals | 0.74 | 7 | 8 |
| **ST-GCN, simulation only, brisk normals (chosen)** | **0.81** | **14** | **20** |
| ST-GCN, mix, brisk normals | 0.66 | 5 | 5 |
| PoseConv3D, simulation only, brisk normals | 0.54 | 1 | 1 |
| PoseConv3D distilled from the chosen ST-GCN | 0.75 | 3 | 3 |

What this showed:

- **The live models score camera angle more than action.** Seen from a high camera behind the person, simulated walking scores like simulated attacks; from the front, nearly everything scores near 0.
- **Real attacks + simulated-only normals teaches a shortcut.** In that mix every real clip is an attack and every normal is simulated, so "looks real" points to "attack". v1 flagged 40-74 % of live clips. This explanation fits the numbers but was not tested on its own. The 49 real normal clips added to the mix (55 windows) were too few to fix it.
- **Calm simulated normals were a realism bug.** Their wrist speed was 0.05 torso/frame against 0.13 in the lab. Adding brisk activities (arm circles, waving, reaching, jogging, fidgeting, looking around) raised it to 0.11.
- **PoseConv3D does not transfer from simulation.** Calibrated as a discard filter, it dropped 82 % of real attack windows. Why is unknown.

## 3. What is in the repo

| Item | Purpose |
|---|---|
| `scripts/train_stage2_sim.py` | Stages: `prepare`, `train`, `calibrate`, `deploy`, `evaluate`. The docstring lists the exact commands. |
| `weights/*_sim_v2.{pth,onnx}`, `weights/reference_data_*_sim_v2.pt` | The new models and their references. Same architectures as the live models. |
| `models/ensemble_super.py` | New registry keys and `STAGE2_MODEL_SETS` (`live`, `sim`). |
| `src/stage2b_staircase.py`, `src/inference_production_pipeline.py` | Engine paths and `--stage2-models {live,sim}`. Default: `live`. |
| `run_webcam_tensorrt_sim.bat`, `run_webcam_tensorrt_sim_debug.bat` | One-click launchers for the new set. The existing launchers are unchanged. |
| `src/skeleton_sim/` | Six brisk normal activities added; normal clips now default to a faster speed. |

The sim cascade order and calibration:

| Slot | Model | Behaviour |
|---|---|---|
| T1 | Distilled PoseConv3D | Pass-through: p is always 0.5, so it never discards or alarms |
| T2 | PoseConv3D | Pass-through, same way |
| T3 | Simulation-only ST-GCN | Decides. Alarms above the 98th percentile of normal activity from lab take 001 and 252 live clips |

- The engine is unchanged. Bounds are placed by scaling the reference matrices with a temperature.
- The two pass-through models still run, which wastes their inference time.
- Verified on the dev PC: ONNX export works, and the PyTorch and TensorRT backends load the sim set and agree on the same clip (p3 0.4939 vs 0.4939).

### Threshold trade-off

ST-GCN alone on take 002. Each threshold is computed from the same normal windows as above.

| Quantile | Attacks caught | Normal windows | False-alarm events/min |
|---|---|---|---|
| 0.95 | 21 of 31 | 2.3 % | 1.24 |
| 0.97 | 20 of 31 | 2.1 % | 1.24 |
| **0.98 (default)** | 14 of 31 | 1.1 % | 0.83 |
| 0.99 | 11 of 31 | 0.5 % | 0.83 |
| 0.995 | 10 of 31 | 0.5 % | 0.83 |

## 4. Files not in git (dev PC only)

| Path | Contents |
|---|---|
| `data/Simulated_v2/` | The dataset these models used: 7,060 clips, including 3,000 normal. `data/Simulated/` is the older set. |
| `data/live_clips_2026-10-31/` | Copies of 296 live debug clips (some are excluded when calibrating or scoring, see below). The originals sat in a temporary folder. Thrust ranges to exclude: `s111247:30-40,s114942:34-42,s1031:28-34,s110035:29-35`. Those ranges are partly from memory and deliberately wide. |
| `results/train_sim/` | Window cache (414 MB), logs, `evaluation_v2.json`, and experiment weights in `experiment_weights/`. |

To re-run `train --model t2`, `calibrate` or `deploy` with `--tag _v2simonly`, first copy `results/train_sim/experiment_weights/*_v2simonly*` back into `weights/`.

## 5. Next steps

1. On the Edge PC, `git pull` and run `run_webcam_tensorrt_sim_debug.bat`. TensorRT engines compile on the first run. This checks real-time speed, Stage 1 and the gates together, and real false alarms.
2. Record a new labelled test with the live camera. Take 002 has been used for selection, and the Edge PC was off for this work.
3. Find out why the live thrusts stay below the alarm bound, then review the other six alarmed live clips.
4. Recover training-camera detection (Validate at 0 %) without bringing back the shortcut. The likely route is real normal clips in similar numbers to the real attacks, so that real no longer implies attack.
5. Make the cheap T1 useful again, or remove the wasted pass-through inference.
6. Consider 3D lifting of real motion (MotionBERT or VideoPose3D). This needs permission before downloading weights.
