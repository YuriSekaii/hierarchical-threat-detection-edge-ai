# Modification Report: Corrected Simulator and the sim3 Blend

**Status: checkpoint, not finished.** `sim3` is an option (`run_webcam_tensorrt_sim3.bat`, `--stage2-models sim3`). The live models stay the default. `sim3` has not run on the Edge PC or on the live camera, and there is still no clean test.

## 1. Result

`sim3` averages the scores of three ST-GCNs. It replaces the two wasted pass-through tiers of `sim`.

Full pipeline, on the dev PC: Stage 1, gates, pose and cascade on the raw video of lab take 002. PyTorch backend, same settings as the launcher, 31 labelled attacks, 2.4 minutes of normal activity.

| Full pipeline | live | sim (v2) | **sim3** |
|---|---|---|---|
| Attacks with at least one alarm | 1 of 31 | 9 of 31 | **14 of 31** |
| Normal clips alarmed | 2 of 425 | 5 of 424 | **0 of 422** |
| False-alarm events (5 s alert hold) | 2 | 2 | **0** |
| Worker exceptions | 0 | 0 | 0 |

This is the first full-pipeline number for these models. The gates matter: `sim` catches 14 of 31 on saved skeletons but 9 of 31 here.

Saved skeletons, through the real engine code, TensorRT on the dev PC:

| Skeleton replay | live | sim (v2) | **sim3** |
|---|---|---|---|
| Take 002 attacks caught | 1 of 31 | 14 of 31 | **15 of 31** |
| Attack windows alarmed | 0.5 % | 10.4 % | **18.8 %** |
| Normal windows alarmed | 2 of 435 | 5 of 435 | **0 of 435** |
| False-alarm events/min | 0.83 | 0.83 | **0** |
| Other live clips alarmed (279) | 5 | 7 | **0** |
| Known live thrusts alarmed (8) | 0 | 0 | 0 |
| Validate Thrust, with / without coat | 32 % / 55 % | 0 % / 0 % | 0 % / 0 % |
| Validate Stab, with / without coat | 85 % / 79 % | 0 % / 0 % | 0 % / 0 % |
| Validate Cut-Down, with / without coat | 88 % / 88 % | 31 % / 35 % | **25 % / 18 % (worse than sim)** |
| Validate OOD false alarms (24 clips) | 0 % | 17 % | 8 % |
| Cascade time, median, RTX 3090 | 2.2 ms (mostly exits at T1) | 5.8 ms | 5.5 ms |

### What did not improve or got worse

- **It still misses more than half of the attacks** (14 of 31; 95 % interval about 29-62 %).
- **Validate Thrust and Stab stay at 0 %.** The live models remain far better on the training camera.
- **Validate Cut-Down fell** from 31 % / 35 % to 25 % / 18 %.
- **The known live thrusts are still not detected** (0 of 8).
- On the PyTorch backend, `sim3` is slower than `sim` (16.9 vs 13.9 ms median).

### How far to trust the numbers

- **Take 002 is not a clean test.** It was already used for selection, and this work used it again to choose the recipes, the blend rule and the bound. These numbers are optimistic. A new recording is needed.
- **Zero false alarms is not a zero rate.** With 0 events in 2.4 minutes, the rate could still be about 1.25 per minute (95 % upper bound).
- **The bound is thin.** It is the highest blended score of 191 held-out normal windows, which are mostly walking.
- **Seeds:** `sim3` uses seed 1 for all three models, fixed before its own result was seen. Over seeds 1-3 this rule gives 15.3 +- 1.2 attacks and 0.14 +- 0.20 events/min.
- **Labels** are by eye from event strips. The known thrust ranges are partly from memory.

## 2. What was found

### The old simulator did not look like real skeletons

Median values in torso lengths, compared without using take 002 or the known thrusts:

| Measure | Real | Old simulator |
|---|---|---|
| Shin length | 0.52-0.59 | 0.77 |
| Thigh length | 0.62-0.64 | 0.80 |
| Body height | 2.1 | 2.5 |
| Face point acceleration (fast jitter) | 0.004-0.006 | 0.011 |
| Shin length wobble within a clip | 0.084-0.099 | 0.020 |
| Free-arm speed during attacks | 0.029-0.046 | 0.013-0.018 |

Changes in `src/skeleton_sim`:

| Change | File |
|---|---|
| Longer torso (0.29 to 0.325) and forearm (0.150 to 0.172), to match where YOLO-pose places the joints | `body.py` |
| Half of the shots use a close camera (1.2-2.5 m, height 1.5-3.2 m) | `camera.py` |
| Less fast jitter (x 0.4), new slow wobble per joint, less noise on hidden joints | `camera.py` |
| Lower overhand wind-ups; the free arm keeps moving during attacks | `actions.py`, `scene.py` |

Skeletons are generated directly. Nothing is rendered or passed through YOLO. The noise model stays because the live models receive YOLO output.

Ruled out on the way: the ST-GCN is not a speed detector (score and wrist speed correlate 0.09 on attack windows), and body size in the frame does not change its score.

### One run is not enough

Re-running the old recipe on a new sample gave a fit-check AUC of 0.710 against 0.847 for the shipped `sim` model, and 1.66 against 0.83 false-alarm events/min. So part of the earlier `sim` result was a lucky pick. The tables below use three seeds each, with the bound set on normals that no model trained on.

| Recipe (3 seeds, bound at q98) | Take 002 AUC | Caught of 31 | Events/min | Validate Thrust / Stab / Cut | OOD |
|---|---|---|---|---|---|
| Old simulator | 0.79 +- 0.03 | 16.3 | 0.8 | 2 / 0 / 25 % | 11 % |
| Corrected simulator | 0.85 +- 0.02 | 17.0 | 1.4 | 6 / 9 / 24 % | 15 % |
| Corrected + real clips | 0.85 +- 0.02 | 25.3 | **4.4** | **70 / 73 / 100 %** | 10 % |

- The corrected simulator improves ranking (AUC), but not the number caught at the bound, and it has more false alarms.
- **Adding real clips brings Validate back**, as the previous report expected. But false alarms on the live camera reach 4.4 events/min, so it cannot be used alone.

Blends, mean of three seeds, with the bound above every held-out normal window:

| Rule | Take 002 AUC | Caught of 31 | Events/min | Validate Thrust / Stab / Cut | OOD |
|---|---|---|---|---|---|
| Old alone | 0.79 | 11.7 | 0.28 | 1 / 0 / 15 % | 7 % |
| Corrected + mix, mean | 0.88 | 16.7 | 1.10 | 7 / 17 / 53 % | 14 % |
| **Old + corrected + mix, mean (sim3)** | **0.87** | **15.3** | **0.14** | 2 / 2 / 29 % | 10 % |
| Old AND corrected AND mix | 0.84 | 0.7 | 0 | 0 / 0 / 9 % | 7 % |

There is a real trade-off here. "Corrected + mix" keeps much more of Validate (7 / 17 / 53 %), at about 8 times the false alarms on the live camera. `sim3` was chosen for the live camera. A better bound from varied real normal activity could change this choice.

Half of Validate OOD trains the mix model, as in earlier recipes. Only the unseen half is scored.

### The known live thrusts

Clip 38 of session 114942 (one of the 8) shows a person walking and turning, with the knife low. The hand moves slightly outward at hip level, and no fast strike is visible in the stills. Tracking is good. Wrist speed in these clips is 0.04-0.13 against 0.16-0.26 for real attacks, and within the range of normal activity. So these clips may be nearly indistinguishable from walking by skeleton alone. Only one clip was viewed, and whether it is a real thrust needs confirming by whoever performed it.

The `e2e` live clips (7) may be real attacks: the live models alarm on 5 of them. They count as normals in the `sim` bound, which may have raised that bound. `sim3` does not use them.

### Two silent failures found

| Failure | What happened | State |
|---|---|---|
| TensorRT inference fails | A freshly compiled engine run on `cuda:1` failed on every clip. TensorRT only logged the error, the engine read uninitialised memory as scores, and `sim3` reported 0 attacks. | **Fixed:** `TensorRTActionModel` now raises an error. A fresh compile on `cuda:0` gives the correct result. Recompiled engines on `cuda:1` work once they exist. |
| Stage 1 engine has the wrong input size | On the dev PC the Stage 1 engine is fixed at 384x640 and the 4:3 launcher setting sends 480x640. Every frame failed, 1,092 exceptions, yet the process ended with exit code 0 and 0 detections. | **Not fixed.** The `.onnx` files in git are 384x640, while an earlier report records the Edge PC at 480x640. Check the debug summary's "Worker exceptions" line on the Edge PC. |

### Blender

Blender 5.2 runs headless here (`D:\Blender\Blender-5.2.4\blender.exe --background`), so the MCP add-on is not needed. It was not used to generate data. The gaps found were body proportions, camera position and noise, all fixed directly in the Python simulator. Keyframing in Blender would give the same kind of motion. Blender becomes useful if real motion capture is added, since it imports BVH.

## 3. What is in the repo

| Item | Purpose |
|---|---|
| `weights/stgcn_sim3_{old,fixed,mix}.{pth,onnx}`, `weights/reference_data_stgcn_sim3_*.pt` | The three models and their references |
| `run_webcam_tensorrt_sim3.bat`, `run_webcam_tensorrt_sim3_debug.bat` | One-click launchers. Existing launchers are unchanged |
| `models/ensemble_super.py`, `src/inference_production_pipeline.py` | The `sim3` set and option |
| `src/stage2b_staircase.py` | `blend` mode (`evaluate_clip` only; `evaluate_video` refuses it) and the TensorRT failure check |
| `scripts/train_stage2_sim.py` | `blend` stage, `prepare --real-normal-live`, `HTD_TRAIN_OUT`. The docstring has the recipe |
| `scripts/export_onnx_models.py` | `--stage2-sim` also exports `sim3` |
| `tests/test_stage2_blend.py` | Checks the blend calculation |

`simulate_dataset.bat` now generates the corrected simulator. The `old` model used the generator from commit e6bfc07.

The three models are:

| Name | Training data |
|---|---|
| `old` | Earlier simulator only |
| `fixed` | Corrected simulator only |
| `mix` | Corrected simulator + real attacks + real normals, batch mix `real:3,sim:12,normal:6,real_normal:7` |

Real normals used by `mix`: first half of take 001, sessions s1031 and s110035 minus their attack-test ranges, plus the earlier False Detected and half of Validate OOD clips. The bound uses the second half of take 001 and sessions s111247 and s114942 minus their ranges. Take 002, the known thrusts and the `e2e` clips are in no training set.

## 4. Files not in git (dev PC only)

| Path | Contents |
|---|---|
| `data/exp_A_old/`, `data/exp_H_all/` | The earlier and corrected simulated datasets |
| `results/train_sim/exp_A_old`, `exp_H_all`, `exp_H_mix` | Window caches |
| `results/train_sim/experiment_weights/` | All experiment weights, seeds 1-3 |
| `results/train_sim/e2e_002/{sim3,sim,live}` | Full-pipeline debug sessions on take 002, with contact sheets (3.5 GB) |

## 5. Next steps

1. On the Edge PC: `git pull`, then `run_webcam_tensorrt_sim3_debug.bat`. Engines compile on the first run. Check real-time speed, and that the summary shows "Worker exceptions: 0" and a non-zero Stage 1 detection count.
2. Record a new labelled test with the live camera, with **varied normal activity** (gestures, bending, reaching, not only walking). This gives the unbiased number, and a better source for the alarm bound.
3. Confirm what the known thrust clips really show.
4. Decide what `sim3` should be tuned for. Validate Thrust and Stab stay at 0 %, while the corrected + mix blend keeps more of them at more false alarms.
5. Make a Stage 1 failure stop the pipeline, and build its engine for the input size in use.
6. Consider 3D lifting of real motion, or motion capture through Blender, if strikes remain the weak point. Both need downloads.
