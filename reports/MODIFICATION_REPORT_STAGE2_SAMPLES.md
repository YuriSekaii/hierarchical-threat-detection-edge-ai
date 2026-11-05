# Modification Report: Motion Samples and the Generator

**Status: unfinished, and nothing new is deployed.** No model trained on the new samples beats `sim3` at a low false-alarm rate, so `sim3` stays the opt-in set and the live models stay the default. The likely missing piece is real normal activity, which is not recorded yet.

## 1. What was recorded

On 2026-11-05 the Edge PC recorded 30 of the 84 planned rows' worth of attack takes: 29 usable takes and one false start (file 026). About 1,500 s of video and 250 strikes. **`train_normal`, `test_attack` and `test_normal` are not recorded**, and neither is `walking_thrust_lefthand`.

- Files 010-013 are labelled `overhand_stab` but show cut-downs, and files 014-017 the reverse. The user mixed them up. Hand travel confirms the swap (1.5-2.0 torso lengths against 0.6-1.3). The originals are not renamed; the script handles it.
- The recorder dropped frames in long takes, because 2560x1440 encodes at only 30 fps on the Edge PC. It now saves the 1920x1440 crop (43 fps). A 3-minute camera test dropped 0 of 5,400 frames. Earlier takes with dropped frames use only their first 30 s.

## 2. The generator

`src/skeleton_sim/from_samples.py` works on real joint coordinates. Nothing is rendered, and no 3D lifting or download is used.

| Variation | How |
|---|---|
| Speed and timing | Resampled at 0.75-1.35 x, with a smoothly varying rate |
| Body | Joint directions kept, bone lengths changed by up to 15 % |
| View | Mix of two recorded directions 90 degrees apart, plus squash and shear |
| Motion | Blend of two performances of the same attack |
| Legs | Upper body placed on the legs of another clip |

Normals go through the same steps, so "looks generated" carries no label.

Eight takes were held out before any training: 003, 008, 010, 017, 020, 023, 028, 031. That is 77 strikes across every attack type and direction.

## 3. Results

All tables are means of seeds 1-3 on saved skeletons, except section 3.3. "Held-out strikes" means at least one window of that strike alarms. "Output" means the network's own output instead of the RMD score the engine uses.

### 3.1 With the alarm bound above every held-out normal window

| Models | Take 002 AUC | Take 002 caught of 31 | Events/min | Validate Thrust / Stab / Cut | Held-out AUC | Held-out strikes |
|---|---|---|---|---|---|---|
| **sim3** | 0.871 | **15.3** | **0.14** | 2 / 2 / 29 % | 0.818 | 32 % |
| sim3 + generated | 0.887 | 15.3 | 0.14 | 2 / 1 / 42 % | 0.868 | 32 % |
| Real windows, no generator | 0.829 | 12.0 | 1.93 | 20 / 70 / 77 % | 0.861 | 34 % |
| Generated | 0.865 | 10.3 | 0.83 | 13 / 28 / 62 % | 0.900 | 24 % |
| Earlier `mix`, output | 0.875 | 20.7 | 2.62 | 62 / 59 / 93 % | 0.946 | 84 % |
| Generated + normals, output | 0.891 | 15.7 | 2.48 | 20 / 46 / 68 % | 0.955 | 60 % |
| Real windows, output | 0.865 | 12.7 | 1.66 | 32 / 41 / 53 % | 0.960 | 54 % |
| old + mix + generated, output | **0.903** | 23.0 | 2.48 | 35 / 43 / 88 % | 0.935 | 75 % |

### 3.2 With the same false-alarm budget

The bound moves until take 002 gives at most N false-alarm events in its 2.4 minutes of normal activity. This uses take 002 to set the bound, so it is only a comparison, not an expected rate.

| Models | N=0: caught of 31 / held-out strikes | N=1 | N=2 |
|---|---|---|---|
| **sim3** | **15.3 / 37 %** | 15.3 / 37 % | 17.3 / 54 % |
| sim3 + generated | 15.3 / 40 % | 15.3 / 40 % | 18.0 / 46 % |
| Earlier `mix`, output | 0.0 / 6 % | 0.0 / 12 % | 1.3 / 23 % |
| old + mix + generated, output | 1.3 / 7 % | 1.3 / 10 % | 2.3 / 15 % |

### 3.3 Full pipeline on the held-out takes

TensorRT on the dev PC at the launcher's settings, with Stage 1, the gates, pose and cascade on the raw video. 0 worker exceptions on every completed run.

| | live | sim3 |
|---|---|---|
| Strikes that reached the cascade | 70 of 77 (91 %) | 69 of 77 (90 %) |
| **Strikes alarmed** | **9 of 77 (12 %)** | **16 of 77 (21 %)** |
| Seen from behind | 4 of 21 | 1 of 21 |
| Side, left | 0 of 18 | 7 of 18 |
| Slash | 1 of 22 | 1 of 22 |

## 4. What this shows

- **The generator is not yet proven.** It raises take 002 ranking and steadies results across seeds (at the q98 bound, 25.7 +- 2.5 caught against 17.0 +- 11.9 for real windows alone). But it does not raise the number caught at a low false-alarm rate, and real windows rank the held-out strikes about as well (0.960 against 0.955).
- **The network learned the new takes, and RMD loses part of that.** Held-out ranking goes from 0.86-0.90 with RMD to 0.94-0.96 with the output.
- **The output is not a free gain.** It catches far more only with far more false alarms. Its highest scores on normal activity are extreme, so at N=0-2 it catches almost nothing. This corrects the first reading of these results.
- **False alarms come from specific activities**, not walking. The table gives the share of each activity's windows that score above the median attack window, on small counts (5-12 windows for the top rows):

| Activity on take 002 | sim3 | `mix`, output | Generated + normals, output |
|---|---|---|---|
| Bend, pick up from floor | 57 % | 43 % | 52 % |
| Hand to face | 50 % | 39 % | 56 % |
| Hand to head, bend | 44 % | 58 % | 64 % |
| Inspect knife | 13 % | 40 % | 67 % |
| Walk, knife low | 2 % | 1 % | 0 % |
| Stretch arms | 0 % | 3 % | 1 % |

These are the activities in the unrecorded `train_normal` takes. Training on them should reduce these false alarms, but that is **untested**.

- **Stage 1 is not the main limit here.** 90 % of strikes reach the cascade, even from behind (81 %), although the knife is detected in only 7 % of sampled frames in one back take. Repeated strikes keep Stage 2 armed for 3 s after each detection, so a **single** attack from behind may do worse. The `test_attack` takes will show this.

## 5. What did not work or got worse

- **No new model beats `sim3` at a low false-alarm rate.**
- `sim3` still misses most strikes: 21 % caught in the full pipeline, 12 % for live.
- Slash is almost never caught in the full pipeline (1 of 22 for both), and from behind `sim3` catches 1 of 21 against 4 of 21 for live.
- Adding generated data lowered Validate Stab slightly (2 % to 1 %), though it raised Cut (29 % to 42 %). Both changes are within what seeds alone can cause.
- Known live thrusts are still not caught at the strict bound (0 of 8).

## 6. An open crash

The pipeline crashed once on the dev PC with `0xC0000409`, a native fail-fast crash, with no Python error and no summary saved. This was `sim3` on take 020, in 1 of 20 `sim3` runs and none of 8 live runs. The trace stops at the first cascade call after `HAND_CHECK`. Three reruns of that take did not crash, and crash tracing did not catch another. **The cause is unknown.** A live camera run could die the same way, and nothing restarts it.

## 7. How far to trust the numbers

- **Same person, room and session.** Held-out strikes are not an independent test. They are likely to look better than a new person would.
- **Strikes are found automatically** from hand speed, not checked by eye. Counts per take (4-18) are plausible but not verified.
- Held-out takes guided the reading of results here, so they are no longer fully clean. Take 002 was already biased.
- **The bound rests on normals that are mostly walking**, so it says little about gestures.
- The view mix uses different performances aligned in time, so it only approximates a real view.
- Differences of 1-3 caught attacks between rules are within seed noise.

## 8. In the repo and on the dev PC only

| Item | Where |
|---|---|
| Generator, tests | `src/skeleton_sim/from_samples.py`, `tests/test_from_samples.py` |
| Cache builder and recipes | `scripts/generate_from_samples.py` |
| Skeleton extraction, as the Edge engine sees frames | `scripts/extract_take_skeletons.py` |
| New pools `gen` and `gen_normal` | `scripts/train_stage2_sim.py` |
| Recorder: 4:3 crop, samples plan resumes across days | `scripts/record_takes.py`, `tests/test_record_takes.py` |
| Recordings, dev PC only | `data/lab_recordings/2026-11-05/` |
| Skeletons, held-out windows, full-pipeline sessions, scoring scripts | `results/samples/` |
| Caches and weights, seeds 1-3 | `results/train_sim/exp_S_raw`, `exp_S_gen`, `results/train_sim/experiment_weights/*_S_*` |
| Edge PC debug logs of 2026-11-05 | `results/edge_debug_2026-11-05/` |

## 9. Next steps

1. On the Edge PC: `git pull`, then `record_samples.bat`. It resumes at take 30. Record the rest, 55 takes: `walking_thrust_lefthand`, `train_normal`, then the `test_*` takes, in other clothes or with another person if possible.
2. Train with the normal takes, and set the bound on real varied normals. This tests whether the false alarms in section 4 fall.
3. Score on the `test_*` takes only, through the full pipeline. That is the first clean number.
4. Then decide whether to use the output instead of RMD. Both need checking at a fixed false-alarm rate.
5. Find the cause of the crash in section 6 if it appears again.
