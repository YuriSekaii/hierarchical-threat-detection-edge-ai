# Why Stage 1 misses lab knives (2026-11-19)

**Status: analysis only. Nothing shipped was changed.**

## Question

The best lab student (YOLO26s distilled from the YOLO26x teacher trained on `lab_gt`, `weights/stage1_repro846_kdgt_p250_newteacher_last.pt`, see `MODIFICATION_REPORT_STAGE1_MODEL_COMPARISON.md` 4i) finds 79 % of the lab test knife boxes at conf 0.40. Which knives does it miss, and why?

## What was run

```bash
DET=kdgtNL T=0.40 .venv/Scripts/python.exe results/samples/poc/analyze_stage1_misses.py      # causes, features, contact sheets
DET=kdgtNL DEV=0 .venv/Scripts/python.exe results/samples/poc/zoom_stage1_wrists.py         # second pass on wrist crops
DETS=kdgtNL,labgtL,teachergtL T=0.40 .venv/Scripts/python.exe results/samples/poc/held_recall_stage1.py   # recall with the weapon hold
```

All three use the lab test labels (2,842 frames, 1,376 knife boxes; same set as 4i) and the detection cache of `sweep_stage1_rules.py`; only the zoom test runs the model again. Box matching is IoU 0.2, 384x640 input, as in 4i.

Each knife box the model misses at conf 0.40 gets one cause:

- **unseen:** no box at IoU ≥ 0.2 at any conf the cache keeps (≥ 0.01).
- **low_score:** a box in the right place, below 0.40.
- **bad_box:** a box at ≥ 0.40 touches the knife, but at IoU < 0.2.

## Results

### Causes (283 missed boxes of 1,376)

| Cause | Boxes | Share | Teacher finds it (0.25) | Moving > 2 box sides in 10 frames | Dark crop (grey mean < 120) | Box < 16 px |
|---|---|---|---|---|---|---|
| low_score | 140 | 49 % | 44 % | 39 of 117 | 64 | 12 |
| unseen | 126 | 45 % | 13 % | 58 of 107 | 61 | 16 |
| bad_box | 17 | 6 % | 18 % | 11 of 15 | 4 | 11 |

`low_score` conf: 57 below 0.10, 29 at 0.10–0.20, 25 at 0.20–0.30, 29 at 0.30–0.40. 98 % of missed knives have a wrist within 0.5 torso lengths, the same share as found ones, so the hand gate is not the cause.

### Recall by feature

| Knife shift to the labelled frame 10 earlier or later (box sides) | Boxes | Recall |
|---|---|---|
| < 0.5 | 261 | 91.2 % |
| 0.5–1 | 328 | 86.6 % |
| 1–2 | 367 | 82.6 % |
| 2–4 | 222 | 72.1 % |
| ≥ 4 | 88 | **47.7 %** |

| Crop brightness (grey mean) | Boxes | Recall |
|---|---|---|
| 60–100 | 200 | **66.5 %** |
| 100–140 | 535 | 76.1 % |
| 140–180 | 580 | 87.1 % |
| 180–256 | 54 | 75.9 % |

| Box side at the 384x640 input | Boxes | Recall |
|---|---|---|
| < 16 px | 79 | **50.6 %** |
| 16–28 px | 511 | 82.3 % |
| 28–40 px | 578 | 81.0 % |
| ≥ 40 px | 208 | 79.3 % |

By take, the swing takes are worst: overhand stab 61.5 %, cut down 65.9 %, slash 68.1 %, thrust low (031) 64.7 %, normal mixed 66.7 %. Slow takes are 85–100 %: thrust chest, walking thrust, Oct 31 take 002. Box shape, crop contrast and Laplacian sharpness of the crop showed no clear link. Laplacian sharpness measures the background as much as the knife.

### What the crops show

Contact sheets of every miss (green = label, red = the model's nearest box, the model's best conf in yellow, `T` = teacher finds it, `W` = wrist near) show four kinds of miss:

1. **Motion blur during a swing.** The knife is a faint smear along the arm, mostly in normal mixed, slash and overhand stab. Many of these are hard to see by eye; the label comes from the frames around them.
2. **Dark knife on dark clothes or the black chair.** A black-handled knife held at the side against the navy shirt, black trousers or the office chair. The model often puts a box in the right place at conf 0.01–0.39 (the `low_score` group).
3. **Knife hidden in the hand or pointing at the camera.** Only the handle or a few pixels of blade show.
4. **Small knives (< 16 px at the input)**, mostly far from the camera. The `bad_box` group is mostly these, where a box is drawn but too loose.

A few clear knives are also missed: a blade held upright in front of the white wall in normal mixed, conf 0.00. That pose may be rare in training.

### More resolution: second pass on wrist crops

The same model is run again on a crop centred on each wrist the pose model finds (wrist conf ≥ 0.3; no knife label is used). The crop is 1280x720 at "2x" (twice the full-frame pixels per knife) or 640x360 at "4x" (native pixels). Its boxes are merged with the full-frame boxes.

| Variant | 0.40: P / R | False boxes | Best R at P ≥ 0.90 (conf) | Best R at P ≥ 0.90, gate (conf) |
|---|---|---|---|---|
| Full frame | 90.5 / 79.1 | 114 | 79.1 % (0.40) | 77.7 % (0.40) |
| **Full + wrist crops 2x** | 89.0 / 83.3 | 142 | **81.7 % (0.50)** | **80.3 % (0.50)** |
| Wrist crops 2x only | 91.8 / 74.8 | 92 | 76.1 % (0.35) | 75.2 % (0.35) |
| Full + wrist crops 4x | 85.8 / 80.4 | 183 | 75.7 % (0.55) | 74.4 % (0.55) |
| Wrist crops 4x only | 84.5 / 20.2 | 51 | 7.3 % | 7.3 % |

| Full-frame miss | Boxes | Found by 2x crops at 0.40 | 4x |
|---|---|---|---|
| low_score | 140 | 45 | 10 |
| unseen | 126 | 11 | 8 |
| bad_box | 17 | 6 | 0 |

At 4x the knife is larger than anything in training and the model mostly fails.

### Recall with the pipeline's weapon hold

The pipeline holds a weapon for 3 s (`weapon_hold=3.0s`, `src/debug_stage2_tracer.py`; 90 frames at 30 fps). A knife box counts as found if it is found in its frame, or if a box at ≥ 0.40 was found near it (centre within 3 box sides) in a labelled frame of the same take up to N frames earlier. Labelled frames are 10 apart, so this is an estimate.

| Detector (conf) | No hold | Hold 10 frames | 30 | 90 (3 s) |
|---|---|---|---|---|
| **KD from new teacher, last (0.40)** | 79.4 % | 86.0 % | 89.8 % | **94.6 %** |
| New teacher, last (0.40) | 78.0 % | 85.9 % | 89.5 % | 93.2 % |
| Student `labgtL` (0.40) | 75.8 % | 84.1 % | 87.9 % | 92.6 % |
| Student `labgtL` (0.25) | 77.6 % | 85.5 % | 89.2 % | 93.7 % |

With the 3 s hold the worst takes are overhand stab 83.1 %, thrust chest 027 84.2 % and slash 88.3 %; all the others are 90–100 %. Of 106 runs of knife frames (labelled frames at most 30 frames apart), the model finds 98. The median delay to the first detection is 0 s; 15 runs take more than 0.5 s and 4 take more than 1 s, at most 2.3 s. The 8 runs never found are all short, 1–5 labelled frames, in stab, slash, cut down, thrust chest 027 and normal mixed.

## Findings

- **Most misses are fast swings and dark knives, not small boxes.** Above 16 px, recall is flat at about 80 %. It falls to 48 % when the knife moves more than 4 box sides between labelled frames, and to 67 % on dark crops. Small boxes (< 16 px) are only 79 of 1,376.
- **Half the misses are seen but not trusted** (`low_score`, conf 0.01–0.39). The other half (`unseen`) are mostly invisible to the teacher too (13 % found), so a better teacher or more distillation will not fix them. They need training data with blurred and dark knives, or another kind of evidence such as the motion of the arm.
- **Resolution helps only the `low_score` group.** A 2x wrist-crop pass finds 45 of the 140 low-score knives and 11 of the 126 unseen. It adds 2.6 points of recall at 90 % precision, for one extra model run per wrist per frame (not timed on the Edge PC). A 4x pass is worse.
- **Per-frame recall understates what the pipeline sees.** With the existing 3 s weapon hold, 94.6 % of knife frames are covered. A knife is found at its first labelled frame in most runs. The open gaps are short swing bursts (overhand stab, slash) and the start of a run, where the hold has nothing yet to hold.

## Proposed next steps

1. Judge Stage 1 recall at the pipeline level (hold included), with a full-pipeline run on the Nov 14 test takes. This needs the TensorRT export of the KD student at 384x640.
2. For the swing takes: add blurred and dark knife frames to training. Motion-blur augmentation of knife crops is cheap; more recordings with swings against the dark chair and clothes are better.
3. If recall must rise further, try the 2x wrist-crop pass only on wrists whose full-frame box is between 0.05 and 0.40 (the `low_score` group). This costs far less than cropping every wrist, but the Edge PC timing must be checked first.

## Follow-up: blur and dark augmentation (2026-11-19)

**Question:** step 2 above, without new recordings. Does adding blurred and darkened copies of the lab knife frames raise recall on fast and dark knives, and does it cost confidence on sharp knives?

**Data.** `results/samples/poc/augment_stage1_blur_dark.py` builds `data/stage1_negatives/lab_gt_aug` from `lab_gt`:
- **What is copied.** Every lab training frame with a knife box gets one copy: 1,840 copies, added to the 4,418 originals, which are all kept (6,258 images).
- **Blur.** A motion blur in a random direction, applied in a feathered region around the knife: 1,571 copies.
- **Darkening.** Lower brightness and contrast in a softly feathered region: 898 copies.
- **Both.** 629 copies get the blur and the darkening.
- **What stays out.** No test frames (Oct 31 take 002, `_test_` takes) and no val images go into train; the script asserts this.

The blur length is 0.3–1.2 box sides. Longer blurs erased the knife on the contact sheet `results/stage1_kd/aug_samples.jpg`.

**What was run.** The `labgtL` recipe (YOLO26s, Ultralytics 8.4.6, MuSGD lr 0.002, 250 epochs, no early stop) on the new set:

```bash
CUDA_VISIBLE_DEVICES=1 PATIENCE=250 DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . labgtaug
CUDA_VISIBLE_DEVICES=1 DEV=0 EXTRA_DETS="labgtaug=weights/stage1_repro846_labgtaug_p250.pt;labgtaugL=weights/stage1_repro846_labgtaug_p250_last.pt" \
  .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . cache lab      # and gtv2
EXTRA_DETS="..." .venv/Scripts/python.exe results/samples/poc/compare_stage1_cached.py labgtL teachergtL kdgtNL labgtaug labgtaugL
DET=labgtaugL T=0.25 .venv/Scripts/python.exe results/samples/poc/analyze_stage1_misses.py   # and DET=labgtL
DETS=labgtL,labgtaugL T=0.25 .venv/Scripts/python.exe results/samples/poc/held_recall_stage1.py
```

Training took 11.3 h on one RTX 3090, with the other GPU training the YOLO26x teacher.

**Best recall at a precision target (conf).** `labgtL` is the same recipe without augmentation; `kdgtNL` is the current best student.

| Detector | Lab, P ≥ 0.90 | Lab, P ≥ 0.90, gate | Lab, P ≥ 0.92, gate | GTv2, P ≥ 0.90 | GTv2, P ≥ 0.90, gate |
|---|---|---|---|---|---|
| `labgtL` | 77.3 % (0.25) | 75.9 % (0.25) | **74.8 %** (0.35) | 65.2 % (0.50) | 65.5 % (0.40) |
| **`labgtaugL`** (augmented, last) | 78.7 % (0.25) | 77.3 % (0.25) | 72.0 % (0.55) | 63.5 % (0.60) | **73.1 %** (0.25) |
| `labgtaug` (augmented, `best.pt`) | 71.9 % (0.50) | 70.9 % (0.50) | 61.3 % (0.70) | 68.1 % (0.45) | 72.2 % (0.35) |
| `kdgtNL` | 79.1 % (0.40) | **77.7 %** (0.40) | 75.7 % (0.50) | **70.2 %** (0.45) | 69.6 % (0.40) |
| `teachergtL` (YOLO26x) | **79.6 %** (0.25) | 78.3 % (0.25) | 75.5 % (0.50) | **71.3 %** (0.25) | 72.5 % (0.10) |

At conf 0.25 the lab precision is about equal: 90.6 % with 112 false boxes for `labgtaugL`, 90.9 % with 107 for `labgtL`. On GTv2 without the gate, `labgtaugL` has more false boxes: 41 against 26 at 0.45. With the gate they drop to 14 against 19.

**Lab recall at conf 0.25, `labgtL` → `labgtaugL`:**

| Group | Boxes | `labgtL` | `labgtaugL` |
|---|---|---|---|
| All | 1,376 | 77.6 % | 79.0 % |
| Slow knife (shift < 0.5 box sides) | 261 | 87.7 % | **91.2 %** |
| Shift 0.5–1 | 328 | 86.3 % | 87.5 % |
| Shift 1–2 | 367 | 82.3 % | 79.8 % |
| Shift 2–4 | 222 | 66.2 % | 68.5 % |
| Fast knife (shift ≥ 4) | 88 | 45.5 % | **55.7 %** |
| Dark crop (grey mean 60–100) | 200 | 58.5 % | 61.0 % |
| Slash take 034 | 94 | 59.6 % | 66.0 % |
| Overhand stab 032 | 65 | 60.0 % | 69.2 % |
| Normal mixed 036 | 246 | 61.8 % | 65.4 % |
| Thrust low 030 | 71 | 84.5 % | 77.5 % |
| With 3 s hold | 1,376 | 93.7 % | 94.3 % |

Misses by cause: `low_score` 136 → 116, `unseen` 156 → 156, `bad_box` 16 → 17.

**Findings:**

- **Sharp knives did not lose confidence.** Recall on slow knives rose from 87.7 % to 91.2 %. The originals are all still in training, and the copies add to them.
- **The gain on the lab camera is small: +1.4 points at 90 % precision**, about the same as distilling from the new teacher (`kdgtNL`, 77.7 % gated against 77.3 %). It is largest where it was aimed: fast knives (+10 points), overhand stab and slash (+6 to +9). Dark knives gain only 2.5 points.
- **The unseen knives stay unseen** (156 both times). The augmentation turned `low_score` knives into hits, not invisible ones. The blur reaches 1.2 box sides, which is much less than the ≥ 4-side swings.
- **On the original camera it helps more with the gate:** 73.1 % against 65.5 % at 90 % precision, the best YOLO26s so far. Without the gate it has more false boxes than `labgtL`.
- **Precision at 92 % is worse** (72.0 % against 74.8 % gated). The augmented model is less sure of its top boxes, so a strict threshold costs recall.
- `best.pt` (chosen on val) is again worse than `last.pt` on the lab test.
### YOLO26x teacher on the augmented set (2026-11-20)

**What was run:** the `teachergt` recipe (YOLO26x, batch 12, nbs 64, 250 epochs, no early stop) on `lab_gt_aug`, then cached and compared as above.

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=250 PATIENCE=250 DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . teachergtaug
```

**Training time: 22.5 h.** The first 80 epochs took about 540 s each, while the plain student trained on the other GPU. The two runs share the CPU that loads the images. Alone, an epoch took 227 s. `best.pt` is epoch 153.

**Best recall at a precision target (conf).** `teachergtL` is the teacher trained without augmentation.

| Detector | Lab, P ≥ 0.90 | Lab, P ≥ 0.90, gate | Lab, P ≥ 0.92, gate | GTv2, P ≥ 0.90 | GTv2, P ≥ 0.90, gate |
|---|---|---|---|---|---|
| `teachergtL` | 79.6 % (0.25) | 78.3 % (0.25) | 75.5 % (0.50) | 71.3 % (0.25) | 72.5 % (0.10) |
| **`teachergtaug`** (`best.pt`) | **82.6 %** (0.25) | **81.2 %** (0.25) | 76.7 % (0.50) | **74.3 %** (0.25) | **74.3 %** (0.15) |
| `teachergtaugL` (last) | 81.5 % (0.10) | 80.1 % (0.10) | **77.9 %** (0.20) | 63.2 % (0.30) | 67.3 % (0.10) |
| Old teacher (`orig_v2`) | not reached | 71.7 % (0.55) | 64.0 % (0.70) | 81.9 % (0.12) | 81.3 % (0.10) |

At conf 0.45 on the lab camera:
- `teachergtaug`: 91.9 % precision, 78.8 % recall, 96 false boxes.
- `teachergtaugL`: 94.6 / 75.1, 59 false boxes.
- `teachergtL`: 92.0 / 77.4, 93 false boxes.

**Findings:**

- **The augmented teacher (`best.pt`) is the best lab detector so far: 82.6 % recall at 90 % precision**, 3.0 points above the teacher without augmentation. It is also better on the original camera (74.3 % against 71.3 %).
- **Here `last.pt` is worse than `best.pt`.** It scores lower overall: fewer false boxes, but it needs conf 0.10 for its best recall, and it loses 11 points on the original camera. This is the reverse of every earlier run, where `last.pt` won.
- `last.pt` at conf 0.25 on the lab camera:
  - slow knives 90.0 %, fast knives 55.7 %, dark crops 63.5 %;
  - with the 3 s hold, 94.4 %, against 94.7 % for `teachergtL`.
  
  The miss analysis was not run for `best.pt` or for `teachergtL`.
- **The distillation uses `best.pt` as the teacher** (`SUFFIX=_augteacherbest`, GPU 0, about 9–10 h). A run from `last.pt` was stopped at epoch 3.

```bash
CUDA_VISIBLE_DEVICES=0 TEACHER=weights/stage1_repro846_teachergtaug_p250.pt SUFFIX=_augteacherbest PATIENCE=250 DEV=0 \
  .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . kdgtaug
```

Choosing `best.pt` over `last.pt` was based on these test numbers, so it is not a clean choice.

### YOLO26s distilled from the augmented teacher (2026-11-20): negative

**What was run:** the command above (250 epochs, 8.2 h on one RTX 3090, the other GPU idle), then cached as `kdgtaugF` (`best.pt`) and `kdgtaugFL` (`last.pt`) and compared:

```bash
EXTRA_DETS="..." .venv/Scripts/python.exe results/samples/poc/compare_stage1_cached.py kdgtNL labgtaugL teachergtaug kdgtaugF kdgtaugFL
DET=kdgtaugFL T=0.45 REF=teachergtaug .venv/Scripts/python.exe results/samples/poc/analyze_stage1_misses.py
```

**Best recall at a precision target (conf):**

| Detector | Lab, P ≥ 0.90 | Lab, P ≥ 0.90, gate | GTv2, P ≥ 0.90 | GTv2, P ≥ 0.90, gate | Lab, 3 s hold (conf) |
|---|---|---|---|---|---|
| `kdgtNL` (KD from `teachergtL`, no augmentation) | **79.1 %** (0.40) | **77.7 %** (0.40) | **70.2 %** (0.45) | 69.6 % (0.40) | **94.6 %** (0.40) |
| `labgtaugL` (augmented, no teacher) | 78.7 % (0.25) | 77.3 % (0.25) | 63.5 % (0.60) | **73.1 %** (0.25) | 94.3 % (0.25) |
| `kdgtaugFL` (KD from `teachergtaug`, last) | 75.1 % (0.45) | 73.9 % (0.45) | 67.5 % (0.40) | 69.0 % (0.35) | 92.3 % (0.45) |
| `kdgtaugF` (KD from `teachergtaug`, `best.pt`) | 70.8 % (0.50) | 72.7 % (0.45) | 61.7 % (0.50) | 68.4 % (0.40) | – |
| `teachergtaug` (YOLO26x) | 82.6 % (0.25) | 81.2 % (0.25) | 74.3 % (0.25) | 74.3 % (0.15) | – |

At conf 0.25 the distilled student has many lab false boxes: 257 for `kdgtaugFL`, against 112 for `labgtaugL` and 232 for `kdgtNL`.

**Miss causes for `kdgtaugFL` at 0.45:**

| Cause | Boxes |
|---|---|
| `low_score` | 199 |
| `unseen` | 125 |
| `bad_box` | 15 |

**Recall by group:**

| Group | Recall |
|---|---|
| Slow knives | 85.8 % |
| Fast knives | 44.3 % |
| Dark crops | 60.5 % |

**Findings:**

- **Distilling the best teacher made the student worse:** 4 points below `kdgtNL` and 3.6 below `labgtaugL` (no teacher) at 90 % lab precision. Its fast-knife recall is back to the level without augmentation.
- **It sees more knives but scores them low.** It has the fewest `unseen` knives of any student (125), but 199 are `low_score`. Its scores spread out, so the false boxes rise before the recall does.
- **The cause is not known.** One guess: the teacher is `best.pt` from epoch 153, while the learning rate was still high, so its outputs may be less sharp than a final-epoch teacher's. Another: the KD loss on the blurred copies, where the teacher itself is unsure, may teach the student to be unsure. Neither was tested.
- **For the Edge PC the best YOLO26s is still `kdgtNL`** on the lab camera, and `labgtaugL` is level with it there. With the gate, `labgtaugL` is better on the original camera (73.1 % against 69.6 %).

Weights (dev PC only): `weights/stage1_repro846_kdgtaug_p250_augteacherbest{,_last}.pt`. Logs (gitignored): `results/stage1_kd/repro846_kdgtaug_p250_augteacherbest.log`, `results/stage1_kd/rules/compare_kdgtaugF.log`.

## Not checked

- The causes come from one model at one threshold, on test labels made by the same process as the training labels. The 408 left-out hard test frames are not in the set, so real recall is lower.
- Brightness and speed are rough measures. Speed uses labelled frames 10 apart, and a box with no labelled neighbour frame has no speed (110 of 1,376 boxes).
- The hold estimate counts a knife as covered if a box was near it earlier in the same take. The pipeline holds the weapon on the tracked person, which may differ. The hold also keeps false boxes alive; precision under the hold was not measured.
- The zoom test chose its threshold on the test set. No Edge PC timing.

## Files

- Scripts: `results/samples/poc/analyze_stage1_misses.py`, `zoom_stage1_wrists.py`, `held_recall_stage1.py`.
- In git: `results/stage1_kd/misses/kdgtNL/{summary.md, zoom_summary.md, misses.json}` and the contact sheets `{unseen,low_score,bad_box}_*.jpg`.
- Augmentation: `results/samples/poc/augment_stage1_blur_dark.py`; contact sheet `results/stage1_kd/aug_samples.jpg`; miss summaries `results/stage1_kd/misses/{labgtaugL,labgtL}/summary.md` (in git). Data `data/stage1_negatives/lab_gt_aug` and weights `weights/stage1_repro846_labgtaug_p250{,_last}.pt` are on the dev PC only. Logs (gitignored): `results/stage1_kd/repro846_labgtaug_p250.log`, `results/stage1_kd/rules/compare_labgtaug.log`.
