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

### Is the distillation correct? (2026-11-20)

**Question:** the augmented teacher is the best detector, yet its student is worse than a student trained without a teacher. Is the distillation broken?

**What was run:**

```bash
CUDA_VISIBLE_DEVICES=0 TEACHER=weights/stage1_repro846_teachergtaug_p250.pt DEV=0 .venv/Scripts/python.exe results/samples/poc/check_stage1_kd.py
```

The script makes four checks:

1. **The right teacher in the right form.**
   - The KD run's teacher file has the same SHA-1 as the teacher run's `best.pt`.
   - Loaded the way `train_stage1_repro.py` loads it (`YOLO(pt).model`, eval), on 48 lab frames letterboxed to 640x640, its one2one knife score matches `model.predict()` on the same image: mean difference 0.001, largest 0.026 (fp16).
   - In eval mode the YOLO26 head returns the raw logits and box distances the training loss uses (`ultralytics/nn/modules/head.py`, `Detect.forward` returns `(y, preds)`), so the KD loss compares like with like.
   - **The wiring is correct.**
2. **The student copies the teacher**, but only a little more than a student trained without one:
   - A distilled student finds 93–95 % of the knives its teacher finds at 0.25; one trained without a teacher finds 90–91 %.
   - The correlation of their confs on the test knives is about 0.72–0.74 in both cases.
3. **The false boxes away from knives are the same.** Count only boxes at ≥ 0.25 that touch no test knife:

   | Student | Teacher | Boxes away from knives |
   |---|---|---|
   | `kdgtNL` | `teachergtL` | 105 |
   | `labgtL` | `teachergtL` | 100 |
   | `kdgtaugFL` | `teachergtaug` | 114 |
   | `labgtaugL` | `teachergtaug` | 105 |

   About 60 % of each student's boxes away from knives are also false boxes of its teacher.
4. **The extra "false boxes" of the distilled students are duplicates.** The scoring counts a second box on an already matched knife as a false box. YOLO26 has no NMS: its one2one head is meant to put one box on each object. The distilled students break that.

**Lab test, false boxes at conf 0.25 (duplicates on a found knife in brackets), and best recall at P ≥ 0.90: R (conf, P):**

| Detector | 0.25: P / R, false boxes (duplicates) | P ≥ 0.90, as scored before | P ≥ 0.90, with NMS 0.5 |
|---|---|---|---|
| `labgtL` | 90.9 / 77.3, 107 (7) | 77.3 % (0.25, P 90.9) | 78.3 % (0.20, P 90.8) |
| **`kdgtNL`** | 83.0 / 82.1, 232 (**127**) | 79.1 % (0.40, P 90.5) | **85.0 %** (0.10, P 90.3) |
| `labgtaugL` | 90.6 / 78.7, 112 (7) | 78.7 % (0.25, P 90.6) | 79.5 % (0.20, P 90.3) |
| `kdgtaugFL` | 81.2 / 80.7, 257 (**143**) | 75.1 % (0.45, P 91.6) | 82.5 % (0.15, P 90.4) |
| `kdgtaugF` | 78.9 / 79.9, 295 (167) | 70.8 % (0.50, P 91.5) | 79.9 % (0.25, P 90.5) |
| `teachergtL` | 90.5 / 79.6, 115 (7) | 79.6 % (0.25, P 90.5) | 81.1 % (0.15, P 90.0) |
| `teachergtaug` | 90.0 / 82.6, 126 (14) | 82.6 % (0.25, P 90.0) | 83.0 % (0.20, P 90.4) |

**GTv2, the same columns:**

| Detector | 0.25: P / R, false boxes (duplicates) | P ≥ 0.90, as scored before | P ≥ 0.90, with NMS 0.5 |
|---|---|---|---|
| `labgtL` | 80.8 / 72.5, 59 (3) | 65.2 % (0.50, P 91.8) | 66.4 % (0.45, P 90.1) |
| `kdgtNL` | 77.2 / 75.1, 76 (38) | 70.2 % (0.45, P 92.0) | 71.1 % (0.40, P 91.4) |
| `labgtaugL` | 72.6 / 75.1, 97 (1) | 63.5 % (0.60, P 91.9) | 63.5 % (0.60, P 91.9) |
| `kdgtaugFL` | 77.7 / 74.6, 73 (30) | 67.5 % (0.40, P 90.9) | 70.8 % (0.35, P 91.3) |
| `teachergtL` | 90.4 / 71.3, 26 (3) | 71.3 % (0.25, P 90.4) | 73.4 % (0.15, P 90.3) |
| `teachergtaug` | 91.0 / 74.3, 25 (3) | 74.3 % (0.25, P 91.0) | 75.4 % (0.20, P 90.2) |

**Findings:**

- **The distillation runs as written, but the KD loss has a flaw.** It also distills the one2one head, and the students come out with two boxes on many knives. A duplicate box on a real knife cannot raise a false alarm. But the box scoring counted it as a false box, so every distilled student's precision was understated, and its working point was pushed up to 0.40–0.50.
- **With duplicates removed, distillation helps on the lab camera.** `kdgtNL` reaches 85.0 % recall at 90 % precision, above its own teacher (81.1 %) and above every model without NMS. On GTv2 it gains less: 71.1 % against 70.2 %.
- **The augmented teacher's student is still not better** with NMS: 82.5 % lab and 70.8 % GTv2, against 85.0 % and 71.1 % for `kdgtNL`. The teacher's gain from the blur/dark copies does not reach its student. Why was not tested.
- **The pipeline is NMS-free too** (`src/stage1_detector.py`). Box-level precision there has the same duplicates.

**Possible fixes:**

- **Add NMS in the pipeline.** No retraining, about 0.1 ms for a few boxes.
- **Distill only the one2many head.** This keeps the one-box-per-object head trained only on the labels. It needs a retrain, about 8 h per student.

**Not checked:**

- The NMS conf thresholds (0.10–0.25) were chosen on the test set.
- Alarm-level precision, where only boxes away from a knife matter, was not measured. It is what the pipeline actually needs.

### Why the augmentation helps the teacher but not its student (2026-11-20)

**Question:** the blur/dark copies target the misses, and the augmented teacher is the best teacher. Why is its distilled student not better than the student distilled from the teacher without augmentation?

**What was run:**

```bash
CUDA_VISIBLE_DEVICES=0 GPU=1 DEV=0 .venv/Scripts/python.exe results/samples/poc/investigate_stage1_aug.py
```

The script scores each model at its own working point: NMS 0.5, then the conf with the best recall at precision ≥ 0.90 on the lab test. It then reads off, per knife, hit or miss, and groups the knives by speed and crop brightness (as `analyze_stage1_misses.py`). It also runs the teachers on 300 blurred/darkened training copies. A contact sheet of knives that no model finds is in `results/stage1_kd/aug_invest/never_found_sample.jpg`.

Model letters:

| Letter | Model | Name in the tables above |
|---|---|---|
| A | YOLO26s, `lab_gt` | `labgtL` |
| C | YOLO26s, `lab_gt_aug` | `labgtaugL` |
| E | YOLO26x teacher, `lab_gt` | `teachergtL` |
| F | YOLO26x teacher, `lab_gt_aug` | `teachergtaug` |
| B | KD from E | `kdgtNL` |
| H | KD from F | `kdgtaugFL` |

**Working points (lab, NMS 0.5):**

| Model | Conf | P | R | False boxes |
|---|---|---|---|---|
| A | 0.20 | 90.8 | 78.3 | 109 |
| C | 0.20 | 90.3 | 79.5 | 117 |
| E | 0.15 | 90.0 | 81.1 | 124 |
| F | 0.20 | 90.4 | 83.0 | 121 |
| B | 0.10 | 90.3 | **85.0** | 125 |
| H | 0.15 | 90.4 | 82.5 | 120 |

**Recall at that working point, by knife group:**

| Group | Knives | A | C | E | F | B | H |
|---|---|---|---|---|---|---|---|
| Slow (< 0.5 box sides per 10 frames) | 264 | 87.9 % | 91.3 % | 89.8 % | 91.3 % | 93.9 % | 92.0 % |
| Medium (0.5–2) | 700 | 84.6 % | 83.6 % | 85.7 % | 87.4 % | 89.3 % | 87.9 % |
| Fast (2–4) | 223 | 67.3 % | 69.5 % | 72.2 % | 74.9 % | 78.5 % | 73.5 % |
| Very fast (≥ 4) | 92 | 48.9 % | 55.4 % | 56.5 % | 59.8 % | 58.7 % | 53.3 % |
| Dark (grey < 100) | 207 | 60.9 % | 62.3 % | 68.6 % | 71.5 % | 74.9 % | 69.1 % |
| Bright (≥ 140) | 634 | 86.1 % | 86.3 % | 88.2 % | 88.8 % | 89.4 % | 88.6 % |
| Fast and dark | 94 | 50.0 % | 55.3 % | 54.3 % | 63.8 % | 66.0 % | 60.6 % |

**Churn: knives found by one model and not the other:**

| Without → with | Gained | Lost | Net | Median conf of the lost knives: before → after |
|---|---|---|---|---|
| A → C (augmentation) | 74 | 58 | +16 | 0.73 → 0.02 |
| E → F (augmentation) | 74 | 48 | +26 | 0.54 → 0.05 |
| B → H (augmentation + KD) | 35 | 70 | −35 | 0.58 → 0.03 |
| E → B (KD) | 95 | 41 | +54 | |
| F → H (KD) | 58 | 65 | −7 | |

**Teacher score on 300 training copies (302 knives).** The second column says whether the teacher was trained on these copies.

| Teacher | Trained on the copies | Median score, unchanged frame | Median score, copy | Copies < 0.5 |
|---|---|---|---|---|
| E | no | 0.96 | 0.06 | 63 % |
| F (`best.pt`) | yes | 0.94 | 0.95 | 1 % |
| G (F, `last.pt`) | yes | 0.95 | 0.96 | 0 % |

**Lab training data.** 3,159 lab frames with 1,856 knife boxes in `lab_gt`. Every swing type of the test takes is in training:
- slash, about 200 boxes;
- overhand stab, about 230;
- cut down, about 180;
- thrust low, about 160.

**Findings:**

- **The augmentation does what it was meant to, without distillation.** It raises recall on fast and dark knives:
  - fast-and-dark: A → C +5 points, E → F +10 points;
  - very fast: A → C +6.5 points.

  Sharp, slow knives did not lose confidence. They rose for both: A → C 87.9 % to 91.3 %, E → F 89.8 % to 91.3 %. The copies did not teach "a knife must look blurred".
- **Each retrain changes 100–130 knives, much more than the net gain.** A → C gains 74 knives and loses 58. The lost ones were not borderline: A had them at a median conf of 0.73, and C has them at 0.02. That means single runs differ by about ±2–3 points on this test set (1,376 knives) from training noise alone. The B (85.0 %) vs H (82.5 %) gap is 35 knives, inside that range. No recipe here was trained twice, so the noise was not measured directly.
- **The teacher's signal on the copies is not the problem.** F scores the copies as confidently as the originals (0.95), so distillation pushes the student toward the label there, not against it. Without augmentation, E gives most copies a score below 0.06: the copies are hard examples, as intended.
- **Training data covers the swings.** The test misses are not motions missing from training.
- **About half the knives no model finds are not blurred.** 117 test knives are missed by all six models:
  - 57 are fast (≥ 2 box sides), 31 are dark;
  - the rest are slow knives hidden in the hand, held at the side against dark trousers or the chair, or pointing at the camera;
  - on several of them no knife is visible in the frame, and the label comes from the frames around it.

  Augmentation cannot teach those, and a sharper or softer blur will not change them.
- **Where distillation from F loses** (B → H): the knives it loses are mostly in slash (10 lost, 1 gained), normal mixed (16 lost, 7 gained) and Oct 31 take 002 (24 lost, 10 gained). Their conf in H is near 0, not just under the threshold.
- **The B-vs-H gap has two possible causes:** the training noise above, or teacher F being `best.pt` from epoch 153 while E was `last.pt` from epoch 250. These cannot be told apart without repeat runs.

**Not checked:**
- No recipe was trained twice with a different seed, so the noise level is inferred from churn, not measured.
- The working points were chosen on the test set.

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
