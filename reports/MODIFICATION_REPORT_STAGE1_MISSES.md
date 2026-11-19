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

## Not checked

- The causes come from one model at one threshold, on test labels made by the same process as the training labels. The 408 left-out hard test frames are not in the set, so real recall is lower.
- Brightness and speed are rough measures. Speed uses labelled frames 10 apart, and a box with no labelled neighbour frame has no speed (110 of 1,376 boxes).
- The hold estimate counts a knife as covered if a box was near it earlier in the same take. The pipeline holds the weapon on the tracked person, which may differ. The hold also keeps false boxes alive; precision under the hold was not measured.
- The zoom test chose its threshold on the test set. No Edge PC timing.

## Files

- Scripts: `results/samples/poc/analyze_stage1_misses.py`, `zoom_stage1_wrists.py`, `held_recall_stage1.py`.
- In git: `results/stage1_kd/misses/kdgtNL/{summary.md, zoom_summary.md, misses.json}` and the contact sheets `{unseen,low_score,bad_box}_*.jpg`.
