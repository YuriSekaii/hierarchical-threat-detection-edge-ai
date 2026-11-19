# Modification Report: Stage 1 alternatives for high recall with the phone rejected (2026-11-13 to 2026-11-14)

**Status: experiment. The shipped `weights/yolo_weapon_distilled.pt` and the pipeline are unchanged.** All new weights are separate files named `weights/stage1_*.pt`. Nothing was exported to TensorRT, run through the full pipeline or tried on the Edge PC.

## 1. Question

Stage 1 flags a phone held like a knife, and retraining with phone frames as negatives costs recall (`MODIFICATION_REPORT_STAGE1_PHONE_NEGATIVES.md`). Recall matters most, because the hand gate, the static filter and Stage 2 remove false detections later. Which model or combination gives the highest recall with the best precision and the phone rejected?

Ideas came from the first phone-negative test and from `Edge Knife Detection Literature Review.md` (phone as its own class, training at the deployed scale, wrist-region zoom, temporal voting).

## 2. What was run

**Note (2026-11-16): all detector retrains below ran under Ultralytics 8.4.9 and lost recall for that reason, see 4e. The crop classifier results are not affected.**

**Benchmark:** `results/samples/poc/bench_stage1.py`. Every system gets the same three measurements. All detectors run at 384x640.

```bash
DEV=0 .venv/Scripts/python.exe results/samples/poc/bench_stage1.py "det,weights/yolo_weapon_distilled.pt,0.45" "crop,weights/yolo_weapon_distilled.pt,0.25,weights/stage1_cropcls2.pt,0.3"
```

In Git Bash set `MSYS2_ARG_CONV_EXCL="*"` first when a spec contains a `D:/` path. Raw rows, with per-take values, are appended to `results/stage1_phoneneg/bench.jsonl`.

- **Ground truth:** the 715 images of `data/Ground_Truth_Dataset`, same counting as `scripts/validate_stage1_ground_truth.py` (IoU 0.20). The harness reproduces the shipped detector exactly: precision 78.76 %, recall 74.17 %, 19 OOD false detections.
- **Phone takes:** Nov 5 takes 006–009, share of frames (every 3rd) with a detection. Take 008 was never used for training.
- **Knife takes:** Nov 5 takes 002, 005, 010, 014, 018, 022, never used for training. Higher is better.
- **Windows:** share of 10-sample windows (about one second) with at least 3 detections.

**Training:** `results/samples/poc/train_stage1_variants.py <REPO> <variant>`, on `data/stage1_negatives/ft` (backup Stage 1 set of 752 + 189 images, 497 phone frames, 617 lab knife frames) or `ft_phonecls` (the same with the phone boxed as class `phone`: 431 train, 64 validation boxes).

| Variant | What it is | Weights |
|---|---|---|
| Phone negatives | shipped detector fine-tuned, phone frames as background | `yolo_weapon_distilled_phoneneg.pt` |
| Frozen fine-tune | same, backbone frozen, 20 epochs | `stage1_freeze.pt` |
| Fresh two-class | COCO YOLO26s trained on `ft`, 80 epochs | `stage1_scratch2.pt` |
| Phone class | COCO YOLO26s trained on `ft_phonecls`, 80 epochs, 640 px | `stage1_phonecls.pt` |
| Phone class, deployed scale | same at 512 px (a 4:3 frame in the 384x640 engine is 512 px wide) | `stage1_phonecls512.pt` |
| Phone class added to shipped | shipped detector fine-tuned on `ft_phonecls`, 512 px | `stage1_phonecls_ft.pt` |
| Extra-large phone class | COCO YOLO26x on `ft_phonecls`, 60 epochs | `stage1_big.pt` |
| Crop classifier | knife / other classifier on the detected box enlarged 50 % per side, 128 px | `stage1_cropcls.pt` |
| Crop classifier v2 | plus 172 low-confidence lab knife crops, 224 px, validation accuracy 94.9 % | `stage1_cropcls2.pt` |

Systems without new training: confidence threshold, a second detector that must confirm the box, a stock COCO detector that vetoes a box it calls "cell phone", and wrist-region zoom (pose gives the wrist, the detector runs again on the enlarged region; the detector was not retrained on such crops).

## 3. Results

"Phone" is take 008 / mean of 006, 007, 009. "Knife" is the worst take / mean of the six knife takes.

| System | Precision | Recall | F1 | OOD false detections | Phone frames | Knife frames | Phone windows (max) | Knife windows (min) |
|---|---|---|---|---|---|---|---|---|
| **Shipped, 0.45 (today)** | 78.8 % | 74.2 % | 76.4 % | 19 | 19 % / 35 % | 36 % / 61 % | 88 % | 59 % |
| Shipped, 0.25 | 69.3 % | 83.8 % | 75.9 % | 44 | 24 % / 43 % | 48 % / 68 % | 97 % | 65 % |
| **Shipped 0.25 + crop classifier v2 at 0.3** | 73.7 % | 81.7 % | 77.5 % | 26 | 1 % / 2 % | 39 % / 60 % | 2 % | 65 % |
| Shipped 0.25 + crop classifier v2 at 0.2 | 73.1 % | 82.5 % | 77.5 % | 28 | 3 % / 2 % | 41 % / 62 % | 2 % | 65 % |
| Shipped 0.15 + crop classifier v2 at 0.3 | 68.1 % | 84.6 % | 75.5 % | 34 | 3 % / 3 % | 42 % / 62 % | 2 % | 67 % |
| Shipped 0.45 + crop classifier v2 at 0.5 | 83.6 % | 72.1 % | 77.4 % | 7 | 0 % / 1 % | 25 % / 51 % | 0 % | 42 % |
| Shipped 0.25 + crop classifier v1 at 0.5 | 74.1 % | 81.3 % | 77.5 % | 25 | 2 % / 2 % | 36 % / 51 % | 2 % | 64 % |
| Phone negatives, 0.45 | 91.3 % | 52.5 % | 66.7 % | 3 | 3 % / 2 % | 28 % / 45 % | 6 % | 49 % |
| Frozen fine-tune, 0.25 | 73.0 % | 72.1 % | 72.5 % | 26 | 10 % / 2 % | 29 % / 47 % | 18 % | 53 % |
| Fresh two-class, 0.25 | 85.4 % | 60.8 % | 71.1 % | 4 | 5 % / 3 % | 36 % / 52 % | 11 % | 59 % |
| Phone class, 0.15 | 76.1 % | 70.4 % | 73.2 % | 16 | 2 % / 1 % | 43 % / 53 % | 3 % | 62 % |
| Phone class, deployed scale, 0.15 | 74.5 % | 63.3 % | 68.5 % | 16 | 1 % / 2 % | 40 % / 53 % | 4 % | 59 % |
| Phone class added to shipped, 0.25 | 76.4 % | 74.2 % | 75.3 % | 13 | 3 % / 3 % | 26 % / 42 % | 10 % | 42 % |
| Phone class added to shipped, 0.10 | 63.2 % | 80.8 % | 70.9 % | 37 | 6 % / 4 % | 28 % / 47 % | 12 % | 51 % |
| Extra-large phone class, 0.15 | 73.9 % | 72.1 % | 73.0 % | 14 | 0 % / 2 % | 37 % / 53 % | 9 % | 62 % |
| Shipped 0.25, confirmed by phone-negative model at 0.05 | 84.3 % | 67.1 % | 74.7 % | 6 | 10 % / 2 % | 36 % / 55 % | 18 % | 58 % |
| Shipped 0.25, COCO "cell phone" veto (YOLO26s) | 69.2 % | 83.3 % | 75.6 % | 44 | 24 % / 42 % | 48 % / 68 % | 97 % | 65 % |
| Shipped 0.25, COCO veto (YOLO26x, 0.10) | 67.8 % | 77.9 % | 72.5 % | 44 | 24 % / 32 % | 48 % / 67 % | 95 % | 65 % |
| Wrist zoom only, shipped 0.25 | 60.2 % | 47.9 % | 53.4 % | 30 | 31 % / 43 % | 44 % / 58 % | 81 % | 59 % |
| Full frame + wrist zoom, shipped 0.25 | 59.2 % | 87.1 % | 70.5 % | 65 | 37 % / 62 % | 51 % / 73 % | 100 % | 66 % |
| Full frame + wrist zoom 0.25 + crop classifier v1 at 0.5 | 66.9 % | 81.7 % | 73.6 % | 36 | 3 % / 3 % | 40 % / 53 % | 3 % | 65 % |

## 4. Findings

- **Best system: shipped detector at 0.25, then crop classifier v2 at 0.3.** Against today's setting, recall rises from 74.2 % to 81.7 %, the phone drops from 19–35 % of frames to 1–2 % (also on the take never trained on), and the lab knife is detected as often as today (60 % against 61 % of frames; worst take 39 % against 36 %). Precision is 73.7 % against 78.8 %, with 26 OOD false detections against 19.
- **Why it works where retraining did not:** the shipped detector is left as it is, so its recall is kept. The classifier only removes boxes.
- **Every retrained detector loses recall** (52–72 % at usable thresholds), whether the phone is background or its own class, small or extra-large, 640 or 512 px. They were trained on the 752-image backup set without the teacher, so they do not reach the shipped detector, which was distilled on more data.
- **Phone as its own class is better than phone as background** for the same training (recall 70.4 % against 60.8 % at similar phone rejection), as the literature review predicted, but not enough to beat the shipped detector.
- **The stock COCO phone veto does not work:** COCO models do not see the phone in these frames and remove real knives instead.
- **Wrist zoom raises recall to 87.1 % but adds many false detections** (65 on OOD images, phone flagged more). With the classifier behind it, it is no better than the classifier alone. The detector was not retrained on zoomed crops, which the literature review says is needed.
- **Cost:** the classifier has 5.4 M parameters and takes about 5.8 ms per detected box in PyTorch on an RTX 3090. It runs only on frames where Stage 1 fires.

## 4b. Why 640x640 scores 86.9 % and 384x640 scores 76.4 % (2026-11-14)

All rows of section 3 were measured at 384x640.

### What was run

```bash
.venv/Scripts/python.exe scripts/validate_stage1_ground_truth.py --weights weights/yolo_weapon_distilled.pt --imgsz 416 640
```

Repeated for the sizes below, confidence 0.45. The test images are 2560x1600 (16:10). The shrink factor is `min(H / 1600, W / 2560)`.

For the lab camera: a one-off script ran the detector on Nov 5 knife takes 002, 005, 010, 014, 018, 022 and phone takes 006, 008 (every 6th frame, confidence 0.45) with the frame shrunk four ways and pasted unscaled into a 384x640 input.

### Ground-truth set

| Input size | Shrink factor | Precision | Recall | F1 |
|---|---|---|---|---|
| 384x640 (deployed) | 0.240 | 78.8 % | 74.2 % | 76.4 % |
| 384x672 | 0.240 | 78.8 % | 74.2 % | 76.4 % |
| 384x608 | 0.2375 | 79.3 % | 76.7 % | 78.0 % |
| 416x608 | 0.2375 | 79.7 % | 76.7 % | 78.1 % |
| 400x640 | 0.250 | 88.2 % | 84.2 % | 86.1 % |
| 416x640 | 0.250 | 88.2 % | 84.2 % | 86.1 % |
| 480x640 | 0.250 | 89.0 % | 84.2 % | 86.5 % |
| 640x640 | 0.250 | 89.8 % | 84.2 % | 86.9 % |
| 448x704 | 0.275 | 75.5 % | 73.3 % | 74.4 % |
| 512x864 | 0.3375 | 67.5 % | 75.4 % | 71.3 % |
| 768x1280 | 0.480 | 67.7 % | 61.3 % | 64.3 % |

- Every size that shrinks the image by exactly 0.25 gives recall 84.2 % and F1 86–87 %. 640x640 is the same pixels as 416x640 plus padding.
- Every other factor is about 10 F1 points lower, including larger inputs. The score does not rise with resolution.
- 384x640 gives factor 0.24 only because the test images are 16:10; the image height, not the width, limits the fit.

Knife size at the deployed input (240 knives): median 26 px long and 17 px wide, none longer than 48 px.

| Knife long side at 384x640 | Knives | Recall at 384x640 | Recall at 640x640 |
|---|---|---|---|
| under 16 px | 5 | 60 % | 100 % |
| 16–24 px | 84 | 75 % | 86 % |
| 24–32 px | 92 | 79 % | 88 % |
| 32–48 px | 59 | 75 % | 78 % |

### Lab camera

Share of frames with a detection. The lab camera is 2560x1440 (16:9); the pipeline analyses the 1920x1440 centre crop, which a 384x640 input shrinks by 0.267.

| Take | 4:3 crop as fed today (0.267) | 4:3 crop at 0.25 | Full 16:9 frame (0.25) | 4:3 crop at 0.30 |
|---|---|---|---|---|
| 002 knife | 88 % | 81 % | 81 % | 83 % |
| 005 knife | 74 % | 76 % | 77 % | 74 % |
| 010 knife | 66 % | 63 % | 64 % | 65 % |
| 014 knife | 35 % | 34 % | 35 % | 39 % |
| 018 knife | 41 % | 45 % | 41 % | 42 % |
| 022 knife | 62 % | 54 % | 62 % | 58 % |
| 006 phone | 61 % | 58 % | 57 % | 55 % |
| 008 phone | 19 % | 21 % | 19 % | 16 % |

No factor is better than another on the lab camera.

### Conclusion

- The 10-point gap is a property of the test set, not of the input resolution. The detector is tuned to how the test-set images look after a 4x shrink, which suggests those images are close to its training images.
- 416x640 is the right size for scoring this 16:10 test set. It is not better for the live camera: a 16:9 frame already gets factor 0.25 in a 384x640 input, and the measured lab takes show no gain at any factor.
- Moving the pipeline to 416x640 would need new TensorRT engines for the detector and the pose model on both PCs and about 8 % more pixels per frame, for no measured gain on the camera. It was not done. The size function `calculate_rectangular_imgsz` now rounds the height up instead of to the nearest 32, so a 16:10 source gets 416x640 with PyTorch weights; a fixed 384x640 TensorRT engine is not affected.
- The 86.9 % is an upper figure for images like the test set. The lab-camera detection share (36–88 % of frames) is the figure that describes the deployed system.


The training set explains it: `D:\Intern AI Project\Clean\Weapon_Detection_SingleGPU` holds the original set and distillation script. All 941 images are 2560x1600 and training ran at 640, so the detector only ever saw knives at the 0.25 factor.

Main rows re-run at 416x640 (`IMGSZ=416,640` before the benchmark command):

| System | Precision | Recall | F1 | OOD false detections | Phone frames |
|---|---|---|---|---|---|
| Shipped, 0.45 | 88.6 % | 83.8 % | 86.1 % | 18 | 18 % / 33 % |
| Shipped, 0.25 | 73.4 % | 90.8 % | 81.2 % | 44 | 24 % / 41 % |
| Shipped 0.25 + crop classifier v2 at 0.3 | 78.8 % | 88.3 % | 83.3 % | 25 | 2 % / 2 % |
| Shipped 0.45 + crop classifier v2 at 0.5 | 93.4 % | 76.7 % | 84.2 % | 6 | 0 % / 1 % |
| Phone class added to shipped, 0.25 | 75.6 % | 78.8 % | 77.1 % | 18 | 3 % / 3 % |
| Extra-large phone class, 0.15 | 70.7 % | 71.3 % | 71.0 % | 17 | 0 % / 2 % |
| Phone negatives, 0.45 | 91.5 % | 53.8 % | 67.7 % | 2 | 2 % / 2 % |

The order is the same as at 384x640: the crop classifier behind the shipped detector keeps the recall and removes the phone, and every retrained detector is below the shipped one. At this size the shipped detector at 0.45 has the highest F1 (86.1 %); the classifier setting trades 10 points of precision for 4.5 points of recall and the phone rejection.

## 4c. The original training recipe, and what "distilled" meant (2026-11-14)

**Superseded by 4e (2026-11-16): the retrains in this section ran under Ultralytics 8.4.9, whose optimizer differs from the shipped run. Their recall loss comes from that, not from the data.**

The original set and script are in `D:\Intern AI Project\Clean\Weapon_Detection_SingleGPU` (`training/train_yolo_distill.py`, `Weapon_Detection_Split`): 941 images (752 train, 189 validation), one class, all 2560x1600. It is the same set as the backup used above.

- The script's `KnowledgeDistillationTrainer` loads the YOLO26x teacher and freezes it, but nothing uses it in the loss. The older copies in `Messy (Don't Touch)` are the same. The shipped "distilled" student is a plain YOLO26s trained for up to 250 epochs (patience 30, batch 32, 640 px, rotation 15°, mixup 0.1).
- So the student did not beat its teacher through distillation; the smaller model simply did better on 752 images.
- The retrains in section 2 used 20–80 epochs and default augmentation, not this recipe. `results/samples/poc/train_stage1_distill_phone.py <REPO> ft_phonecls|ft` repeats the original student recipe on the phone-class and the phone-negative sets. Results are below.


### Result: the original recipe with phone data still loses recall

Both runs stopped early (patience 30): phone as its own class after 91 epochs (best at 61, 1.9 h), phone as background after 144 epochs (best at 114, 3.0 h). Benchmarked with `bench_stage1.py`, at 384x640 and with `IMGSZ=416,640`.

| System | Conf | Size | Precision | Recall | F1 | OOD FP | Phone 008 | Phone 006/007/009 | Knife frames, mean |
|---|---|---|---|---|---|---|---|---|---|
| Shipped detector | 0.45 | 384x640 | 78.8 % | 74.2 % | 76.4 % | 19 | 19 % | 35 % | 61 % |
| Shipped detector | 0.25 | 416x640 | 73.4 % | 90.8 % | 81.2 % | 44 | 24.3 % | 41.2 % | |
| Shipped 0.25 + crop classifier 0.3 | | 416x640 | 78.8 % | 88.3 % | 83.3 % | 25 | 2.0 % | 2.0 % | |
| `stage1_distill_ft_phonecls.pt` (phone as class) | 0.15 | 384x640 | 73.0 % | 67.5 % | 70.1 % | 21 | 3.1 % | 1.9 % | 52.3 % |
| | 0.25 | 384x640 | 79.3 % | 59.2 % | 67.8 % | 11 | 2.6 % | 1.2 % | 48.6 % |
| | 0.45 | 384x640 | 90.8 % | 49.2 % | 63.8 % | 5 | 1.5 % | 0.8 % | 43.5 % |
| | 0.15 | 416x640 | 72.6 % | 65.0 % | 68.6 % | 22 | 1.5 % | 1.9 % | 47.7 % |
| | 0.25 | 416x640 | 79.8 % | 55.8 % | 65.7 % | 10 | 0.4 % | 1.1 % | 44.7 % |
| | 0.45 | 416x640 | 88.8 % | 46.3 % | 60.8 % | 4 | 0.2 % | 0.4 % | 38.5 % |
| `stage1_distill_ft.pt` (phone as background) | 0.15 | 384x640 | 71.6 % | 57.9 % | 64.1 % | 36 | 5.9 % | 2.7 % | 62.7 % |
| | 0.25 | 384x640 | 79.3 % | 55.8 % | 65.5 % | 21 | 4.2 % | 1.8 % | 58.4 % |
| | 0.45 | 384x640 | 89.1 % | 47.5 % | 62.0 % | 7 | 2.0 % | 0.9 % | 52.1 % |
| | 0.15 | 416x640 | 67.6 % | 59.2 % | 63.1 % | 47 | 6.6 % | 2.6 % | 64.4 % |
| | 0.25 | 416x640 | 74.0 % | 53.3 % | 62.0 % | 31 | 5.9 % | 1.8 % | 59.6 % |
| | 0.45 | 416x640 | 86.3 % | 44.6 % | 58.8 % | 7 | 2.6 % | 0.9 % | 53.3 % |
| `stage1_distill_ft.pt` 0.15 + crop classifier 0.3 | | 416x640 | 75.5 % | 57.9 % | 65.6 % | 24 | 3.1 % | 2.0 % | 61.9 % |

- Negative result. Even at confidence 0.15 the best retrain reaches 67.5 % recall, against 74.2 % (0.45) and 83.8 % (0.25) for the shipped detector at the same size. The short recipe was not the reason the earlier retrains lost recall.
- The phone false detections do drop, to 0–7 % of frames, as in every other retrain.
- These detectors do not gain from 416x640; the 0.25 shrink effect of section 4b is specific to the shipped weights.
- Raw tables: `results/stage1_phoneneg/bench_distill_384.md`, and the rows in `bench.jsonl`.

### Control: the same recipe on the original 752 images alone

`train_stage1_distill_phone.py <REPO> orig` (data yaml `data/stage1_negatives/orig/data.yaml`, pointing at `D:\Backup\Weapon_Detection_Split`). Stopped after 177 epochs (best at 147, 1.6 h). Weights: `weights/stage1_distill_orig.pt`.

| System | Conf | Size | Precision | Recall | F1 | OOD FP | Phone 008 | Phone 006/007/009 | Knife frames, mean |
|---|---|---|---|---|---|---|---|---|---|
| Shipped detector | 0.45 | 384x640 | 78.8 % | 74.2 % | 76.4 % | 19 | 19 % | 35 % | 61 % |
| `stage1_distill_orig.pt` | 0.15 | 384x640 | 76.2 % | 65.4 % | 70.4 % | 16 | 14.0 % | 23.1 % | 59.1 % |
| | 0.25 | 384x640 | 80.3 % | 59.6 % | 68.4 % | 13 | 11.2 % | 17.9 % | 53.8 % |
| | 0.45 | 384x640 | 90.8 % | 49.2 % | 63.8 % | 5 | 7.2 % | 12.5 % | 45.8 % |
| | 0.15 | 416x640 | 74.2 % | 60.0 % | 66.4 % | 17 | 15.8 % | 22.2 % | 57.7 % |
| | 0.25 | 416x640 | 82.1 % | 57.5 % | 67.7 % | 10 | 12.1 % | 17.3 % | 53.2 % |
| | 0.45 | 416x640 | 87.8 % | 47.9 % | 61.99 % | 4 | 9.2 % | 10.8 % | 45.8 % |
| `stage1_distill_orig.pt` 0.25 + crop classifier 0.3 | | 384x640 | 83.4 % | 58.8 % | 69.0 % | 8 | 0.9 % | 2.3 % | 47.4 % |

- The control does not reproduce the shipped detector: 49.2 % recall at 0.45 against 74.2 %. The recipe and the 752 images found on disk do not give the shipped weights.
- Adding the phone as a class costs no recall against this like-for-like control (49.2 % at 0.45 and 59.2 % against 59.6 % at 0.25 for both), and removes the phone detections (7–23 % of frames down to 1–3 %). So the phone data works as intended; the recall gap is between any retrain here and the shipped weights.
- Correction (2026-11-15): an earlier version of this section said the shipped weights have two classes. They have one (`{0: 'knife'}`); the checkpoint is dated 2026-01-22, Ultralytics 8.4.6, student recipe, data `D:\Weapon_Detection_SingleGPU\Weapon_Detection_Split`, a folder that no longer exists. So the class list does not explain the gap; the image set at that path may have differed from the copy on disk.
- Not checked: a second seed of the control.

## 4d. A real teacher and real distillation, with Nov 14 lab frames (2026-11-15)

**Superseded by 4e (2026-11-16): the retrains in this section ran under Ultralytics 8.4.9, whose optimizer differs from the shipped run. Their recall loss comes from that, not from the data.**

**Status: test only. No shipped weight was changed. Result: better on the lab camera, worse on the ground-truth set; not a replacement yet.**

### Question

The intended recipe was: fine-tune YOLO26x as teacher, distil it into YOLO26s. The original script never used the teacher (4c). Does the recipe, done for real and with lab frames of Nov 14 added, beat the shipped detector on data no detector here was trained on? Also: is YOLO27 usable?

- **YOLO27:** no. The Ultralytics page (https://docs.ultralytics.com/models/yolo27, read 2026-11-15) says the model is not released: no weights, no date. It points to YOLO26.

### What was run

Script `results/samples/poc/train_stage1_kd.py` (new), each with `DEV=<gpu> .venv/Scripts/python.exe results/samples/poc/train_stage1_kd.py . <mode>`:

| Mode | What | Result |
|---|---|---|
| `build` | Set `data/stage1_negatives/lab1114` = `ft_phonecls` + every 15th frame of the Nov 14 `train_attack`, `train_normal` and `confuser` takes. Knife takes: frame kept when the shipped detector fires at 0.5 or more, box = class `knife`. Object takes: box at 0.25 or more = class `phone`, other frames as background. Takes 020, 041, 050, 056 go to validation. | 3565 train, 369 validation images |
| `teacher` | YOLO26x, original teacher recipe (200 epochs, patience 20, batch 12, 640 px, rotation 15°, mixup 0.1) | stopped at epoch 72, best 52, validation mAP50 0.855 → `weights/stage1_kd_teacher.pt` |
| `student 0` | YOLO26s, original student recipe (250 epochs, patience 30, batch 32), no teacher | stopped at 147, best 117, mAP50 0.860 → `weights/stage1_kd_control.pt` |
| `student` | Same, plus a distillation loss (weight 1.0): on every anchor of both heads, the teacher's class scores as soft targets and the distance to the teacher's box, weighted by the teacher's score | stopped at 121, best 91, mAP50 0.839 → `weights/stage1_kd_student.pt` |

Scoring, all at 384x640: `bench_stage1.py "det,<weights>,<conf>" ...` (ground-truth set, Nov 5 takes) and `bench_stage1_holdout.py` (new, same arguments): share of frames with a knife detection on the Nov 14 test takes 025–036 and Oct 31 take 002, and on the Nov 14 object takes 050 (second phone) and 056 (remote), where lower is better.

### Results

Ground-truth set (715 images, original camera) and Nov 5 takes:

| System | Conf | Precision | Recall | F1 | OOD FP | Phone 008 | Phone 006/007/009 | Nov 5 knife frames, mean |
|---|---|---|---|---|---|---|---|---|
| Shipped | 0.15 | 61.3 % | 87.1 % | 71.9 % | 61 | 27.4 % | 49.3 % | 70.6 % |
| | 0.25 | 69.3 % | 83.8 % | 75.8 % | 44 | 23.5 % | 43.1 % | 67.9 % |
| | 0.45 | 78.8 % | 74.2 % | 76.4 % | 19 | 19 % | 35 % | 61 % |
| Teacher (YOLO26x) | 0.15 | 74.6 % | 79.6 % | 77.0 % | 17 | 4.0 % | 0.6 % | 67.6 % |
| | 0.25 | 79.4 % | 77.1 % | 78.2 % | 9 | 3.3 % | 0.5 % | 65.8 % |
| | 0.45 | 87.0 % | 72.5 % | 79.1 % | 3 | 1.5 % | 0.3 % | 61.7 % |
| Student, no teacher | 0.15 | 60.3 % | 80.8 % | 69.0 % | 54 | 2.0 % | 1.0 % | 66.2 % |
| | 0.25 | 73.1 % | 77.9 % | 75.4 % | 16 | 1.3 % | 0.8 % | 63.7 % |
| | 0.45 | 83.3 % | 70.8 % | 76.6 % | 6 | 0.7 % | 0.8 % | 59.4 % |
| Student, distilled | 0.15 | 68.1 % | 80.0 % | 73.6 % | 16 | 7.9 % | 0.3 % | 77.0 % |
| | 0.25 | 80.7 % | 72.9 % | 76.6 % | 4 | 4.8 % | 0.3 % | 74.6 % |
| | 0.45 | 90.9 % | 57.9 % | 70.7 % | 0 | 1.5 % | 0.0 % | 66.1 % |

Lab takes none of these detectors was trained on:

| System | Conf | Nov 14 test takes, frames (mean / worst take) | ~1 s windows (mean / worst) | Oct 31 take 002 | Second phone 050 | Remote 056 |
|---|---|---|---|---|---|---|
| Shipped | 0.15 | 50 % / 29 % | 68 % / 49 % | 65 % | 19 % | 22 % |
| | 0.25 | 47 % / 27 % | 66 % / 43 % | 61 % | 13 % | 19 % |
| | 0.45 | 39 % / 21 % | 59 % / 35 % | 54 % | 8 % | 13 % |
| Shipped 0.25 + crop classifier v2 0.3 | | 45 % / 26 % | 64 % / 42 % | 58 % | 13 % | 16 % |
| Teacher (YOLO26x) | 0.15 | 55 % / 35 % | 70 % / 53 % | 72 % | 6 % | 6 % |
| | 0.25 | 53 % / 33 % | 68 % / 50 % | 70 % | 5 % | 5 % |
| | 0.45 | 49 % / 28 % | 64 % / 46 % | 65 % | 3 % | 4 % |
| Student, no teacher | 0.15 | 56 % / 35 % | 71 % / 52 % | 73 % | 6 % | 4 % |
| | 0.25 | 54 % / 30 % | 69 % / 45 % | 70 % | 3 % | 3 % |
| | 0.45 | 49 % / 25 % | 65 % / 40 % | 66 % | 3 % | 3 % |
| Student, distilled | 0.15 | 58 % / 36 % | 72 % / 56 % | 75 % | 4 % | 5 % |
| | 0.25 | 54 % / 34 % | 69 % / 52 % | 71 % | 3 % | 5 % |
| | 0.45 | 47 % / 26 % | 62 % / 41 % | 62 % | 1 % | 4 % |

### Findings

- **Lab camera: all three retrains beat the shipped detector.** At 0.25 they see the knife in 53–54 % of the Nov 14 test frames against 47 %, and in 70–71 % of Oct 31 take 002 against 61 %. They fire on the second phone and the remote in 3–5 % of frames against 13–19 %; the crop classifier of section 3 does not reject these two objects.
- **Ground-truth set: recall is still lower than shipped**, 80–81 % at 0.15 against 87 %, and 73–78 % at 0.25 against 84 %. The gap is smaller than in 4c (best there 67.5 % at 0.15) and precision is higher: at about 80 % recall the teacher and the distilled student have 17 and 16 OOD false detections, the shipped detector at 84 % recall has 44.
- **Distillation gives a small gain, not a clear one.** Against the no-teacher student, the distilled one is better on the worst Nov 14 test take (34 % against 30 % of frames at 0.25) and on Nov 5 knife frames (74.6 % against 63.7 %), with fewer OOD false detections. But its ground-truth recall falls faster with confidence (57.9 % against 70.8 % at 0.45), and it is one run per model. The teacher is no better than the students on lab takes, so there is little for the student to learn from it there.
- **Still far from enough on the lab camera:** the best system sees the knife in 58 % of the Nov 14 test frames and 36 % of the worst take.

### Not checked

- The lab boxes are the shipped detector's own detections, not hand labels. Lab knives the shipped detector misses are not in the training set, and frames where the knife is hidden were not separated, so the frame shares are not recall. Hand-labelled lab frames are the missing piece.
- The Nov 14 test takes are held-out takes of the same person, room and day as the Nov 14 training frames. Oct 31 take 002 is another day; no other person or room was tested.
- The Nov 5 knife takes and phone takes 006, 007, 009 are in the training set of the three new models, so those two columns are not held-out for them.
- One run per model, one distillation weight (1.0). The confidence each model would run at was not tuned.
- No TensorRT export, no timing on the Edge PC, no run of the full pipeline (hand gate, static filter, Stage 2).

### Files

- Scripts: `results/samples/poc/train_stage1_kd.py`, `results/samples/poc/bench_stage1_holdout.py`.
- Rows: `results/stage1_kd/holdout.jsonl` (in git); logs and training runs in `results/stage1_kd/`, benchmark rows in `results/stage1_phoneneg/bench.jsonl` (dev PC only).
- Data `data/stage1_negatives/lab1114` and weights `weights/stage1_kd_{teacher,control,student}.pt`: dev PC only.

## 4e. Why every retrain lost recall: the Ultralytics version (2026-11-15 to 2026-11-16)

**Status: cause found and fixed in the training script. No shipped weight was changed. The retrains of sections 2, 4c and 4d ran with the wrong optimizer; their recall numbers do not describe the data.**

### Question

With more data, every retrain scored below the shipped detector on the ground-truth set. Is the dataset wrong, the benchmark wrong, or the training wrong?

### What was checked

| Check | How | Result |
|---|---|---|
| Benchmark | `bench_stage1.py` on the shipped weights | 74.2 % recall at 0.45 and 384x640; at 640x640 the README's 84 %. The test set has one class and 240 boxes in 715 images. No fault found. |
| Original dataset | Label cache in `Clean\Weapon_Detection_SingleGPU\Weapon_Detection_Split\labels\train.cache`, written at the original path `D:\Weapon_Detection_SingleGPU` | Same as on disk: 752 images, 485 label files, 450 boxes. The data was not lost or changed. |
| Shipped weights | md5 | Identical to `D:\Backup\YOLO_Weapon_Training_Results\distilled_yolo26s_from_yolo26x\weights\best.pt`. Three old runs with the same arguments have the same curve (best epoch 98, mAP50 0.90426), so the original training repeats itself. |
| Training arguments | `args.yaml` of that run against ours | The same. |
| Training curve | `results.csv` of that run against the control of 4c | **Different.** Epoch 98: box loss 1.21 against 1.73. The learning-rate columns differ: 8 parameter groups, peak 0.0055 and 0.0018, against 3 groups at 0.0018. |
| Optimizer code | `ultralytics/engine/trainer.py`, 8.4.6 against 8.4.9 | **The cause.** With `optimizer=auto`, 8.4.6 (shipped run) uses MuSGD at lr 0.002 with muon 0.5 / sgd 0.5 on a set under 10 000 iterations. 8.4.9 (`.venv`) uses AdamW at 0.002 there, and MuSGD at lr 0.01 with muon 0.2 / sgd 1.0 above 10 000 iterations. |
| Lab labels | 24 random knife boxes and 24 object boxes of `lab1114` drawn on the frames | Object boxes all right. Knife boxes: 20 of 24 show a knife in the hand, about 4 are on a hand with no knife visible. |

So the control of 4c trained with AdamW, and the three runs of 4d with MuSGD at five times the learning rate. None used the optimizer of the shipped run.

### What was run

`results/samples/poc/train_stage1_repro.py` (new). It loads Ultralytics 8.4.6 from `.ul846/` (install once: `.venv/Scripts/python.exe -m pip install --no-deps --target .ul846 ultralytics==8.4.6`). For the lab set, which has more than 10 000 iterations, it forces the small-set values (MuSGD, lr 0.002, muon 0.5, sgd 0.5).

```bash
DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . orig      # original 752 images, original arguments
DEV=1 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . lab       # lab1114 set of 4d, YOLO26s
DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . teacher   # lab1114, YOLO26x
DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . kd        # lab1114, YOLO26s + distillation loss of 4d
```

### Results

**Reproduction.** `orig` gives the original `results.csv` line for line (all 127 epochs, five decimals) and the same benchmark numbers as the shipped weights (78.76 % precision, 74.17 % recall, 19 OOD false detections at 0.45). The shipped detector is reproducible.

Ground-truth set and Nov 5 takes, 384x640:

| System | Conf | Precision | Recall | F1 | OOD FP | Phone 008 | Phone 006/007/009 | Nov 5 knife frames, mean |
|---|---|---|---|---|---|---|---|---|
| Shipped (= `stage1_repro846_orig.pt`) | 0.15 | 61.3 % | 87.1 % | 71.9 % | 61 | 27.4 % | 49.3 % | 70.6 % |
| | 0.25 | 69.3 % | 83.8 % | 75.8 % | 44 | 23.5 % | 43.1 % | 67.9 % |
| | 0.45 | 78.8 % | 74.2 % | 76.4 % | 19 | 18.9 % | 34.7 % | 61.4 % |
| Student on lab set, 8.4.9 (4d control) | 0.15 | 60.3 % | 80.8 % | 69.0 % | 54 | 2.0 % | 1.0 % | 66.2 % |
| | 0.45 | 83.3 % | 70.8 % | 76.6 % | 6 | 0.7 % | 0.8 % | 59.4 % |
| **Student on lab set, shipped optimizer** (`stage1_repro846_lab.pt`) | 0.15 | 56.4 % | 91.3 % | 69.8 % | 77 | 4.4 % | 0.9 % | 63.7 % |
| | 0.25 | 63.2 % | 86.7 % | 73.1 % | 51 | 3.3 % | 0.7 % | 61.8 % |
| | 0.45 | 70.9 % | 80.0 % | 75.2 % | 30 | 2.9 % | 0.4 % | 58.8 % |
| **Teacher, YOLO26x, shipped optimizer** (`stage1_repro846_teacher.pt`) | 0.15 | 69.6 % | 90.4 % | 78.6 % | 7 | 6.4 % | 0.1 % | 64.6 % |
| | 0.25 | 74.2 % | 87.5 % | 80.3 % | 2 | 5.7 % | 0.0 % | 62.8 % |
| | 0.45 | 83.9 % | 76.3 % | 79.9 % | 0 | 4.2 % | 0.0 % | 59.0 % |
| **Student distilled from that teacher** (`stage1_repro846_kd.pt`) | 0.05 | 41.3 % | 93.3 % | 57.2 % | 59 | 1.8 % | 0.1 % | 70.9 % |
| | 0.10 | 54.8 % | 88.3 % | 67.6 % | 20 | 1.3 % | 0.0 % | 67.8 % |
| | 0.15 | 63.6 % | 84.6 % | 72.6 % | 9 | 0.9 % | 0.0 % | 65.9 % |
| | 0.25 | 71.7 % | 77.9 % | 74.7 % | 3 | 0.2 % | 0.0 % | 62.7 % |
| | 0.45 | 86.9 % | 63.8 % | 73.6 % | 2 | 0.0 % | 0.0 % | 51.5 % |

Lab takes none of these detectors was trained on:

| System | Conf | Nov 14 test takes, frames (mean / worst take) | ~1 s windows (mean / worst) | Oct 31 take 002 | Second phone 050 | Remote 056 |
|---|---|---|---|---|---|---|
| Shipped | 0.15 | 50 % / 29 % | 68 % / 49 % | 65 % | 19 % | 22 % |
| | 0.25 | 47 % / 27 % | 66 % / 43 % | 61 % | 13 % | 19 % |
| | 0.45 | 39 % / 21 % | 59 % / 35 % | 54 % | 8 % | 13 % |
| **Student on lab set, shipped optimizer** | 0.15 | 53 % / 34 % | 69 % / 51 % | 73 % | 11 % | 11 % |
| | 0.25 | 51 % / 30 % | 66 % / 46 % | 70 % | 5 % | 4 % |
| | 0.45 | 47 % / 26 % | 63 % / 40 % | 67 % | 3 % | 3 % |
| **Teacher, YOLO26x, shipped optimizer** | 0.15 | 57 % / 37 % | 71 % / 53 % | 71 % | 5 % | 4 % |
| | 0.25 | 54 % / 33 % | 69 % / 48 % | 68 % | 4 % | 3 % |
| | 0.45 | 49 % / 28 % | 66 % / 43 % | 62 % | 2 % | 2 % |
| **Student distilled from that teacher** | 0.05 | 62 % / 43 % | 74 % / 60 % | 75 % | 7 % | 5 % |
| | 0.10 | 58 % / 38 % | 72 % / 54 % | 73 % | 5 % | 4 % |
| | 0.15 | 55 % / 34 % | 70 % / 50 % | 71 % | 4 % | 3 % |
| | 0.25 | 51 % / 28 % | 65 % / 43 % | 66 % | 2 % | 2 % |
| | 0.45 | 43 % / 21 % | 59 % / 33 % | 54 % | 1 % | 2 % |

Training: student stopped at epoch 111 (best 81, validation mAP50 0.855); teacher stopped at 37 (best 17, mAP50 0.887); distilled student stopped at 138 (best 108, mAP50 0.851).

### Findings

- **The dataset and the benchmark are right; the training was wrong.** The retrains ran under a newer Ultralytics that silently picks another optimizer. With the shipped run's optimizer the shipped detector comes back exactly.
- **With that fixed, more data helps.** The student trained on the lab set has higher ground-truth recall than the shipped detector at every confidence (80.0 % against 74.2 % at 0.45, 91.3 % against 87.1 % at 0.15). Its precision at the same confidence is lower; at equal recall the two are close (86.7 % recall at 63.2 % precision against 87.1 % at 61.3 %).
- It sees the lab knife more often (Nov 14 test frames 47 % against 39 % at 0.45, Oct 31 take 002 67 % against 54 %) and fires far less on the phone, the second phone and the remote (3 % of frames against 8–35 %).
- The 8.4.9 runs of 4d scored a little higher on lab frames (54 % against 51 % at 0.25) and lower on the ground-truth set. Both sets of numbers come from one run each.
- **The teacher is now the best detector measured here.** At 0.25 it has 87.5 % ground-truth recall at 74.2 % precision with 2 OOD false detections; the shipped detector has 83.8 %, 69.3 % and 44. On the Nov 14 test takes it sees the knife in 54 % of frames against 47 %. It is YOLO26x, about six times the parameters of YOLO26s; its speed on the Edge PC was not measured.
- **Distillation does not bring the student up to the teacher.** The distilled student gives lower scores, so its recall falls fast with confidence (63.8 % at 0.45). At a lower threshold it matches the shipped detector's recall with fewer false detections outside the set: at 0.10, 88.3 % recall and 20 OOD false detections against 87.1 % and 61 for the shipped detector at 0.15, and 58 % of Nov 14 test frames against 50 %. Its precision inside the ground-truth set is lower there (54.8 % against 61.3 %).
- **Between the two students there is no clear winner.** At about 87 % recall the plain lab student (0.25) has the higher precision (63.2 % against 54.8 %), the distilled one (0.10) the fewer OOD false detections (20 against 51) and more lab frames (58 % against 51 %). One run each.

### Not checked

- The lab boxes are still the shipped detector's own detections; about 1 in 6 knife boxes sampled is on an empty hand. Hand-labelled lab frames were not made.
- One run per model, one distillation weight (1.0). The forced small-set optimizer values on the larger lab set are a choice, not a tuned setting.
- The phone-class, phone-background and crop-classifier runs of sections 2 to 4c were not repeated under 8.4.6.
- No TensorRT export, no Edge PC timing (the teacher's above all), no full-pipeline run. The Edge PC has its own Ultralytics version; only training depends on 8.4.6.

### Files

- Script: `results/samples/poc/train_stage1_repro.py`. Ultralytics 8.4.6: `.ul846/` (not in git).
- Weights (dev PC only): `weights/stage1_repro846_{orig,lab,teacher,kd}.pt`. Runs and logs: `results/stage1_kd/repro846_*`.
- Rows: `results/stage1_kd/holdout.jsonl`.

## 4f. Why one class does not reach 90–100 % here: labels, knife size, early stopping (2026-11-16)

**Status: investigation. No shipped weight and no label was changed. Main result: the test set and the original training set are labelled incompletely, so the precision and F1 numbers of every section above are too low, and recall is measured only on knives an earlier detector had already found.**

### Question

The detector has one class, yet no model reaches 90–100 % recall, and more data helps little. Is the data fed in wrong? Is the benchmark right? Detectors with 100 classes reach a high F1; why not this one?

Short answers to the side questions:

- The pipeline runs Stage 1 at confidence 0.45 (`src/stage1_detector.py:54`). The 0.45 rows are in every table; lower thresholds are shown because recall comes first for Stage 1.
- The shipped detector was trained on the original 752 images (450 knife boxes) only. The lab models of 4d and 4e were trained on those 752 images plus Nov 5 and Nov 14 lab frames (3565 images).

### What was run

`results/samples/poc/analyse_stage1_misses.py` (new):

```bash
DEV=0 .venv/Scripts/python.exe results/samples/poc/analyse_stage1_misses.py . run shipped=weights/yolo_weapon_distilled.pt teacher=weights/stage1_repro846_teacher.pt ...
DEV=0 IMGSZ=800,1280 .venv/Scripts/python.exe results/samples/poc/analyse_stage1_misses.py . run shipped=... teacher=...    # also 1600,2560
.venv/Scripts/python.exe results/samples/poc/analyse_stage1_misses.py . sheet results/stage1_kd/misses/teacher_384x640.json fp 0.25 out.jpg   # and: miss 0.25
DEV=0 .venv/Scripts/python.exe results/samples/poc/analyse_stage1_misses.py . trainneg weights/stage1_repro846_teacher.pt weights/yolo_weapon_distilled.pt weights/stage1_repro846_lab.pt
DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . clean                  # original recipe on the cleaned set
PATIENCE=100 DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . clean     # and: orig
```

`run` writes one row per ground-truth box (best score of a knife detection that overlaps it at IoU 0.20) and one per detection that overlaps no box. `sheet` draws those boxes for checking by eye. `trainneg` builds `data/stage1_negatives/orig_clean`.

### What was checked

| Check | Result |
|---|---|
| How the test labels were made (`D:\Backup\auto_label.py`) | An earlier YOLO26s student marked knives at confidence 0.4 on every 10th frame of the validation videos; the boxes were then checked by hand. A frame where that detector found nothing got an empty label file. |
| Test frames with a box | All 596 non-OOD frames come from knife-attack videos. With a box: cut-down 41 of 189, stab 97 of 200, thrust 97 of 207. **361 frames of a person holding a knife have no box.** |
| Detections with no box, by eye | Teacher at 0.25: 65 such detections on non-OOD frames, 63 of them on frames with no label at all. Of the 48 drawn, about 40 show the blade clearly and the rest are the knife hand blurred by motion; none is on another object. Shipped at 0.45: 27 such detections, 23 on frames with no label. |
| Missed boxes, by eye | Teacher at 0.25 misses 25 boxes. All show a knife: dark blade against dark trousers, or motion blur. |
| Knife size as the network sees it | Test images are 2560x1600 and are shrunk to 384x640. Longer side of the knife box: median 26 px (10th–90th percentile 20–36). Original training set 30 px, lab frames 32–34 px. |
| Larger input | Worse, not better. Boxes found at 0.45: shipped 76.2 % at 384x640, 54.2 % at 800x1280, 4.2 % at 1600x2560; teacher 78.3 %, 72.1 %, 22.1 %. The detectors only know the knife at the size they were trained on. |
| Lowest threshold | At 0.01 every detector finds 95–98 % of the labelled boxes. The knives are seen, but many with a low score. |
| Original training set | 302 of the 752 images have no box and are trained as "no knife". In 85 of them a detector (teacher, shipped or lab student) sees a knife at 0.05 or more. Of the teacher's 30 detections at 0.25 or more, about 25 show a knife or its handle in the hand. |
| Early stopping | The validation set has 189 images and 119 boxes. The shipped run stopped at epoch 127 of 250 (best 98). The first run on the cleaned set stopped at 47 (best 17). With patience 100 both sets train all 250 epochs, best at 237 and 238. Where patience 30 stops is partly chance. |
| What the lab frames add | A lab knife frame is in the training set only when the shipped detector fires on it at 0.5 or more (4d). So the lab data holds knives the shipped detector already finds, not the ones it misses. |

### Results

Counted box by box at 384x640. "Found" = a knife detection at that score or more overlaps the box. (`bench_stage1.py` matches one detection to one box in detection order and gives slightly lower recall: 74.2 % for the shipped detector at 0.45.)

| Detector | Trained on | Found at 0.45 | at 0.25 | at 0.15 | at 0.01 | Detections with no box at 0.45 (of them OOD) | at 0.25 |
|---|---|---|---|---|---|---|---|
| Shipped | 752 images, stopped at 127 | 76.2 % | 85.8 % | 89.2 % | 96.7 % | 46 (19) | 85 (44) |
| `stage1_repro846_orig_p100.pt` | 752 images, 250 epochs | 78.8 % | 83.3 % | 86.7 % | 96.7 % | 67 (30) | 109 (60) |
| `stage1_repro846_clean.pt` | 667 images, stopped at 47 | 57.9 % | 73.8 % | 82.9 % | 97.9 % | 46 (22) | 109 (50) |
| `stage1_repro846_clean_p100.pt` | 667 images, 250 epochs | 79.6 % | 86.7 % | 88.8 % | 95.4 % | 39 (7) | 79 (29) |
| `stage1_repro846_lab.pt` | lab set | 82.1 % | 88.3 % | 92.9 % | 97.5 % | 75 (30) | 113 (51) |
| `stage1_repro846_teacher.pt` (YOLO26x) | lab set | 78.3 % | 89.6 % | 92.5 % | 97.1 % | 33 (0) | 67 (2) |
| `stage1_repro846_kd.pt` | lab set + teacher | 64.6 % | 79.2 % | 85.4 % | 96.2 % | 22 (2) | 56 (3) |

Same detectors with the benchmark of the other sections (`bench_stage1.py`, `bench_stage1_holdout.py`), 384x640:

| Detector | Conf | Precision | Recall | F1 | OOD FP | Nov 14 test takes, frames (mean / worst take) | Oct 31 take 002 | Second phone 050 | Remote 056 |
|---|---|---|---|---|---|---|---|---|---|
| Shipped | 0.25 | 69.3 % | 83.8 % | 75.8 % | 44 | 47 % / 27 % | 61 % | 13 % | 19 % |
| | 0.45 | 78.8 % | 74.2 % | 76.4 % | 19 | 39 % / 21 % | 54 % | 8 % | 13 % |
| Original set, 250 epochs (`orig_p100`) | 0.25 | 63.8 % | 81.7 % | 71.7 % | 60 | 44 % / 27 % | 64 % | 16 % | 17 % |
| | 0.45 | 73.4 % | 77.1 % | 75.2 % | 30 | 38 % / 22 % | 59 % | 8 % | 12 % |
| Cleaned set, 250 epochs (`clean_p100`) | 0.25 | 72.1 % | 85.0 % | 78.0 % | 29 | 45 % / 28 % | 67 % | 14 % | 17 % |
| | 0.45 | 82.7 % | 77.9 % | 80.3 % | 7 | 37 % / 25 % | 62 % | 9 % | 13 % |
| Cleaned set, stopped at 47 (`clean`) | 0.25 | 59.4 % | 72.5 % | 65.3 % | 50 | 54 % / 35 % | 69 % | 20 % | 21 % |
| | 0.45 | 73.9 % | 56.7 % | 64.2 % | 22 | 42 % / 25 % | 55 % | 11 % | 13 % |

### Findings

- **The benchmark code is right, the test labels are not complete.** Most detections counted as false on the non-OOD frames are real knives with no label. So precision and F1 on this set are too low for every detector, and more so for a better one: the teacher's 33 "false" detections at 0.45 are all on knife frames, none on the OOD images. Only the OOD count is a clean false-detection number.
- **Recall on this set is recall on the knives an earlier detector found at 0.4**, plus what was added by hand. The real recall over all frames with a visible knife is not known; 361 knife-video frames have no box.
- **Why not 90–100 %:** the knife is about 26 px long and often dark on dark; there are 450 training boxes; and the training set tells the detector "no knife" on frames that show one. A 100-class detector with a high F1 has large objects, thousands of hand-made boxes per class and a completely labelled test set. Here the detector does find 95–98 % of the labelled knives, but with scores below the threshold.
- **Removing the 85 wrong negatives helps precision, not recall.** With the same 250 epochs and old data only, the cleaned set at 0.45 has 77.9 % recall, 82.7 % precision and 7 OOD false detections; the uncleaned set has 77.1 %, 73.4 % and 30; the shipped detector 74.2 %, 78.8 % and 19. The recall difference is 2 boxes, inside the noise. On the lab takes all three are the same (37–39 % of Nov 14 test frames at 0.45). One run each.
- **Early stopping decides more than it should.** The first cleaned run stopped after 17 useful epochs and lost 20 points of recall. Any comparison of two runs here must fix the number of epochs or use a long patience.
- **More lab data cannot fix what the shipped detector misses**, because the lab frames are chosen and boxed by the shipped detector itself.

### What this means for the next step

1. Label the 361 test frames without a box by hand (the detections at 0.05 can be shown as suggestions to accept or delete). Until then, compare detectors by labelled-box recall and OOD false detections, not by precision or F1.
2. Label lab frames by hand, including frames where no detector fires.
3. Label or remove the knife frames without a box in the training set, and train with a fixed number of epochs.

### Not checked

- The by-eye counts are from picture sheets of the detections, not a relabel of the set. Frames where no detector fires were not looked at.
- The validation set (189 images) was made the same way as the training set and was not checked.
- One run per setting. The cleaned set drops frames by detector score; a frame with a knife no detector sees stays in as "no knife".
- The lab models of 4e were not retrained with a long patience or on a cleaned set.

### Files

- Scripts: `results/samples/poc/analyse_stage1_misses.py`, `results/samples/poc/train_stage1_repro.py` (modes `clean`, `PATIENCE`).
- Rows per box: `results/stage1_kd/misses/*.json`; cleaned set and scores: `data/stage1_negatives/orig_clean` (both dev PC only). Lab-take rows: `results/stage1_kd/holdout.jsonl`.
- Weights (dev PC only): `weights/stage1_repro846_{clean,clean_p100,orig_p100}.pt`. Runs and logs: `results/stage1_kd/repro846_*`.

## 4g. Relabelled test set and training set (2026-11-16)

**Status: new label sets written next to the old ones; no old label and no shipped weight was changed. Main result: on the completed test set the shipped detector finds 58 % of the visible knives at 0.45, not 74 %, and nearly all its detections on knife frames are right. The labels were completed by Claude by eye, not by the project owner.**

### Question

4f found that frames with a visible knife have no box in the test set and in the original training set. What do the detectors score once those frames are labelled, and does training on completed labels help?

### What was run

`results/samples/poc/relabel_stage1.py` (new):

```bash
DEV=0 .venv/Scripts/python.exe results/samples/poc/relabel_stage1.py . propose test <5 detector weights>           # hand tiles + detector boxes as suggestions
DEV=0 .venv/Scripts/python.exe results/samples/poc/relabel_stage1.py . hand test                                    # drop decisions on the left hand
.venv/Scripts/python.exe results/samples/poc/relabel_stage1.py . apply test                                          # -> data/Ground_Truth_Dataset_v2
HAND=R DEV=0 .venv/Scripts/python.exe results/samples/poc/relabel_stage1.py . propose train <weights>               # and: val
.venv/Scripts/python.exe results/samples/poc/relabel_stage1.py . apply train                                         # and: val -> data/stage1_negatives/orig_v2
.venv/Scripts/python.exe results/samples/poc/relabel_stage1.py . lab all                                             # -> data/stage1_negatives/lab_v2
PATIENCE=100 DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . v2                           # original recipe on orig_v2
PATIENCE=50 DEV=1 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . labv2                         # lab student on lab_v2
GT_DIR=<repo>/data/Ground_Truth_Dataset_v2 DEV=0 .venv/Scripts/python.exe results/samples/poc/analyse_stage1_misses.py . run name=weights ...
GT_DIR=<repo>/data/Ground_Truth_Dataset_v2 DEV=0 .venv/Scripts/python.exe results/samples/poc/bench_stage1.py "det,<weights>,<conf>" ...
```

How the labels were completed:

- For every image without a box, the script cuts a 300-pixel tile around each hand (pose wrists) at full resolution and draws the boxes of five detectors (teacher, lab student, shipped, cleaned-set student, distilled student; score 0.02 or more) into it.
- Each tile was looked at and given one of three decisions: knife (a suggested box, or a box drawn by eye when no detector had one), no visible knife, or cannot tell. An image with a "cannot tell" tile and no accepted box is left out of the new set, so it counts neither for nor against a detector.
- Check against the old labels: in the hand-checked boxes the knife is in the right hand in 230 of 234 test boxes and 436 of 446 training boxes (nearest wrist of the pose model). So a decision on a left-hand tile is dropped. This removed 6 of the 109 boxes accepted by eye on the test set; three of them were sleeve ends the detectors score at 0.78 to 0.88.

### Label counts

| Set | Images without a box | Get a knife box | Cannot tell (left out) | Stay "no visible knife" | Boxes before → after |
|---|---|---|---|---|---|
| Test (With_Coat, Without_Coat) | 361 | 102 | 90 | 169 | 240 → 342 |
| Original training set | 302 | 80 | 93 | 129 | 450 → 530 |
| Original validation set | 70 | 20 | 19 | 31 | 119 → 139 |

The 119 OOD test images are unchanged. Of the 102 new test boxes, 11 were drawn by eye where no detector had a box.

### Results on the relabelled test set

Box by box (`analyse_stage1_misses.py`), 342 boxes, 384x640. "No box" = detections that overlap no knife box; the number in brackets is how many of them are on the OOD images.

| Detector | Trained on | Found at 0.45 | at 0.25 | at 0.15 | at 0.01 | No box at 0.45 | at 0.25 |
|---|---|---|---|---|---|---|---|
| Shipped | 752 images, old labels, stopped at 127 | 58.2 % | 67.0 % | 71.1 % | 84.5 % | 26 (19) | 51 (44) |
| `stage1_repro846_orig_p100.pt` | old labels, 250 epochs | 62.3 % | 67.5 % | 71.3 % | 84.8 % | 36 (30) | 66 (60) |
| `stage1_repro846_clean_p100.pt` | 85 doubtful images removed, 250 epochs | 63.2 % | 70.8 % | 73.4 % | 83.0 % | 13 (7) | 39 (29) |
| **`stage1_repro846_v2_p100.pt`** | **relabelled set `orig_v2`, 250 epochs** | **66.1 %** | **72.5 %** | **75.7 %** | **88.9 %** | 30 (15) | 49 (28) |
| `stage1_repro846_lab.pt` | lab set, old labels | 67.3 % | 74.0 % | 78.7 % | 89.8 % | 35 (30) | 60 (51) |
| `stage1_repro846_labv2_p50.pt` | lab set on `orig_v2` (`lab_v2`), patience 50, stopped at 95 (best 45) | 69.3 % | 76.9 % | 80.7 % | 90.1 % | 30 (20) | 62 (48) |
| `stage1_repro846_teacher.pt` (YOLO26x) | lab set, old labels | 63.5 % | 77.8 % | 83.0 % | 93.3 % | 2 (0) | 6 (2) |
| `stage1_repro846_kd.pt` | lab set + teacher | 50.0 % | 67.0 % | 73.1 % | 91.5 % | 6 (2) | 12 (3) |

With `bench_stage1.py` (one detection per box, OOD false detections count against precision):

| Detector | Conf | Precision | Recall | F1 | OOD FP |
|---|---|---|---|---|---|
| Shipped | 0.25 | 80.3 % | 65.5 % | 72.1 % | 44 |
| | 0.45 | 87.4 % | 56.7 % | 68.8 % | 19 |
| `orig_p100` | 0.45 | 85.0 % | 61.1 % | 71.1 % | 30 |
| `clean_p100` | 0.25 | 85.9 % | 69.6 % | 76.9 % | 29 |
| | 0.45 | 94.2 % | 62.0 % | 74.8 % | 7 |
| **`v2_p100`** | 0.25 | 81.9 % | 71.3 % | 76.3 % | 28 |
| | 0.45 | 87.8 % | 64.9 % | 74.6 % | 15 |
| Lab student | 0.25 | 78.3 % | 72.8 % | 75.5 % | 51 |
| | 0.45 | 84.9 % | 65.8 % | 74.1 % | 30 |
| Lab student on `lab_v2` | 0.25 | 79.6 % | 75.2 % | 77.3 % | 48 |
| | 0.45 | 88.6 % | 67.8 % | 76.8 % | 20 |
| Teacher (YOLO26x) | 0.25 | 95.6 % | 76.3 % | 84.9 % | 2 |
| | 0.45 | 98.2 % | 62.0 % | 76.0 % | 0 |
| Distilled student | 0.10 | 70.6 % | 78.1 % | 74.2 % | 20 |
| | 0.25 | 87.6 % | 66.1 % | 75.3 % | 3 |
| | 0.45 | 96.0 % | 49.4 % | 65.3 % | 2 |

Lab takes none of these detectors was trained on (`bench_stage1_holdout.py`); the first three trained rows use old data only:

| Detector | Conf | Nov 14 test takes, frames (mean / worst take) | Oct 31 take 002 | Second phone 050 | Remote 056 |
|---|---|---|---|---|---|
| Shipped | 0.25 | 47 % / 27 % | 61 % | 13 % | 19 % |
| | 0.45 | 39 % / 21 % | 54 % | 8 % | 13 % |
| `orig_p100` | 0.45 | 38 % / 22 % | 59 % | 8 % | 12 % |
| `clean_p100` | 0.45 | 37 % / 25 % | 62 % | 9 % | 13 % |
| **`v2_p100`** | 0.25 | 51 % / 34 % | 76 % | 25 % | 22 % |
| | 0.45 | 48 % / 29 % | 72 % | 18 % | 15 % |
| Lab student | 0.45 | 47 % / 26 % | 67 % | 3 % | 3 % |
| Lab student on `lab_v2` | 0.25 | 46 % / 25 % | 66 % | 3 % | 2 % |
| | 0.45 | 39 % / 21 % | 61 % | 2 % | 1 % |
| Teacher (YOLO26x) | 0.25 | 54 % / 33 % | 68 % | 4 % | 3 % |
| | 0.45 | 49 % / 28 % | 62 % | 2 % | 2 % |

Lab student on `lab_v2`: `PATIENCE=50 DEV=1 ... train_stage1_repro.py . labv2`, stopped early at epoch 95 (best validation epoch 45); logs `results/stage1_kd/{misses_labv2_p50_v2,bench_gtv2_labv2p50,holdout_repro846_labv2_p50}.log`.

### Findings

- **The shipped detector's real recall at 0.45 is about 58 %.** The 74 % of the old test set came from counting only the knives an earlier detector had found. At the lowest threshold it reaches 84.5 %, so about one visible knife in six is not seen at all.
- **Precision was never the problem.** On knife frames the shipped detector has 7 detections without a box at 0.45, the teacher 2. Almost all false detections are on the OOD images.
- **Completing the training labels helps a little:** same recipe, same 250 epochs, 66.1 % found at 0.45 against 62.3 % with the old labels and 63.2 % with the doubtful images removed. It has more detections without a box on knife frames (15 against 6).
- **Completed labels also carry over to the lab camera, with old data only:** `v2_p100` sees the knife in 48 % of the Nov 14 test frames at 0.45 against 38 % with the old labels, and in 72 % of Oct 31 take 002 against 59 %. It also fires more on the second phone and the remote (18 % and 15 % of frames against 8 % and 12 %): a dark object in a fist now counts as a knife, and the old data has no phone examples.
- **Completed labels in the lab set help on the test set but not on the lab camera:** the lab student on `lab_v2` finds 69.3 % at 0.45 (best of all YOLO26s models, 88.6 % precision, 20 OOD false detections) against 67.3 % on the old lab set. On the Nov 14 test takes it drops to 39 % of frames at 0.45 against 47 %, and on take 002 to 61 % against 67 %. It stopped early with its best validation epoch at 45, so this run is not at a fixed epoch count (5 below) and may be undertrained; the lab frames it adds are still only frames the shipped detector already finds (3 below).
- **The teacher is the best detector at any threshold below 0.45:** 77.8 % found at 0.25 with 6 detections without a box, 83.0 % at 0.15 with 15. The shipped detector at 0.45 finds 58.2 % with 26.
- **The detectors are confidently wrong on one thing:** with the coat on, the empty left sleeve end is scored up to 0.88.

### Proposed solution

1. **Use the completed sets from now on**: `Ground_Truth_Dataset_v2` to compare detectors, `orig_v2` / `lab_v2` to train. They should be checked by the project owner first; the decisions are in `results/stage1_kd/relabel/` and can be changed tile by tile.
2. **Run Stage 1 at a lower confidence with a detector that has few false detections.** Recall comes first for Stage 1 and the later stages remove false detections. The teacher at 0.15 finds 83 % against 58 % for the shipped detector at 0.45, with fewer false detections. Whether YOLO26x is fast enough on the Edge PC has to be measured; if not, a YOLO26m student is the next size to try.
3. **Label lab frames the same way**, including frames where no detector fires: hand tiles, suggestions from the teacher at 0.02, decisions by eye. The lab set now only holds knives the shipped detector already finds.
4. **Look at the hand, not the whole frame.** The knife is 25 px long in a 384x640 frame, and one visible knife in six gets no detection at all. In a 300-pixel tile around the right hand at full resolution it is about 110 px, large enough to label by eye. A small detector trained on hand tiles (wrists from the pose model) is the change most likely to move recall towards 90 %. Not tested yet.
5. Train with a fixed number of epochs under Ultralytics 8.4.6 (4e, 4f).

### Not checked

- The new labels are one pass by Claude from picture tiles, with detector boxes shown. They are not checked by a person. Boxes drawn by eye are approximate.
- The new boxes come mostly from detector suggestions, so the completed test set still favours the detectors that made them, above all the lab-trained ones. Frames where a knife is visible but no detector fires and the eye missed it are still "no knife".
- 90 test frames are left out as "cannot tell"; they are the hard ones (blade tip under a sleeve, motion blur), so real recall over all knife frames is lower than the numbers above.
- A training image with no right wrist found shows only the frame centre in its tile; it stays "no knife" unless a detector has a box of 0.10 or more elsewhere (3 images).
- One run per setting. No TensorRT export, no Edge PC timing, no full-pipeline run.

### Files

- Scripts: `results/samples/poc/relabel_stage1.py`; `GT_DIR` in `analyse_stage1_misses.py` and `bench_stage1.py`; modes `v2`, `labv2` in `train_stage1_repro.py`.
- Decisions and new boxes (in git): `results/stage1_kd/relabel/{test,train,val}_{items,decisions,labels_v2}.json`, `test_decisions_before_hand.json`, `test_knife_hand_votes.json`. Tile sheets `*_sheet_NN.jpg` are rebuilt by `propose` (dev PC only).
- Sets (dev PC only): `data/Ground_Truth_Dataset_v2`, `data/stage1_negatives/orig_v2`, `data/stage1_negatives/lab_v2`.
- Weights (dev PC only): `weights/stage1_repro846_v2_p100.pt`, `weights/stage1_repro846_labv2_p50.pt`.

## 4h. Knife labels for the lab recordings (2026-11-17)

**Status: new label set `data/lab_stage1_gt` (dev PC only); decisions in git. No old label, recording or weight changed. Labels are by Claude agents (Haiku first pass, Sonnet review), not by a person.**

### Question

The lab set used so far only holds knives the shipped detector already found (4g, point 3). How do the detectors score on lab frames labelled independently of them, including frames where no detector fires?

### What was run

`results/samples/poc/label_lab_stage1.py`:

```bash
DEV=0 .venv/Scripts/python.exe results/samples/poc/label_lab_stage1.py . propose     # 6968 frames -> 8370 hand tiles, 524 sheets
DEV=0 .venv/Scripts/python.exe results/samples/poc/label_lab_stage1.py . persons     # 685 whole-person tiles where no wrist was found, sheets 524-566
.venv/Scripts/python.exe results/samples/poc/label_lab_stage1.py . apply
SKIP=results/stage1_kd/lab_label/cart_shadow_ids.json .venv/Scripts/python.exe results/samples/poc/label_lab_stage1.py . suspects   # 31 review sheets
.venv/Scripts/python.exe results/samples/poc/label_lab_stage1.py . apply             # with lab_decisions_fix.json
```

- Frames: test split = Nov 14 test takes (attack and normal) and Oct 31 take 002, every 10th frame (3,250 frames); train split = Nov 5 and Nov 14 training takes, every 20th / 45th frame (3,718 frames).
- First pass: 29 Haiku agents, one decision per tile with `labeller_prompt.md` (knife box a/b/c, `k` knife but no box, `n` no knife, `u` cannot tell).
- Suspect tiles (label and detector scores disagree): 1,813. 730 of them are one spot, a knife-shaped shadow on the white cart that the detectors score 0.55–0.88 even in an empty room (26 takes); a random sheet of 36 was checked by eye, all 36 are the shadow, so they stay "no knife". The other 1,083 were re-judged by four Sonnet agents (`reviewer_prompt.md`): 155 labels changed, most of them to "cannot tell" (n→u 53, a→u 20, n→a 20, b→a 13). Five misread tile ids were remapped by hand.
- Audit: 36 random test tiles still labelled "no knife" though the teacher scores ≥ 0.5 or the shipped detector ≥ 0.7 were checked by eye. About 31 are right (empty hand, hand on a chair armrest, seated person); about 5 may be a knife.

### Label counts

| Split | Frames | With a knife box | No knife | Left out (cannot tell) | Knife boxes |
|---|---|---|---|---|---|
| Test | 3,250 | 1,061 | 1,422 | 767 | 1,069 |
| Train | 3,718 | 1,555 | 1,465 | 698 | – |

### Detectors on the lab test labels

Box by box as `bench_stage1.py` (IoU 0.2), from `sweep_stage1_rules.py` (`MODIFICATION_REPORT_STAGE1_CONF_RULES.md`). "Hand gate" = the pipeline's gate (`src/weapon_gate.py`): keep a box only if a wrist is within 0.5 torso lengths of it.

| Detector | Conf | Precision | Recall | With hand gate: precision | recall |
|---|---|---|---|---|---|
| Lab student `labv2_p50` | 0.45 | 75.0 % | 78.3 % | 88.4 % | 77.3 % |
| | 0.25 | 66.5 % | 84.8 % | 84.7 % | 83.6 % |
| Teacher (YOLO26x) | 0.45 | 64.9 % | 82.8 % | – | – |
| | 0.25 | 62.2 % | 87.6 % | 79.5 % | 86.2 % |
| Shipped | 0.45 | 69.2 % | 72.5 % | 86.6 % | 71.7 % |
| Distilled student `kd` | 0.45 | 63.3 % | 73.7 % | – | – |

### Findings

- **On the lab camera the teacher is not the most precise detector.** At 0.45 it has 64.9 % precision against 75.0 % for the lab student; with the cart shadow removed, 85.2 % against 88.3 %. Its 98 % on `Ground_Truth_Dataset_v2` does not carry over: it fires on empty hands and chair armrests.
- **The cart shadow is the largest single false-alarm source on the lab camera**, and the hand gate already removes it.
- Recall at 0.45 is higher here (78 % lab student) than on `Ground_Truth_Dataset_v2` (68 %), because 767 hard frames are left out as "cannot tell".

### Not checked

- No person checked these labels. About 1 in 7 high-score "no knife" tiles may be a knife (audit above), so lab precision is a little too low.
- The labels started from detector suggestions; a knife no detector boxed and no agent saw stays "no knife".
- The train split was labelled but no detector was retrained on it yet.

### Files

- Script: `results/samples/poc/label_lab_stage1.py` (`SKIP=` in `suspects`).
- In git: `results/stage1_kd/lab_label/{lab_items.json, decisions/, review_fix/, lab_decisions_fix.json, lab_labels.json, cart_shadow_ids.json, labeller_prompt.md, reviewer_prompt.md}`.
- Dev PC only: `data/lab_stage1_gt` (images, labels, `test.txt`, `train.txt`, `data.yaml`); sheets in `results/stage1_kd/lab_label/{sheets,review_all,review_round1,review_audit,review_cart_sample}`.

### Three-voter check of these labels (2026-11-17 to 2026-11-18)

**Question:** how many first-pass labels does a second and third independent vote agree with, and can Claude settle the rest without a person?

**What was run** (`results/samples/poc/vote_lab_labels.py`):

```bash
.venv/Scripts/python.exe results/samples/poc/vote_lab_labels.py tally    # first pass + haiku1 + haiku2 -> votes.csv, 68 check sheets of disagreements
# Opus agents answer claude_check/sheets/check_NN.jpg with checker_prompt.md -> claude_check/answers/check_NN.json (label + confidence)
.venv/Scripts/python.exe results/samples/poc/vote_lab_labels.py claude   # merge answers into votes.csv
.venv/Scripts/python.exe results/samples/poc/vote_lab_labels.py final    # -> lab_decisions_final.json
```

- Two more Haiku votes per tile on all 567 sheets, next to the first pass. Tiles where the three disagree go to a Claude checker (Opus).
- Claude settles a tile alone at confidence ≥ 0.8, or at ≥ 0.6 when two of the three earlier votes say the same. The rest wait for a person in `label_review_ui.py`.
- The Opus run stopped at the weekly limit with sheets 07, 14 and 62 not done; they were finished by hand in a later session, so all 68 sheets (2,425 tiles) have an answer.

| Outcome | Tiles |
|---|---|
| All three votes agree | 6,630 |
| Disagreement, settled by Claude | 945 |
| Disagreement, waits for a person (Claude < 0.6, or 0.6–0.8 against the majority) | 1,480 |
| **Total** | **9,055** |

Claude's answers on the 2,425 checked tiles: 1,002 a knife box, 43 knife with no box, 784 no knife, 596 cannot tell; 1,012 are below 0.6 confidence.

`lab_decisions_final.json` (tiles still waiting count as "cannot tell", so they are left out): 2,689 knife box, 11 knife with no box, 4,048 no knife, 2,307 cannot tell.

**Findings**

- **Three in four tiles (73 %) get the same label from all three votes.** The disagreements are mostly fast swings with motion blur, hands cut off by the frame, and fists holding a small dark object.
- **Claude settles only 39 % of the disagreements.** The rest are too blurred to call from the tile; a person is needed for them, or they stay out of the label set.

**Not checked**

- `data/lab_stage1_gt` was not rebuilt from `lab_decisions_final.json`, and the detectors were not re-scored on it; the numbers above still use the 4h labels. (Done in 4i.)
- No person has reviewed the 1,480 waiting tiles yet. (Done on 2026-11-18, see 4i.)
- Claude's confidence is self-reported and not calibrated against a person.

**Files:** `results/samples/poc/vote_lab_labels.py`, `label_review_ui.py`; `results/stage1_kd/lab_label/{votes.csv, lab_decisions_final.json, claude_check/}`; Haiku votes in `results/stage1_kd/lab_label/votes/`.

## 4i. Retrained on the person-checked lab labels (2026-11-18)

**Status: new weights `weights/stage1_repro846_labgt_p250_last.pt` (dev PC only). Nothing shipped was changed. Proposed as the next lab-camera student; not yet exported to TensorRT or run in the pipeline.**

### Question

The lab knife labels are final: a person settled all 1,480 tiles that waited after the three-voter check (4h; 810 train tiles, 670 test tiles). Does a student trained on them do better on the lab camera than the same recipe on the old lab labels (`lab_v2`, knives the shipped detector already found)?

### What was run

```bash
.venv/Scripts/python.exe results/samples/poc/vote_lab_labels.py final          # votes.csv -> lab_decisions_final.json
.venv/Scripts/python.exe results/samples/poc/label_lab_stage1.py . apply        # -> data/lab_stage1_gt
.venv/Scripts/python.exe results/samples/poc/label_lab_stage1.py . trainset     # -> data/stage1_negatives/lab_gt
PATIENCE=250 DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . labgt   # new labels
PATIENCE=250 DEV=1 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . labv2   # control, old labels
EXTRA_DETS="labgt=weights/stage1_repro846_labgt_p250.pt;labgtL=results/stage1_kd/repro846_labgt_p250/weights/last.pt;..." \
  DEV=0 .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . cache lab      # and: cache gtv2
EXTRA_DETS="labv2p250=x;labv2L=x;labgt=x;labgtL=x" .venv/Scripts/python.exe results/samples/poc/compare_stage1_cached.py \
  shipped teacher student labv2p250 labv2L labgt labgtL > results/stage1_kd/rules/compare_labgt.log
```

- **Final labels** (`lab_decisions_final.json`): 6,630 tiles by three agreeing votes, 945 by Claude, 1,480 by the person. Test split: 1,362 frames with a knife (1,376 boxes), 1,480 without, 408 left out. Train split: 1,913 / 1,378 / 427.
- **`lab_gt` training set:** `lab_v2` with its old lab knife frames (619 Nov 5 + 1,576 Nov 14, boxes from the shipped detector) replaced by 3,159 person-checked train frames (class `knife`). Kept: the 659 `orig_v2` images and 600 phone frames. 132 new frames from the five validation takes are left out, and 18 phone frames that are also in the new set keep only the new copy. Validation is `lab_v2`'s 350 images for both runs. No test take and no Oct 31 take 002 frame is in training (asserted in `trainset`).
- **Same recipe for both runs:** YOLO26s, Ultralytics 8.4.6, MuSGD lr 0.002, 250 epochs, no early stop. `best.pt` came from epoch 85 (`labgt`) and 49 (`labv2`), chosen on the small validation set. The final epoch (`last.pt`, "last" below) is the fixed-epoch model and is better for both.
- Scoring: box by box, IoU 0.2, 384x640, as `bench_stage1.py`. "Gate" = the pipeline's hand gate (wrist within 0.5 torso lengths). Re-adding the shipped detector under a new name through `EXTRA_DETS` gave exactly its numbers.

### Lab camera test labels (2,842 frames, 1,376 knife boxes; 1,480 frames have no knife)

| Detector | 0.45: P / R | false boxes | 0.45 + gate: P / R | false boxes | 0.25 + gate: P / R |
|---|---|---|---|---|---|
| Shipped | 73.0 / 64.7 | 330 | 89.6 / 63.9 | 102 | 84.3 / 71.7 |
| Teacher (YOLO26x) | 69.9 / 76.7 | 455 | 89.0 / 75.6 | 129 | 83.9 / 80.7 |
| Lab student `labv2_p50` (until now) | 79.1 / 70.8 | 257 | 92.3 / 69.9 | 80 | 89.0 / 76.6 |
| Control: old labels, 250 epochs, last | 73.8 / 77.2 | 378 | 91.2 / 76.2 | 101 | 89.7 / 79.1 |
| **New labels, 250 epochs, last** | **92.8 / 74.8** | **80** | 92.7 / 73.4 | 79 | 90.8 / 75.9 |
| New labels, `best.pt` (epoch 85) | 90.3 / 73.7 | 109 | 90.7 / 72.6 | 102 | 86.9 / 76.8 |

Best recall at a precision target (thresholds 0.01–0.8; threshold in brackets):

| Detector | P ≥ 0.90, no gate | P ≥ 0.90, gate | P ≥ 0.92, gate |
|---|---|---|---|
| Lab student `labv2_p50` | 67.9 % (0.55) | 73.2 % (0.35) | 69.9 % (0.45) |
| Control, old labels, last | not reached | **78.2 % (0.30)** | 73.8 % (0.60) |
| **New labels, last** | **77.3 % (0.25)** | 75.9 % (0.25) | **74.8 % (0.35)** |
| Teacher | not reached | 71.7 % (0.55) | 64.0 % (0.70) |

Oct 31 take 002 alone (572 frames, 499 boxes), 0.25 + gate: new labels 94.6 / 83.8, control 93.1 / 86.0, `labv2_p50` 93.1 / 81.2.

### Original camera, `Ground_Truth_Dataset_v2` (625 images, 342 boxes)

| Detector | 0.45: P / R | P ≥ 0.90, gate |
|---|---|---|
| Teacher | 98.1 / 62.0 | 81.3 % |
| Lab student `labv2_p50` | 88.5 / 67.8 | 63.7 % |
| Control, old labels, last | 89.5 / 64.9 | 63.2 % |
| New labels, last | 89.7 / 66.4 | 65.5 % |

### Findings

- **The new labels teach the student what the hand gate did.** Without any gate it has 92.8 % precision at 0.45 on the lab camera, against 73.8 % for the same recipe on the old labels. False boxes on frames with no knife drop from 351 to 60. It is the only detector that reaches 90 % precision without the gate (77.3 % recall at 0.25). The cart shadow and empty hands are now negatives in training.
- **Behind the gate the gain is small, and recall is a little lower.** At 90 % gated precision the control finds 78.2 %, the new model 75.9 %; at 92 %, 73.8 % against 74.8 %. At 0.25 with the gate the new model finds 75.9 % against 79.1 %. The gate already removes most of what the new labels fix. The new model helps where the gate fails: a wrist the pose model does not find, or a knife held away from the wrist.
- **Training the full 250 epochs matters as much as the labels.** The same old labels at 250 epochs find 78.2 % at 90 % gated precision, against 73.2 % for `labv2_p50` (stopped at 95). For both runs the final epoch beats the validation-chosen `best.pt`.
- **No loss on the original camera:** the new model is level with the control and `labv2_p50` on `Ground_Truth_Dataset_v2`. The teacher stays far ahead there.

### Proposed next steps

1. Try `stage1_repro846_labgt_p250_last.pt` as the lab-camera student at 0.25 with the hand gate (or about 0.10 without it: 81.2 % recall at 85 % precision). Export to TensorRT at 384x640 and run the whole pipeline on the Nov 14 test takes and take 002 before shipping it. If recall behind the gate matters most, `stage1_repro846_labv2_p250_last.pt` at 0.30 is the alternative.
2. Retrain the teacher (YOLO26x) on `lab_gt`: best on the original camera but least precise on the lab camera. About 10 h on one RTX 3090. Done below: it is now the best lab detector.
3. Train with no early stop and keep the final epoch from now on (`PATIENCE=250`).

### Not checked

- The test labels come from the same labelling process and rules as the training labels (cart shadow = no knife, blurred swing = cannot tell), so a model trained on them is favoured on this test set. The 408 left-out test frames are the hardest ones; real recall is lower.
- Validation still holds 116 lab frames with the old labels; only checkpoint choice depends on it.
- One run per setting. No TensorRT export, no Edge PC timing, no full-pipeline (Stage 2) run, no frame-share holdout benchmark (`bench_stage1_holdout.py`).

### Files

- Scripts: `results/samples/poc/label_lab_stage1.py` (`trainset`), `train_stage1_repro.py` (mode `labgt`), `sweep_stage1_rules.py` (`EXTRA_DETS`), `compare_stage1_cached.py`.
- In git: `results/stage1_kd/lab_label/{votes.csv, lab_decisions_final.json, lab_labels.json}`, `results/stage1_kd/rules/sweep_lab_final.jsonl`.
- Dev PC only: `data/lab_stage1_gt`, `data/stage1_negatives/lab_gt`, `results/stage1_kd/rules/compare_labgt.log`, caches `results/stage1_kd/rules/cache_{lab,gtv2}.json`, runs `results/stage1_kd/repro846_{labgt,labv2}_p250/`, weights `weights/stage1_repro846_{labgt,labv2}_p250{,_last}.pt`.

### Teacher retrained on the new labels, and distillation (2026-11-19)

**Question:** step 2 above. Does a YOLO26x teacher trained on `lab_gt` fix the old teacher's lab-camera false boxes, and does distilling a teacher into YOLO26s on `lab_gt` beat the plain student `labgtL`?

**What was run** (same recipe as above: Ultralytics 8.4.6, MuSGD lr 0.002, 250 epochs, no early stop; the teacher at batch 12, nbs 64; the KD runs pin the GPU with `CUDA_VISIBLE_DEVICES`, because the teacher is loaded before Ultralytics picks the device):

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=250 PATIENCE=250 DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . teachergt
CUDA_VISIBLE_DEVICES=1 SUFFIX=_oldteacher PATIENCE=250 DEV=0 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . kdgt
CUDA_VISIBLE_DEVICES=0 TEACHER=weights/stage1_repro846_teachergt_p250_last.pt SUFFIX=_newteacher PATIENCE=250 DEV=0 \
  .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . kdgt
EXTRA_DETS="teachergt=...;teachergtL=...;kdgtO=...;kdgtOL=..." DEV=0 .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . cache lab   # and gtv2
```

The teacher took 11.5 h on one RTX 3090; `best.pt` is epoch 132. The KD runs took 5.8–7.5 h; `best.pt` is epoch 143 (old teacher) and 65 (new teacher).

Lab camera (2,842 frames, 1,376 boxes):

| Detector | 0.45: P / R | false boxes | 0.25 + gate: P / R | P ≥ 0.90, no gate | P ≥ 0.90, gate | P ≥ 0.92, gate |
|---|---|---|---|---|---|---|
| Old teacher | 69.9 / 76.7 | 455 | 83.9 / 80.7 | not reached | 71.7 % (0.55) | 64.0 % (0.70) |
| **New teacher, last** | **92.0 / 77.4** | 93 | 90.4 / 78.3 | **79.6 % (0.25)** | 78.3 % (0.25) | **75.5 % (0.50)** |
| New teacher, `best.pt` | 91.5 / 78.1 | 100 | 88.2 / 80.2 | 79.7 % (0.35) | **78.5 % (0.35)** | 74.9 % (0.55) |
| Student, new labels, last (`labgtL`) | 92.8 / 74.8 | 80 | 90.8 / 75.9 | 77.3 % (0.25) | 75.9 % (0.25) | 74.8 % (0.35) |
| Control, old labels, last (`labv2L`) | 73.8 / 77.2 | 378 | 89.7 / 79.1 | not reached | 78.2 % (0.30) | 73.8 % (0.60) |
| KD from old teacher, last | 70.1 / 63.5 | 373 | 84.4 / 76.1 | not reached | 69.4 % (0.35) | 66.6 % (0.40) |
| **KD from new teacher, last** | 92.0 / **78.2** | 94 | 83.0 / 80.7 | 79.1 % (0.40) | 77.7 % (0.40) | **75.7 % (0.50)** |
| KD from new teacher, `best.pt` (epoch 65) | 89.9 / 76.4 | 118 | 76.4 / 81.5 | 74.5 % (0.50) | 73.3 % (0.50) | 67.3 % (0.60) |

`Ground_Truth_Dataset_v2` (625 images, 342 boxes), best recall at P ≥ 0.90 with the gate: old teacher 81.3 %, new teacher `best.pt` 74.9 %, last 72.5 %, **KD from new teacher, last 69.6 %** (`best.pt` 67.8 %), `labgtL` 65.5 %, KD from old teacher 62.3 %.

Oct 31 take 002 alone (499 boxes) at 0.45: KD from new teacher, last 93.1 / 84.4, `labgtL` 95.3 / 81.2, new teacher 95.0 / 80.0.

**Findings:**

- **The new labels fix the teacher too.** On the lab camera its precision without the gate goes from 69.9 % to 92.0 % at 0.45 (false boxes 455 to 93), and it is now the best lab detector: 79.6 % recall at 90 % precision without the gate, 2.3 points above `labgtL`.
- **It loses some of the old teacher's lead on the original camera** (81.3 % to 72.5–74.9 % at 90 % gated precision), but stays well above every YOLO26s.
- **Distilling the old teacher hurts on the lab camera.** The student copies the old teacher's lab false boxes: 70 % precision without the gate, against 92.8 % for `labgtL` trained on the same data without a teacher. On the original camera it is no better than `labgtL` at 90 % precision. Distill only from a teacher trained on the lab labels.
- **Distilling the new teacher gives the best YOLO26s.** On the lab camera it is close to the teacher: 79.1 % recall at 90 % precision without the gate (`labgtL` 77.3 %, teacher 79.6 %), 77.7 % with the gate (`labgtL` 75.9 %). On the original camera it gains about 4 points over `labgtL` at 90 % gated precision (69.6 % against 65.5 %).
- **Its threshold is higher.** Its scores are less sharp than `labgtL`'s: at 0.25 it has 228 gated false boxes (`labgtL` 106), so its working point is about 0.40, where it has 79 % lab recall at 90 % precision with or without the gate. The final epoch again beats `best.pt` (chosen at epoch 65), by 4–8 points.
- **Proposed lab-camera student:** `weights/stage1_repro846_kdgt_p250_newteacher_last.pt` at conf 0.40, with the hand gate kept for safety (it changes little for this model). Not yet exported to TensorRT or run in the pipeline.

Logs (dev PC only, gitignored): `results/stage1_kd/rules/compare_{kdgtO,teachergt,kdgtN}.log`. Weights (dev PC only): `weights/stage1_repro846_teachergt_p250{,_last}.pt`, `weights/stage1_repro846_kdgt_p250_{oldteacher,newteacher}{,_last}.pt`.

Not checked: one run per setting; no TensorRT export or Edge PC timing; no full-pipeline run; the KD student's threshold was chosen on the same test set it is scored on.

## 5. Not checked

- The classifier's lab knife crops are the shipped detector's own boxes, not hand labels.
- The ground-truth set is the original camera; the lab camera is covered only by the frame shares on six takes of one person.
- The classifier was not exported to TensorRT or timed on the Edge PC (RTX 3050), and the system was not run with the hand gate, static filter and Stage 2.
- Only one confuser object (one phone, one session) was tested. Other handheld objects are untested.
- Two harness limits: the confirm system counts a `phone` box of a three-class confirmer as a confirmation, and the crop system does not apply the phone-class filter. Rows affected by this (confirm with `stage1_big.pt`, crop behind a phone-class detector) are left out of the table.
- Not tried: the Sohas phone and wallet dataset (needs a download), a focal-type loss, retraining the detector on wrist crops.

## 6. Files

- Scripts: `results/samples/poc/bench_stage1.py`, `train_stage1_variants.py`, `train_stage1_phoneneg.py`, `extract_phone_negatives.py`.
- Data: `data/stage1_negatives/` (`phone_2026-11-05`, `ft`, `ft_phonecls`, `crops`, `crops2`).
- Weights: `weights/stage1_*.pt`, `weights/yolo_weapon_distilled_phoneneg.pt`. Training runs: `results/stage1_phoneneg/`.
