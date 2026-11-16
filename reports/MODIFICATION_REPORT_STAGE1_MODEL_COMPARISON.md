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
