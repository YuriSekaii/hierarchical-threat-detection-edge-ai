# Modification Report: Stage 1 alternatives for high recall with the phone rejected (2026-11-13 to 2026-11-14)

**Status: experiment. The shipped `weights/yolo_weapon_distilled.pt` and the pipeline are unchanged.** All new weights are separate files named `weights/stage1_*.pt`. Nothing was exported to TensorRT, run through the full pipeline or tried on the Edge PC.

## 1. Question

Stage 1 flags a phone held like a knife, and retraining with phone frames as negatives costs recall (`MODIFICATION_REPORT_STAGE1_PHONE_NEGATIVES.md`). Recall matters most, because the hand gate, the static filter and Stage 2 remove false detections later. Which model or combination gives the highest recall with the best precision and the phone rejected?

Ideas came from the first phone-negative test and from `Edge Knife Detection Literature Review.md` (phone as its own class, training at the deployed scale, wrist-region zoom, temporal voting).

## 2. What was run

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

## 4b. Input size

All rows above were measured at 384x640, which shrinks the 16:10 test images by 0.24. The detector scores about 10 F1 points higher at exactly 0.25 (416x640). See `MODIFICATION_REPORT_STAGE1_INPUT_SIZE_GAP.md`.

## 5. Not checked

- The classifier's lab knife crops are the shipped detector's own boxes, not hand labels.
- The ground-truth set is the original camera; the lab camera is covered only by the frame shares on six takes of one person.
- The classifier was not exported to TensorRT or timed on the Edge PC (RTX 3050), and the system was not run with the hand gate, static filter and Stage 2.
- Only one confuser object (one phone, one session) was tested. Other handheld objects are untested.
- Two harness limits: the confirm system counts a `phone` box of a three-class confirmer as a confirmation, and the crop system does not apply the phone-class filter. Rows affected by this (confirm with `stage1_big.pt`, crop behind a phone-class detector) are left out of the table.
- Not tried: the Sohas phone and wallet dataset, cross-fitted teacher distillation, a focal-type loss, retraining the detector on wrist crops. The first needs a download, the others a rebuilt teacher.

## 6. Files

- Scripts: `results/samples/poc/bench_stage1.py`, `train_stage1_variants.py`, `train_stage1_phoneneg.py`, `extract_phone_negatives.py`.
- Data: `data/stage1_negatives/` (`phone_2026-11-05`, `ft`, `ft_phonecls`, `crops`, `crops2`).
- Weights: `weights/stage1_*.pt`, `weights/yolo_weapon_distilled_phoneneg.pt`. Training runs: `results/stage1_phoneneg/`.
