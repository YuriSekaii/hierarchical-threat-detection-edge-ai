# Modification Report: Stage 1 retrained with phone frames as negatives (2026-11-13)

**Status: experiment. The shipped `weights/yolo_weapon_distilled.pt` is unchanged.** New weights are in separate files. Not exported to TensorRT, not run through the full pipeline or on the Edge PC.

## 1. Question

Nov 5 takes 006–009 (`thrust_low` front, left, back, right) were performed with a phone instead of a knife. Does Stage 1 flag the phone as a knife, and if so, does retraining with those frames as negatives fix it?

## 2. What was run

- `results/samples/poc/extract_phone_negatives.py <REPO>`: runs the shipped detector on takes 006–009 and saves every 5th flagged frame (confidence ≥ 0.25) with an empty label file to `data/stage1_negatives/phone_2026-11-05` (497 images).
- `results/samples/poc/train_stage1_phoneneg.py <REPO> build|train|check [weights]`:
  - **build:** `data/stage1_negatives/ft` = backup Stage 1 set `D:\Backup\Weapon_Detection_Split` (752 train, 189 validation images, one class, mapped to class 1 `knife` of the shipped two-class model) + phone frames (takes 006, 007, 009 for training, 008 for validation) + 617 lab knife frames from the Nov 5 knife takes.
  - **train:** fine-tune of the shipped weights, 40 epochs, image size 640, learning rate 0.001, no teacher.
  - **check:** share of frames with a detection at confidence 0.45 on phone and knife takes.
- `scripts/validate_stage1_ground_truth.py --weights <w> --conf <c>`: 715-image ground-truth set.
- Knife takes 002, 005, 010, 014, 018, 022 and phone take 008 were kept out of training.
- The lab knife boxes are the shipped detector's own detections at confidence ≥ 0.5, not hand labels.

## 3. Results at the live threshold 0.45

All detections on the phone takes are class `knife`.

| | Shipped | Run 1: phone negatives only | Run 2: phone negatives + lab knife frames |
|---|---|---|---|
| Phone take 008 (never trained on) | 19 % | 0 % | 3 % |
| Phone takes 006 / 007 / 009 | 61 / 20 / 23 % | 0 / 0 / 0 % | 0 / 5 / 0 % |
| Knife takes 002 / 005 / 010 | 88 / 76 / 65 % | 0 / 11 / 0 % | 51 / 72 / 53 % |
| Knife takes 014 / 018 / 022 | 36 / 42 / 61 % | 0 / 0 / 0 % | 28 / 39 / 30 % |
| Ground-truth precision | 78.8 % | 92.8 % | 90.7 % |
| Ground-truth recall | 74.2 % | 37.5 % | 52.5 % |
| Ground-truth F1 | 76.4 % | 53.4 % | 66.5 % |
| False detections on OOD images | 19 | 2 | 3 |

Run 1 learned that the lab camera means "no knife", because its only lab frames were phone frames. It is kept as `results/stage1_phoneneg/phone_only_run1.pt`. Run 2 is `weights/yolo_weapon_distilled_phoneneg.pt`.

## 4. Confidence threshold sweep

| Detector, threshold | Recall | Precision | OOD false detections | Phone frames flagged | Knife frames detected |
|---|---|---|---|---|---|
| Shipped, 0.45 | 74.2 % | 78.8 % | 19 | 19–61 % | 36–88 % |
| Shipped, 0.35 | 77.1 % | 75.2 % | 30 | 21–67 % | 42–92 % |
| Shipped, 0.25 | 83.8 % | 69.6 % | 44 | 23–71 % | 48–93 % |
| Shipped, 0.15 | 87.1 % | 62.4 % | 58 | 27–78 % | 52–94 % |
| Run 2, 0.45 | 52.5 % | 90.7 % | 3 | 0–5 % | 28–72 % |
| Run 2, 0.35 | 55.4 % | 86.4 % | 4 | 0–5 % | 31–75 % |
| Run 2, 0.25 | 60.4 % | 82.4 % | 7 | 0–6 % | 35–77 % |
| Run 2, 0.15 | 65.4 % | 71.0 % | 17 | 1–7 % | 39–78 % |

Window rule (at least 3 of 10 sampled frames at confidence 0.25, about one second): shipped gives 65–100 % of windows on knife takes and 34–97 % on phone takes; run 2 gives 53–92 % on knife takes and 0–7 % on phone takes.

## 5. Conclusion

- Stage 1 does flag the phone as a knife, in 19–61 % of frames.
- Negatives remove the phone detections, also on the take the model never saw, and cut OOD false detections from 19 to 3.
- Recall drops from 74 % to 53 %, and no threshold on run 2 recovers it (65 % at 0.15).
- The shipped detector at 0.25 has the best recall (84 %) but keeps flagging the phone.

## 6. Not checked

- The retrain is a plain fine-tune without the teacher; the original distillation script is no longer in the repo. Class `hand_grip_knife` got no examples.
- The ground-truth set is the original camera only, so its recall is not a lab-camera number.
- How many extra detections at threshold 0.25 survive the hand, static-object and Stage 2 rules.

Follow-up comparison of other models: `MODIFICATION_REPORT_STAGE1_MODEL_COMPARISON.md`.
