# Stage 1: confidence rules that keep precision and raise recall

**Status: offline sweep, no pipeline code and no weight changed. Date: 2026-11-17.**

## Question

At conf 0.01 the detectors find 85–92 % of the labelled knives; at 0.45 the teacher (YOLO26x) has about 98 % precision and the lab student about 88 %. Can a rule use both: take low-score boxes for recall, and keep only the ones a second signal confirms, so precision stays where it is at 0.45?

## What was run

`results/samples/poc/sweep_stage1_rules.py` (new). Every detector runs once per image at conf 0.01 (384x640, half); the boxes, a crop-classifier score per box and the pose wrists are cached, then every rule is scored offline.

```bash
DEV=0 .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . cache gtv2   # data/Ground_Truth_Dataset_v2, 625 images, 342 knife boxes
DEV=0 .venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . cache lab    # lab test frames, labels of section 4h in MODIFICATION_REPORT_STAGE1_MODEL_COMPARISON.md
.venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . sweep gtv2
.venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . sweep lab
.venv/Scripts/python.exe results/samples/poc/sweep_stage1_rules.py . sweep gtv2 --apply-to lab --min-precision P
```

Detectors: student `stage1_repro846_labv2_p50.pt` (YOLO26s, the 88 % one), teacher `stage1_repro846_teacher.pt` (YOLO26x), shipped `yolo_weapon_distilled.pt`, distilled student `stage1_repro846_kd.pt`.

Rules ("hi/lo" rule = keep a box if its score is ≥ hi, or if it is ≥ lo **and** the gate passes; hi = "never" means only gated boxes are kept):

| Gate | Passes when |
|---|---|
| `teacher@c` | the teacher has a box with score ≥ c overlapping it (IoU ≥ 0.2) |
| `cls@p` | the crop classifier `stage1_cropcls2.pt` gives knife ≥ p on the box crop (+50 % margin) |
| `wrist@k` | the box centre is within k torso lengths of a wrist (pose model, keypoint conf ≥ 0.3) |
| `wrist@k+cls@p` | both |

Grid: single thresholds 0.01–0.8; hi 0.3–0.7 and "never"; lo 0.01–0.25; c 0.05–0.45; p 0.3–0.95; k 0.5–1.0. Matching as in `bench_stage1.py`: one detection per box, IoU ≥ 0.2, highest score first; any other detection counts as false.

## Results

### Ground_Truth_Dataset_v2 (342 knife boxes, 119 OOD images)

Single threshold:

| Detector | Conf | Precision | Recall | False det. | of them OOD |
|---|---|---|---|---|---|
| Student | 0.01 | 35.9 % | 88.6 % | 540 | 404 |
| | 0.10 | 68.2 % | 80.1 % | 128 | 97 |
| | 0.25 | 79.6 % | 75.4 % | 66 | 48 |
| | 0.45 | 88.5 % | 67.8 % | 30 | 20 |
| Teacher | 0.01 | 58.8 % | 91.8 % | 220 | 79 |
| | 0.10 | 87.6 % | 82.7 % | 40 | 11 |
| | 0.15 | 92.4 % | 81.6 % | 23 | 7 |
| | 0.25 | 95.6 % | 76.3 % | 12 | 2 |
| | 0.45 | 98.1 % | 62.0 % | 4 | 0 |
| Shipped | 0.45 | 87.4 % | 56.7 % | 28 | 19 |
| Distilled student | 0.45 | 96.0 % | 49.4 % | 7 | 2 |

Best rule at each precision floor (highest recall):

| Precision ≥ | Student, best rule | P / R | Teacher, best rule | P / R |
|---|---|---|---|---|
| 88 % | hi 0.70, lo 0.05, `teacher@0.10` | 89.3 % / 77.8 % | hi 0.30, lo 0.10, `wrist@0.5` | 90.7 % / 82.5 % |
| 92 % | hi 0.70, lo 0.10, `teacher@0.10` | 92.8 % / 75.7 % | single 0.15 | 92.4 % / 81.6 % |
| 95 % | hi 0.70, lo 0.15, `teacher@0.25` | 95.1 % / 73.1 % | single 0.25 | 95.6 % / 76.3 % |
| 96 % | hi 0.70, lo 0.25, `teacher@0.35` | 96.3 % / 67.5 % | single 0.30 | 96.1 % / 72.8 % |

Student rules without the teacher: the best at ≥ 88 % precision is hi 0.70, lo 0.20, `wrist@0.5+cls@0.7` at 88.6 % / 74.9 %, against 88.5 % / 67.8 % for the student alone at 0.45.

### Lab test frames (1,069 knife boxes, 2,483 frames; labels of section 4h)

"No-knife FP" = false detections on frames with no knife. "Cart shadow removed" drops every box on the knife-shaped shadow of the white cart (`results/stage1_kd/rules/static_lab.json`), as a stand-in for the pipeline's static filter.

| Rule | Precision | Recall | False det. | No-knife FP | Cart shadow removed: P / R |
|---|---|---|---|---|---|
| Student 0.45 (today's threshold, no gate) | 75.0 % | 78.3 % | 279 | 247 | 88.3 % / 78.3 % |
| Student 0.25 | 66.5 % | 84.8 % | 457 | 406 | 83.0 % / 84.8 % |
| Student 0.15 | 60.8 % | 86.6 % | 596 | 469 | 75.6 % / 86.5 % |
| **Student 0.45 + hand gate 0.5** | **88.4 %** | **77.3 %** | 108 | 79 | 88.7 % / 77.3 % |
| **Student 0.35 + hand gate 0.5** | **86.9 %** | **80.7 %** | 130 | 100 | – |
| **Student 0.30 + hand gate 0.5** | **85.8 %** | **81.9 %** | 145 | 113 | – |
| **Student 0.25 + hand gate 0.5** | **84.7 %** | **83.6 %** | 161 | 126 | 85.0 % / 83.5 % |
| Student 0.15 + hand gate 0.5 | 81.9 % | 85.4 % | 202 | 151 | 82.1 % / 85.3 % |
| Student 0.25 + hand gate + classifier 0.7 | 86.6 % | 77.2 % | 128 | 98 | 86.8 % / 77.1 % |
| Student: hi 0.70, lo 0.05, `teacher@0.10` (best on GT v2) | 61.8 % | 88.5 % | 584 | 501 | 78.4 % / 88.4 % |
| Student: hi 0.70, lo 0.20, `teacher@0.45` | 69.0 % | 82.4 % | 396 | 361 | 88.0 % / 82.3 % |
| Teacher 0.45 | 64.9 % | 82.8 % | 479 | 431 | 85.2 % / 82.7 % |
| Teacher 0.25 + hand gate 0.5 | 79.5 % | 86.2 % | 238 | 178 | 79.7 % / 86.2 % |
| Shipped 0.45 + hand gate 0.5 | 86.6 % | 71.7 % | 119 | 84 | 86.9 % / 71.7 % |

The same student + hand gate rules on `Ground_Truth_Dataset_v2`:

| Student conf + hand gate 0.5 | Precision | Recall | OOD FP |
|---|---|---|---|
| 0.45 | 89.8 % | 66.7 % | 18 |
| 0.35 | 87.3 % | 70.5 % | 23 |
| 0.30 | 84.6 % | 72.2 % | 29 |
| 0.25 | 83.2 % | 73.7 % | 35 |
| 0.15 | 77.7 % | 77.5 % | 54 |

## Findings

- **The two high numbers belong to different boxes and different cameras, so no rule gets both.** The 92–95 % recall at 0.01 comes with 34–59 % precision; the teacher's 96–98 % precision at 0.45 is measured on `Ground_Truth_Dataset_v2` only. On the lab camera the teacher has 65 % precision at 0.45 (85 % without the cart shadow): it fires on empty hands and chair armrests.
- **Teacher confirmation works on the internet test set and fails on the lab camera.** On GT v2, "student ≥ 0.70, or student ≥ 0.05 confirmed by the teacher at 0.10" keeps the student's 89 % precision and raises recall from 68 % to 78 %. Picked there and applied to the lab frames it gives 61.8 % precision. On GT v2 the teacher alone is still better than any student + teacher rule (92.4 % / 81.6 % at 0.15), so running both models buys nothing over running the teacher.
- **The pipeline's hand gate is the filter that holds precision while the threshold drops.** It removes the cart shadow and most no-knife detections on the lab camera: the student at 0.45 goes from 75.0 % to 88.4 % precision for 1 point of recall. With the gate on, lowering the threshold costs precision slowly: 0.45 → 0.25 gives +6.3 points recall for −3.7 points precision on the lab frames (+7.0 / −6.6 on GT v2).
- **The crop classifier and a tighter gate (0.25 torso) do not help:** at equal precision they keep fewer knives than a lower threshold with the plain gate.
- **Sweet spot: lab student at 0.30 with the hand gate on.** Lab frames: 85.8 % precision, 81.9 % recall (against 88.4 % / 77.3 % at 0.45). GT v2: 84.6 % / 72.2 % (against 89.8 % / 66.7 %). If precision must stay at the 0.45 level, 0.35 keeps about 87 % and still gains 3–4 points of recall. Below 0.25 each extra point of recall costs about two points of precision.

## Proposed solution

1. Run Stage 1 with the lab student at conf 0.30 and keep `hand_gate` (0.5 torso) and `static_filter` on, as `inference_production_pipeline.py` already does. This changes only the `--conf` value.
2. Do not add the teacher as a confirmer: it costs a YOLO26x pass per frame and does not raise precision on the lab camera.
3. Next lever for recall without precision loss: a temporal rule (a track opened by one box at ≥ 0.45 near a wrist is kept at ≥ 0.15 in the next frames). Not tested here: the labelled lab frames are every 10th frame, so a track cannot be followed.

## Not checked

- Thresholds were chosen by looking at both sets, not on a separate tuning set. The two sets agree on the ordering, but the exact sweet spot can move by ±0.05.
- The lab labels are by Claude agents, not a person; about 1 in 7 high-score "no knife" tiles may hold a knife (section 4h of `MODIFICATION_REPORT_STAGE1_MODEL_COMPARISON.md`), so lab precision is a little too low for every rule.
- Box-level numbers only. Stage 2 is armed by detections in a window of frames, so the alarm-level effect of 0.30 (false alarms per hour on normal takes) was not measured, and no full-pipeline run was made.
- The hand gate here keeps a box near any person's wrist; the pipeline uses the tracked threat actor's wrists. The static filter is approximated by removing one known box.
- PyTorch weights at 384x640, not the TensorRT engines; no Edge PC run.

## Files

- Script: `results/samples/poc/sweep_stage1_rules.py` (modes `cache`, `sweep`, `--apply-to`, `DROP_STATIC=`).
- Results in git: `results/stage1_kd/rules/sweep_{gtv2,lab,lab_static}.jsonl` (one row per rule), `static_lab.json`.
- Dev PC only: caches `results/stage1_kd/rules/cache_{gtv2,lab}.json` (rebuilt by `cache`) and the printed tables `*.log` (gitignored).
