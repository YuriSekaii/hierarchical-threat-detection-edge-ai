# Stage 1 training label audit: 2-fold out-of-fold, feature-space, goal-alignment, and `/swarm` eye check (2026-11-25)

**Status: audit only. No dataset in `data/` and no weights in `weights/` were modified.**

## Question

Student B (`kdgtNL`, trained on `data/stage1_negatives/lab_gt`) meets both lab test targets after cleaning the test labels (`MODIFICATION_REPORT_STAGE1_TARGETS.md`). Since `lab_gt` was labelled by the same process, which training frames are actually safe, which are corrupted, and can a hybrid out-of-fold + feature-space + goal-alignment pipeline shrink the `/swarm` eye-inspection pool before looking at crops?

## What was run

Script: `results/samples/poc/audit_stage1_train.py`:

```bash
.venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py selftest
.venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py split
DEV=0 EPOCHS=60 WORKERS=2 .venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py train a   # GPU 0, 0.600 h
DEV=1 EPOCHS=60 WORKERS=2 .venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py train b   # GPU 1, 0.588 h
DEV=0 .venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py score
.venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py triage
# /swarm: 7 inspectors (inspector-1..7) checked sheets 0000..0069 (829 tiles) by eye -> eye_votes/shard_*.json
.venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py tally_eye
```

How the hybrid triage works:
1. **Take-grouped 2-fold split (`split`):** `lab_gt/train.txt` (4,418 images) split into Fold A (2,234 images: 1,164 positive, 322 human-checked) and Fold B (2,184 images: 1,206 positive, 328 human-checked) grouped by recording take (`006_thrust_low...` phone frames and `2026-11-05_006_...` lab frames share a group so no take leaks across folds).
2. **Out-of-fold models (`train a/b`):** 60-epoch YOLO26s (Ultralytics 8.4.6 MuSGD recipe). Each model scores only the unseen fold.
3. **Signals (`score` + `triage`):**
   - Out-of-fold predictions (`conf >= 0.05`) and `E2ELoss` (`box`, `cls`, `dfl`) on letterboxed 640×640 frames, paired with external `v2_p100` (`cls == 0`, never saw lab frames) and `stage1_cropcls2.pt`.
   - **Goal alignment (detect-head TracIn):** cosine similarity of each frame's `Detect` head gradient against the unseen fold's human-checked clean anchor frames (leave-one-out when the frame is itself in the anchor).
   - **YOLO neck feature outliers:** P3 (layer 16) + P4 (layer 19) RoI-pooled 384-d embeddings; cross-take cosine similarity to clean knife boxes.
   - **Geometry & temporal:** multi-box `IoU` / `cover` check (`lab_gt` and `lab_gt_aug_hn` additions), pose wrist distance (`yolo26s-pose.pt`), and single-frame `knife ↔ no_knife` temporal flips within each lab take.

## Results

### 1. Pool reduction

| Set | Total | Safe (skipped) | Suspect (to `/swarm`) | Contact sheets |
|---|---|---|---|---|
| `lab_gt/train.txt` | 4,418 | **3,589 (81.2 %)** | **829 (18.8 %)** — 323 pos, 506 neg | 70 (`sheet_0000`–`0069`) |

| Source | Safe pos | Safe neg | Suspect pos | Suspect neg |
|---|---|---|---|---|
| `lab_gt` (3,159) | 1,581 | 954 | 259 | 365 |
| `orig_v2` (659) | 466 | 115 | 64 | 14 |
| `phone` + `other` (600) | 0 | 473 | 0 | 127 |

### 2. `/swarm` eye-inspection verdicts on the 829 suspect crops

| Source | `ok` | `wrong_pos` | `bad_box` | `missed_knife` | `unclear` | Total |
|---|---|---|---|---|---|---|
| `lab_gt` | 492 | **25** | **20** | **30** | 57 | 624 |
| `orig_v2` | 67 | 0 | **2** | 0 | 9 | 78 |
| `phone` (`ft_phonecls`) | 99 | 0 | 0 | 13* | 0 | 112 |
| `other` | 13 | 0 | 0 | 0 | 2 | 15 |
| **Total** | **671** | **25** | **22** | **43 (30 real + 13\*)** | **68** | **829** |

*\*Note on the 13 `phone` items (`#466, #468–#479`, `006_thrust_low_front_164412`):* `ft_phonecls` correctly labels take `006` as `phone` (no knife). Held edge-on, the black phone resembles a narrow tactical blade and fooled both `v2_p100` and the eye inspector.

**What the 77 real frame-level corruptions (`25 wrong_pos + 22 bad_box + 30 missed_knife`) are:**
- **`wrong_pos` (25, all `lab_gt`):** black wristwatch boxed as a knife (`#4, #25, #167`), empty fists (`#158, #660, #723, #743, #766, #819`), bent elbow/sleeve (`#2, #666, #672`), background shelf/chair leg (`#635, #643`), and black smartphones (`#242, #266, #285`). 15 of the 25 were in tiles marked `human`.
- **`bad_box` (22: 20 `lab_gt`, 2 `orig_v2`):** **15 of the 16 multi-box frames in `lab_gt`** (`93.8 %`) are duplicate overlapping boxes on one knife; the rest are shifted boxes sitting on the wristwatch/wrist instead of the blade (`#52, #172, #184`).
- **`missed_knife` (30 in `lab_gt`):** frames labeled `no knife` where a knife is clearly visible in the hand or on the desk (`#7, #9, #12, #13, #20, #64–#74, #97, #137, #237, #241, #359, #748–#765`). 13 were in tiles marked `human`.
- **`unclear` (68):** heavy motion blur or occlusion that should be excluded from training rather than labeled positive or negative.

### 3. Systemic take-level bug uncovered: 88 phone-take frames labeled as `knife` in `lab_gt`

Nov 5 takes `006`, `007`, `008`, `009`, and `028` (`thrust_low`) were recorded with a **phone in hand, not a knife** (`MODIFICATION_REPORT_LAB_SEGMENTS.md`, `MODIFICATION_REPORT_STAGE1_PHONE_NEGATIVES.md`). However, `label_lab_stage1.py` ran on `2026-11-05/train_attack/*.avi` without skipping them:

| Take (phone-only) | Total in `lab_gt` | Labeled `knife` (class 1) | Human-checked | Flagged suspect by OOF audit |
|---|---|---|---|---|
| `2026-11-05_006_thrust_low_front` | 79 | **35** | 1 | 8 |
| `2026-11-05_007_thrust_low_left` | 46 | **1** | 0 | 0 |
| `2026-11-05_009_thrust_low_right` | 106 | **12** | 7 | 10 |
| `2026-11-05_028_thrust_low_lefthand` | 76 | **40** | 0 | 5 |
| **Total** | **307** | **88** | **8** | **23** |

Because phone takes appeared mislabeled as `knife` in **both** Fold A (`006, 007`) and Fold B (`009, 028`), both OOF fold models learned to treat an edge-on phone as a knife, letting **65 of the 88** pass into the `Safe` bucket. Take-level provenance check is essential alongside OOF scoring. *(Correction from eye verification below: take `028` actually holds a real knife and 6 boxes in `009` sit on a desk knife, leaving **42** true phone-as-knife boxes across `006`, `007`, and `009`.)*

### 4. `lab_gt_aug_hn` multi-box duplicates

Checking `data/stage1_negatives/lab_gt_aug_hn/train.txt` (`fix/` and `neg/` additions) found **`45` multi-box frames** (`hn_dups`) with overlapping boxes (`IoU >= 0.1` or `cover >= 0.3`), confirming the unchecked-duplicate note in `MODIFICATION_REPORT_STAGE1_LABEL_RECHECK.md`.

### 5. Signal precision on the 829 suspect items

| Flag | Flagged | Confirmed corrupt (`wrong_pos` + `bad_box` + `missed_knife`) | Precision | Unclear |
|---|---|---|---|---|
| `pos_multibox` | 16 | 15 | **93.8 %** | 0 |
| `pos_cropcls_low` | 44 | 11 | **25.0 %** | 6 |
| `pos_unseen_oof_v2` | 128 | 25 | **19.5 %** | 28 |
| `pos_temporal_spike` | 32 | 6 | **18.8 %** | 6 |
| `pos_high_oof_loss` | 172 | 32 | **18.6 %** | 23 |
| `neg_high_oof_loss` | 145 | 27 | **18.6 %** | 22 |
| `pos_feat_outlier` | 83 | 13 | **15.7 %** | 8 |
| `neg_det_hit` | 250 | 30 | **12.0 %** | 26 |
| `pos_neg_goal_grad` | 191 | 15 | 7.9 % | 22 |
| `neg_neg_goal_grad` | 27 | 2 | 7.4 % | 1 |
| `pos_far_from_wrist` | 83 | 6 | 7.2 % | 4 |
| `neg_cropcls_hi` | 281 | 17 | 6.0 % | 20 |
| `neg_temporal_sandwich` | 170 | 7 | 4.1 % | 5 |

## Findings

- **Hybrid triage saved 81.2 % of the eye-check work** (829 crops instead of 4,418 frames) and surfaced **77** confirmed frame-level label corruptions plus **88** systemic phone-as-knife labels in `lab_gt`, and **45** duplicate multi-box frames in `lab_gt_aug_hn`.
- **`orig_v2` (659 images) is clean** relative to `lab_gt`: 0 `wrong_pos`, 0 `missed_knife`, only 2 `bad_box` (`#227` duplicate + 1 other).
- **The biggest clean win before retraining Student B** is fixing:
  1. Remove/relabel the **88** `knife` boxes from phone takes `2026-11-05_006..009` and `028` (turn them into `phone` class 2 or exclude from `knife`).
  2. Drop or fix the **77** confirmed `wrong_pos` / `bad_box` / `missed_knife` frames and **68** `unclear` frames (and propagate fixes to `lab_gt_aug` / `lab_gt_aug_hn` + clean its **45** `hn_dups`).
- **Limitation of OOF + feature space:** systematic errors repeated across both folds (`006/007` vs `009/028`) look consistent to OOF models and k-NN; take-level metadata checks are still required. Also, wrist-crop feature k-NN was dropped from negative flagging because pooling a `30%×30%` wrist box vs `5%×8%` blade box encoded box scale rather than object identity.

## Human review + `lab_gt_clean` retraining (2026-11-25 to 2026-11-26)

**Status: new clean dataset `data/stage1_negatives/lab_gt_clean` and retrained weights `weights/stage1_repro846_*clean*.pt` (dev PC only). Shipped weights are unchanged.**

### 1. Human double-check (`review_train_audit.bat`)

All `143` non-`ok` items from `lab_gt` and `orig_v2` (`66 unclear`, `30 missed_knife`, `25 wrong_pos`, `22 bad_box`) were reviewed by the user in `label_review_ui.py` (`results/stage1_kd/train_audit/human_review/votes.csv`): `71 a`, `9 b`, `1 c`, `5 box` (drawn), `34 n`, `23 u`.

**Eye check of takes `006–009` and `028` before building `lab_gt_clean` (`check_028.jpg`, `check_2026-11-05_006..009.jpg`):**
- **`2026-11-05_028_thrust_low_lefthand` has a real knife** (shiny blade and handle clearly visible in `f00240`, `f00320`, `f00340`), contradicting the earlier note in `MODIFICATION_REPORT_LAB_SEGMENTS.md`. All `40` `knife` labels in `028` were kept.
- **`2026-11-05_006`, `007`, `009`** are phone takes (`008` has no `lab_gt` frames): `35` (`006`) + `1` (`007`) + `6` (`009` hand-held phone) = `42` `knife` boxes (1 already set `n` by human -> `41` converted) were moved to `class 2 (phone)`. Six frames in `009` (`f00480`, `f00860`, `f00880`, `f00900`, `f01320`, `f02360`) box the real knife lying on the desk (`xc < 0.25, yc < 0.25`) and stay `class 1 (knife)`.

**Dataset built (`audit_stage1_train.py build_clean`):** `4,395` train images (`2,328` positive, `2,067` negative; `23` `u` dropped, `86` boxes fixed/recovered, `34` set negative, `41` phone-take `knife` boxes moved to `phone`, zero duplicate multi-box frames). Validation (`350` images) is unchanged.

### 2. Clean retraining (`train_stage1_repro.py`, 250 epochs, MuSGD, Ultralytics 8.4.6)

```bash
.venv/Scripts/python.exe results/samples/poc/audit_stage1_train.py build_clean
# GPU 1: plain YOLO26s (resumed from epoch 203 with WORKERS=0 after transient DataLoader worker MemoryError at epoch 204)
DEV=1 EPOCHS=250 PATIENCE=250 WORKERS=2 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . labgtclean
# GPU 0: Teacher YOLO26x (12.09 h) -> KD Student B YOLO26s (6.07 h)
CUDA_VISIBLE_DEVICES=0 DEV=0 EPOCHS=250 PATIENCE=250 WORKERS=2 .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . teachergtclean
CUDA_VISIBLE_DEVICES=0 DEV=0 EPOCHS=250 PATIENCE=250 WORKERS=2 TEACHER=weights/stage1_repro846_teachergtclean_p250_last.pt .venv/Scripts/python.exe results/samples/poc/train_stage1_repro.py . kdgtclean
```

Caching and scoring (`DEV=0/1`): `sweep_stage1_rules.py . cache lab|gtv2`, `eval_stage1_targets.py` (IoU 0.2 and `MATCH=cover`), and `bench_stage1_holdout.py`.

### 3. Benchmark results

**Lab test targets (`eval_stage1_targets.py`, 1,437 knives: 925 clear, 209 blurred, 294 hidden, 9 unsure; NMS 0.5, IoU 0.2):**

| Model | Conf | Clear R | Blurred R | Hidden R | All R | All P (FP) | Fast-hand P (FP) |
|---|---|---|---|---|---|---|---|
| `A labgt` (old `lab_gt`) | 0.10 | 90.3 | 55.0 | 70.7 | 80.8 | 92.1 (99) | 93.9 (23) |
| `A labgt` (old `lab_gt`) | 0.20 | 87.7 | 52.2 | 67.3 | 77.9 | 94.4 (67) | 95.1 (17) |
| **`labgtcleanL`** (plain, last) | 0.10 | **90.7** | **59.3** | **72.1** | **81.9** | **94.4 (70)** | 93.5 (25) |
| **`labgtcleanL`** (plain, last) | 0.20 | **88.5** | **56.9** | **68.0** | **79.3** | **95.6 (53)** | 94.2 (21) |
| `E teacher` (old `lab_gt`) | 0.25 | 89.5 | 54.1 | 67.3 | 79.5 | 94.9 (61) | 95.3 (17) |
| **`teachergtcleanL`** (teacher, last) | 0.25 | **90.3** | 53.6 | **68.7** | **80.2** | **95.7 (52)** | 93.8 (23) |
| `B KD from E` (old `lab_gt`) | 0.10 | 93.0 | 64.1 | 75.9 | 85.0 | 94.4 (73) | 94.2 (23) |
| `B KD from E` (old `lab_gt`) | 0.20 | 91.9 | 60.8 | 72.1 | 83.0 | **95.7 (53)** | **95.8 (16)** |
| `B KD from E` (old `lab_gt`) | 0.25 | 90.9 | 59.3 | 71.4 | 82.0 | **96.2 (47)** | 95.7 (16) |
| **`kdgtclean`** (clean KD, `best.pt`) | 0.10 | **93.8** | **67.0** | **80.3** | **86.9** | 93.7 (84) | **94.8 (22)** |
| **`kdgtclean`** (clean KD, `best.pt`) | 0.20 | **92.4** | **60.8** | **78.2** | **84.7** | 94.9 (65) | **95.8 (17)** |
| **`kdgtclean`** (clean KD, `best.pt`) | 0.25 | **91.7** | **59.8** | **76.9** | **83.8** | 95.7 (54) | **96.0 (16)** |
| **`kdgtclean`** (clean KD, `best.pt`) | 0.30 | **90.7** | 58.4 | **75.5** | **82.6** | 96.0 (50) | **96.1 (15)** |
| `kdgtcleanL` (clean KD, last) | 0.20 | 91.8 | 56.9 | 74.1 | 82.7 | 95.3 (58) | 95.2 (19) |
| `kdgtcleanL` (clean KD, last) | 0.25 | 91.5 | 56.0 | 73.5 | 82.3 | 95.8 (52) | 95.2 (19) |

With `MATCH=cover`: `kdgtclean` at `0.25` reaches **92.0 % clear R**, **97.7 % fast-hand P (9)**, **96.7 % all P (42)**, and **100.0 % away-from-hand P (0)**; at `0.30`: **91.0 % clear R**, **97.9 % fast-hand P (8)**, **96.9 % all P (38)**.

**Original camera (`Ground_Truth_Dataset_v2`, 625 images) and lab cache (`LAB_FIX`, NMS 0.5, IoU 0.2):**

| Model | GTv2 @0.20 P / R (FP) | GTv2 @0.25 P / R (FP) | Lab @0.10 P / R (FP) | Lab @0.20 P / R (FP) |
|---|---|---|---|---|
| `labgtL` | 80.3 / 74.0 (62) | 82.1 / 72.5 (54) | 93.7 / 79.2 (80) | 95.6 / 76.2 (52) |
| `labgtcleanL` | **80.8 / 74.9 (61)** | **84.8 / 73.7 (45)** | **95.7 / 80.1 (54)** | **96.8 / 77.6 (38)** |
| `teachergtL` | 90.8 / 71.9 (25) | 91.4 / 71.3 (23) | 94.5 / 80.6 (70) | 95.6 / 78.2 (53) |
| `teachergtcleanL` | **93.8** / 71.1 (**16**) | **94.8** / 69.3 (**13**) | **95.0** / 80.1 (**63**) | **96.5 / 78.8 (43)** |
| `kdgtNL` (old B) | **85.6** / 76.6 (**44**) | **87.7** / 74.9 (**36**) | **95.6** / 83.1 (**57**) | **97.0** / 81.1 (**38**) |
| **`kdgtclean`** (`best.pt`) | 78.2 / **81.6** (78) | 80.8 / **79.8** (65) | 94.8 / **84.9** (69) | 96.0 / **82.7** (51) |
| `kdgtcleanL` (`last.pt`) | 76.9 / 77.8 (80) | 78.0 / 76.9 (74) | 95.3 / 82.1 (60) | 96.4 / 80.7 (45) |

At `0.30` on `GTv2`, `kdgtclean` reaches `83.1 % P / 77.5 % R (54)`, higher recall than `kdgtNL` at `0.20` (`76.6 %`).

**Holdout takes (`bench_stage1_holdout.py` at conf `0.20`, NMS-free):**

| Model (@0.20) | Nov 14 test frames (mean / worst) | Windows (mean / worst) | Oct 31 take 002 | Second phone 050 | Remote 056 |
|---|---|---|---|---|---|
| `kdgtNL` (old B) | 41 % / 8 % | 57 % / 12 % | 73 % | 2 % | 2 % |
| **`kdgtclean`** (`best.pt`) | **43 % / 9 %** | **59 % / 12 %** | **77 %** | 3 % | 3 % |
| `kdgtcleanL` (`last.pt`) | 42 % / 9 % | 58 % / 12 % | 74 % | 2 % | 4 % |
| `labgtcleanL` | 41 % / 9 % | 57 % / 13 % | 70 % | 3 % | 4 % |
| `teachergtcleanL` | 42 % / 9 % | 58 % / 13 % | 67 % | **1 %** | **2 %** |

### 4. Findings from retraining on `lab_gt_clean`

- **Data cleaning improved every plain model directly:** `labgtcleanL` beats `labgtL` on both lab (`95.7/80.1` vs `93.7/79.2` at `0.10`, `-26` FP; blurred R `+4.3` to `+4.7`) and `GTv2` (`84.8/73.7` vs `82.1/72.5` at `0.25`). `teachergtcleanL` also gained `+1` to `+3.4` precision points with fewer false boxes on both sets.
- **Distilled `kdgtclean` (`best.pt`) is the new strongest student on recall:** at `0.20` it lifts lab clear recall to **92.4 %** (vs `91.9 %` for `B`), hidden-knife recall to **78.2 %** (`+6.1`), overall lab recall to **84.7 %** (`+1.7`), and `GTv2` recall to **81.6 %** (`+5.0`), while matching `B`'s **95.8 %** fast-hand precision. At `0.25` it holds **91.7 %** clear recall with **96.0 %** fast-hand precision (`97.7 %` with `cover`, `0` away-from-hand FP).
- **Phone / remote confusers stay low (`2–3 %`)** without phone-negative collapse.

## Not checked

- The `3,589` `Safe` frames were not inspected by eye (except the `006–009` and `028` take checks).
- The `45` `hn_dups` in `lab_gt_aug_hn` were detected geometrically (`IoU`/`cover`) but not yet folded into a `lab_gt_clean_aug_hn` set.
- `kdgtclean` was not exported to TensorRT (`384x640`), run on the Edge PC, or run through Stage 2; `Ground_Truth_Dataset` (715 raw images) was not re-scored separately from `Ground_Truth_Dataset_v2`.
- Only one seed per clean model was trained; `labgtclean` resumed from epoch 203 with `WORKERS=0` after a Windows DataLoader `MemoryError`.

## Files

- Scripts (in git): `results/samples/poc/audit_stage1_train.py` (`build_clean`), `results/samples/poc/train_stage1_repro.py` (`labgtclean`, `teachergtclean`, `kdgtclean`, `RESUME`, `WORKERS`), `results/samples/poc/eval_stage1_targets.py` (`EXTRA_MODELS`).
- Report (in git): `reports/MODIFICATION_REPORT_STAGE1_TRAIN_AUDIT.md`.
- Audit summaries & human votes (in git): `results/stage1_kd/train_audit/{fold_meta.json,triage_summary.json,eye_audit.json}`, `results/stage1_kd/train_audit/eye_votes/shard_*.json`, `results/stage1_kd/train_audit/human_review/{items.json,votes.csv}`, `review_train_audit.bat`, `results/stage1_kd/holdout.jsonl`.
- Weights & logs (dev PC only, not committed): `weights/stage1_repro846_{labgtclean,teachergtclean,kdgtclean}_p250{,_last}.pt`, `data/stage1_negatives/lab_gt_clean/`, `results/stage1_kd/train_audit/{audit_scores.json,suspect_items.json,fold_a,fold_b,run_fold_a,run_fold_b,sheets,eval_targets_clean*.log}`.
