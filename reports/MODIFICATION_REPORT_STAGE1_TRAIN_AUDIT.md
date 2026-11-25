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

Because phone takes appeared mislabeled as `knife` in **both** Fold A (`006, 007`) and Fold B (`009, 028`), both OOF fold models learned to treat an edge-on phone as a knife, letting **65 of the 88** pass into the `Safe` bucket. Take-level provenance check is essential alongside OOF scoring.

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

## Not checked

- The `3,589` `Safe` frames were not inspected by eye (except the known `65` phone-take frames `006/007/009/028` uncovered by take metadata).
- The `45` `hn_dups` in `lab_gt_aug_hn` were detected geometrically (`IoU`/`cover`) but not yet rendered/reviewed by eye.
- No cleaned dataset (`lab_gt_clean`) was built or used to retrain Student B yet.

## Files

- Script (in git): `results/samples/poc/audit_stage1_train.py`.
- Report (in git): `reports/MODIFICATION_REPORT_STAGE1_TRAIN_AUDIT.md`.
- Audit summaries (in git): `results/stage1_kd/train_audit/{fold_meta.json,triage_summary.json,eye_audit.json}`, `results/stage1_kd/train_audit/eye_votes/shard_*.json`.
- Large generated files (dev PC only, not committed): `results/stage1_kd/train_audit/{audit_scores.json,suspect_items.json,fold_a,fold_b,run_fold_a,run_fold_b,sheets}`.
