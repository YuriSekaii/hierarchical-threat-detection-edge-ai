# Stage 1 scored by target: clear knives and fast swings (2026-11-21)

## Question

One overall precision and recall hides what matters. The user's targets:
1. **Knives a person can clearly see: recall 90–95 %.**
2. **Fast swings: recall may be low, but precision must be high.** A box on a fast-moving hand must really be a knife, not a random blur.

How do the Stage 1 models do on each target?

## What was run

**Step 1: visibility label per test knife** (`results/samples/poc/label_stage1_visibility.py`). Every labelled lab test knife, 1,489 of them including the 113 knives added in `MODIFICATION_REPORT_STAGE1_LABEL_RECHECK.md`, was sorted by eye:
- `c`: clear, you can tell it is a knife;
- `b`: blurred by motion;
- `h`: hidden or barely visible (in the fist, against dark clothes, too small);
- `x`: no knife.

The labelling followed the by-eye rule: four voters (three Haiku agents at high effort, and cheapag). Unanimous answers stand, Claude settles disagreements, and the knives Claude is unsure of go to `review_visibility.bat`.

```bash
.venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py sheets
# voters -> results/stage1_kd/vis_review/votes/{haiku1,haiku2,haiku3,cheapag}.json; Claude -> votes/claude.json
.venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py tally
.venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py review     # unsure ones -> review_visibility.bat
```

| Label | Knives | By vote | By Claude |
|---|---|---|---|
| clear | 816 | 658 | 158 |
| blurred | 190 | 63 | 127 |
| hidden | 250 | 88 | 162 |
| no knife | 1 | 1 | |
| unsure (waits for the human) | 232 | | |

The four voters agreed on 810 of 1,489 knives. Claude looked at the other 679 and was sure of 447.

**Step 2: scoring** (`results/samples/poc/eval_stage1_targets.py`, lab cache, no GPU):

```bash
.venv/Scripts/python.exe results/samples/poc/eval_stage1_targets.py     # FAST=0.5 by default
```

- Boxes go through NMS 0.5, then greedy matching at IoU 0.2, as before.
- **Recall** is reported per knife group.
- **Precision** is reported per box group, by the hand the box sits at:
  - each box belongs to the nearest wrist within half a torso length;
  - the wrist's speed is its mean shift to the labelled frames 10 before and 10 after, in torso lengths;
  - a box at a hand moving ≥ 0.5 torso lengths per 10 frames (1/3 s) is a **fast-hand box**;
  - a box with no wrist that near is **away from a hand**.

**Does hand speed match the blurred label?**

| Knife label | Median hand speed | Share at a fast hand |
|---|---|---|
| clear | 0.36 | 30 % |
| blurred | 0.70 | 72 % |
| hidden | 0.27 | 22 % |

Hand speed separates blurred from clear knives, but not cleanly: 30 % of clear knives are at a fast hand.

## Results

**At each model's highest conf that still gives ≥ 90 % recall on clear knives** (lab test; P with false boxes in brackets):

| Model | Conf | **Clear R** | Blurred R | **Fast-hand P** | All boxes P |
|---|---|---|---|---|---|
| Shipped detector | never: best 86.4 at 0.10 | | | | |
| A YOLO26s `lab_gt` | 0.10 | 90.1 | 55.3 | 96.5 (13) | 93.7 (80) |
| C YOLO26s blur/dark | 0.10 | 90.2 | 60.0 | 97.7 (9) | 94.6 (69) |
| C1 same, seed 1 | 0.10 | 90.6 | 61.1 | 96.4 (14) | 96.5 (44) |
| L half blur | 0.10 | 90.4 | 63.2 | 95.7 (17) | 93.2 (87) |
| **B KD from E** | **0.20** | **90.4** | 61.1 | **98.1 (7)** | **97.0 (38)** |
| H KD from F | 0.10 | 91.2 | 60.0 | 97.2 (11) | 94.3 (73) |
| E teacher YOLO26x | never: best 90.0 at 0.10 | | | | |
| F teacher YOLO26x blur/dark | 0.30 | 90.4 | 55.3 | 97.4 (10) | 96.3 (45) |

**Highest clear-knife recall** of any model and conf: F teacher at 0.10, 92.5 %, with fast-hand P 95.2 (20). No model reaches 95 %.

Recall on hidden knives is 70–81 % for the students and 33–56 % for the shipped detector. Precision on boxes away from a hand is high for the retrained models, but 7–12 % for the shipped detector, which makes 190–446 false boxes there on the lab camera.

The full table (6 confs × 9 models, every group) is in `results/stage1_kd/vis_review/eval_targets.log`.

**Clear knives the best student misses.** B at conf 0.20 misses 78 clear knives:
- B's own best conf on them: 59 under 0.05 (not seen at all), 13 between 0.05 and 0.20, 6 boxes in the wrong place;
- 31 are missed by all six students at 0.10;
- where they are: 21 in Oct 31 take 002, 20 in the normal-mixed take, 10 in slash.

In the contact sheet `results/stage1_kd/vis_review/clear_missed_B.jpg`:
- the knife is held away from the body against the bright floor or wall;
- raised next to the head;
- blade sideways or pointing up;
- or next to a chair or the cart.

Their box size is about the same as other clear knives (median 101 px against 113 px).

## Findings

- **Fast-swing precision is already met.** At fast hands every retrained model is at 95–98 % precision with 7–17 false boxes on the whole lab test. That is as high as or higher than at slow hands. The retrained models do not box random blur.
- **Clear-knife recall stops at about 90 %.** Each retrained student reaches 90–91 % only at conf 0.10–0.20. The best single number is 92.5 % (teacher F, 0.10). The gap to 95 % is about 40 clear knives, and the models do not see most of them at all.
- **B, the student distilled from E, is the best trade-off:** 90.4 % clear recall at conf 0.20 with 97 % precision and 7 fast-hand false boxes. The differences between students are 1–2 points, inside the 3-point training noise measured earlier.
- **The shipped detector misses both targets on the lab camera:** at best 86.4 % clear recall, and hundreds of false boxes away from the hands.
- **The missing clear knives are pose and background cases, not blur:**
  - arm stretched out, knife against the bright floor or wall;
  - knife raised by the head;
  - blade sideways or up;
  - knife next to the chair or cart.

  The next lever is more training frames of these poses, not more blur.

## Not checked

- 232 test knives still wait for the human's visibility label; they are scored as their own "unsure" group. When they are answered, re-run `tally` and `eval_stage1_targets.py`.
- The visibility labels are from crops. A few "clear" knives may be partly hidden in the full frame.
- Hand speed comes from pose 10 frames apart. A swing that starts and stops between two labelled frames looks slow. Boxes with no wrist found are not in the fast or slow groups.
- Conf points are chosen on the test set.
- Not checked on the Edge PC or with TensorRT. Ground_Truth_Dataset was not split this way.

## Files

- In git:
  - scripts `results/samples/poc/label_stage1_visibility.py`, `results/samples/poc/eval_stage1_targets.py`;
  - `review_visibility.bat`; `label_review_ui.py` gained `--vis` (keys c / b / h / x / u);
  - `results/stage1_kd/vis_review/{items.json, visibility.json, votes/*.json, claude_notes/, human/votes.csv, voter_prompt.md, clear_missed_B.jpg}`.
- Gitignored: `results/stage1_kd/vis_review/eval_targets.log`.
- Dev PC only: the sheets `results/stage1_kd/vis_review/{sheets, claude_sheets}/`.
