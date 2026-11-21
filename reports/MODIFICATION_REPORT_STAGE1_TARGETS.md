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

| Label | Knives | By vote | By Claude | By the human (`review_visibility.bat`) |
|---|---|---|---|---|
| clear | 991 | 658 | 158 | 175 |
| blurred | 208 | 63 | 127 | 18 |
| hidden | 272 | 88 | 162 | 22 |
| no knife | 12 | 1 | | 11 |
| unsure (human: cannot tell) | 6 | | | |

The four voters agreed on 810 of 1,489 knives. Claude looked at the other 679 and was sure of 447; the human answered the other 232. Of Claude's 232 unsure guesses the human called 175 clear, including 42 that Claude had guessed "hidden" and 17 "blurred".

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
| clear | 0.38 | 33 % |
| blurred | 0.68 | 68 % |
| hidden | 0.27 | 23 % |

Hand speed separates blurred from clear knives, but not cleanly: a third of clear knives are at a fast hand.

## Label errors among the missed clear knives (2026-11-21, current numbers)

**Question:** are the clear knives the models miss labelled right?

**What was run** (`results/samples/poc/check_stage1_misses.py`):

```bash
.venv/Scripts/python.exe results/samples/poc/check_stage1_misses.py sheets   # 163 knives, 14 sheets
# voters -> vis_review/miss_check/votes/haiku{1,2,3,4}.json; Claude -> votes/claude.json, duplicates.json
.venv/Scripts/python.exe results/samples/poc/check_stage1_misses.py tally
.venv/Scripts/python.exe results/samples/poc/check_stage1_misses.py review    # -> review_misses.bat
.venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py tally  # folds result.json + duplicates.json in
.venv/Scripts/python.exe results/samples/poc/eval_stage1_targets.py            # MATCH=cover for the second table
```

- **Items:** every clear knife that B misses at conf 0.20 or F misses at conf 0.10 (151), plus the 12 knives labelled "no knife". Each cell shows the label box (green) and B's best box near it (yellow, with conf).
- **Votes:** the by-eye rule, with labels c / w (knife there, box misses it) / b / h / x.
  - cheapag could not vote: its current model cannot open images. Its first answer copied the labels from `items.json`, and a second try answered "cannot see images". A fourth Haiku agent voted in its place.
  - The four voters agreed on 98 of 163. Claude settled 48 and was unsure of 17.
  - A result that differs from an earlier human answer goes back to the human. Where Claude is unsure, the human's earlier answer stands.

**Finding 1: duplicate labels.** 51 test frames have two or three knife labels, and every lab take has one knife. Claude checked all 51 by eye (`miss_check/dup_sheets/`):
- 43 labels are a second box on a knife that is already labelled. Only one of the two can be matched, so the other always counted as a miss.
- 34 of the 43 come from the 113 test knives added by `review_stage1_candidates.py`.
  - Cause: the candidate step picked up detector boxes that matched no label at IoU 0.2. A box over the hand and the knife overlaps a knife-only label by less than 0.2, so the knife was added again.
- Fix: one label per knife. The original box is kept, unless it misses the knife (588, 1345); the duplicates are listed in `miss_check/duplicates.json`. Seven frames with boxes on two different hands were left as they are.

**Finding 2: other label errors.**

| Result on the 163 | Knives |
|---|---|
| label right, clear | 86 |
| hidden, not clear | 12 |
| blurred, not clear | 1 |
| no knife (dropped from the labels) | 6 (943 on a face, plus the human's earlier "no knife" kept where Claude was unsure) |
| duplicate (dropped) | 17 here, 43 in all |
| back to the human (`review_misses.bat`) | 41 |

The 41 for the human:
- 28 knives the human called clear but the voters and Claude call hidden or blurred;
- 8 of the human's 11 "no knife" answers. In 6 of them (612, 623, 631, 633, 641, 649) all voters see a clear knife. Those answers were given 10 in 28 seconds and look like a key slip;
- 5 that Claude could not settle.

Until they are answered, these 41 are scored as "unsure".

**Lab test after the fixes.** 1,440 knives: 909 clear, 201 blurred, 283 hidden, 47 unsure; 43 duplicate and 6 no-knife labels dropped. P with false boxes in brackets.

| Model | Clear R @0.10 | Clear R @0.20 | Clear R @0.30 | Fast-hand P @0.20 | All P @0.20 |
|---|---|---|---|---|---|
| Shipped detector | 87.9 | 84.8 | | 90.8 (34) | 66.3 (534) |
| A YOLO26s `lab_gt` | 90.6 | 88.1 | | 94.9 (18) | 94.3 (68) |
| C1 blur/dark, seed 1 | 91.4 | 89.7 | | 95.3 (17) | 96.1 (46) |
| **B KD from E** | 93.4 | **92.6** | 90.9 | 95.8 (16) | 95.8 (52) |
| F teacher blur/dark | 94.6 | 93.1 | | 94.7 (21) | 94.5 (70) |

Dropping the duplicates turns some hand-plus-knife boxes back into false boxes: B's fast-hand false boxes at 0.20 go from 7 to 16. With `MATCH=cover`, a box that holds at least half of the knife label also counts as a hit (one hit per knife):

| Model (`MATCH=cover`) | Highest conf with clear R ≥ 90 % | Clear R | Blurred R | Fast-hand P | All P |
|---|---|---|---|---|---|
| A YOLO26s `lab_gt` | 0.10 | 90.9 | 59.2 | 95.7 (16) | 93.1 (87) |
| C blur/dark | 0.10 | 91.4 | 66.2 | 97.7 (9) | 94.2 (73) |
| C1 blur/dark, seed 1 | 0.10 | 91.6 | 65.7 | 95.8 (16) | 96.0 (50) |
| L half blur | 0.10 | 91.6 | 68.2 | 95.0 (20) | 92.7 (94) |
| **B KD from E** | **0.30** | **91.1** | 63.7 | **98.4 (6)** | **97.7 (28)** |
| H KD from F | 0.30 | 90.1 | 59.2 | 97.8 (8) | 96.2 (46) |
| E teacher | 0.25 | 90.2 | 58.2 | 97.5 (9) | 96.0 (48) |
| F teacher blur/dark | 0.40 | 90.6 | 56.2 | 97.8 (8) | 96.7 (39) |

B at conf 0.20 with `MATCH=cover`: clear R 92.8 %, fast-hand P 98.1 (7), all P 96.9 (39). Shipped detector: best 88.1 %.

**What this shows:**
- **Both targets are met by B on the fixed labels.** Clear-knife recall is 92.6 % at conf 0.20, inside the 90–95 % target. Fast-hand precision is 95.8 %, or 98.1 % when a hand-plus-knife box counts.
- Most of the earlier "stops at 88 %" was the test labels:
  - each duplicate box was a miss no model could avoid;
  - the extra knives added by the review were often the same knife boxed again.
- Every retrained student now reaches 90 % on clear knives. The shipped detector still does not.
- **The 41 knives with the human can move this.** They are found 38–51 % of the time. If all of them end up clear, B's clear recall at 0.20 is 90.0 %; if none do, 92.6 %.

## Results with the human's visibility answers (before the label fixes)

`eval_stage1_targets.py` after `tally` with `vis_review/human/votes.csv` (991 clear, 208 blurred, 272 hidden). P with false boxes in brackets; precision does not depend on the visibility labels and is unchanged.

| Model | Clear R @0.10 | Clear R @0.20 | Clear R @0.25 | Blurred R @0.20 | Fast-hand P @0.20 | All P @0.20 |
|---|---|---|---|---|---|---|
| Shipped detector | 83.7 | 80.5 | 79.1 | 51.4 | 91.9 (30) | 66.9 (525) |
| A YOLO26s `lab_gt` | 86.4 | 83.2 | 82.3 | 52.4 | 97.4 (9) | 95.6 (52) |
| C YOLO26s blur/dark | 86.6 | 84.6 | 83.8 | 56.7 | 97.8 (8) | 95.7 (52) |
| C1 same, seed 1 | 86.7 | 84.8 | 83.6 | 57.2 | 97.5 (9) | 97.3 (32) |
| L half blur | 86.3 | 84.5 | 83.8 | 57.7 | 97.3 (10) | 94.6 (66) |
| **B KD from E** | **88.5** | **87.1** | **86.1** | 61.1 | **98.1 (7)** | **97.0 (38)** |
| H KD from F | 87.9 | 86.7 | 85.5 | 58.2 | 97.9 (8) | 95.9 (51) |
| E teacher YOLO26x | 87.4 | 85.5 | 85.0 | 55.3 | 97.3 (10) | 95.6 (53) |
| F teacher YOLO26x blur/dark | **90.1** | 88.4 | 88.0 | 57.7 | 96.7 (13) | 95.5 (57) |

- **Only teacher F reaches 90 % clear recall, and only at conf 0.10** (fast-hand P 95.2 with 20 false boxes). No student reaches 90 % at any listed conf; the best is B, 88.5 % at 0.10 and 87.1 % at 0.20.
- Clear recall fell 2–4 points against the first run. The reason: the 175 knives the human called clear are hard ones. B finds 71 % of them at 0.20, against 90.4 % of the 816 clear knives labelled before. F finds 76 %.
- **11 test knives are "no knife" by the human's eye** (12 with the voters' one). The detectors still box 55–64 % of them. They sit in "all R" and their boxes count as correct; they are left in the labels until checked.

## First results, before the human's answers

The 232 knives Claude was unsure of were scored as their own group here, so the clear group was the 816 easier knives.

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

With the human's answers (final numbers above):
- **Fast-swing precision: met.** Unchanged, 96–98 % for every retrained model.
- **Clear-knife recall: not met.** Best student B is at 87–88.5 %, best model F at 90.1 % (conf 0.10). The gap to 95 % is about 65 clear knives for B.
- B is still the best student on both targets.

First run (816 clear knives):
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

- 41 knives from the miss check wait in `review_misses.bat`. When they are answered, re-run `check_stage1_misses.py tally`, `label_stage1_visibility.py tally` and `eval_stage1_targets.py`.
- Only the missed clear knives were checked for label errors. The knives the models find, and the blurred and hidden ones, were not re-checked. Duplicates were checked in every frame.
- The duplicate list is Claude's call alone, by eye, without the four voters: one knife per take makes it a geometric check.
- `MATCH=cover` is looser than the IoU 0.2 rule used in every earlier report. It is shown next to the standard numbers, not instead of them.
- The visibility labels are from crops. A few "clear" knives may be partly hidden in the full frame.
- Hand speed comes from pose 10 frames apart. A swing that starts and stops between two labelled frames looks slow. Boxes with no wrist found are not in the fast or slow groups.
- Conf points are chosen on the test set.
- Not checked on the Edge PC or with TensorRT. Ground_Truth_Dataset was not split this way.

## Files

- In git:
  - scripts `results/samples/poc/label_stage1_visibility.py`, `results/samples/poc/eval_stage1_targets.py`;
  - `review_visibility.bat`; `label_review_ui.py` gained `--vis` (keys c / b / h / x / u);
  - `results/stage1_kd/vis_review/{items.json, visibility.json, votes/*.json, claude_notes/, human/votes.csv, voter_prompt.md, clear_missed_B.jpg}`.
  - miss check: `results/samples/poc/check_stage1_misses.py`, `review_misses.bat`, `results/stage1_kd/vis_review/miss_check/{items.json, voter_prompt.md, votes/, duplicates.json, result.json, human/, sheets/, dup_sheets/}`;
  - `label_review_ui.py --vis` gained the `w` key and shows the human's earlier answer.
- Gitignored: `results/stage1_kd/vis_review/eval_targets.log`, `eval_targets_cover.log`.
- Dev PC only: the sheets `results/stage1_kd/vis_review/{sheets, claude_sheets}/`.
