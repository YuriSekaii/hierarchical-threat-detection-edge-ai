# Stage 1 label re-check by eye: missing test knives, training fixes, new hand frames (2026-11-21)

## Question

The lab "false boxes" of the Stage 1 students sit at a hand, and some hold a knife the labels miss (`MODIFICATION_REPORT_STAGE1_MISSES.md`, "Where the false boxes sit in the label review"). Three questions:
1. Which detector boxes on spots labelled "no knife" are really knives?
2. Which are empty hands that can teach the model what is not a knife?
3. How do the test scores change once the missing knives are labelled?

## Rule (user, 2026-11-21)

Every crop is checked by eye:
- Four voters look at each crop: cheapag and three Haiku agents at high effort.
- If all four agree, the vote stands. Exception: when all say "no knife" but a detector scores the spot ≥ 0.40, Claude looks anyway, because the earlier review missed dark knives exactly that way.
- Any disagreement goes to Claude.
- What Claude is unsure of goes to the human, through `review_candidates.bat`.

Labels: `a` = knife, box fits; `k` = knife, box does not fit; `n` = no knife; `u` = cannot tell.

## What was run

`results/samples/poc/review_stage1_candidates.py`:

```bash
.venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py collect test     # lab test caches, no GPU
DEV=0 .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py collect train
DEV=0 MAXNEG=300 .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py collect neg
.venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py sheets       # 74 sheets of 12 numbered crops
# voters write votes/{haiku1,haiku2,haiku3,cheapag}.json; Claude writes votes/claude.json
.venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py tally
.venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py apply
```

Candidates come from three places:
- **test:** boxes at conf ≥ 0.25 that match no test label, from 8 models (A, C, C1, L, B, H, E, F), merged across models. Lab test set only.
- **train:** the same on the lab training frames, with the augmented teacher F plus two detectors that never saw a lab frame (`v2`, shipped).
- **neg:** new frames halfway between the sampled training frames, from training takes only. A frame is kept when F has a box ≥ 0.25 away from the knife, either because both neighbouring labelled frames show no knife, or because there is one box on the knife and another box elsewhere. A single box away from the old knife position is usually the knife after it moved, so that case alone does not count.

| | Items | Frames |
|---|---|---|
| test | 254 | 243 |
| train | 546 | 521 |
| neg (new frames) | 78 | 74 |

## Results

**Votes.** All four voters agreed on 163 of the 878 items. Claude looked at the other 715, plus the 62 unanimous "knife" votes.

| | Knife (`a`) | No knife (`n`) | To the human |
|---|---|---|---|
| test | 40 | 112 | 102 |
| train | 12 | 452 | 82 |
| neg | 50 (with knife), 11 frames with an empty hand | 14 | 15 |

What the confirmed "no knife" spots are:
- **Test:** mostly a hand resting on a dark chair armrest (about 60 crops), hands on the head, sandals, and the cart.
- **Train:** 125 are a black phone in the Nov 5 phone takes 006–009. The rest are fists and open hands next to the cart, chairs and laptop.

**Unanimous votes can be wrong.**
- On 8 new frames from the phone takes, all four voters called a black phone a knife. Claude overrode them, since those takes have no knife in the hand.
- The other 54 unanimous "knife" votes are real knives.

**New frames rarely show an empty hand.** Every lab training take has a knife in the hand, the "normal" takes included. In 58 of the 74 new frames the box is a knife the person is holding, although both neighbouring labelled frames have no knife label. In most of those neighbours the hand had moved out of the crop, so this alone does not prove those labels wrong (contact sheet `results/stage1_kd/cand_review/neg_neighbours.jpg`). It was not checked further.

**Lab test, with the 40 recovered knives** (`LAB_FIX=results/stage1_kd/cand_review/test_label_fix.json`). NMS 0.5; best recall at P ≥ 0.90 (conf, P):

| Model | Old labels | + 40 knives | P / R at conf 0.25, old | P / R at conf 0.25, + 40 |
|---|---|---|---|---|
| A `labgtL` | 78.3 (0.20, 90.8) | 80.4 (0.10, 90.3) | 91.4 / 77.3 | 93.0 / 76.5 |
| C `labgtaugL` | 79.5 (0.20, 90.3) | 81.6 (0.10, 91.0) | 91.2 / 78.7 | 93.1 / 78.0 |
| C1 `labgtaugS1L` | 82.3 (0.10, 91.2) | 83.8 (0.05, 90.4) | 92.9 / 78.3 | 94.7 / 77.5 |
| L `labgtaugltL` (half blur) | 75.3 (0.40, 90.2) | 80.2 (0.15, 90.6) | 89.3 / 78.4 | 91.6 / 78.1 |
| B `kdgtNL` | 85.0 (0.10, 90.3) | 84.2 (0.10, 92.0) | 92.2 / 82.1 | 94.0 / 81.4 |
| H `kdgtaugFL` | 82.5 (0.15, 90.4) | 82.7 (0.10, 91.3) | 91.3 / 80.7 | 93.0 / 79.9 |
| E `teachergtL` | 81.1 (0.15, 90.0) | 81.9 (0.10, 91.3) | 91.0 / 79.6 | 93.0 / 79.0 |
| F `teachergtaug` | 83.0 (0.20, 90.4) | 84.4 (0.10, 90.2) | 90.9 / 82.6 | 92.6 / 81.7 |

## Findings

- **The lab test set misses knives.** 40 of 254 checked spots hold a knife with no label. The 102 spots still with the human may add more.
  - With the 40 added, precision at conf 0.25 rises about 2 points for every model.
  - The half-blur student L moves from 75.3 % to 80.2 % at P ≥ 0.90. Its extra "false boxes" were partly real knives.
  - The spread between the students shrinks from 75.3–85.0 % to 80.2–84.2 %. The earlier verdict "half blur is worse" does not hold on the fixed labels.
- **Most real false boxes are a hand on a dark object.** The dark chair armrest is the most common in the test set; in training it is the phone. Training already holds these spots as background (452 confirmed), and the models still fire on them.
- **Few new empty-hand frames exist in the lab recordings.** The person always carries the knife. Only 11 new frames add an empty-hand box. Another 49 add a held knife located by eye, often held against the trousers, which is the kind the models miss.
- **Training set built (not trained yet):** `data/stage1_negatives/lab_gt_aug_hn`. It is `lab_gt_aug` plus:
  - 12 added knives on training frames, with the same box on their blur/dark copy;
  - 60 new frames: 49 with a knife, 11 with an empty hand.

  Validation is unchanged, and train.txt has 6,318 lines.

## After the human review (2026-11-21)

The human answered all 199 items in `review_candidates.bat`:
- test: 69 knives (plus 4 drawn boxes), 23 no knife, 6 cannot tell;
- train: 23 knives (plus 16 drawn boxes), 33 no knife, 10 cannot tell;
- new frames: 13 knives, 1 no knife, 1 cannot tell.

The `b` key does nothing on these tiles (they have box `a` only). The 20 "b" answers are boxes the human drew.

`apply` then gives:
- test: **113 knives added** in 111 frames, 135 spots confirmed empty, 6 undecided (cannot tell, left out);
- training set `lab_gt_aug_hn`:
  - 51 knives added on training frames;
  - 72 new frames: 61 with a knife, 11 with an empty hand;
  - train.txt has 6,330 lines.

**Lab test with all 113 added knives.** NMS 0.5; best recall at P ≥ 0.90; P / R / false boxes at conf 0.25:

| Model | Best R at P ≥ 0.90 (conf, P) | At conf 0.25 |
|---|---|---|
| A `labgtL` | 81.8 (0.05, 91.6) | 96.1 / 75.2 / 45 |
| C `labgtaugL` | 82.2 (0.05, 92.5) | 96.5 / 76.9 / 42 |
| C1 `labgtaugS1L` | 83.0 (0.05, 94.2) | 97.8 / 76.2 / 25 |
| L `labgtaugltL` | 82.9 (0.05, 90.9) | 95.1 / 77.2 / 59 |
| B `kdgtNL` | 84.9 (0.05, 93.1) | 97.4 / 80.2 / 32 |
| H `kdgtaugFL` | 83.2 (0.05, 91.8) | 96.1 / 78.5 / 47 |
| E `teachergtL` | 82.4 (0.05, 92.7) | 96.3 / 77.8 / 45 |
| F `teachergtaug` | 85.2 (0.05, 91.5) | 95.9 / 80.5 / 51 |

**What this shows:**
- At conf 0.25 every model is now at 95–98 % precision. The old 90 % rule is reached at the lowest conf tried (0.05) for all of them, so it no longer separates the models.
- All students sit at 81.8–85.2 %, inside the 3-point training noise.

Overall recall mixes knives no one can see with clear ones. `MODIFICATION_REPORT_STAGE1_TARGETS.md` scores them apart.

**Correction (later on 2026-11-21).** 34 of the 113 added test knives are a second box on a knife that was already labelled. A detector box over the hand and the knife overlaps a knife-only label by less than IoU 0.2, so `collect` counted it as an unlabelled spot. They are dropped through `results/stage1_kd/vis_review/miss_check/duplicates.json` (`MODIFICATION_REPORT_STAGE1_TARGETS.md`, "Label errors among the missed clear knives").
- The 51 knives added on training frames and the 61 new knife frames in `lab_gt_aug_hn` were not checked for the same problem.
- `collect` should test overlap with "box holds half the label" as well as IoU before a spot is called unlabelled.

## Not checked

- 6 test and 11 training/new-frame items stay "cannot tell" and are left out.
- Spots no detector boxed are not in the review, so knives that every model misses stay unlabelled.
- Whether the training frames labelled "no knife" next to the new knife frames really hold no knife.
- No model was trained on `lab_gt_aug_hn` yet.
- The test working points are still chosen on the test set.

## Files

- In git:
  - `results/samples/poc/review_stage1_candidates.py`;
  - `review_candidates.bat` (opens `label_review_ui.py --dir results/stage1_kd/cand_review`);
  - `results/stage1_kd/cand_review/{items.json, votes.csv, votes/*.json, test_label_fix.json, voter_prompt.md, claude_notes/}`;
  - `results/stage1_kd/cand_review/neg_neighbours.jpg`.
- Changed:
  - `label_review_ui.py` (`--dir`, reads `items.json`);
  - `sweep_stage1_rules.py` (`LAB_FIX`).
- Dev PC only: the sheets (`sheets/`, `claude_sheets/`, `vote_a_sheets/`, 34 MB), the new frames `data/lab_stage1_hn/images` (17 MB) and the set `data/stage1_negatives/lab_gt_aug_hn`.
