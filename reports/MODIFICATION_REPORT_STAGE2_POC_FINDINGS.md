# Modification Report: Stage 2 proof of concept, what was measured (2026-11-12)

**Status: measurements only. No source file, weight or model was changed.** All experiments ran on skeletons on the dev PC, none through the full pipeline (Stage 1, gates, pose). The scripts are in `results/samples/poc/` (section 9).

## 1. Question

Can the current Stage 2 recognise a knife attack at all, before collecting more data? The earlier reports show 1 of 31 attacks on lab take 002 (original models), 14 of 31 for sim3, and 0 % on Validate Thrust and Stab for the simulated sets. This session checked what the original 1.0000 score covered, then tried to get the same recipe working on the lab camera with real recordings only.

## 2. What the 1.0000 score covered

`scripts/test_modular_validation_suite.py` reproduces F1 1.0000 (77 videos, 33 attack, 44 normal). The attack side is only Cut-Down: `ActionDataset` takes the first sub-folder it finds in `With_Coat` and `Without_Coat` (`src/dataset.py:108`). The 39 Thrust and 32 Stab clips are never loaded.

All 104 attack clips and 44 OOD clips of `data/Validate`, original models, caught clips:

| | Suite-style (clip max, dataset heatmaps) | Live-style (50-frame windows, stride 10, per-window cascade) |
|---|---|---|
| Thrust | 23 of 39 | 16 of 39 |
| Stab | 28 of 32 | 26 of 32 |
| Cut-Down | 33 of 33 | 29 of 33 |
| OOD false alarms | 0 of 44 | 0 of 44 (5 clips are shorter than 50 frames and have no window) |

- The thresholds of the shipped cascade were chosen among 19,359 combinations on those same 77 clips (`results/tuning/champion_multitier_staircase_v5.10.json`), so a perfect score on them was expected.
- A veto chain loses attacks: a typical missed thrust has p1 0.98, p2 0.02, p3 0.98, and tier 2 discards it. Without the vetoes, the mean of the three tier scores catches 79 of 104 at zero false alarms (AUC 0.962, 39 normal clips with windows).
- The Validate OOD clips are mostly 360x640 internet clips, the attacks are 2560x1600 lab clips. The test never asked a model to separate normal from attack on the same camera and person.
- The live code path is not broken: frame rate, body proportions and hand speed match between Validate and the lab recordings.

## 3. Original models on the lab camera

Take 002 (Oct 31): the shipped cascade catches 1 of 31. Per model, window ranking of attack against normal windows is at chance (AUC T1 0.55, T2 0.74, T3 0.49, mean 0.64). On the Nov 5 attack takes against Oct 31 normal windows the AUC is 0.54-0.66 per action, and by facing direction 0.38-0.81. The models trained on the Validate camera do not transfer.

## 4. Real lab takes, plain model

Setup unless stated: train on the Nov 5 training takes (strikes = attack, standing pauses = normal) plus lab take 001 and the live normal clips; test on the take 002 segments (31 attack, 24 normal); alarm bound set from attacks only (keeps 90 % of the strikes of the 8 held-out Nov 5 takes), so nothing is tuned on take 002. Means over 3-5 seeds.

| Step | Result |
|---|---|
| One ST-GCN, own output, bound 0.5 | 30 of 31 attacks, 77 of 78 held-out strikes, but 16 of 24 normal segments alarmed |
| Same, attack-only bound | 21.7 of 31 attacks, 6.0 of 24 normal segments, AUC 0.80 |
| Score by distance to the nearest trained attack only ("reject what is not trained") | AUC 0.33, worse than chance: normal walking is closer to the attack examples than the take 002 attacks are |
| Attack split by type and facing direction (16 classes) | no gain |
| Drop the walking-thrust takes | AUC 0.794 to 0.835 (most of their windows are walking) |

## 5. Labels

A window used to count as attack if it overlapped a strike by 15 frames, but the forward hand movement lasts a median of 13 frames (90th percentile 40) in a 50-frame window.

| Labelling | Attack windows | Attacks of 31 | Normal segments of 24 | AUC |
|---|---|---|---|---|
| Old overlap rule | 1,177 | 21.7 | 6.7 | 0.787 |
| Whole forward movement inside the window | 638 | 24.7 | 9.0 | 0.799 |
| Same, plus windows with no hand trip as normal | 638 | 27.3 | 10.0 | 0.816 |

Precise labels help slightly. They also lower the bound, so more normal movement passes.

## 6. What did not help

| Idea | Result |
|---|---|
| Stage 1 knife box as extra input (centre, offset to the nearer wrist, size, seen flag), 5 seeds | AUC 0.820 body only, 0.835 with knife, 0.825 arm + knife; normal segments alarmed 9.6, 9.4, 7.6. Within seed noise: the knife moves with the wrist in strikes and in stretches alike. Boxes for the Nov 5 takes are in `results/samples/knife/` |
| Oct 31 normal segments as training negatives, one left out each time | False alarms 9.7 to 4.0 of 24, attacks 26.7 to 19.7 of 31 (the model sees that recording only as "normal"). Stretch arms, walk-turn and inspect knife still alarm |
| Shorter window, small temporal CNN, precise labels | 50 frames AUC 0.792 (27.2 attacks, 9.4 normal); 30 frames 0.742 (29.6, 16.4); 30 frames stride 5 0.740 (30.0, 18.4); 20 frames 0.685 (31.0, 21.2) |

The take 002 normal activity is 284 windows of walking, 39 of stretching and 16 of everything else (hand to head while bending 4, pick up from floor 3, hand to face 2, hand to chest 2, one each of five more).

## 7. The project's own recipe on lab takes

Three models as in `scripts/train_stage2_sim.py` (ST-GCN, PoseConv3D, PoseConv3D distilled by relational KD; attack-type cross-entropy, attack/normal BCE, batch-hard triplet; same augmentation), RMD reference built per model, 3 seeds, precise labels. Nothing saved to `weights/`.

**RMD reference background (found in the code).** `rmd_reference` (`scripts/train_stage2_sim.py:391`) builds the background Gaussian from all training windows. With attack-heavy lab data (72 % attacks) the background is nearly the attack set, the two distances cancel, and RMD ranks take 002 below chance (AUC 0.25-0.37 for every model and combination). Building the background from the not-attack windows only gives 0.45 to 0.76 on one ST-GCN, equal to its plain output (0.77).

With the normal-only background, window 50, 3 seeds:

| Score | AUC | Attacks of 31 | Normal segments of 24 |
|---|---|---|---|
| ST-GCN, output | 0.785 | 24.7 | 9.0 |
| Three blended, output | 0.840 | 24.7 | 6.7 |
| ST-GCN, RMD | 0.741 | 27.3 | 11.3 |
| PoseConv3D, RMD | 0.760 | 27.0 | 10.7 |
| Distilled, RMD | 0.776 | 26.7 | 9.7 |
| Three blended, RMD | 0.767 | 27.3 | 10.3 |
| All three must agree, RMD | 0.782 | 27.7 | 9.3 |

A repeat run of the 50-frame models (separate training) gave 0.819 for the blended output and 0.782 for the blended RMD; run-to-run spread is about 0.02.

Window 30 against 50, three seeds, bound from attacks only, then the best single threshold tuned on take 002 itself (an upper limit, not an expected result):

| Window | Score | AUC | Attacks / normal segments | Caught at 0 / <=2 / <=4 false-alarm segments (tuned) |
|---|---|---|---|---|
| 50 | Three blended, output | 0.819 | 22.3 / 7.0 | 2.3 / 13.7 / 18.7 |
| 50 | Three blended, RMD | 0.782 | 26.7 / 9.0 | 0.0 / 11.0 / 11.7 |
| 30 | Three blended, output | 0.756 | 30.7 / 15.0 | 0.0 / 8.0 / 13.7 |
| 30 | Three blended, RMD | 0.767 | 31.0 / 13.3 | 2.0 / 6.3 / 13.3 |

At 30 frames the models catch everything and alarm on more than half of the normal segments, as the earlier work found. RMD removes about two of those false alarms. With a threshold tuned on take 002 itself, at most about 14 of 31 attacks are caught at two false-alarm segments. For reference sim3 catches 14-15 of 31 with almost no false alarms.

## 8. What this shows

- The method and the live path work. A model trained on real lab takes ranks held-out lab strikes well (77 of 78 caught, AUC about 0.95 in earlier reports) and gets 22-31 of 31 take 002 attacks depending on the bound.
- Every variant tried (labels, classes, knife input, window length, three models, RMD, tuned thresholds) leaves the same movements alarmed: stretch arms (every run), hand to head while bending, walk-turn, and often pick up from floor, hand to face, arms out. They move the hand away from its rest position as a strike does, and the skeleton does not tell them apart.
- RMD is not better than the plain output on this data. It measures unfamiliarity, and the take 002 attack variants (lunges, crouching thrusts, backhand slash) are further from the trained attacks than its walking is.
- The data gap is concrete: 638 attack windows against no repeated non-attack arm movements in the same session. Plain walking is not a problem.

## 9. Files and what is missing

Scripts, in `results/samples/poc/`: `poc_validate.py` (section 2), `poc_diag.py`, `poc_auc.py`, `poc_take002.py` and `poc_nov5.py` (section 3), `poc_inputs.py`, `poc_same_camera.py`, `poc_ood.py`, `poc_ood2.py`, `poc_trim.py`, `extract_knife.py`, `poc_knife.py`, `poc_oct31neg.py`, `poc_window.py`, `poc_three.py`, `poc_three_bg.py`, `poc_rmdcheck.py`, `poc_three_win.py`. They take the repo path as their first argument.

Not done or not trusted:

- Take 002 has been used for choices before, and one person is in training and test, so all numbers are optimistic. A new recording is still needed for an unbiased number.
- The 3-5 seeds and 24 normal segments mean one or two segments of difference is noise.
- Sections 4-6 use small models (ST-GCN or a three-layer temporal CNN), not the full cascade; the cascade's five thresholds cannot be set without normal examples. Section 7 uses the project's three models with a single best threshold, not the 19,359-combination search.
- The strike finder is automatic, its labels were not checked by eye.
- The knife is seen in 65-87 % of frames only, and Stage 1 gives a box, not the blade direction.
- Nothing ran through Stage 1, the gates or the TensorRT engines.

Next step: record the missing `train_normal` takes (repeated stretching, reaching, hand to head and face, picking up, pointing, inspecting the knife, turning) in the same session as a few attack takes, keep one recording untouched for testing, then repeat sections 7 and 2 with them. Public skeleton data as a substitute is untested.
