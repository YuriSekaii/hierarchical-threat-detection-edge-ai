# Modification Report: Stage 2 trained on relabelled lab recordings plus every dataset (2026-11-12 to 2026-11-13)

**Status: measurements only. No shipped weight or source file was changed.** Skeleton-only tests on the dev PC, not the full pipeline.

## 1. Question

Oct 31 and Nov 5 already contain a lot of normal movement. If all of it is cut out and relabelled (attack and normal) and used for training, does Stage 2 separate attack from normal on the lab camera? If not, does adding every other dataset and the simulation help?

## 2. What was run

Script: `results/samples/poc/poc_relabel3.py` (three-model recipe: ST-GCN, PoseConv3D, distilled PoseConv3D; attack-type loss plus binary loss plus triplet loss; 700 steps, batch 48; RMD reference built from normal windows only).

```bash
python results/samples/poc/poc_relabel3.py <REPO> [--extra] [--sim N] [--cache exp_S_gen|exp_FH_all|exp_H_all] [--seeds 1,2,3] [--violent label1,label2]
```

- **Test:** take 002 (Oct 31), 55 labelled segments (31 attack, 24 normal), 5 folds by segment. Training windows that overlap a held-out segment are dropped. Results are means over 3 seeds.
- **Base:** the same models trained on Nov 5 takes only, without take 002.
- **Bound:** set from attacks only (10 % quantile of the held-out Nov 5 strike scores).
- **`--extra`:** adds `data/Validate` clips (636 attack, 108 normal windows). `data/Ground_Truth_Dataset` is single frames and cannot be used for motion.
- **`--sim N`:** adds N simulated attack and N simulated normal windows plus the original clips from the named cache. `exp_FH_all` is snapped to the OpenSim body by `results/samples/tools/body_filter.py`; `exp_H_all` is the same data without snapping.
- An earlier version with a small CNN (`poc_relabel.py`) gave AUC 0.824 against 0.825 without relabelling, and 0.743 with Validate added. It is superseded by the three-model runs below.

## 3. Results

Segment-level, take 002 left-out segments. "Attacks at ≤2 FA" is the number of attack segments caught when the threshold allows at most 2 normal segments to alarm.

| Training data | AUC, output blend | AUC, RMD blend | Attacks at ≤2 FA (output / RMD) |
|---|---|---|---|
| Nov 5 only (base) | 0.833 | not kept | not kept |
| + relabelled take 002 | 0.837 | not kept | 14.0 of 31 |
| + Validate + original clips + 3,000 + 3,000 sim + 2D generated (`exp_S_gen`) | 0.849 | 0.854 | 16.3 / 16.0 of 31 |
| + Validate + original clips + 3,000 + 3,000 OpenSim-snapped sim (`exp_FH_all`) | 0.828 | 0.857 | 17.0 / 17.3 of 31 |
| same, unsnapped (`exp_H_all`) | 0.836 | 0.845 | 18.7 / 14.3 of 31 |

At the attack-only bound, relabel only: 18.3 of 31 attacks caught, 3.7 of 24 normal segments alarmed.

- Relabelling take 002 changes the AUC by 0.004. Adding everything else adds about 0.02.
- Snapping the simulation to the OpenSim body gives +0.02 to +0.03 AUC when take 002 is not in training, and no difference when it is.
- Normal movements that still alarm in every run: stretch arms, hand to head while bending, walk-turn, pick up from floor.
- Attacks that are still missed: slow overhand wind-ups.

## 4. Stretch and low knife swing counted as violent

Decision by the project owner (2026-11-13): a stretch with a weapon in hand should be flagged, because it looks like a swing. Run: `--extra --sim 3000 --cache exp_FH_all --violent stretch_arms,knife_swing_low` (those take 002 labels become attack, class swing).

| | Stretch = normal | Stretch = violent |
|---|---|---|
| Attack / normal segments | 31 / 24 | 34 / 21 |
| AUC, RMD blend | 0.857 | 0.840 |
| AUC, output blend | 0.828 | 0.822 |
| Attacks at ≤2 FA (RMD) | 17.3 of 31 | 20.7 of 34 |
| Attacks caught at the bound (RMD) | not kept | 30.0 of 34 |
| Normal segments alarmed at the bound (RMD) | not kept | 10.3 of 21 |

- `stretch_arms` is flagged every time and leaves the error list.
- The gain equals the three moved segments; the separation itself does not improve.
- Normal labels still alarmed (count over 3 seeds, output blend): walk_turn 9, hand_to_head_bend 6, walk_knife_low 4, pick_up_from_floor 3, hand_to_face 3, arms_out 3, inspect_knife 2, knife_point_down 2.
- Attack labels still missed: overhand_stab_slow_windup 3, overhand_big_windup 3, knife_swing_low 3, overhand_stab_bending 1.

## 5. Conclusion

More labels from the same recordings and more simulated data do not close the gap: the AUC stays between 0.82 and 0.86. The remaining errors are walking and turning with the knife, hand-to-head movements and slow wind-ups.

## 6. Not checked

- The choice of `stretch_arms` and `knife_swing_low` as the violent labels was made by Claude; `arms_out` stayed normal.
- Take 002 is used both to train (other folds) and to test, so these numbers are optimistic for a new recording.
- Nothing was run through Stage 1, the gates or pose estimation.

## 7. Retrained on the cut clips of all three days, scored on the Nov 14 test takes (2026-11-14 to 2026-11-15)

**Status: measurements only. No shipped weight was changed.** Skeleton-only, dev PC, not the full pipeline.

### Question

The earlier runs used raw Nov 5 takes with swapped stab and cut-down labels and had no recorded normal activity. With the clips of `MODIFICATION_REPORT_LAB_SEGMENTS.md` (Oct 31, Nov 5, Nov 14), does the same three-model recipe give good results on takes it never saw?

### What was run

Script: `results/samples/poc/poc_segments.py` (same models, losses, 700 steps and batch 48 as `poc_relabel3.py`). `scripts/train_stage2_sim.py` got `lab_segments()`, which reads the `segments.csv` of every day.

```bash
DEV=cuda:1 .venv/Scripts/python.exe results/samples/poc/poc_segments.py . --train nov5|clips|all [--seeds 1,2,3] [--save]
```

| Role | Takes |
|---|---|
| Training, `nov5` | Nov 5 clips (labels fixed, clips with unreliable timing dropped) + Oct 31 take 001 |
| Training, `clips` | the above + Nov 14 training takes (first repeat of each normal take, attack takes 001, 037–041) |
| Training, `all` | the above + `data/Validate` clips + 3,000 + 3,000 simulated windows and the original clips of `exp_FH_all` |
| Calibration, sets the alarm bound, never trained on | Nov 5 takes 003, 008, 010, 017, 020, 023, 028, 031; second repeat of each Nov 14 normal take |
| Test, never trained on | Nov 14 `test_*` takes 025–036 (60 strikes, 6.1 min normal); Oct 31 take 002 without clip 043 (30 attacks, 2.3 min normal); Nov 14 object takes 048–057 (25 strikes with a phone, pen, wallet or remote) |

- Whole takes are replayed like the live pipeline (50-frame windows, stride 10). A strike is caught if a window that overlaps its clip by 15 frames or more alarms. False alarms are counted as events (alarmed normal windows less than 5 s apart count once).
- Windows within 1 s of a strike clip are neither trained as normal nor scored: they hold the wind-up or the recovery.
- Shipped models on the same windows: the live cascade with its own thresholds, and the three `sim3` models with their shipped bound.
- "Bound q99" is the 99 % quantile of the calibration normal windows; "strict bound" is above every calibration normal window. Retrained rows are means of seeds 1–3.

### Results on the Nov 14 test takes (60 strikes)

"Outside pauses" counts only take 036 (`test_normal_mixed`, 4.3 min), because the pauses inside attack takes were cut automatically.

| Models | Window AUC (all / outside pauses) | Caught, false-alarm events (outside pauses) | Best caught at ≤ 2 events, threshold tuned on the test itself |
|---|---|---|---|
| Live cascade, shipped thresholds | 0.50 / 0.50 | 4, 8 (6) | 0 |
| `sim3`, shipped bound | 0.765 / 0.819 | 31, 20 (6) | 0 |
| `sim3`, bound q99 | same | 19, 8 (3) | same |
| `nov5`, output blend, bound q99 | 0.611 / 0.579 | 1.3, 4.0 (3.3) | 0 |
| `clips`, output blend, bound q99 | 0.761 / 0.796 | 37.3, 14.3 (3.3) | 4.7 |
| `clips`, RMD blend, bound q99 | 0.764 / 0.811 | 24.0, 11.7 (3.0) | 3.3 |
| **`all`, output blend, bound q99** | **0.813 / 0.839** | **40.0, 14.7 (5.3)** | 8.0 |
| `all`, output blend, strict bound | same | 23.0, 8.3 (3.3) | same |
| `all`, RMD blend, bound q99 | 0.808 / 0.832 | 33.3, 13.3 (5.3) | 9.3 |

### Results on Oct 31 take 002 (30 attacks) and the object takes (25 strikes)

| Models | Take 002 AUC | Take 002 caught, events | Object takes AUC (all / outside pauses) | Object strikes alarmed, events |
|---|---|---|---|---|
| Live cascade, shipped thresholds | 0.50 | 1, 2 | 0.49 / 0.49 | 2, 5 |
| `sim3`, shipped bound | 0.880 | 15, 0 | 0.756 / 0.826 | 11, 12 |
| `nov5`, output blend, bound q99 | 0.874 | 0.3, 1.7 | 0.886 / 0.936 | 1.7, 0 |
| `clips`, output blend, bound q99 | 0.876 | 18.7, 5.0 | 0.926 / 0.976 | 22.7, 8.3 |
| `all`, output blend, bound q99 | 0.905 | 23.3, 3.7 | 0.898 / 0.970 | 23.0, 8.7 |
| `all`, RMD blend, bound q99 | 0.896 | 20.3, 3.0 | 0.874 / 0.950 | 20.0, 11.0 |

### Findings

- **The complete dataset helps, but the result is not good.** Best retrained set: 40 of 60 test strikes caught with about 15 false-alarm events in 6 minutes, against 31 with 20 for `sim3`. Ranking rises from 0.765 to 0.813. No setting catches most strikes at a low false-alarm rate: with at most 2 events, 8 of 60.
- **The Nov 14 normal takes are what helps.** Without them (`nov5`) the model ranks the Nov 14 test near chance (0.61); with them 0.76; with the other datasets and simulation 0.81.
- **On take 002 `sim3` is still the only model that catches attacks with no false alarm** (15 of 30). The retrained models catch more (23) with 3–4 events, and 4 at zero events.
- **Stage 2 alarms on strikes with a phone, pen, wallet or remote as on knife strikes** (23 of 25), as expected from a skeleton. Rejecting those is Stage 1's job.
- **What the highest-scoring "normal" windows of the test show** (frames checked by eye for six of them): take 036 at 261–270 s is overhead arm stretching and wide arm swings, which the project owner decided on 2026-11-13 should be flagged; take 036 at 165 s is bending to pick the knife up from the floor; take 032 at 14 s is an overhand strike seen from behind that has no attack clip. So part of the counted false alarms are label gaps, and bend-and-pick-up stays a real one.
- Weakest strike types at the strict bound: walking thrust (0.7 of 8 caught) and overhand stab (1.7 of 4).

### Not checked

- Take 036 has one label for 5 minutes of mixed activity (mostly sitting in a swivel chair, which no training take shows). Stretches inside it are counted as normal here. It was not relabelled.
- Pauses inside attack takes may hold strikes the cutter missed or rejected; only six windows were looked at.
- One person, one room. Test takes were recorded on the same day as part of the training takes.
- The calibration takes are held out of training, so the models saw only one repeat of each Nov 14 normal activity. Training on both repeats was not tried.
- Nothing ran through Stage 1, the gates, pose estimation or TensorRT. The cascade's five thresholds were not fitted; one blended score and one bound were used.
- The Nov 5 `thrust_low` clips done with a phone in hand are trained as attacks (the motion is a strike).

### Files

- `results/samples/poc/poc_segments.py`; `scripts/train_stage2_sim.py` (`lab_segments`, `LAB_DAYS`).
- Raw results and logs: `results/stage2_segments/{nov5,clips,all}.json` and `.log`; per-window scores of seed 1: `scores_all_*.npz`. Seed-1 models of `all`: `results/train_sim/exp_segments_all/` (dev PC only).
