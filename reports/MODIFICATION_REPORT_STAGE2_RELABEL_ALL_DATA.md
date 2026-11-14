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
