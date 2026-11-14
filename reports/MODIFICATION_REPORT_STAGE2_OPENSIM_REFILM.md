# Modification Report: original dataset fitted to the OpenSim body and re-filmed from all directions (2026-11-13)

**Status: measurements only. No shipped weight or source file was changed.** Skeleton-only tests on the dev PC.

## 1. Question

The lab recordings (Oct 31, Nov 5) differ from the original dataset mainly by camera angle. If the original skeletons are fitted to the OpenSim 3D body and re-filmed from every direction with the lab camera, can models trained on the original dataset alone predict the lab recordings? And is the training code itself correct?

## 2. What was run

Script: `results/samples/poc/poc_rotate3.py` (three-model recipe, 3 seeds).

```bash
python results/samples/poc/poc_rotate3.py <REPO>             # take 002 attack against normal
python results/samples/poc/poc_rotate3.py <REPO> --actions   # same-action benchmark
python results/samples/poc/poc_rotate3.py <REPO> --selftest  # score the training data itself
```

- **Source:** 1,316 windows of the original dataset (attack clips, Validate, Validate OOD and False_Detected normals). Fit to the OpenSim Rajagopal body with `src/skeleton_sim/body_model.py`: median fit error 0.000, 95th percentile 0.002; 1,250 windows kept. The fit is cached in `results/samples/body_model/fit_original_windows.npz`.
- **Three training sets:** original as recorded; re-filmed with the lab camera (tilt 13°) without turning; re-filmed from 8 directions.

| Training set | Normal | Thrust | Stab | Cut-down |
|---|---|---|---|---|
| Original | 125 | 436 | 305 | 450 |
| Re-filmed, no turn | 105 | 427 | 305 | 413 |
| Re-filmed, 8 directions | 840 | 3,416 | 2,440 | 3,304 |

## 3. Results

**Attack against normal on take 002 (Oct 31), segment AUC**

| Training set | AUC |
|---|---|
| Original | 0.413 |
| Re-filmed, no turn | 0.363 |
| Re-filmed, 8 directions | 0.330 |

All three are below chance.

**Same-action benchmark (attacks only: thrust, stab, cut-down on the lab camera)**

| Training set | Nov 5 strikes: right action | Nov 5 strikes: seen as attack | Take 002: right action |
|---|---|---|---|
| Original | 74.4 % | 92.8 % | 45.8 % |
| Re-filmed, no turn | 63.4 % | 92.0 % | 48.6 % |
| Re-filmed, 8 directions | 73.7 % | 79.9 % | 37.5 % |

**Self-test: each model scored on its own training data**

| Training set | F1 (3 seeds) | Attacks caught | Normal alarmed |
|---|---|---|---|
| Original | 1.000, 1.000, 1.000 | 1,191 of 1,191 | 0 of 125 |
| Re-filmed, no turn | 1.000, 1.000, 1.000 | 1,144–1,145 of 1,145 | 0 of 105 |
| Re-filmed, 8 directions | 0.993, 0.987, 0.991 | 8,927–9,030 of 9,160 | 0–1 of 840 |
| Lab (Nov 5 training takes) | 0.995, 0.995, 0.997 | 632–634 of 638 | 0 of 163 |

## 4. Conclusion

- The training code is not bugged: every model reproduces its own training data with F1 above 0.98.
- Attack recognition transfers to the lab camera: the original models see 93 % of Nov 5 strike windows as attacks and name the right action on 74 %.
- Re-filming from 8 directions does not improve this, and does not help attack against normal on take 002.
- The gap is normal against attack on the lab camera. The original dataset has only 125 normal windows and none are a person standing or walking with a knife in the lab.

## 5. Not checked

- Only yaw was turned; camera height and distance were kept at the lab values.
- Take 002 action labels were mapped to the three original classes by name; the 45.8 % on take 002 includes actions the original dataset does not have.
