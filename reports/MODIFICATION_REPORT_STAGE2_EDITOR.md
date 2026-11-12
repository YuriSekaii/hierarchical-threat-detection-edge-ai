# Modification Report: Multi-Point Motion Editor

**Status: the editor works as motion simulation, but it does not yet improve detection.** No model trained with it beats `sim3` at a low false-alarm rate, so nothing new is deployed and the full pipeline was not run. `sim3` stays opt-in and the live models stay the default.

## 1. What was built

`src/skeleton_sim/motion_edit.py` re-performs a recorded motion through chosen points. There are no prebuilt models, no downloads and no hand-written poses.

- **Control:** any number of points on any joint at any frame (`follow_keys`), or one landing point for a strike (`reperform`).
- **How the body follows:** among the recorded poses closest to the current one, it learns how each joint moved when the controlled joints moved. It walks to the target in small steps and re-learns this at each step.
- **The one hand-written rule:** an arm bone cannot stretch beyond the person's own length.
- **Timing of a strike edit:** the change follows how far the hand has come along its own trip, so a wind-up adds nothing and a hold stays at the target.

Related methods in the literature, recalled and not checked here: motion warping (Witkin and Popovic 1995) and locally learned pose models (Chai and Hodgins 2005).

## 2. Does it look like real motion?

Yes for strikes, and partly for gestures. Errors are in torso lengths. Lower is better.

### 2.1 One real thrust built from another, given only the real landing point

Six pairs, chest to low and low to chest, seen from the front, back and left. The poses it learns from contain neither the target action nor held-out takes. "Real vs real" compares two real strikes of the target action, which is about the best anyone can do. It is available for 5 of the 6 pairs, because take 7 has only one usable strike.

| Method | Hand | Elbow | Shoulder | Free arm | Head | Legs |
|---|---|---|---|---|---|---|
| No edit | 0.546 | 0.362 | 0.131 | 0.191 | 0.167 | 0.231 |
| Plain geometry (bones only) | 0.287 | 0.439 | 0.131 | 0.191 | 0.167 | 0.231 |
| Same coupling for all poses | 0.287 | 0.238 | 0.124 | 0.210 | 0.220 | 0.256 |
| **Coupling learned near each pose** | **0.287** | **0.230** | **0.110** | 0.191 | **0.151** | 0.236 |
| Real vs real | 0.314 | 0.222 | 0.088 | 0.154 | 0.120 | 0.155 |

The elbow is almost as close as two real strikes. Plain geometry is worse than no edit. Free arm and legs barely change: no point controls them.

### 2.2 Real take 002 gestures rebuilt from a real pause, with points taken from the gesture

Mean of nine activities. Take 002 is only measured against, never learned from. Because the points come from the real gesture, this shows how many points a gesture needs, not that new gestures can be invented accurately.

| Points | Hand | Elbow | Shoulders | Free arm | Head | Legs |
|---|---|---|---|---|---|---|
| No edit | 0.572 | 0.355 | 0.204 | 0.344 | 0.293 | 0.253 |
| Hand, 1 point | 0.543 | 0.401 | 0.188 | 0.357 | 0.289 | 0.262 |
| Hand, 4 points | 0.174 | 0.236 | 0.189 | 0.344 | 0.290 | 0.257 |
| Hand + shoulders, 4 points | 0.174 | 0.200 | 0.054 | 0.292 | 0.243 | 0.245 |
| Hand + shoulders, 8 points | 0.101 | 0.184 | 0.042 | 0.289 | 0.239 | 0.242 |

One point is not enough for a gesture, and it makes the elbow worse. Four points on the hand and shoulders capture most of it.

## 3. Does it help detection?

Means of seeds 1-3 on saved skeletons. The first table moves the bound until take 002 gives at most N false-alarm events in 2.4 minutes, which only compares rules. Each cell is caught of 31 on take 002 / held-out strikes (78) / low thrust strikes (29).

| Models | N=0 | N=1 | N=2 |
|---|---|---|---|
| **`sim3`** | **15.3 / 31 % / 28 %** | 15.3 / 31 % / 28 % | 17.3 / 45 % / 36 % |
| `sim3` + corrected strikes, no editing | 15.7 / 30 % / 25 % | 15.7 / 30 % / 25 % | 17.7 / 36 % / 30 % |
| `sim3` + edited attacks | 16.0 / 33 % / 32 % | 16.0 / 33 % / 32 % | 17.7 / 37 % / 33 % |
| `sim3` + edited + gestures from pauses | 12.3 / 22 % / 20 % | 14.3 / 27 % / 24 % | 16.0 / 33 % / 31 % |
| `sim3` + edited + half gestures from strikes | 13.7 / 26 % / 26 % | 15.7 / 30 % / 36 % | 17.3 / 37 % / 39 % |

With the bound above every held-out normal window:

| Models | Take 002 AUC | Caught of 31 | Events/min | Validate Thrust / Stab / Cut | Held-out strikes |
|---|---|---|---|---|---|
| **`sim3`** | 0.871 | **15.3** | 0.14 | 2 / 2 / 29 % | 26 % |
| `sim3` + corrected strikes | 0.893 | 13.7 | 0.28 | 2 / 1 / 47 % | 24 % |
| `sim3` + edited attacks | 0.891 | 10.3 | **0** | 3 / 1 / 35 % | 19 % |
| `sim3` + edited + gestures from pauses | 0.883 | 12.7 | 0.14 | 5 / 1 / 42 % | 21 % |
| `sim3` + edited + half gestures from strikes | 0.880 | 12.3 | 0.28 | 7 / 1 / 51 % | 26 % |

The 0.7 gain at N=0 for `sim3` + edited attacks is within seed noise, and at the strict bound it catches 5 fewer.

### 3.1 The missing action

All recorded low thrusts were removed from the samples, then real low thrusts were scored. The older `real` and `sim` training data still contains low thrusts, so the action is missing only from the new samples.

| Low thrust strikes caught | N=0 | N=1 | N=2 | Strict bound | AUC |
|---|---|---|---|---|---|
| No editing | 17 % | 17 % | 26 % | 40 % | 0.884 |
| Generated low thrusts | 31 % | 33 % | 37 % | 38 % | 0.837 |

**A weak sign of help, not proof.** Generated data catches more at low false-alarm budgets, but seeds are noisy: 24, 24 and 45 % at N=0 with editing, and 17, 0 and 34 % without. Ranking is worse with editing, and the strict bound is about equal.

### 3.2 False alarms by activity

Share of windows at or above the median attack window of take 002, on 5-12 windows per gesture.

| Activity | `sim3` | Edited attacks only | + gestures from pauses | + half from strikes | `sim3` + half from strikes |
|---|---|---|---|---|---|
| Hand to face | 50 % | 67 % | 72 % | 56 % | 44 % |
| Hand to head, bend | 44 % | 67 % | 69 % | 64 % | 50 % |
| Inspect knife | 13 % | 80 % | 80 % | 80 % | 33 % |
| Pick up from floor | 57 % | 10 % | 0 % | 14 % | 29 % |
| Walk, knife low | 2 % | 3 % | 8 % | 8 % | 1 % |

**Generated gestures did not reduce false alarms on real gestures.** Pick-up does fall, but it falls without gestures too (19 % with no editing), so the likely cause is the real walking and desk windows now used as normals.

## 4. Why the gestures do not help

A model that never saw gestures scored new windows (median z-score, and share at or above the median attack window of take 002):

| Windows | Wrist speed | Hand travel | z | Share |
|---|---|---|---|---|
| Real hand to face | 0.133 | 1.10 | 1.5 | 67 % |
| Real hand to head, bend | 0.272 | 1.86 | 1.3 | 67 % |
| Real inspect knife | 0.205 | 1.40 | 1.4 | 80 % |
| Generated from a pause, 8 gestures | 0.08-0.18 | 0.8-1.7 | 0.0-0.8 | 5-14 % |
| Real held-out strikes | 0.179 | 1.69 | 1.1 | 37 % |
| Generated edited attacks | 0.179 | 1.61 | 1.7 | 91 % |

- **Real gestures move as fast as strikes.** The first generated gestures assumed they were slow, at 0.06-0.12. Correcting that helped only a little.
- **Generated gestures from a pause move too little of the body:** other wrist 0.5-0.9 against 0.6-1.0 for real, and feet stance change 0.4-0.8 against 1.1-1.4. Building them on real strikes fixes the arms but not the feet.
- **Take 002 differs from the Nov 5 takes.** Stance change is 1.0 for its attacks and 0.4 for held-out Nov 5 strikes. Whether this is stepping, camera or tracking was not checked. Part of the gap may come from the session, not the simulator.
- **Edited attacks are too easy for the model.** It scores 91 % of them at or above the median attack, against 37 % of real held-out strikes. They come from takes it trained on, so the comparison is not fair, but the model may be learning the source takes more than the attack.

The share uses take 002 only as a reference point, and the real rows hold 5-12 windows.

## 5. Mistakes found and fixed

| Mistake | Effect | Fix |
|---|---|---|
| The old strike finder used hand speed only | It picked the hand coming back as the landing, merged strikes, and counted desk visits and walking in take 9. **`exp_S_raw`, `exp_S_gen` and `heldout.npz` from the earlier report use these labels** | `find_strikes` follows each trip away from the hand's rest. Checked by eye on thrust takes only |
| Rest taken as the longest-held position | An arm held out counted as rest | Lowest of the positions held for long |
| Editing followed the clock | The hand fell early during a hold and rose early in a wind-up | Follows the hand's own progress |
| **My labelling bug:** windows between strikes counted as normal | 550 windows, mostly real attacks (hand travel 1.31 against 1.49), taught the model that attacks are normal | Only pauses and walks outside the first and last hand trip are normal. The invalid scores are kept in `results/samples/edit/score_edit_BUGGY_normals.txt` |
| **My filter bug:** the side-view wind-up filter also hit overhand strikes | Take 11 kept 0 of 13 | Thrusts only |
| Slow gestures | See section 4 | Timing overlaps strikes, fitted once on take 002 |

I also suspected YOLO was swapping left and right arms in side views. It was not: 0 label jumps across all takes. In the frames checked from take 7, the knife hand pulls back near the shoulder while the free arm reaches forward, which looks like a wind-up.

## 6. What did not work or got worse

- **No editor recipe beats `sim3`;** only the body filter in section 9 does. Adding gestures lowers the number caught at low false-alarm budgets (12.3-13.7 against 15.3 at N=0).
- **Inspect knife false alarms get worse** when blending in gestures: 33 % against 13 % for `sim3`. Hand to head stays about the same.
- **The network's own output still fails at a low false-alarm rate:** 0-2 of 31 at N=0, although gestures were meant to help it.
- **Corrected labels alone do not help:** alone at N=0 they catch 6.7, and at the strict bound 7.3. The earlier `S_raw` caught 12.0 at the strict bound, on a different batch mix.
- **Pick-up bends stay mild,** because the takes hold few deep bends to learn from.
- Known live thrusts are still not caught by `sim3` (0 of 8).
- Take 9 gives no usable thrust and take 7 only one, so low thrust from the right is untested and from the left barely tested.

## 7. How far to trust the numbers

- **One person, one room,** and only two sessions.
- **Take 002 is less clean than before.** It set the N budgets, guided the gesture timing and the choice to build gestures on strikes. Held-out takes also guided decisions.
- Gesture targets and timing are my own choices, not learned.
- 12 % of the walk-in windows counted as normal still have hand travel as large as a strike. They were not checked one by one.
- Strike labels were checked by eye on thrust takes only.
- Differences of 1-3 caught attacks are within seed noise.
- The first clean score still needs the `test_*` takes.
- The open crash from the earlier report (`0xC0000409`) is unchanged, and the full pipeline was not run here.

## 8. In the repo and on the dev PC only

| Item | Where |
|---|---|
| Editor and tests | `src/skeleton_sim/motion_edit.py`, `tests/test_motion_edit.py` (run directly with `PYTHONPATH=.`) |
| One action from another test | `scripts/evaluate_motion_edit.py` |
| Cache builder with the recipes | `scripts/generate_from_edits.py` |
| Scoring, drawings, logs, dev PC only | `results/samples/tools/` (`score_edit.py`, `gap.py`, `motion_stats.py`, `keys_test.py`, `run_edit.py`), `results/samples/edit/` |
| Caches and weights, dev PC only | `results/train_sim/exp_E_*`, `results/train_sim/experiment_weights/*_E_*` |

## 9. 3D body constraint

The OpenSim Rajagopal 2016 full-body model gives bone sizes, joint limits and bone masses. Only its body is used, with no recorded motions. Its license was not found, so the file stays untracked in `results/samples/body_model/`. Code: `src/skeleton_sim/body_model.py`, tests `tests/test_body_model.py`.

- **Fit:** joint angles go through a sigmoid that keeps them inside the model's limits. Bone scale is fitted to the person. The camera has perspective, its tilt is fixed at the measured 13°, and the centre of mass must stay over the feet.
- **Fits the training takes** to 0.02-0.04 torso lengths, about YOLO's own jitter.
- **Building a missing view:** errors are in torso lengths, and lower is better.

| Method | Overall | Wrist | Ankle |
|---|---|---|---|
| Today's mix of two views | 0.276 | 0.415 | 0.379 |
| Body fitted to two views | 0.274 | 0.444 | 0.347 |
| Body from one view only | 0.338 | 0.606 | 0.378 |

  From the front alone, the fit puts every thrust behind the body, because one view cannot tell towards from away.
- **Non-body poses:** this is the share of windows that fit worse than 95 % of real Nov 5 windows. Real data from Oct 31 gives 22 %, edited attacks 26 %, gestures 42 %, and the old simulator inside `sim3` 48-50 %.

**Body filter:** every simulated window in `sim3`'s three caches becomes its body fit, plus the fit miss of a random real training window (`results/samples/tools/body_filter.py`). The three recipes were retrained with seeds 1-3. `sim3` averages 3 models, so it is also shown with two seeds per recipe (6 models) for a fair size comparison.

| Models | Strict: caught of 31 (sd) | Strict: held-out / low | N=0: caught / held-out / low | N=0: other live alarms |
|---|---|---|---|---|
| `sim3`, 3 models | 15.3 (1.2) | 26 % / 24 % | 15.3 / 31 % / 28 % | 1.0 |
| `sim3`, 6 models | 16.3 (0.9) | 31 % / 24 % | 17.7 / 43 % / 34 % | 3.0 |
| **`sim3` + body-filtered, 6 models** | **16.7** (3.4) | **38 % / 30 %** | **20.0 / 53 % / 37 %** | 4.0 |
| Body-filtered alone, 3 models | 16.0 (2.9) | 38 % / 26 % | 8.7 / 16 % / 13 % | 1.0 |

- **Gains over an equal number of models:** about 5 more of 78 held-out strikes and 2 more of 29 low thrusts at the strict bound. At N=0 it catches 2.3 more on take 002. It is the first change in this report that beats `sim3` on held-out takes.
- **Each recipe alone gains at the strict bound** (old 11.7 to 13.3, fixed 8.3 to 9.3, mix 8.3 to 8.7). At N=0 each one alone gets worse (old 9.7 to 6.3, fixed 6.3 to 0).
- **What got worse:**
  - Seeds vary more, with an sd of 3.4 against 0.9.
  - Body-filtered models alone give more false-alarm events on take 002 (0.41 against 0.14 per minute).
  - Activities that rose above the median attack: knife point down 7 to 53 %, knife swing low 7 to 40 %, hand to chest 11 to 28 %. Hand to face (50 to 11 %) and inspect knife (13 to 0 %) fell.
- **Not checked:** the full pipeline, `test_*` takes, and pairing residuals by view. The body has no neck. Held-out takes guided earlier choices.

**Reject filter instead of snapping (idea 3):** drop each simulated window that fits the body worse than 95 % of the cache's real windows (`results/samples/tools/reject_filter.py`). It drops 7.6 % of the fixed simulator's windows and 18 % of the old one's. It is worse than snapping and worse than `sim3` alone:

| Models | Strict: caught of 31 | Strict: held-out / low | N=0: caught / held-out / low |
|---|---|---|---|
| `sim3` rejected | 10.3 | 27 % / 23 % | 9.0 / 20 % / 15 % |
| `sim3` + rejected | 13.7 | 35 % / 29 % | 13.7 / 33 % / 26 % |
| body-filtered + rejected | 16.3 | 43 % / 35 % | 16.3 / 43 % / 28 % |

Out-of-distribution validation clips above the bound rose from 10 to 18 % for `sim3` rejected. Snapping a window keeps its motion, while dropping it loses the motion together with the bad pose.

**Body editor (idea 1, `src/skeleton_sim/body_edit.py`):**
- **How it works:** it re-performs a strike inside the body. The landing pose changes as little as the person's own nearby recorded poses allow, and every pose stays inside the joint limits. Several landings are solved at once.
- **Rebuilding one action from another** (the pairs in `scripts/evaluate_motion_edit.py`; error on the joints other than the hand, mean of elbow, shoulder, free arm, head and legs):

| Editor | Hand | Not hand |
|---|---|---|
| No edit | 0.546 | 0.216 |
| 2D editor (local) | 0.287 | 0.184 |
| Body editor, everything may move | 0.283 | 0.205 |
| Body editor, striking arm and trunk | 0.280 | 0.203 |
| Body editor, striking arm only | 0.295 | 0.193 |
| Two real strikes of the target action (floor) | 0.314 | 0.148 |

  The body editor lands the hand as well as the 2D editor, but it moves the head and shoulder more. Its gain is that every pose is a body pose and can be turned.
- **Turning a one-view fit gives bodies that do not exist:** from the front, how far a person leans is not seen. Front and back takes, turned to the side, lean 12-22° (median), and 10 % of their frames lean more than 55°. Takes filmed from the side lean 4-11°. A pose prior learned from our own takes (lean from side takes, sideways bend from front takes; `body_model.pose_prior`) helps the legs but not the arms:

| Prior weight | Fit of its own view | Turned to another recorded view |
|---|---|---|
| none | 0.048 | 0.338 |
| 1e-4 | 0.054 | 0.320 |
| 1e-3 | 0.066 | 0.313 |

  The wrists stay near 0.62, because how far the knife goes towards the camera is the action itself. So turned strikes come from bodies fitted to two views instead (`results/samples/tools/body_clips.py`): 188 of 216 training strikes and 22 pauses. Each is fitted to its own take and to a take of the same action filmed 90° away. The second take is another performance, so the fit to the own view is looser (0.110). Unturned, the full miss is added back; turned, only its fast part (the tracker's jitter).

**Self-training with snapping (idea 2, `results/samples/tools/self_train.py`):** each round re-performs 512 strikes of training takes to new landing points, snaps them into the body, adds them to the poses the editor learns from, and rebuilds the pairs above. It stops when the not-hand error falls by less than 0.5 %.

| Loop | Round 0 | Round 1 | Round 2 |
|---|---|---|---|
| 2D editor, samples fitted into the body | 0.184 | 0.185, stop | |
| Body editor | 0.203 | 0.196 | 0.196, stop |
| Control: same strike frames added unedited | 0.203 | 0.195 | |

The body loop's gain comes from having more strike poses to learn from, not from the generated samples: the control gains as much. The loop adds no information, because every sample is made from the same takes.

**Detection with body strikes:** `--attacks body` in `scripts/generate_from_edits.py` adds 3,000 strikes per class re-performed in the body. Each strike gets a new landing point and only the striking arm and trunk may change. 70 % are seen from a random direction. There are also 12,000 two-view pauses turned the same way, so a turned view carries no label. `B_body` uses the usual simulated base, `B_bodyF` the body-filtered one. Seeds 1-3 for each.

| Models | Strict: caught of 31 (sd) | Strict: held-out / low | N=0: caught / held-out / low / other alarms |
|---|---|---|---|
| `sim3` | 15.3 (1.2) | 26 % / 24 % | 15.3 / 31 % / 28 % / 1.0 |
| Body strikes alone | 2.0 (1.6) | 12 % / 15 % | 2.3 / 11 % / 14 % / 0.3 |
| `sim3` + body strikes | 14.7 (0.5) | 27 % / 25 % | 15.3 / 31 % / 30 % / 1.3 |
| `sim3` + body-filtered, 6 models | 16.7 (3.4) | 38 % / 30 % | 20.0 / 53 % / 37 % / 4.0 |
| `sim3` + body-filtered + body strikes, 7 models | 17.0 (2.4) | 35 % / 29 % | 20.7 / 59 % / 40 % / 6.0 |

- **Body strikes alone fail on take 002** (2 of 31), like every other recipe built only from the Nov 5 takes. Their ranking (AUC 0.878-0.883) is higher than `sim3`'s 0.871.
- **Added to `sim3`, they change nothing.**
- **Added to `sim3` + body-filtered,** the N=0 numbers rise a little (+0.7 caught, +6 % held-out, +3 % low), but other live alarms rise from 4 to 6 and held-out at the strict bound falls from 38 to 35 %. That is one extra model and within seed noise. It is not a clear gain.
- **About 5 % of turned windows reach past the image edge** by up to 140 px, which real windows never do.

**Why body strikes lose to snapping:**
- **The body strikes look real.** They are the training windows nearest to take 002's real attacks (0.156 against 0.196 for old-simulator attacks of the same library size).
- **But the normals that came with them were 22 still pauses.** Hand speed: body strikes 0.218 torso lengths per frame, body pauses 0.021. So the cheapest rule was "a real-looking hand moving fast is an attack". Real normals in take 002 move as fast (stretch arms 0.219, inspect knife 0.206), and the body models flag them: inspect knife 13 → 67 %, arms out 13 → 53 %.
- **On the Nov 5 takes alone the body models look best** (AUC 0.98 of held-out attacks against held-out normals). The only normals there are still pauses, so that number hides the problem.
- **Snapping avoids this.** It adds no new attacks and snaps attacks and normals alike. The old simulator's normals already move their arms (arm circles, waving, reaching).

**Fix tried: normals as fast as the strikes.** `--body-gestures 12000`: the same strike clips, the same solve and the same turns, with the hand sent to the person's own face, head, chest, shoulder, point, overhead or inspect position instead. Hand speed matches the strikes (median 0.233 against 0.220).

| Models | Strict: caught of 31 (sd) | Strict: held-out / low | Validation clips above bound: thrust / stab / cut / OOD | N=0: caught / held-out / low / other |
|---|---|---|---|---|
| Body strikes | 2.0 (1.6) | 12 % / 15 % | 7 / 9 / 29 / 4 | 2.3 / 11 % / 14 % / 0.3 |
| Body strikes + gestures | 3.7 (0.9) | 15 % / 15 % | 22 / 31 / 71 / 6 | 4.3 / 17 % / 13 % / 0.7 |
| `sim3` + body gestures | 12.0 (0.8) | 18 % / 18 % | 3 / 4 / 49 / 8 | 12.7 / 23 % / 22 % / 1.3 |
| `sim3` + body-filtered + body gestures | 16.7 (2.6) | 37 % / 31 % | 1 / 0 / 35 / 10 | 20.3 / 50 % / 36 % / 4.3 |
| `sim3` + body-filtered (no body data) | 16.7 (3.4) | 38 % / 30 % | 2 / 0 / 30 / 10 | 20.0 / 53 % / 37 % / 4.0 |

- **What it fixed:**
  - Arm activities: inspect knife 67 → 40 %, arms out 53 → 13 %, stretch arms 17 → 8 %.
  - Validation attack clips above the bound tripled (thrust 7 → 22, stab 9 → 31, cut 29 → 71).
- **What got worse:** knife point down 0 → 60 % and walk turn 7 → 12 %. Added to `sim3`, body gestures make it worse (12.0 against 15.3 caught).
- **What it did not fix:** take 002 attacks (3.7 of 31) and held-out strikes stay far below `sim3`. Added to `sim3` + body-filtered, it changes nothing.
- **What is left:** every body window comes from 21 takes of one person in one session. The normals of other sessions (live Oct 31 clips, lab take 001) are unlike anything in the body data, and the detector's score (distance to the training classes) rises for unfamiliar windows. Those normals then set a high bound. The old simulator's random people, cameras and actions cover them; 21 takes cannot. This matches the earlier finding that every recipe built only from the Nov 5 takes fails on take 002.

**Where the four ideas stand:** snapping (idea 4, the body filter) is the only one that beats `sim3`. Rejecting (idea 3) is worse. The body editor (idea 1) gives real body poses at any angle but no detection gain. Self-training (idea 2) gains nothing beyond what weighting strike frames more would give.

## 10. Next steps

1. Record `train_normal` when back at the Edge PC. The editor can then learn gesture points from real gestures, instead of my choices, and generate many variations of them. This is its most promising use, and it stays with little data.
2. Score on the `test_*` takes only, through the full pipeline. That is the first clean number.
3. Check why take 002 differs from the Nov 5 takes, since it decides how much of the gap in section 4 is the simulator.
4. Test the edited attacks on sources held out from training, to see whether the model is learning the source takes.
