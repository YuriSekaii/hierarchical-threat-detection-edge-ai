# Modification Report: Skeleton Simulator for Stage 2 Training Data

## 1. Problem

The Stage 2 action models were trained on 118 real clips (43 Thrust, 28 Stab, 47 Cut-Down) recorded with one camera: high (about 3 m), close, looking steeply down, the person mostly seen from behind or at a diagonal. Training uses no augmentation. On 2026-10-31 the live demo thrusts scored 0.000.

Tests with the current cascade (PyTorch weights, best 50-frame window per clip, as in the live pipeline):

| Data | Clips | Alarm |
|---|---|---|
| Real training clips | 118 | 99 % |
| Real attacks recorded in the lab with the live camera (2026-10-31 segments) | 23 | **0 %** |
| The same training clips with small changes (arm a bit lower or bent, lean, stance, a slightly different view, speed, mirror) | 118 | **21-47 %** |

The models only know the motions and the camera of the training set. A test with simulated people makes this clearer: seen from a high camera behind the person, simulated attacks **and simulated walking** reach T1 of about 0.5-0.6; seen from the front, both drop to about 0. The T1 score depends more on the camera angle than on the action.

Recording enough real data for every attack, speed, body and camera is not practical, so the missing variety is simulated.

## 2. What was built

`src/skeleton_sim/` (no downloads, no new dependencies):

| File | What it does |
|---|---|
| `body.py` | 3D COCO-17 body in metres. Random adult (1.52-1.92 m, limb ratios +-3.5 %). Pose from high-level controls: pelvis position and facing, crouch, trunk lean/side lean/twist, head turn, wrist targets, elbow swivel, foot positions. Arms and legs are solved with two-bone IK (bone lengths never change); the shoulder shrugs and protracts with the hand; the pelvis drops if a foot would be out of reach. |
| `motion.py` | Keyframe timeline. Cubic Hermite curves between keys: "stop" keys (start and end of a strike, holds) and "pass" keys (the middle of a slash arc). Feet lift automatically when they move. Slow random sway (weight shift, breathing, head, hands) so no two takes are identical. |
| `actions.py` | Attack library. Each attack has the phases of a real one (get ready, wind up, fast strike, follow-through or hold, recover) and draws its own random height, direction, reach, speed, lean, stance and step. Left or right hand (25 % left). The free hand hangs, guards, grabs or counter-swings. Normal activities for negatives. |
| `camera.py` | Pinhole camera at any azimuth (0-360 deg around the person), height 0.9-3.6 m, distance 2.5-9 m, field of view 55-95 deg, off-centre framing and small roll; rendered into the 1920x1440 frame the pipeline analyses. YOLO-pose sensor model: jitter per joint measured on the lab takes (about 1 % of torso length, wrists 1.4-1.6 %), slow drift and larger error on joints hidden behind the body, the far side of the face hidden, rare one-joint glitches. |
| `scene.py` | One take = random person + random lead-in (stand still, walk in 1-4 steps, one step, ready stance) + attack + random ending (hold, bring the feet together, step back), cut like the training clips (0.25-1 s before and after the attack) so every 50-frame window contains the attack. Cameras that cut the person off or make them tiny are rejected. |
| `augment2d.py` | Perturbs real 2D clips: the knife arm's path swung lower/higher about the shoulder and/or less extended (the edit grows with arm extension, so the resting arm stays put, elbow re-solved with 2D IK), upper-body lean, stance width, limb and shoulder/hip proportions, view squeeze/squash/shear/roll, uneven speed change, mirror (other hand), jitter. |
| `export.py` | Writes CVAT XML identical in layout to `data/<class>/annotations_yolo26s-pose/*.xml`; contact sheets and animated preview grids. |

Attack types (`--per-action` clips each):

| Folder | Actions |
|---|---|
| `Sim_Thrust` | `thrust_train` (straight arm at shoulder height, like the dataset), `thrust_low` (hip/belly), `thrust_bent` (short bent-arm jab), `thrust_mid`, `walking_thrust` (thrust while walking in, the live demo style); 1-4 repeats, optional lunge |
| `Sim_Stab` | `overhand_stab` (ice-pick, 1-5 repeats, optional deep bend), `upper_stab` (underhand rising), `frenzy` (grab with the free hand, 3-7 fast short stabs) |
| `Sim_Cut-Down` | `cut_down` (wind up above/behind the head, chop through the front, 1-3 repeats), `ground_strike` (bent over / crouched, striking at the floor) |
| `Sim_Slash` | single slash, forehand or backhand, high/mid/low/diagonal down/diagonal up |
| `Sim_Multi-Slash` | 2-5 alternating slashes |
| `Sim_Swing` | big swings with full body turn and a step |
| `Sim_Combo` | 2-4 different attacks chained without a pause; `turn_attack` (turns 70-180 deg toward the target while winding up, then strikes) |
| `Sim_OOD_Normal` | idle, walking (with turns), turning around, pointing with the knife, inspecting the knife, hand to head, arms out, picking up from the floor, talking gestures. "OOD" in the folder name gives label 0 in `ActionDataset`. |

`Aug_Thrust`, `Aug_Stab`, `Aug_Cut-Down`: `--real-aug` perturbed copies of every real training clip.

The lab segments are **not** used for generation; they stay held out as the real test of whatever is trained next.

## 3. Realism checks

The simulator was tuned against the real clips with the motion statistics in `scripts/validate_simulation.py` (torso units after the pipeline's own preprocessing). The first version struck 1.5-2x too fast and raised the knife too high; after tuning (default speed 0.72, wind-ups mostly to head height, shorter steps):

Medians per group (150 clips sampled per simulated folder):

| Group | Reach | Reach range | Wrist speed / frame | Highest wrist (- = above shoulder) | Stance | Noise |
|---|---|---|---|---|---|---|
| Real Thrust | 1.07 | 0.67 | 0.16 | +0.12 | 0.70 | 0.002 |
| Real Stab | 1.04 | 0.75 | 0.20 | -0.33 | 0.77 | 0.002 |
| Real Cut-Down | 1.18 | 0.98 | 0.26 | -0.37 | 0.94 | 0.004 |
| Lab attacks (live camera) | 1.12 | 0.96 | 0.26 | -0.32 | 1.05 | 0.005 |
| Lab normal | 1.02 | 0.69 | 0.13 | +0.10 | 0.77 | 0.003 |
| Sim_Thrust | 1.08 | 0.57 | 0.11 | +0.39 | 0.83 | 0.003 |
| Sim_Stab | 1.08 | 0.70 | 0.15 | +0.17 | 0.74 | 0.003 |
| Sim_Cut-Down | 1.10 | 0.91 | 0.26 | -0.60 | 0.81 | 0.003 |
| Sim_Slash | 1.08 | 0.85 | 0.24 | -0.02 | 0.72 | 0.002 |
| Sim_Multi-Slash | 1.08 | 0.90 | 0.29 | -0.31 | 0.74 | 0.003 |
| Sim_Swing | 1.07 | 0.90 | 0.27 | -0.19 | 0.88 | 0.003 |
| Sim_Combo | 1.12 | 0.95 | 0.25 | -0.65 | 0.84 | 0.003 |
| Sim_OOD_Normal | 1.10 | 0.63 | 0.05 | +0.16 | 0.59 | 0.002 |

Simulated attacks fall in the range of the real ones. Sim_Thrust is wider on purpose (low and bent-arm thrusts, which the real set lacks). Normal activity is about 5x slower at the wrist than attacks, as in the lab takes.

## 4. Dataset generated on the dev PC

`simulate_dataset.bat` defaults (200 clips per attack type, 150 per normal activity, 10 copies per real clip), seed 2026, 73 s on 18 CPU workers, 518 MB in `data/Simulated/` (gitignored, dev PC only):

| Folder | Clips |
|---|---|
| Sim_Thrust | 1000 |
| Sim_Stab | 600 |
| Sim_Cut-Down | 400 |
| Sim_Slash | 200 |
| Sim_Multi-Slash | 200 |
| Sim_Swing | 200 |
| Sim_Combo | 400 |
| Sim_OOD_Normal (label 0) | 1350 |
| Aug_Thrust / Aug_Stab / Aug_Cut-Down | 430 / 280 / 470 |
| **Total** | **5530** |

Simulated clips: 75 % right hand, 25 % left; camera height 0.9-3.6 m (median 2.2), azimuth all the way round; clip length median 113 frames (3.8 s). `ActionDataset` / `PoseConv3DHeatmapDataset` read the folders unchanged (checked: labels 1 for attack folders, 0 for `Sim_OOD_Normal`).

## 5. Current cascade on the generated data (baseline before retraining)

| Group | Clips | Alarm | T1 median | T1 >= 0.20 |
|---|---|---|---|---|
| Real training Thrust / Stab / Cut-Down | 43 / 28 / 47 | 100 / 100 / 98 % | 0.99-1.00 | 98-100 % |
| **Lab attacks (live camera)** | 23 | **0 %** | 0.002 | 4 % |
| Lab normal | 25 | 4 % | 0.090 | 40 % |
| Aug_Thrust / Aug_Stab / Aug_Cut-Down | 150 each | 35 / 34 / 48 % | 0.17 / 0.21 / 0.53 | 48-65 % |
| Sim_Thrust | 150 | 6 % | 0.089 | 39 % |
| Sim_Stab | 150 | 13 % | 0.171 | 49 % |
| Sim_Cut-Down | 150 | 22 % | 0.200 | 50 % |
| Sim_Slash / Multi-Slash / Swing | 150 each | 6 / 10 / 5 % | 0.07-0.12 | 41-45 % |
| Sim_Combo | 150 | 12 % | 0.160 | 46 % |
| **Sim_OOD_Normal** | 142 | **11 %** | 0.116 | 45 % |

Simulated attacks by camera position: back 15-17 %, side 11-13 %, front 2-4 % alarm.

The current models alarm on simulated normal activity about as often as on simulated attacks, and they almost never alarm on attacks seen from the front. They have learned "this camera, these 118 motions", not "a knife attack". This is the baseline the retrained models have to beat on the lab attacks while keeping normal activity low.

## 6. How to use

- One click: `simulate_dataset.bat` (writes `data\Simulated\`, then runs the check). It refuses to overwrite an existing dataset; use `--out` for a new one.
- Look at `data\Simulated\previews\*.jpg` / `*.mp4` before training.
- Train with the real folders plus the simulated ones, e.g. `ActionDataset(root, ["data/Thrust", "data/Stab", "data/Cut-Down", "data/Simulated/Sim_Thrust", ..., "data/Simulated/Sim_OOD_Normal"])`. Keep `data/lab_recordings/2026-10-31/segments` and the live demo clips out of training; they are the test.
- After retraining, run `python scripts/validate_simulation.py` again: the lab attacks should go up from 0 % and lab/simulated normals should stay low.

## 7. Limits

- The motion is keyframed, not motion-captured. Speeds, reach, wrist height and stance match the real clips in the statistics above, but fine detail (hip and shoulder timing, follow-through) is simplified. Real recordings stay the test.
- One person, no victim, no occlusion by other people or furniture.
- `augment2d` view changes are 2D approximations (good for +-30 deg); large view changes come from the 3D simulator only.
- Normal activities are simple; if the retrained T1 alarms on real walking, add more (running, carrying, phone use) the same way.
