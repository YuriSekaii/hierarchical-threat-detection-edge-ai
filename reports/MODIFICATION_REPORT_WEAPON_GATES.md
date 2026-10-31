# Modification Report: Weapon Plausibility Gates (Static Filter + Hand Gate)

## 1. Why

Too many Stage 1 detections reached Stage 2. Two kinds of false alarm are cheap to reject before the action cascade runs:

- an object that sits in the same place scan after scan (on a table, on the floor, a picture), and
- a weapon that is nowhere near the threat actor's hand.

## 2. What changed

| File | Change |
|---|---|
| `src/weapon_gate.py` (new) | `WeaponMotionGate` (static filter) and `weapon_in_hand` (hand gate), shared by production and debug |
| `src/inference_production_pipeline.py` | Stage 1 loop calls `_gate_weapon_detection`; Stage 2 loop calls `_check_weapon_in_hand` before the cascade; new constructor args + CLI flags |
| `src/debug_stage2_tracer.py` | Same gates in the instrumented loops; new events `S1_GATED`, `HAND_CHECK`, `CLIP_SKIPPED`; counts in `summary.json` |

### 2.1 Static filter (Stage 1, before arming)

Each Stage 1 detection is compared with the previous one (within 1.5 s). The move is measured as the box-centre shift divided by the box diagonal.

| Verdict | Rule | Effect |
|---|---|---|
| `moving` | move >= 0.15 | arms / extends Stage 2 |
| `still` | move < 0.15, not yet static | extends an open window only |
| `first` | no detection in the last 1.5 s | extends an open window only (does not arm) |
| `static` | the last 3 detections all within 0.15 of each other | ignored: no arming, no extension |

Arming now needs a second detection that has moved (about 0.3 to 0.7 s later). No motion history is lost, because Stage 2 analyzes the 50 frames before the arming moment.

### 2.2 Hand gate (Stage 2, per clip)

For every non-static Stage 1 detection inside the clip, the distance from the actor's nearest wrist to the weapon box (0 if the wrist is inside the box) is divided by the actor's torso length.

| Verdict | Rule | Effect |
|---|---|---|
| `in_hand` | closest distance <= 0.5 torso | cascade runs |
| `far` | measured, but never <= 0.5 torso | clip dropped |
| `no_person` | weapon seen only on frames with no person | clip dropped |
| `unknown` | no detection in the clip, or wrists not visible | cascade runs (benefit of the doubt: fast attacks blur the knife) |

### 2.3 Flags

| Flag | Default |
|---|---|
| `--no-static-filter` | filter on |
| `--static-min-move` | 0.15 box diagonals |
| `--no-hand-gate` | gate on |
| `--hand-max-dist` | 0.5 torso lengths |

## 3. Evidence used for the thresholds (Edge PC sessions)

- Session 2026-10-31: every detection of the held knife was 0.01 to 0.21 torso from a wrist. A threshold of 0.5 leaves more than 2x margin.
- The held knife moved 0.03 to 3.1 box diagonals between scans. It moved less than 0.15 only while the person stood still.
- The f0 detection (conf 0.46, nobody in frame) used to arm Stage 2. It is now an unconfirmed single sighting and does not arm.

## 4. Verification

| Test | Before | After |
|---|---|---|
| Replay of 10/31 detections | armed at f0 (false) and f900 | armed at f920 only |
| Replay of 10/22 detections | armed at f453 | armed at f463 |
| 10/31 clips through the hand gate | - | 49 in hand, 1 dropped (no person), 4 unknown |
| End-to-end, 4 held-out attack videos (`--debug`, PyTorch) | - | armed twice, all 7 clips in hand, 5 alarms |
| End-to-end, same videos, production path | - | 10 alarms |
| End-to-end, one knife frame frozen for 6 s | would arm | 24 detections gated as static, never armed, 0 alarms |

**Not yet covered end to end:** the `far` path (a weapon seen while a person stands away from it) is only covered by a synthetic unit check. A live test with a knife on a table and a person walking past would confirm it.

## 5. Known trade-off

A person who holds a knife completely still for about 1 s stops extending the Stage 2 window. Stage 2 re-arms as soon as the knife moves again, and it analyzes the 50 frames before re-arming.
