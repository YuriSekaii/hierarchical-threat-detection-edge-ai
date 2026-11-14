# Modification Report: lab recordings cut and relabelled into clips (2026-11-14)

**Status: data preparation only. No model was trained and no shipped weight was changed.**

## 1. Question

The takes recorded on Oct 31, Nov 5 and Nov 14 did not follow the recording guide exactly: extra strikes, walking in and out, wrong labels, empty takes. Which version did the Stage 2 work use, raw takes or cut clips, and are the cut clips correct? Where no cut version existed, cut and relabel the takes so they can be used for training, testing and simulation.

## 2. What the Stage 2 work used

| Day | Used by Stage 2 work | Cut version before today | Result |
|---|---|---|---|
| Oct 31 | `segments/` (59 hand-made clips) | yes | Checked with 8-frame samples per clip. Labels are right, except clip 043: it is a kick labelled `attack`. About 11 attack clips overlap their neighbours in time. Left unchanged (hand-made). |
| Nov 5 | raw takes, trimmed only in code (60 frames off each end; first 30 s when frames dropped) | no | Cut now, see below |
| Nov 14 | not used yet | no | Cut now, see below |

## 3. What was run

Script: `scripts/cut_segments.py` (new). It finds strikes from the knife-hand wrist speed (`hand_speed`, `strike_events` in `src/skeleton_sim/from_samples.py`), makes a 1.7 s attack clip around each one (30 frames before the speed peak, 21 after), drops strikes within 45 frames (1.5 s) of walking in or out, and cuts the rest of each take into `normal` clips. Normal takes become `walk` (first and last 2 s) plus pieces of at most 10 s. Labels and rejected strikes come from a hand review in `<day>/segments/review.csv`; the strikes were checked on strip images of 6 frames around each detected strike.

```bash
.venv/Scripts/python scripts/extract_take_skeletons.py ...        # skeletons, YOLO26s-pose 384x640
.venv/Scripts/python scripts/cut_segments.py data/lab_recordings/2026-11-05 --kp results/samples/kp
.venv/Scripts/python scripts/cut_segments.py data/lab_recordings/2026-11-14 --kp results/samples/kp_1114
.venv/Scripts/python scripts/cut_segments.py --selftest
```

Output layout is the same as Oct 31: `<day>/segments/NNN_<category>_<label>.avi` (MJPG, original resolution), a matching `.npz` (`kpts`, `take`, `start_frame`, `end_frame`, `fps`, `crop`) and `segments.csv`.

## 4. Results

| Day | Clips | attack | confuser | normal |
|---|---|---|---|---|
| Nov 5 | 526 | 268 | 0 | 258 |
| Nov 14 | 530 | 98 | 46 | 386 |

Corrections written in the review files:

- **Nov 5:** 010–013 relabelled `cut_down_*` and 014–017 relabelled `overhand_stab_*` (the known label swap). 001 (calibration) and 026 (false start) dropped. 006–009 and 028 noted: thrust_low done with a phone in hand, not the knife. 16 strikes rejected as walking in or out, standing still or out of view. Clips that end after 30 s in takes with dropped frames carry a note that their timing is unreliable.
- **Nov 14:** 004 dropped (camera recorded no frames). Test takes 025–035 hold several strikes each, not one; each strike is its own clip. 5 strikes rejected as walking or hand-to-head. 027 has no person from about 20 s to 103 s. 036 (`test_normal_mixed`) is mostly sitting in a swivel chair from about 25 s to 143 s.

## 5. What was not checked

- Strike strips of 034 (slash), 001, 037–041 and the confuser `*_strike` takes on Nov 14 were not checked strike by strike, and the second strip sheet of several Nov 5 takes was not either. A few `attack`/`confuser` clips may be walking.
- Nov 14 normal takes 037–057 were checked only through the strip images, not 1 fps contact sheets.
- The clip npz files have no `knife` key (Oct 31 has knife boxes); knife detection was not run.
- `scripts/train_stage2_sim.py` still reads only Oct 31 (`LAB` is hardcoded). Using Nov 5 and Nov 14 clips in training needs a change there.
- Test takes (Nov 14 `test_*`) and Oct 31 take 002 must stay out of training.

## 6. Files

- `scripts/cut_segments.py`, `scripts/extract_take_skeletons.py` (fixed a crash on an empty take)
- `data/lab_recordings/2026-11-05/segments/{review.csv,segments.csv}` and `2026-11-14/segments/{review.csv,segments.csv}` are in git; clips and npz files are on the dev PC only.
- Skeletons: `results/samples/kp/` (Nov 5), `results/samples/kp_1114/` (Nov 14), dev PC only.
