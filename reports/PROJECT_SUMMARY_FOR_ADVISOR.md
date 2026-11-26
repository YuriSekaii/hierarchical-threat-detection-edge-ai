# Knife-Attack Detection on Edge GPUs: Project Summary (Jan to Nov 2026)

This summary is built from all 52 `.md` reports and the 100 or so ground-truth `.txt` logs in `D:\Intern AI Project`. The timeline comes from file modification dates. The repository has 109 commits.

**Goal:** raise an alarm when a person attacks with a knife, in real time, on an edge GPU.

**System:**
- **Stage 1** finds the knife (YOLO26s).
- **Stage 2a** tracks the person and extracts the pose: 17 keypoints over 50 frames.
- **Stage 2b** decides whether that motion is an attack.

The project ran in four phases, and each phase grew out of the one before it:
1. Build a detector that works (January to September).
2. Make it fast (September).
3. Run it on the real camera (October).
4. Find out why Stage 2 failed there (November).

---

## 1. Timeline

| Date (2026) | Phase | What happened |
|---|---|---|
| Jan 27 to Feb 2 | Stage 1 distillation | YOLO26x teacher distilled into a YOLO26s student. 23 attempts at adding a P2 head all failed. |
| Sep 4 to 8 | Stage 2 baseline | ST-GCN with k-NN vs Mahalanobis; the k-NN version was chosen. |
| Sep 16 to 17 | Phase 1: Stage 2 search | 14 versions (v1.00 to v3.10): normalization, tracking, 4 out-of-distribution (OOD) gates, 5 backbones. |
| Sep 16 to 18 | Phase 2: Stage 1 search | Literature review, then 5 alternatives to YOLO tested (v4.10 to v4.50). |
| Sep 17 to 18 | Phase 2: Stage 2 optimisation | PoseConv3D scaling and distillation, super-ensembles, Staircase Cascade v5.10. |
| Sep 19 to 21 | Engineering | Modular pipeline v5.20, TensorRT v6.00, 384x640 engines, cross-platform launchers. |
| Oct 8 to 22 | First run on the real camera | Camera and UI fixes. A debug tracer showed that Stage 2 was blind on this camera. |
| Oct 31 to Nov 5 | Live fixes and simulated data | Weapon gates, catch-up mode, tracker fix, skeleton simulator, model sets `sim` and `sim3`, 29 new lab recordings. |
| Nov 7 to 12 | Root cause | Motion editor, then measurements of what Stage 2 can really do. |

---

## 2. Stage 1: Knife Detector (Jan to Sep)

**Test set:** 715 images containing 240 knives (359 with a coat, 235 without, 121 with no weapon). A detection counts when it overlaps the knife box by IoU 0.20.

### 2.1 Distillation (Jan)

A small student model copied a large teacher. The student was **better as well as faster**:

| Model | F1 | Precision | Recall | False alarms | Missed knives | Latency |
|---|---|---|---|---|---|---|
| YOLO26x teacher | 73.7% | 72.8% | 74.6% | 67 | 61 | 15.3 ms |
| **YOLO26s distilled** | **86.9%** | **89.8%** | **84.2%** | **23** | **38** | **11.0 ms** |

Then the team tried to make the student even better with a P2 head (a higher-resolution output layer). **23 attempts failed:**
- F1 ranged from 0.05 to 0.81.
- The root cause was a `reg_max` 1 vs 16 mismatch between the transplanted weights.
- The best attempt (attempt 21) reached F1 0.809 at confidence 0.20. It never beat the plain student at confidence 0.45 (0.869).

**Effect of the confidence threshold on the plain student:**

| Confidence | 0.45 | 0.25 | 0.20 | 0.10 |
|---|---|---|---|---|
| F1 | 0.869 | 0.818 | 0.789 | 0.704 |
| Recall | 84% | 91% | 92% | 93% |
| False alarms | 23 | 75 | 98 | 172 |

### 2.2 Architecture search (Sep)

The next question was whether a stronger model could also reject fists and clothing seams that look like blades. A literature review listed four candidates and estimated their benefit. Each was then measured on the same 715 images:

| Version | Method | F1 | False alarms | Latency (RTX 3090) | Result |
|---|---|---|---|---|---|
| v4.00 | YOLO26s distilled | **86.9%** | **23** | **12.9 ms** | **kept** |
| v4.10 | Sa2VA-7B language model + SAM-2 | 35.3% (only about 75 images tested) | 5 | 1,911 ms, 16 GB of GPU memory | rejected |
| v4.20 | Grounding DINO | 30.5% | 154 | 186 ms | rejected |
| v4.30 | + hand-contact check (DINOv2) | 84.1% | 44 (false alarms on no-weapon images -84%) | 33.9 ms | optional guard |
| v4.40 | + depth filter | 86.1%, best recall 89.2% | 43 | 113 ms | rejected (too slow) |
| v4.50 | Greedy NMS instead of NMS-free | 82.3% | 53 (+130%) | 12.5 ms | NMS-free confirmed |

What these tests showed:
- The review predicted that Sa2VA would suppress 97 to 99% of false alarms. In practice it was **148 times slower** and missed 74% of knives.
- The literature estimates did not match reality, so **measuring every candidate on our own data was necessary.**
- Bugs fixed along the way:
  - Sa2VA silently loaded none of its 1,632 weight tensors and ran on random weights.
  - The word "cutter" in the Grounding DINO prompt caused 219 false alarms.

**Stage 1 was settled at 86.9% F1.** The work then moved to Stage 2.

---

## 3. Stage 2: Action Recognition (Sep)

**Data:**
- Training: 297 clips (Cut-Down 129, Stab 57, Thrust 111).
- Validation: 77 videos split into 148 clips (33 attacks, 44 normal).

### 3.1 Phase 1: 14 versions

Each version fixed the main weakness of the one before it:

| Version | Change | F1 | Missed /33 | False alarms /44 | Why it was the next step |
|---|---|---|---|---|---|
| v1.00 | ST-GCN + k-NN | 0.794 | 8 | 17 | buffer needs 1,055 MB, too big for the edge |
| v1.01 | JPEG frame buffer | 0.794 | 8 | 17 | memory down to 47 MB (-95.5%) |
| v1.02 | Torso-scale normalization | 0.765 | **2** | 17 | far-away people were scaled wrongly |
| v1.03 | 384x640 input | 0.765 | 2 | 17 | 40% fewer pixels, 127.6 ms saved on CPU |
| v1.04 | ByteTrack | 0.765 | 2 | 17 | ID swaps 15 → 0 |
| v1.05 | Cosine k-NN gate | 0.774 | 9 | 5 | the following versions try to cut false alarms |
| v1.06 | ViM gate | 0.805 | 2 | 13 | |
| v1.07 | ASH-B + energy gate | 0.789 | 5 | 10 | |
| **v1.08** | **Relative Mahalanobis (RMD) gate** | **0.917** | **0** | 6 | best single model |
| v2.00 | CTR-GCN | 0.901 | 1 | 6 | 1.33M parameters (-56%) |
| v2.10 | PoseConv3D | 0.867 | 7 | **1** | most precise, but misses fast wrist flicks |
| v2.20 | SkateFormer | 0.788 | 7 | 7 | overfits 297 clips |
| v3.00 | Skeleton-OOD trained end to end | 0.713 | 2 | 23 | pruning during training blocked learning |
| **v3.10** | **ST-GCN + CTR-GCN + PoseConv3D ensemble** | **0.971** | **0** | **2** | the three models' errors cancel out |

Note: v1.00 to v1.08 used 39 normal videos and v2.10 onward used 44, so early and late scores are not fully comparable.

### 3.2 Phase 2: from 0.971 to 1.000

**PoseConv3D downscaling:** smaller models generalised better.

| Model | Parameters | F1 | False alarms | Missed |
|---|---|---|---|---|
| Scaled up | 3.55M | 0.841 | 7 | 4 |
| Tier 3 | 598k | **0.886** | 6 | 2 |
| Tier 5 | 132k | 0.862 | **0** | 8 |

**Distillation** from ST-GCN to Tier 3 (relational knowledge distillation) gave **F1 0.800**: recall improved to 32/33, but false alarms rose to 15.

**Ensembles:**
- 2 models reached 0.985.
- **3 models (ST-GCN + CTR-GCN + Tier 3) reached 1.000**, but ran at only 22 FPS.

**Staircase Cascade v5.10:**
- It reaches the same 1.000 at 34.8 FPS.
- Each clip passes through the tiers only until one is confident, so most clips stop early:
  - Tier 1 (PoseConv3D Tier 3) settles 54.5% of clips.
  - Tier 2 (distilled Tier 3) settles 23.4%.
  - Tier 3 (ST-GCN) handles the remaining 22.1%.
- 77.9% of clips therefore never run the heavy ST-GCN.
- Median action latency is 3.07 ms.
- The thresholds were chosen from **1.25M combinations**; 333 of them gave F1 1.000.

---

## 4. Making It Fast (Sep 19 to 21)

With accuracy looking solved, the work turned to speed and deployment:

| Change | Before | After |
|---|---|---|
| TensorRT FP16: Stage 1 / pose / ST-GCN | 12.05 / 14.80 / 3.99 ms | 5.80 / 6.10 / 0.86 ms |
| 384x640 Stage 1 engine | 5.80 ms, 91 MiB | **3.86 ms (259 FPS), 62 MiB** |
| Buffer of 1,200 frames at 1080p | 7,464 MB | 188 MB |
| Stage 1 F1 at 640x640 with TensorRT | 86.9% | 84.9% |
| Stage 1 F1 at 384x640 | – | **76.4% on the first test; never re-measured** |

Supporting work:
- A cross-platform launcher works on NVIDIA GPUs and on Intel laptops.
- Packages were moved to a project environment in 2.8 s.
- The pipeline stops with an error if no CUDA GPU is available, instead of running slowly on the CPU.

---

## 5. Running on the Real Camera (Oct)

On the Edge PC (an RTX 3050 with the lab camera), the problems came one after another. Each fix exposed the next problem:

| Problem | Before | After |
|---|---|---|
| Camera discovery / GUI freeze when switching | 10 to 12 s / 3.5 s | 0.14 s / 0 s |
| Display frame rate (driver locked to 15 FPS, `waitKey` 15 ms, CPU drawing) | 15.1 FPS | 30 FPS |
| Heatmap body size (coordinates normalized twice) | 6 cells, every clip rejected at tier 1 | 19 cells, as in training |
| False arming on a knife lying still | armed | 0 alarms (static filter + hand gate within 0.5 torso lengths) |
| Pose per frame (default tracker instead of ByteTrack) | 34.0 ms | **11.5 ms** |
| Frames never analyzed (43.5 s test) | 163 (5.4 s) | **0** |
| Stage 2 backlog behind live (median / max) | 25 / 73 frames | 0 / 5 frames |
| Engine fed 480x640 instead of 384x640 | 1,092 silent exceptions | input size read from the engine |

**At this point the pipeline itself ran cleanly. The models still caught almost nothing.**

---

## 6. Why Stage 2 Fails on the Live Camera (Oct 31 to Nov 12)

### 6.1 The gap

| Test | Detection rate |
|---|---|
| Training clips (118) | 99% |
| Same clips with small changes (arm lower, lean, other view) | 21 to 47% |
| **Real lab attacks on the live camera** | **0 of 23**; take 002: **1 of 31** |
| AUC per tier on lab windows (0.5 means chance) | 0.55 / 0.74 / 0.49 |

### 6.2 Why the 1.000 was misleading (found Nov 12)

- **Only Cut-Down was tested.** The data loader read only the first subfolder of each class, so the 39 Thrust and 32 Stab clips were never loaded.
- **The thresholds were tuned on the same 77 videos they were scored on** (1.25M combinations).
- **Normal and attack clips came from different cameras.** The normal clips were 360x640 internet videos and the attacks were 2560x1600 lab clips, so a model could tell them apart by the camera instead of the action.
- **On the full validation set, live-style:** Thrust **16/39**, Stab **26/32**, Cut-Down **29/33**, false alarms 0/44.

### 6.3 Attempts to close the gap

Test: take 002, which has 31 attacks and 2.4 minutes of normal activity.

| Model set | Data | Attacks caught | False-alarm events | Validation set, Thrust / Stab |
|---|---|---|---|---|
| Original | 118 real clips | 1 / 31 | 2 | 32 to 55% / 79 to 85% |
| `sim` | 5,530 simulated clips | 14 / 31 | 2 | 0% / 0% |
| **`sim3`** (3 ST-GCNs, corrected body model; opt-in) | simulated + corrected | **15.3 / 31** | **0** | 2% / 2% |
| `sim3` + motion editor | + edited real strikes | 16.0 / 31 | 0 | – |
| `sim3` + body filter (OpenSim) | 6 models | 20.0 / 31 | at most 1 allowed | – |
| Trained on real lab takes | 29 lab takes, 259 strikes | 22 to 31 / 31 | **6.7 to 10 of 24 normal segments** | – |

**Full pipeline on 8 held-out takes (77 strikes):**

| | Original | `sim3` |
|---|---|---|
| Strikes alarmed | 9 (12%) | **16 (21%)** |

The pattern across these attempts:
- **Simulated data** trades accuracy on the training camera for accuracy on the live camera.
- **Real lab data** catches the attacks but also fires on normal arm movements.
- Changes to labels, window length, the knife box as an extra input, RMD, or blending models moved AUC by **only ±0.02**.
- The same normal movements always trigger alarms: stretching (every run), hand to head, picking something up, and walk-turn.

---

## 7. Conclusions and Challenges

**Conclusions**
1. **Stage 1 works:** 86.9% F1, 3.86 ms with TensorRT. Distillation beat the teacher by 13 F1 points; seven alternative designs were measured and none beat it.
2. **The real-time pipeline works:** 30 FPS on the camera, 0 skipped frames, and pose at 11.5 ms on an RTX 3050.
3. **Stage 2's 1.000 F1 was an artefact of the test, not real performance.** It came from a Cut-Down-only test set, thresholds tuned on that same set, and normal clips from a different camera.
4. **The models learned the camera and the 118 training motions, not the attack itself.** On the real camera they catch 1 of 31 attacks.
5. **The method itself is sound.** Trained on real lab takes, the model catches 22 to 31 of 31 attacks (held-out strike AUC about 0.95).
6. **The limit is data, not architecture.** Every model change since September moved AUC by at most ±0.02.

**Challenges**
1. **No real normal arm movements were recorded.** There are 638 attack windows and 0 repeated non-attack gestures, and only 29 of the 84 planned takes were recorded.
2. **No clean test set.** Take 002 was used to choose models, and the same person appears in training and test, so every number above is optimistic.
3. **Small sample.** With 31 attacks and 24 normal segments, a difference of 1 or 2 segments is noise. 0 false alarms in 2.4 minutes still allows a rate of up to 1.25 per minute (95% bound).
4. **Some numbers were never measured on the deployed setting:**
   - Stage 1 F1 at 384x640.
   - All Jetson Nano and GTX 1060 latencies, which are estimates.
   - Several archive figures that disagree between reports: CTR-GCN F1 0.901 vs 0.865, and the distilled Tier 3 at 0.886 vs 0.800.

**Next step:** record the missing normal-activity takes in the same session as some attack takes, keep one recording untouched for testing, then retrain and re-measure. Re-measure Stage 1 F1 at 384x640.

---

## 8. Datasets

Root: `D:\Intern AI Project\hierarchical-threat-detection-edge-ai\data\`. The counts below were taken from disk on 2026-11-13.

| Path | Contents | Used for |
|---|---|---|
| `Ground_Truth_Dataset/` (`With_Coat`, `Without_Coat`, `OOD`, `labels/`) | 715 images: 351 / 245 / 119 on disk, 240 knife boxes | Stage 1 test (section 2). The reports list the split as 359 / 235 / 121. |
| `Cut-Down/`, `Stab/`, `Thrust/` (`videos/` + `annotations_yolo26s-pose/*.xml`) | 47 / 28 / 43 = 118 videos, each with a pose XML file | Stage 2 training. The 297 training clips are windows cut from these. |
| `Validate/` (`With_Coat`, `Without_Coat`, `OOD`, each with `Cut_Down`, `Stab`, `Thrust`) | 48 + 56 attack videos and 46 normal videos | Stage 2 validation. The suite reads only Cut-Down, which gives the 77 videos (section 6.2). |
| `False Detected Clip/` | 5 skeleton XML files plus the videos | Earlier false-alarm cases |
| `Simulated/` | 5,530 simulated skeleton clips (11 classes, including `Aug_*` real clips with small changes) | Model set `sim` (sim v2) |
| `Simulated_v2/` | 7,180 clips from the corrected simulator | Model set `sim3` |
| `exp_A_old/`, `exp_H_all/` | 6,000 clips each | Simulator experiments |
| `lab_recordings/2026-10-31/` (`raw`, `segments`, `analysis`) | Lab takes 001 and 002 on the live camera (002: 31 attacks + 2.4 min of normal activity) | Live test set (section 6) |
| `lab_recordings/2026-11-05/` (`train_attack`, `calib`, `takes.csv`) | 29 attack takes, about 1,500 s, 259 strikes; 8 takes held out | Training on real lab data, and the held-out test |
| `live_clips_2026-10-31/` (`s1031`, `s110035`, `s111247`, `s114942`, `e2e`) | Clips from 4 live sessions plus 1 end-to-end run | Checking live false alarms |
| `../results/samples/` (`heldout.npz`, `poc/`, `edit/`, `body_model/`, ...) | Saved skeletons and proof-of-concept scripts | Sections 6.3 and 7 |

---

## 9. References

All paths are relative to `D:\Intern AI Project\`. **Archive** means `HTD_Archive_Reports_and_Experiments\reports\`, and **Repo** means `hierarchical-threat-detection-edge-ai\reports\`.

| Date | File | What it is | Used in |
|---|---|---|---|
| Jan 27 to Feb 2 | `Messy (Don't Touch)\Weapon_Detection_SingleGPU\Inference_Results_GroundTruth\*.txt` | About 100 Stage 1 ground-truth runs (teacher, student, P2 attempts) | 2.1 |
| Jan 31 | `Messy (Don't Touch)\Weapon_Detection_SingleGPU\Technical_Handout_P2_Hybrid.md` | Log of the 23 attempts to add a P2 head | 2.1 |
| Feb 2 | `Messy (Don't Touch)\*\Workflow.md` | First pipeline design | 1 |
| Sep 8 | Archive `BENCHMARK_EXPLANATION.md` | Mahalanobis vs k-NN for ST-GCN | 1, 3 |
| Sep 15 | `Clean\*\README.md` (3 files) | Clean copies of the January to September code | 2.1 |
| Sep 16 | Archive `Modern OOD Action Recognition Models.md` | Literature review of Stage 2 backbones and OOD gates | 3.1 |
| Sep 16 | Archive `Stage1_SOTA_Architectures_2026.md`, `STAGE1_DEEP_RESEARCH_PROMPT.md` | Stage 1 literature search and the brief written for it | 2.2 |
| Sep 16 to 17 | Archive `ROADMAP_AND_HANDOFF.md`, `ROADMAP_AND_HANDOFF_PHASE2.md`, `HANDOFF_NEXT_STEPS.md` | Plans and handoffs between phases | 3 |
| Sep 17 | Archive `VERSIONS.md` | Changelog for v1.00 to v3.10 (Stage 2, Phase 1) | 3.1 |
| Sep 17 | Archive `VERSIONS_PHASE2.md` | Changelog for v4.00 to v4.50 and v5.10 (Stage 1 search, cascade) | 2.2, 3.2 |
| Sep 17 | Archive `PROJECT_REPORT_PHASE1_PHASE2.md` | Combined report for Phases 1 and 2 | 2, 3 |
| Sep 17 to 18 | Archive `PROJECT_REPORT_` and `COMPACT_REPORT_` files on POSECONV3D_SCALING, F1_MAXIMIZATION and STAIRCASE_CASCADE | PoseConv3D scaling and distillation, ensembles, cascade, edge latency | 3.2 |
| Sep 18 | Archive `FULL_FINAL_REPORT.md` | Master report up to v5.10 | 2 to 4 |
| Sep 18 | `HTD_Archive_Reports_and_Experiments\ARCHIVE_MANIFEST*.md` | What was moved into the archive | – |
| Sep 19 | Archive `MODIFICATION_REPORT_MODULAR_PIPELINE_V5.20_AND_README.md` | Modular pipeline v5.20 | 4 |
| Sep 19 | Archive `MODIFICATION_REPORT_TENSORRT_ACCELERATION.md` | TensorRT v6.00 | 4 |
| Sep 20 | Archive `MODIFICATION_REPORT_RECTANGULAR_TENSORRT_INFERENCE.md` | 384x640 engines | 4 |
| Sep 20 | Archive `MODIFICATION_REPORT_CROSS_PLATFORM_DEPLOYMENT_AND_PACKAGE_MIGRATION.md` | Launchers for the Edge PC and the laptop | 4 |
| Sep 18 to 20 | `hierarchical-threat-detection-edge-ai\README.md` and `README_v5.10/v5.20/v6.00_*archive.md` | Project README at each version | 4 |
| Sep 17 to 21 | Archive `AgentRule*.md`, `UniversalAgentRule.md` | Working rules for the coding agent (no results) | – |
| Oct 8 | Repo `MODIFICATION_REPORT_CAMERA_HOTSWAP_PYTORCH.md` | Camera discovery and hot-swap | 5 |
| Oct | Repo `opencv_ui_bottleneck_and_refactoring_report.md` | 15 to 30 FPS display fix | 5 |
| Oct 22 | Repo `MODIFICATION_REPORT_STAGE2_DEBUG_TRACER.md` | Debug tracer; double-normalization bug | 5 |
| Oct 31 | Repo `MODIFICATION_REPORT_WEAPON_GATES.md` | Static filter and hand gate | 5 |
| Nov 5 | Repo `MODIFICATION_REPORT_STAGE2_CATCHUP.md` | Catch-up mode and the tracker fix | 5 |
| Nov 5 | Repo `MODIFICATION_REPORT_SKELETON_SIMULATOR.md` | Skeleton simulator; 0 of 23 live attacks caught | 6.1, 6.3 |
| Nov 5 | Repo `MODIFICATION_REPORT_STAGE2_RETRAIN_SIM.md` | `sim` (sim v2) retraining | 6.3 |
| Nov 5 | Repo `MODIFICATION_REPORT_STAGE2_SIM3.md` | Corrected simulator, `sim3` | 6.3 |
| Nov 5 | Repo `MODIFICATION_REPORT_STAGE2_SAMPLES.md` | Nov 5 lab recordings and generator | 6.3 |
| Nov 5 | Repo `NEXT_CONVERSATION_PROMPT.md` | Handoff note | – |
| Nov 7 | Repo `MODIFICATION_REPORT_STAGE2_EDITOR.md` | Motion editor and body filter | 6.3 |
| Nov 12 | Repo `MODIFICATION_REPORT_STAGE2_POC_FINDINGS.md` | Why the 1.000 was misleading; real-data proof of concept | 6.2, 6.3, 7 |
