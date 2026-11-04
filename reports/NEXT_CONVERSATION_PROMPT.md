Continue my project: hierarchical-threat-detection-edge-ai (knife-violence detection, edge AI). Reply in plain English and read files before making claims.

Repo: YuriSekaii/hierarchical-threat-detection-edge-ai, branch main.
Dev PC (this machine, 2 x RTX 3090): D:\Intern AI Project\hierarchical-threat-detection-edge-ai, with .venv.
Edge PC (RTX 3050, camera attached): D:\Code\hierarchical-threat-detection-edge-ai. SSH: ssh -o BatchMode=yes User@100.94.121.118 (remote shell is cmd). It may be off; ask me.
Workflow: check that both PCs are on the same commit -> change code on the dev PC -> commit and push -> git pull on the Edge PC. I hate opening CMD, so give me one-click .bat launchers. Never overwrite my lab recordings in data/lab_recordings.

Read first: reports/MODIFICATION_REPORT_STAGE2_RETRAIN_SIM.md (current state, numbers, next steps), then reports/MODIFICATION_REPORT_SKELETON_SIMULATOR.md and the docstring of scripts/train_stage2_sim.py.

Where things stand:
- Pipeline: Stage 1 YOLO knife detector with static and hand gates -> Stage 2a YOLO26s-pose + ByteTrack -> Stage 2b 3-tier cascade on 50-frame clips, one every 10 frames.
- The original ("live") cascade is still the default. It caught 1 of 31 attacks in lab take 002 and scores camera angle more than action.
- I built a skeleton simulator (src/skeleton_sim, simulate_dataset.bat) and retrained. The new "sim v2" set is an option: run_webcam_tensorrt_sim.bat, or --stage2-models sim. Its ST-GCN decides; the two PoseConv3D tiers are pass-through because they dropped 82 % of real attack windows.
- On take 002 sim v2 caught 14 of 31 attacks at the same false-alarm rate as the live models (0.83 events/min).
- Sim v2 is NOT ready to replace the live models: my known live thrusts stay below its alarm bound (0.21-0.47 vs 0.55), it detects 0 % of Validate Thrust and Stab, and it alarmed on 7 of 280 live clips (one reviewed, it looks false).
- It has never run end to end on the Edge PC.
- Take 002 helped choose the recipe and threshold, so it is no longer a clean test. A new recording is needed for an unbiased number.
- Files not in git, dev PC only: data/Simulated_v2, data/live_clips_2026-10-31, results/train_sim.

Your first steps: check git status and which tools are available in this conversation (a new tool was added, tell me how it could help this project). Then give me a short plan from the report's "Next steps" and ask which to do first, unless I say otherwise below. Ask before downloading model weights. Keep lab take 002, new test recordings and known live thrust clips out of training. Report results honestly, including what got worse.

My instruction for this conversation:
