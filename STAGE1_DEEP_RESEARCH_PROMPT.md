# Master Prompt Brief for Gemini Deep Research: Stage 1 Weapon Detection Overhaul

> **HOW TO USE THIS WITH GEMINI DEEP RESEARCH:**  
> 1. Go to **[Gemini](https://gemini.google.com)** (make sure "Deep Research" is enabled).  
> 2. Copy and paste the entire prompt below into Gemini.  
> 3. Gemini Deep Research will browse the web for 10-20 minutes, analyzing academic papers, GitHub repositories, and benchmarks.  
> 4. When it finishes, save the markdown report into this repository as `Stage1_Weapon_Detection_Architectures.md`.  
> 5. In your new chat, tell the agent: *"Read `Stage1_Weapon_Detection_Architectures.md` and start with Item 1."*

---

### COPY THE PROMPT BELOW:

```markdown
You are acting as an elite Computer Vision and Edge-AI Research Scientist specializing in Real-Time Object Detection and Threat Recognition. I need an exhaustive, deep-dive academic and engineering research whitepaper to guide the architectural overhaul of Stage 1 (Weapon / Threat Object Detection) in our live edge surveillance system.

Below is the complete context of our current architecture, empirical baseline, physical failure modes, past negative results, and strict hardware constraints. Use this exact context to drive your deep literature and architectural research.

---

### 1. SYSTEM CONTEXT & VISION: HIERARCHICAL EDGE-AI SURVEILLANCE
Our project is a hierarchical two-stage edge surveillance pipeline designed for live CCTV camera streams:
- **Stage 1 (Primary Gatekeeper - Low-Power Always-On Scanner):**
  - Ingests video at 30 FPS into a 1,200-frame circular buffer.
  - Subsamples frames at an effective rate of 3 FPS (1 frame every 10) to run a lightweight weapon/knife detector.
  - Compute goal: Keep continuous idle power draw minimal.
  - Security role: If an armed attacker brandishes or draws a weapon, Stage 1 MUST detect it (Conf >= 0.45). If Stage 1 misses the weapon, the downstream action recognition engine is NEVER triggered.
- **Stage 2 (Triggered Kinematic Action Auditing):**
  - When Stage 1 fires, a secondary worker audits the preceding 50 frames using YOLO-Pose keypoints + a 9-block Spatial-Temporal Graph Convolutional Network (ST-GCN) and Relative Mahalanobis Distance (RMD) to verify active assault trajectories (Cut-Down, Stab, Thrust).
  - NOTE: Stage 2 is 100% complete and achieves 0 missed attacks (100% recall), 94.3% precision, and 0.9706 F1 with only 2 civilian false alarms.
  - **The entire remaining vulnerability of our system lies in Stage 1!**

---

### 2. OUR CURRENT STAGE 1 BASELINE (EMPIRICALLY VERIFIED ON 715 REAL SURVEILLANCE FRAMES)
- **Current Deployed Model:** Distilled YOLO26s student (~9.4M parameters) trained via response-based knowledge distillation from a heavy YOLO26x teacher (~60M parameters).
- **Inference Latency:** 11.18 ms on NVIDIA GTX 1060 (3GB) / 12.06 ms on NVIDIA RTX 3090 (Batch=1, FP16, 640x640).
- **Empirical Metrics at Production Operating Point (Conf = 0.45, IoU = 0.20):**
  - **Accuracy:** 91.62%
  - **F1-Score:** 86.88%
  - **Precision:** 89.78% (23 False Positives on civilian handheld items)
  - **Recall:** 84.17% (202 out of 240 threat frames detected)
  - **CRITICAL FAILURE:** **38 MISSED WEAPONS (15.83% False Negative Rate)**. In a security context, missing 38 knife attacks is unacceptable.
- **High-Sensitivity Operating Point (Conf = 0.20, IoU = 0.20):**
  - Recall increases to 91.67% (220/240, 20 missed), but Precision drops to 69.18% as False Positives explode by +326% from 23 to 98.

---

### 3. ROOT PHYSICAL FAILURE MODES & ABLATION LESSONS
1. **Extreme Blade Anisotropy & Spatial Quantization:**
   - Knives are elongated and narrow (often 1 to 5 pixels wide in surveillance video).
   - Standard CNN feature pyramids (downsampled by stride 8 at P3, stride 16 at P4, stride 32 at P5) blur or erase these micro-edge gradients before deep semantic layers can represent them.
2. **The Hand-Blade Occlusion & Non-Maximum Suppression (NMS) Dilemma:**
   - Weapons are gripped in hands. When held, the hand bounding box and blade bounding box heavily overlap.
   - Traditional Greedy NMS often suppresses the blade prediction in favor of the hand or person detection box.
3. **High-Velocity Motion Blur:**
   - Slashing and stabbing attacks move at 8–15 m/s, causing motion blur that drops confidence scores below the 0.45 threshold.
4. **Specular Reflections & Ambient Camouflage:**
   - Metallic stainless steel blades reflect room lighting or disappear against light-colored clothing.
5. **PAST NEGATIVE RESULT (CRITICAL LESSON): Naive P2 Head Failed:**
   - We tested adding a high-resolution P2 feature map (stride 4, 160x160) to YOLO26s (YOLO26s-P2).
   - **Result:** Latency spiked by +73% (from 11.18 ms to 19.34 ms), while recall dropped from 84.17% down to 73.75% (-25 weapons detected!). Gradient dilution across 4 multi-scale heads destroyed the semantic representation of the compact backbone.

---

### 4. DEPLOYMENT & HARDWARE CONSTRAINTS
- **Edge Deployment Targets:** NVIDIA Jetson Orin Nano / NX, edge IPCs, and low-cost host PCs (GTX 1060 3GB / RTX 3090).
- **Latency Budget:** Maximum 15.0 ms per frame on desktop GPU, target >= 60 FPS in FP16.
- **Operating System:** Windows 10/11 & Linux.
- **Framework:** PyTorch 2.0+ (Ultralytics ecosystem or clean PyTorch implementation).
- **Class Setup:** Single target class (weapon / knife), with heavy background OOD distractors (smartphones, pens, vape devices, wallets, metallic keys).

---

### 5. RESEARCH OBJECTIVES & QUESTIONS TO INVESTIGATE IN DEPTH
Please conduct a rigorous, multi-source literature review covering the latest state-of-the-art developments (2024–2026) and address the following questions in detail:

1. **Real-Time Detection Transformers (RT-DETRv2 / RT-DETRv3):**
   - How does bipartite Hungarian matching (NMS-free inference) resolve the hand-knife bounding box suppression problem?
   - What is the exact computational cost and latency of RT-DETRv2-S / RT-DETRv2-M on edge hardware?
   - How does the hybrid encoder (AIFI + CCFM) perform on small, occluded, anisotropic objects compared to standard CNN PANets?
2. **Next-Generation NMS-Free YOLOs (YOLOv10 / YOLOv11 / YOLOv12):**
   - How does YOLOv10's Consistent Dual Assignment (one-to-many training, one-to-one inference) compare in latency and small-object recall against RT-DETR?
   - Does YOLOv12's Area Attention or YOLOv11's C3k2 feature extraction improve micro-edge representation without suffering the gradient dilution of our failed P2 head?
3. **Open-Vocabulary & Vision-Language Knowledge Distillation:**
   - We already have pre-trained YOLO-World v2 (yolov8x-worldv2.pt) weights available in our repository.
   - Can we distill rich semantic text-vision representations (separating "knife blade" from "smartphone in hand") into a compact student without adding runtime latency?
   - What specific distillation loss formulations (feature hint matching, logit distillation, relation distillation) work best for small weapon defense?
4. **Dynamic & Multi-Scale Attention Heads (DyHead / Weighted BiFPN):**
   - How can dynamic scale-aware, spatial-aware, and task-aware attention be applied to P3–P5 to capture fine blade contours without introducing a heavy P2 head?
5. **Short-Window Temporal Motion Cues:**
   - In surveillance video, static knives are hard to detect, but a knife being drawn or swung creates rapid specular shifts and high-acceleration trajectories.
   - Are there lightweight temporal feature differencing or 3-frame temporal attention mechanisms that can boost detection recall during active strikes while maintaining sub-15ms speed?
6. **Architectural Comparison Matrix & Implementation Recommendations:**
   - Provide a structured technical comparison table ranking candidate architectures by:
     - Expected Weapon Recall (targeting >92%)
     - Expected Civilian False Positive Rate
     - Inference Latency (ms on modern GPUs)
     - Parameter Count & Model Size
     - Ease of PyTorch training and deployment on Windows
   - Rank the top 3 concrete architectures we should prioritize for implementation, detailing exact PyTorch/Ultralytics configuration blocks.
```
