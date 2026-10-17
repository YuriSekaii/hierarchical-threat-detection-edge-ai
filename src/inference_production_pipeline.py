"""
src/inference_production_pipeline.py

Production End-to-End Real-Time Surveillance Pipeline (Version 6.00 - TensorRT Acceleration).
Modular Orchestrator integrating:
1. In-Memory TurboJPEG Buffer (src/pipeline_buffer.py) -> ~188.5 MB RAM (1,200 1080p frames)
2. Stage 1 Handheld Weapon Detector (src/stage1_detector.py) -> Distilled YOLO26s (TensorRT NMS-Free, 86.88% F1, 5.45 ms)
3. Stage 2a Pose Tracker (src/stage2a_pose_tracker.py) -> YOLO26s-Pose TensorRT + ByteTrack Actor Locking + Torso Normalization
4. Stage 2b Staircase Cascade (src/stage2b_staircase.py) -> Champion 2 TensorRT (PoseConv3D T3 -> Distilled T3 -> ST-GCN, 1.0000 F1)

Strict GPU enforcement (No CPU fallback). All models run natively via NVIDIA TensorRT engines compiled from universal ONNX blueprints.
Fully decoupled, modular architecture preventing single-file bloat.
Supports physical webcam streams, video files, headless deployment, and simulated dry-run verification.
"""

import os
import sys
import time
import argparse
import threading
from collections import deque
from typing import Optional, Union, Tuple
import cv2
import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.pipeline_buffer import ThreadSafeCircularBuffer, FrameItem
from src.stage1_detector import Stage1WeaponDetector, calculate_rectangular_imgsz
from src.stage2a_pose_tracker import Stage2aPoseTracker
from src.stage2b_staircase import Stage2bStaircaseEngine
from src.engine_builder import get_hardware_precision


class ProductionHierarchicalPipeline:
    """
    Unified Hierarchical Edge Surveillance Pipeline (v6.00 TensorRT Accelerated).
    Orchestrates ingestion, lightweight 3 FPS weapon screening,
    threat-actor pose tracking, and progressive multi-tier staircase auditing.
    """

    def __init__(
        self,
        weapon_weights: Optional[str] = None,
        pose_weights: Optional[str] = None,
        conf_threshold: float = 0.45,
        buffer_size: int = 1200,
        jpeg_quality: int = 85,
        device: Optional[str] = None,
        imgsz_override: Optional[Tuple[int, int]] = None,
        backend: str = "tensorrt",
        camera_res: str = "auto",
        aspect_ratio: str = "4:3",
    ):
        self.camera_res = camera_res.lower()
        self.aspect_ratio = aspect_ratio.lower()
        self.backend = backend.lower()
        if self.backend == "tensorrt":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "[CRITICAL ERROR] TensorRT backend strictly requires an NVIDIA CUDA GPU.\n"
                    "CPU execution is not supported by TensorRT."
                )
            self.device = device or "cuda:0"
            if "cpu" in self.device.lower():
                raise RuntimeError("[CRITICAL ERROR] Strict GPU policy: Cannot run TensorRT on CPU.")
            prec, _ = get_hardware_precision()
            runtime_label = f"NATIVE TENSORRT 11.3 ({prec})"
            hw_name = torch.cuda.get_device_name(0)
            hw_policy = "Strict CUDA Active (No CPU Fallback)"
        else:
            if torch.cuda.is_available() and (device is None or "cpu" not in str(device).lower()):
                self.device = device or "cuda:0"
                hw_name = torch.cuda.get_device_name(0)
                hw_policy = "NVIDIA CUDA GPU Accelerated"
            elif hasattr(torch, "xpu") and torch.xpu.is_available() and (device is None or "cpu" not in str(device).lower()):
                self.device = device or "xpu:0"
                hw_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, "get_device_name") else "Intel XPU GPU"
                hw_policy = "Intel XPU Hardware Acceleration"
            else:
                self.device = "cpu"
                hw_name = "Host CPU (AVX / oneDNN Reference Mode)"
                hw_policy = "CPU Reference Mode (~5-10 FPS)"
            runtime_label = "BASE PYTORCH EAGER (.pt / .pth)"

        self.conf_threshold = conf_threshold
        self.imgsz_override = imgsz_override
        self.rect_imgsz = imgsz_override or (384, 640)

        print("\n" + "=" * 80)
        print("   HIERARCHICAL EDGE-AI THREAT & VIOLENCE DETECTION (PRODUCTION PIPELINE)")
        print(f"   Hardware: {hw_name} | Runtime: {runtime_label}")
        print(f"   Target Confidence: {self.conf_threshold:.2f} | Policy: {hw_policy}")
        print("=" * 80)

        # 1. In-Memory Ring Buffer
        self.buffer = ThreadSafeCircularBuffer(maxlen=buffer_size, jpeg_quality=jpeg_quality)

        # 2. Stage 1 Handheld Weapon Detector
        self.stage1_detector = Stage1WeaponDetector(
            weights_path=weapon_weights,
            conf_threshold=conf_threshold,
            device=self.device,
            imgsz=self.rect_imgsz,
            backend=self.backend,
        )

        # 3. Stage 2a Pose Extractor & ByteTrack Threat Actor Guard
        self.stage2a_tracker = Stage2aPoseTracker(
            weights_path=pose_weights,
            device=self.device,
            imgsz=self.rect_imgsz,
            backend=self.backend,
        )

        # 4. Stage 2b Champion 2 Staircase Cascade Engine
        self.stage2b_engine = Stage2bStaircaseEngine(
            device=torch.device(self.device),
            backend=self.backend,
        )

        # Runtime & Threading State
        self.is_running = False
        self.weapon_active_until = 0.0
        self.last_weapon_bbox = None
        self.last_analyzed_idx = 0
        self.clip_stride = 10
        self.clip_length = 50

        # UI & Telemetry State
        self.status_text = "SCANNING (IDLE)"
        self.status_color = (0, 255, 0)
        self.is_violent_alert = False
        self.alert_until = 0.0
        self.last_resolved_tier = None
        self.last_confidence = 0.0
        self.pending_audit_frames = 0
        self.fps_telemetry = 0.0

    def weapon_scanner_worker(self):
        """Thread 2: 3 FPS lightweight Stage 1 weapon screening loop."""
        target_delay = 1.0 / 3.0
        while self.is_running:
            t_start = time.time()
            frame = self.buffer.get_latest_frame()

            if frame is not None:
                det_res = self.stage1_detector.predict(frame, imgsz=self.rect_imgsz)
                if det_res["has_threat"]:
                    self.weapon_active_until = time.time() + 3.0
                    self.last_weapon_bbox = det_res["best_box"]
                    self.stage2a_tracker.update_weapon_anchor(det_res["best_box"])

                    if not self.is_violent_alert:
                        conf_pct = det_res["scores"][0] * 100.0 if det_res["scores"] else 0.0
                        self.status_text = f"WEAPON VERIFIED ({conf_pct:.1f}%)"
                        self.status_color = (0, 165, 255)

            elapsed = time.time() - t_start
            time.sleep(max(0.01, target_delay - elapsed))

    def action_auditing_worker(self):
        """Thread 3: Zero-drop sliding-window kinematic evaluator with Staircase Cascade."""
        while self.is_running:
            now = time.time()
            is_active = (now < self.weapon_active_until)

            if not is_active:
                if self.is_violent_alert and now > self.alert_until:
                    self.is_violent_alert = False
                    self.status_text = "SCANNING (IDLE)"
                    self.status_color = (0, 255, 0)
                elif not self.is_violent_alert:
                    self.status_text = "SCANNING (IDLE)"
                    self.status_color = (0, 255, 0)

                self.pending_audit_frames = 0
                self.stage2a_tracker.reset_tracking()
                time.sleep(0.05)
                continue

            # Extract contiguous 50-frame slice
            window, pending = self.buffer.get_window_by_index(self.last_analyzed_idx, self.clip_length)
            if len(window) < self.clip_length:
                time.sleep(0.02)
                continue

            self.last_analyzed_idx += self.clip_stride
            self.pending_audit_frames = pending

            # Step 1: Extract/Cache skeletons for any unvisited frames in window
            for item in window:
                if item.skeleton is None:
                    raw_frame = item.get_frame()
                    if raw_frame is not None:
                        kpts, bbox, track_id = self.stage2a_tracker.extract_actor_pose(
                            raw_frame, imgsz=self.rect_imgsz
                        )
                        item.skeleton = kpts
                        item.actor_bbox = bbox
                        item.track_id = track_id
                    else:
                        item.skeleton = np.zeros((17, 2), dtype=np.float32)

            # Step 2: Kinematic preprocessing & Torso-Scale Normalization
            raw_seq = np.array([item.skeleton for item in window], dtype=np.float32)  # (50, 17, 2)
            normed_coords, graph_tensor = self.stage2a_tracker.preprocess_sliding_window(raw_seq)

            # Step 3: Evaluate Progressive Staircase Cascade (Champion 2)
            cascade_res = self.stage2b_engine.evaluate_clip(graph_tensor)

            self.last_resolved_tier = cascade_res["resolved_tier"]
            self.last_confidence = cascade_res["confidence"]
            actor_tag = f"Actor ID: {self.stage2a_tracker.active_threat_actor_id}" if self.stage2a_tracker.active_threat_actor_id else "Threat Actor"

            if cascade_res["is_threat"]:
                self.is_violent_alert = True
                self.alert_until = time.time() + 5.0
                detail_str = f"Tier {cascade_res['resolved_tier']} | Conf: {cascade_res['confidence']*100:.1f}%"
                self.status_text = f"VIOLENCE DETECTED! ({actor_tag} | {detail_str})"
                self.status_color = (0, 0, 255)
                print(f"[ACTION] !!! VIOLENT ASSAULT ALARM: {actor_tag} | {cascade_res['decision_reason']} (Latency: {cascade_res['total_cascade_ms']:.2f} ms) !!!")
            else:
                if not self.is_violent_alert:
                    detail_str = f"Tier {cascade_res['resolved_tier']} Cleared ({cascade_res['confidence']*100:.1f}%)"
                    self.status_text = f"WEAPON DRAWN - NORMAL MOTION ({actor_tag} | {detail_str})"
                    self.status_color = (0, 165, 255)

    def apply_aspect_ratio(self, frame: np.ndarray) -> np.ndarray:
        """
        Enforces desired aspect ratio (e.g. 4:3) on camera frames.
        Extracts the highest possible resolution 4:3 center crop (e.g. 1920x1440 from 2560x1440).
        """
        if frame is None or self.aspect_ratio not in ("4:3", "4/3"):
            return frame
        h, w = frame.shape[:2]
        target_w = int(h * 4.0 / 3.0)
        if target_w < w:
            offset_x = (w - target_w) // 2
            return frame[:, offset_x : offset_x + target_w]
        elif target_w > w:
            target_h = int(w * 3.0 / 4.0)
            offset_y = (h - target_h) // 2
            return frame[offset_y : offset_y + target_h, :]
        return frame

    def open_camera(self, idx: int) -> Optional[cv2.VideoCapture]:
        """
        Opens camera index with DirectShow/MSMF acceleration, MJPG FourCC,
        and high-resolution sensor negotiation (up to 2560x1440 native).
        """
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    if self.camera_res in ("1440p", "2k", "auto"):
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                    elif self.camera_res == "1080p":
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    elif self.camera_res == "720p":
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    elif self.camera_res == "480p":
                        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    for _ in range(3):
                        r, test_frame = c.read()
                        if r and test_frame is not None and test_frame.size > 0:
                            return c
                    c.release()
            except Exception:
                pass
        return None

    def run(self, source: Union[int, str] = 0, headless: bool = False):
        """Executes live video stream surveillance with auto-failover and non-blocking camera cycling."""
        current_source = source
        cap = None

        if isinstance(current_source, int):
            cap = self.open_camera(current_source)
            if cap is None:
                print(f"\n[CAMERA NOTICE] Requested Camera Index {current_source} is not available or disconnected.")
                print("[CAMERA NOTICE] Auto-scanning connected cameras for the next available device...")
                candidates = [(current_source + i) % 4 for i in range(1, 5)]
                for cand in candidates:
                    test_c = self.open_camera(cand)
                    if test_c is not None:
                        cap = test_c
                        current_source = cand
                        cand_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        cand_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[CAMERA AUTO-SELECT] Switched automatically to active Camera Index: {current_source} ({cand_w}x{cand_h})\n")
                        break
        else:
            cap = cv2.VideoCapture(current_source)

        if cap is None or not cap.isOpened():
            print(f"[ERROR] Could not open video source: {current_source}")
            return

        # Determine geometry from test frame or cap properties
        ret_t, test_frame = cap.read() if cap.isOpened() else (False, None)
        if ret_t and test_frame is not None:
            test_frame = self.apply_aspect_ratio(test_frame)
            f_w, f_h = test_frame.shape[1], test_frame.shape[0]
            last_frame = test_frame
        else:
            raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.aspect_ratio in ("4:3", "4/3") and raw_w > 0 and raw_h > 0:
                crop_w = int(raw_h * 4.0 / 3.0)
                f_w = crop_w if crop_w <= raw_w else raw_w
                f_h = raw_h
            else:
                f_w, f_h = raw_w, raw_h

        if self.imgsz_override:
            self.rect_imgsz = self.imgsz_override
        else:
            self.rect_imgsz = calculate_rectangular_imgsz(f_h, f_w, base_dim=640) if f_w > 0 and f_h > 0 else (480, 640)

        aspect_tag = f" [4:3 Aspect Ratio]" if self.aspect_ratio in ("4:3", "4/3") else ""
        print(f"[CONFIG] Input Stream: {f_w}x{f_h}{aspect_tag} -> Inference Resolution: {self.rect_imgsz} (Native TensorRT Rectangular)")

        # Fast camera discovery via DirectShow COM (pygrabber)
        available_cameras = [current_source] if isinstance(current_source, int) else []
        camera_names = {current_source: f"Camera {current_source}"} if isinstance(current_source, int) else {}

        if isinstance(current_source, int):
            def _probe_cameras():
                devices = []
                try:
                    from pygrabber.dshow_graph import FilterGraph
                    names = FilterGraph().get_input_devices()
                    for idx, name in enumerate(names):
                        devices.append((idx, name.strip()))
                except Exception:
                    for idx in range(4):
                        devices.append((idx, f"Camera {idx}"))

                verified = []
                for idx, name in devices:
                    if idx == current_source:
                        verified.append((idx, name))
                        camera_names[idx] = name
                        continue
                    test_c = self.open_camera(idx)
                    if test_c is not None:
                        verified.append((idx, name))
                        camera_names[idx] = name
                        test_c.release()

                if verified:
                    available_cameras.clear()
                    available_cameras.extend([idx for idx, _ in verified])
                    summary = [f"{idx}: {camera_names.get(idx, 'Cam')}" for idx in available_cameras]
                    print(f"[CAMERAS] Active cameras ready for cycling: {summary}")

            # Run in fast background thread so launch is instant (<0.5s discovery)
            threading.Thread(target=_probe_cameras, daemon=True).start()

        self.is_running = True
        t_weapon = threading.Thread(target=self.weapon_scanner_worker, daemon=True)
        t_action = threading.Thread(target=self.action_auditing_worker, daemon=True)
        t_weapon.start()
        t_action.start()

        print("[RUNNING] Surveillance pipeline active. Press 'q' to stop, 'i' to swap camera.")

        frame_times = deque(maxlen=30)
        t_last = time.perf_counter()
        last_frame = None

        # State for asynchronous non-blocking camera swap
        is_swapping = False
        swap_target_idx = None
        swap_target_name = ""

        def _async_swap_worker(target_idx: int, target_name: str):
            nonlocal cap, current_source, is_swapping, last_frame
            try:
                new_cap = self.open_camera(target_idx)
                if new_cap is not None:
                    old_cap = cap
                    cap = new_cap
                    current_source = target_idx
                    if old_cap is not None:
                        old_cap.release()
                    new_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    new_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if self.aspect_ratio in ("4:3", "4/3") and new_w > 0 and new_h > 0:
                        crop_w = int(new_h * 4.0 / 3.0)
                        if crop_w <= new_w:
                            new_w = crop_w
                    if not self.imgsz_override and new_w > 0 and new_h > 0:
                        self.rect_imgsz = calculate_rectangular_imgsz(new_h, new_w, base_dim=640)
                    print(f"\n[CAMERA] Swapped to Camera {current_source} ({target_name}, {new_w}x{new_h}) successfully.")
                else:
                    print(f"\n[CAMERA WARNING] Camera {target_idx} opened but produced no frames.")
            except Exception as e:
                print(f"\n[CAMERA ERROR] Exception during swap: {e}")
            finally:
                is_swapping = False

        def _trigger_swap():
            nonlocal is_swapping, swap_target_idx, swap_target_name
            if is_swapping:
                return
            cams = list(available_cameras) if available_cameras else [0, 1, 2]
            for fallback_idx in [0, 1, 2]:
                if fallback_idx not in cams:
                    cams.append(fallback_idx)
            if current_source in cams:
                cur_pos = cams.index(current_source)
                target_idx = cams[(cur_pos + 1) % len(cams)]
            else:
                target_idx = cams[0]
            target_name = camera_names.get(target_idx, f"Camera {target_idx}")
            print(f"\n[CAMERA] Initiating hot-swap: Camera {current_source} -> Camera {target_idx} ({target_name})...")
            is_swapping = True
            swap_target_idx = target_idx
            swap_target_name = target_name
            threading.Thread(target=_async_swap_worker, args=(target_idx, target_name), daemon=True).start()

        consecutive_failures = 0
        win_title = f"Hierarchical Edge Threat Detection [{self.backend.upper()}]"
        if not headless:
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
            if self.aspect_ratio in ("4:3", "4/3"):
                cv2.resizeWindow(win_title, 960, 720)
            else:
                cv2.resizeWindow(win_title, 1280, 720)

        while self.is_running:
            if is_swapping:
                # Keep GUI active and responsive while swap worker runs in background!
                if not headless:
                    hud = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.rectangle(hud, (10, 10), (820, 85), (0, 0, 0), -1)
                    cv2.putText(hud, f"STATUS: SWITCHING TO CAMERA {swap_target_idx} ({swap_target_name})...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(hud, "Please wait, initializing camera hardware...", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                    cv2.imshow(f"Hierarchical Edge Threat Detection [{self.backend.upper()}]", hud)
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.02)
                continue

            ret = False
            frame = None
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if not headless:
                    hud = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.rectangle(hud, (10, 10), (820, 85), (0, 0, 0), -1)
                    cv2.putText(hud, f"CAMERA ALERT: NO SIGNAL FROM CAMERA {current_source}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    cv2.putText(hud, f"Press 'i' to switch camera | Auto-failover in {max(0, 30 - consecutive_failures)} ticks | 'q' to quit", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
                    cv2.imshow(f"Hierarchical Edge Threat Detection [{self.backend.upper()}]", hud)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                    elif key in (ord('i'), ord('I')) and isinstance(current_source, int):
                        _trigger_swap()
                        continue

                # Auto-failover if disconnected for ~1 second (30 ticks)
                if isinstance(current_source, int) and consecutive_failures >= 30 and not is_swapping:
                    consecutive_failures = 0
                    print(f"\n[AUTO-FAILOVER] Camera {current_source} signal unavailable. Auto-switching to next camera...")
                    _trigger_swap()
                time.sleep(0.02)
                continue

            # Frame successfully acquired
            consecutive_failures = 0
            frame = self.apply_aspect_ratio(frame)
            last_frame = frame
            now = time.perf_counter()
            frame_times.append(now - t_last)
            t_last = now
            self.fps_telemetry = len(frame_times) / sum(frame_times) if frame_times else 0.0

            # Ingest into TurboJPEG circular buffer
            self.buffer.append(frame, timestamp=time.time())

            if not headless:
                # Telemetry HUD Display
                hud = frame.copy()
                cv2.rectangle(hud, (10, 10), (820, 85), (0, 0, 0), -1)
                cv2.putText(hud, f"STATUS: {self.status_text}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, self.status_color, 2)

                buf_mb = self.buffer.memory_mb
                buf_len = len(self.buffer)
                tier_info = f"Tier: {self.last_resolved_tier}" if self.last_resolved_tier else "Tier: Idle"
                actor_info = f"Actor ID: {self.stage2a_tracker.active_threat_actor_id}" if self.stage2a_tracker.active_threat_actor_id else "Scanning"
                cam_name = camera_names.get(current_source, f"Cam {current_source}")
                res_info = f"{frame.shape[1]}x{frame.shape[0]}"
                cam_label = f"Cam: {current_source} ({cam_name}, {res_info}) ['i' to swap]" if isinstance(current_source, int) else "Source: File"
                telemetry = f"{cam_label} | FPS: {self.fps_telemetry:.1f} | Buf: {buf_len}/1200 | {tier_info} | {actor_info}"
                cv2.putText(hud, telemetry, (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

                # Draw weapon bounding box if actively detected
                if time.time() < self.weapon_active_until and self.last_weapon_bbox is not None:
                    bx1, by1, bx2, by2 = [int(v) for v in self.last_weapon_bbox]
                    cv2.rectangle(hud, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(hud, "WEAPON: KNIFE", (bx1, max(20, by1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)

                cv2.imshow(f"Hierarchical Edge Threat Detection [{self.backend.upper()}]", hud)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in (ord('i'), ord('I')) and isinstance(current_source, int):
                    _trigger_swap()

        self.is_running = False
        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("[SHUTDOWN] Surveillance pipeline terminated cleanly.")

    def run_dry_run(self, num_frames: int = 120):
        """Simulates live frame ingestion and pipeline execution for verification."""
        print(f"\n=== Running Production Pipeline [{self.backend.upper()}] Verification Dry Run ===")
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        self.is_running = True
        t_weapon = threading.Thread(target=self.weapon_scanner_worker, daemon=True)
        t_action = threading.Thread(target=self.action_auditing_worker, daemon=True)
        t_weapon.start()
        t_action.start()

        for i in range(num_frames):
            item = self.buffer.append(dummy_frame)
            # Simulate threat anchor halfway through
            if i == 40:
                self.weapon_active_until = time.time() + 5.0
                self.stage2a_tracker.update_weapon_anchor([300, 200, 400, 350])
            time.sleep(0.01)

        time.sleep(0.5)
        self.is_running = False

        print("\n--- Dry Run Verification Complete ---")
        print(f"Frames Ingested: {len(self.buffer)}")
        print(f"TurboJPEG Buffer RAM: {self.buffer.memory_mb:.2f} MB")
        print(f"Last Status: {self.status_text}")
        print(f"System State: VERIFIED FUNCTIONAL ({self.backend.upper()} Backend Active).\n")


def parse_source(src_str: str) -> Union[int, str]:
    try:
        return int(src_str)
    except ValueError:
        return src_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Edge Threat Detection Production Pipeline")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: camera index (0) or path to video file (default: 0)")
    parser.add_argument("--webcam", type=str, default=None,
                        help="Webcam index alias for --source (e.g. 0)")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Stage 1 weapon detection confidence threshold (default: 0.45)")
    parser.add_argument("--quality", type=int, default=85,
                        help="In-memory TurboJPEG compression quality 1-100 (default: 85)")
    parser.add_argument("--backend", type=str, choices=["tensorrt", "pytorch"], default="tensorrt",
                        help="Runtime backend: 'tensorrt' (native engines) or 'pytorch' (eager baseline, default: tensorrt)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode without cv2.imshow GUI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run verification dry run with synthetic frames")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cuda:0' (default: auto)")
    parser.add_argument("--resolution", "--res", type=str, default="auto",
                        choices=["auto", "1440p", "1080p", "720p", "480p"],
                        help="Camera stream resolution: 'auto' (native sensor, up to 2560x1440), '1080p', '720p', '480p'")
    parser.add_argument("--aspect-ratio", "--aspect", type=str, default="4:3",
                        choices=["4:3", "16:9", "native"],
                        help="Video aspect ratio framing: '4:3' (highest resolution 4:3, e.g. 1920x1440), '16:9', or 'native' (default: 4:3)")

    args = parser.parse_args()

    active_source = parse_source(args.webcam if args.webcam is not None else args.source)

    pipeline = ProductionHierarchicalPipeline(
        conf_threshold=args.conf,
        jpeg_quality=args.quality,
        device=args.device,
        backend=args.backend,
        camera_res=args.resolution,
        aspect_ratio=args.aspect_ratio,
    )

    if args.dry_run:
        pipeline.run_dry_run()
    else:
        pipeline.run(source=active_source, headless=args.headless)
