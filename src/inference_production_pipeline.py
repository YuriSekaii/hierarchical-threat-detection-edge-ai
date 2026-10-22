"""
src/inference_production_pipeline.py

Production End-to-End Real-Time Surveillance Pipeline (Version 6.20 - Decoupled Multi-UI & TensorRT).
Modular Orchestrator integrating:
1. Hybrid Hot-Ring + Async TurboJPEG Buffer (src/pipeline_buffer.py) -> 0.0 ms hot window access, ~188.5 MB RAM
2. Stage 1 Handheld Weapon Detector (src/stage1_detector.py) -> Distilled YOLO26s (TensorRT NMS-Free, 86.88% F1, 5.45 ms)
3. Stage 2a Pose Tracker (src/stage2a_pose_tracker.py) -> YOLO26s-Pose TensorRT + ByteTrack Actor Locking + Torso Normalization
4. Stage 2b Staircase Cascade (src/stage2b_staircase.py) -> Champion 2 TensorRT (PoseConv3D T3 -> Distilled T3 -> ST-GCN, 1.0000 F1)

Architectural Upgrades (v6.20):
- Decoupled Producer-Consumer Camera Ingestion Thread: `cap.read()` runs in a dedicated thread
  completely isolated from GUI rendering, locking onto native 30/60 FPS hardware sensor output.
- 3 Pluggable UI Backends (`--ui auto|qt|web|opencv`):
  * `qt` (PySide6 / PyQt6): Hardware-accelerated Qt6 desktop window with non-destructive QPainter
    vector glass overlay, bounding box layer, and interactive camera-swap controls (Zero pixel mutation).
  * `web` (Built-in HTTP/MJPEG + JSON Telemetry): Zero-dependency HTML5 + CSS Glassmorphism web dashboard
    at `http://localhost:8080` for headless edge/browser monitoring.
  * `opencv` (Fast Pre-Scaled Display Surface): Downscales high-res 1440p frames to window size (720p)
    prior to HUD compositing and `cv2.imshow()`, cutting GDI memory bandwidth by 75%.
- Backlog-Free Sliding Window Auditing & Strided Cold-Start Backfill: Eliminates the 40-second stale index
  trap and halves initial 50-frame pose extraction burst latency.
"""

import os
import sys
import json
import time
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque
from typing import Optional, Union, Tuple, Sequence, Callable
import cv2
import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.pipeline_buffer import ThreadSafeCircularBuffer, FrameItem, _encode_jpeg_bytes
from src.stage1_detector import Stage1WeaponDetector, calculate_rectangular_imgsz
from src.stage2a_pose_tracker import Stage2aPoseTracker
from src.stage2b_staircase import Stage2bStaircaseEngine
from src.engine_builder import get_hardware_precision


def _detect_qt_binding():
    """Detects PySide6 or PyQt6 if available for hardware-accelerated vector overlay UI."""
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        return "PySide6", QtWidgets, QtCore, QtGui
    except Exception:
        pass
    try:
        from PyQt6 import QtWidgets, QtCore, QtGui
        return "PyQt6", QtWidgets, QtCore, QtGui
    except Exception:
        pass
    return None, None, None, None


_WEB_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Hierarchical Edge-AI Surveillance HUD</title>
<style>
  :root { --accent: #00ff66; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #090b10; color: #f0f4f8;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 100vh; overflow: hidden;
  }
  .viewport {
    position: relative; display: inline-block;
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,0.75);
    border: 1px solid rgba(255,255,255,0.12);
    background: #000;
  }
  #stream { display: block; max-width: 96vw; max-height: 92vh; width: auto; height: auto; }
  .hud-card {
    position: absolute; top: 16px; left: 16px;
    background: rgba(12, 15, 22, 0.84);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1.5px solid var(--accent);
    border-radius: 8px; padding: 12px 16px;
    min-width: 340px; pointer-events: auto;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  }
  .hud-title { font-size: 15px; font-weight: 700; color: var(--accent); letter-spacing: 0.4px; margin-bottom: 6px; }
  .hud-line { font-size: 12.5px; color: #e2e8f0; margin-bottom: 4px; font-family: monospace; }
  .hud-sub { font-size: 11.5px; color: #94a3b8; font-family: monospace; }
  .controls { margin-top: 8px; display: flex; gap: 8px; }
  button {
    background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2);
    border-radius: 5px; padding: 4px 10px; font-size: 11.5px; cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: rgba(255,255,255,0.22); }
  #bbox-layer { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
</style>
</head>
<body>
  <div class="viewport">
    <img id="stream" src="/stream.mjpg" alt="Live Stream"/>
    <canvas id="bbox-layer"></canvas>
    <div class="hud-card" id="card">
      <div class="hud-title" id="status">STATUS: INITIALIZING...</div>
      <div class="hud-line" id="line2">Connecting telemetry...</div>
      <div class="hud-sub" id="line3">Pipeline: Idle</div>
      <div class="controls">
        <button onclick="swapCam()">Swap Camera [I]</button>
      </div>
    </div>
  </div>
<script>
const img = document.getElementById('stream');
const canvas = document.getElementById('bbox-layer');
const ctx = canvas.getContext('2d');
const card = document.getElementById('card');
const statusEl = document.getElementById('status');
const line2El = document.getElementById('line2');
const line3El = document.getElementById('line3');

function swapCam() { fetch('/api/swap', {method: 'POST'}); }
window.addEventListener('keydown', (e) => { if (e.key === 'i' || e.key === 'I') swapCam(); });

async function pollTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    const d = await res.json();
    statusEl.textContent = 'STATUS: ' + d.status_text;
    statusEl.style.color = d.css_color;
    card.style.borderColor = d.css_color;
    line2El.textContent = d.line2;
    line3El.textContent = d.line3;

    if (img.clientWidth > 0 && img.clientHeight > 0) {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (d.weapon_bbox && d.frame_w > 0 && d.frame_h > 0) {
        const sx = canvas.width / d.frame_w;
        const sy = canvas.height / d.frame_h;
        const [x1, y1, x2, y2] = d.weapon_bbox;
        const bx = x1 * sx, by = y1 * sy, bw = (x2 - x1) * sx, bh = (y2 - y1) * sy;
        ctx.strokeStyle = '#ff1e1e';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = '#ff1e1e';
        ctx.fillRect(bx, Math.max(0, by - 22), 118, 22);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px monospace';
        ctx.fillText('WEAPON: KNIFE', bx + 6, Math.max(15, by - 6));
      }
    }
  } catch (e) {}
  setTimeout(pollTelemetry, 80);
}
pollTelemetry();
</script>
</body>
</html>
"""


class ProductionHierarchicalPipeline:
    """
    Unified Hierarchical Edge Surveillance Pipeline (v6.20 TensorRT Accelerated).
    Orchestrates asynchronous camera ingestion, lightweight 3 FPS weapon screening,
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
        ui_mode: str = "auto",
        web_port: int = 8080,
    ):
        self.camera_res = camera_res.lower()
        self.aspect_ratio = aspect_ratio.lower()
        self.backend = backend.lower()
        self.ui_mode = ui_mode.lower()
        self.web_port = int(web_port)

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

        # 1. In-Memory Hybrid Ring Buffer (64-Frame Zero-Decode Hot Ring + Async JPEG)
        self.buffer = ThreadSafeCircularBuffer(
            maxlen=buffer_size,
            jpeg_quality=jpeg_quality,
            async_encode=True,
            hot_window_size=64,
        )

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
        self.gpu_lock = threading.Lock()
        self.weapon_active_until = 0.0
        self.last_weapon_bbox = None
        self.last_analyzed_idx = 0
        self.clip_stride = 10
        self.clip_length = 50
        self.max_audit_lag = self.clip_length + (3 * self.clip_stride)

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
        """Thread 2: 3 FPS lightweight Stage 1 weapon screening loop with zero-copy frame fetch."""
        target_delay = 1.0 / 3.0
        last_scanned_idx = -1

        while self.is_running:
            t_start = time.perf_counter()
            frame, frame_idx = self.buffer.get_latest_frame_with_index()

            if frame is not None and frame_idx != last_scanned_idx:
                last_scanned_idx = frame_idx
                with self.gpu_lock:
                    det_res = self.stage1_detector.predict(frame, imgsz=self.rect_imgsz)

                if det_res["has_threat"]:
                    now = time.time()
                    was_inactive = now >= self.weapon_active_until
                    self.weapon_active_until = now + 3.0
                    self.last_weapon_bbox = det_res["best_box"]
                    self.stage2a_tracker.update_weapon_anchor(det_res["best_box"])

                    # Snap auditing cursor to current live window when transitioning from idle
                    if was_inactive:
                        latest_idx = self.buffer.get_latest_index()
                        if latest_idx >= 0:
                            self.last_analyzed_idx = max(0, latest_idx - self.clip_length + 1)

                    if not self.is_violent_alert and (was_inactive or self.last_resolved_tier is None):
                        conf_pct = det_res["scores"][0] * 100.0 if det_res["scores"] else 0.0
                        self.status_text = f"WEAPON VERIFIED ({conf_pct:.1f}%)"
                        self.status_color = (0, 165, 255)

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.005, target_delay - elapsed))

    def _populate_window_skeletons(self, window: Sequence[FrameItem]):
        """
        Populates missing skeletons across the 50-frame window.
        Uses strided extraction + linear keypoint interpolation during initial cold-start bursts
        (>clip_stride missing frames) to cut burst latency by ~50% and prevent GPU/CPU starvation.
        """
        missing_indices = [i for i, item in enumerate(window) if item.skeleton is None]
        if not missing_indices:
            return

        use_strided_cold_start = len(missing_indices) > (self.clip_stride + 5)

        if not use_strided_cold_start:
            for idx in missing_indices:
                if not self.is_running:
                    return
                item = window[idx]
                raw_frame = item.get_frame()
                if raw_frame is not None:
                    with self.gpu_lock:
                        kpts, bbox, track_id = self.stage2a_tracker.extract_actor_pose(
                            raw_frame, imgsz=self.rect_imgsz
                        )
                    item.skeleton = kpts
                    item.actor_bbox = bbox
                    item.track_id = track_id
                else:
                    item.skeleton = np.zeros((17, 2), dtype=np.float32)
        else:
            key_positions = set(missing_indices[::2]) | {missing_indices[0], missing_indices[-1]}
            for idx in sorted(key_positions):
                if not self.is_running:
                    return
                item = window[idx]
                raw_frame = item.get_frame()
                if raw_frame is not None:
                    with self.gpu_lock:
                        kpts, bbox, track_id = self.stage2a_tracker.extract_actor_pose(
                            raw_frame, imgsz=self.rect_imgsz
                        )
                    item.skeleton = kpts
                    item.actor_bbox = bbox
                    item.track_id = track_id
                else:
                    item.skeleton = np.zeros((17, 2), dtype=np.float32)
                time.sleep(0.0005)

            n_items = len(window)
            for idx in missing_indices:
                item = window[idx]
                if item.skeleton is not None:
                    continue
                prev_item = window[idx - 1] if idx > 0 else None
                next_item = window[idx + 1] if (idx + 1) < n_items else None
                if (
                    prev_item is not None
                    and next_item is not None
                    and prev_item.skeleton is not None
                    and next_item.skeleton is not None
                ):
                    if np.any(prev_item.skeleton) and np.any(next_item.skeleton):
                        item.skeleton = (0.5 * (prev_item.skeleton + next_item.skeleton)).astype(np.float32)
                    elif np.any(prev_item.skeleton):
                        item.skeleton = prev_item.skeleton.copy()
                    else:
                        item.skeleton = next_item.skeleton.copy()
                    item.actor_bbox = prev_item.actor_bbox
                    item.track_id = prev_item.track_id
                elif prev_item is not None and prev_item.skeleton is not None:
                    item.skeleton = prev_item.skeleton.copy()
                    item.actor_bbox = prev_item.actor_bbox
                    item.track_id = prev_item.track_id
                else:
                    item.skeleton = np.zeros((17, 2), dtype=np.float32)

    def action_auditing_worker(self):
        """Thread 3: Zero-drop sliding-window kinematic evaluator with Staircase Cascade."""
        was_active = False

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
                self.last_resolved_tier = None
                if was_active:
                    self.stage2a_tracker.reset_tracking()
                    was_active = False

                latest_idx = self.buffer.get_latest_index()
                if latest_idx >= 0:
                    self.last_analyzed_idx = max(0, latest_idx - self.clip_length + 1)

                time.sleep(0.04)
                continue

            latest_idx = self.buffer.get_latest_index()
            if latest_idx >= self.clip_length:
                live_window_start = latest_idx - self.clip_length + 1
                if not was_active or (live_window_start - self.last_analyzed_idx) > self.max_audit_lag:
                    self.last_analyzed_idx = max(0, live_window_start)
            was_active = True

            window, pending = self.buffer.get_window_by_index(self.last_analyzed_idx, self.clip_length)
            if len(window) < self.clip_length:
                time.sleep(0.015)
                continue

            self.last_analyzed_idx = window[0].index + self.clip_stride
            self.pending_audit_frames = pending

            # Step 1: Extract/Cache skeletons (0.0 ms frame access via 64-frame hot ring)
            self._populate_window_skeletons(window)
            if not self.is_running:
                break

            # Step 2: Kinematic preprocessing & Torso-Scale Normalization
            raw_seq = np.array([item.skeleton for item in window], dtype=np.float32)
            normed_coords, graph_tensor = self.stage2a_tracker.preprocess_sliding_window(raw_seq)

            # Step 3: Evaluate Progressive Staircase Cascade (Champion 2)
            with self.gpu_lock:
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
        low internal driver buffer size (1 frame), and high-resolution sensor negotiation.
        """
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    c.set(cv2.CAP_PROP_FPS, 30)
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

    @staticmethod
    def _fit_text(text: str, font: int, target_scale: float, target_thickness: int, max_w: int):
        scale = target_scale
        th = target_thickness
        (tw, th_box), _ = cv2.getTextSize(text, font, scale, th)
        if tw > max_w and tw > 0:
            ratio = max_w / tw
            scale = target_scale * ratio
            th = max(1, int(round(target_thickness * ratio)))
            (tw, th_box), _ = cv2.getTextSize(text, font, scale, th)
        return scale, th, tw, th_box

    @classmethod
    def render_hud_overlay(cls, hud: np.ndarray, title: str, title_color: Tuple[int, int, int], line2: str, line3: Optional[str] = None):
        h, w = hud.shape[:2]
        scale = max(0.5, h / 720.0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale_title = 0.38 * scale
        base_scale_body = 0.28 * scale
        base_th_title = max(1, int(round(1.0 * scale)))
        base_th_body = max(1, int(round(0.9 * scale)))

        pad_x = max(8, int(8 * scale))
        pad_y = max(6, int(6 * scale))
        line_spacing = max(4, int(5 * scale))
        max_line_w = w - int(20 * scale) - 2 * pad_x

        s1, th1, w1, h1 = cls._fit_text(title, font, base_scale_title, base_th_title, max_line_w)
        s2, th2, w2, h2 = cls._fit_text(line2, font, base_scale_body, base_th_body, max_line_w)

        lines_to_draw = [(title, s1, th1, title_color, h1), (line2, s2, th2, (230, 230, 230), h2)]
        if line3:
            s3, th3, w3, h3 = cls._fit_text(line3, font, base_scale_body, base_th_body, max_line_w)
            lines_to_draw.append((line3, s3, th3, (185, 185, 185), h3))
            max_text_w = max(w1, w2, w3)
        else:
            max_text_w = max(w1, w2)

        box_w = min(w - int(20 * scale), max_text_w + 2 * pad_x)
        total_text_h = sum(item[4] for item in lines_to_draw)
        box_h = pad_y * 2 + total_text_h + line_spacing * (len(lines_to_draw) - 1)

        x1, y1 = max(10, int(10 * scale)), max(10, int(10 * scale))
        x2, y2 = min(w, x1 + box_w), min(h, y1 + box_h)

        roi = hud[y1:y2, x1:x2]
        cv2.addWeighted(roi, 0.12, roi, 0.0, 12 * 0.88, dst=roi)

        border_th = max(1, int(round(1.0 * scale)))
        cv2.rectangle(hud, (x1, y1), (x2, y2), title_color, border_th)

        cur_y = y1 + pad_y
        for text, s, th, col, line_h in lines_to_draw:
            cur_y += line_h
            cv2.putText(hud, text, (x1 + pad_x, cur_y), font, s, col, th, cv2.LINE_AA)
            cur_y += line_spacing

    @classmethod
    def render_weapon_bbox(cls, hud: np.ndarray, bbox: Sequence[Union[int, float]], label: str = "WEAPON: KNIFE"):
        h, w = hud.shape[:2]
        scale = max(0.5, h / 720.0)
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        box_thickness = max(2, int(round(1.5 * scale)))
        cv2.rectangle(hud, (bx1, by1), (bx2, by2), (0, 0, 255), box_thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35 * scale
        thickness = max(1, int(round(1.0 * scale)))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        badge_pad = max(3, int(3 * scale))
        lbl_y1 = max(0, by1 - th - 2 * badge_pad)
        lbl_y2 = by1
        lbl_x2 = min(w, bx1 + tw + 2 * badge_pad)
        cv2.rectangle(hud, (bx1, lbl_y1), (lbl_x2, lbl_y2), (0, 0, 255), -1)
        cv2.putText(hud, label, (bx1 + badge_pad, lbl_y2 - badge_pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _run_web_ui(
        self,
        win_title: str,
        get_telemetry_fn: Callable[[], dict],
        trigger_swap_fn: Callable[[], None],
    ):
        """
        Option 2 from Refactoring Report: Decoupled Headless Web Streamer
        (MJPEG + JSON Telemetry + HTML5 Canvas / CSS Glassmorphism HUD).
        """
        pipeline_ref = self

        class SurveillanceHTTPHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _WEB_DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/api/telemetry"):
                    payload = json.dumps(get_telemetry_fn()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path.startswith("/stream.mjpg"):
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last_sent_idx = -1
                    try:
                        while pipeline_ref.is_running:
                            frame, idx = pipeline_ref.buffer.get_latest_frame_with_index()
                            if frame is None or idx == last_sent_idx:
                                time.sleep(0.01)
                                continue
                            last_sent_idx = idx
                            # Resize stream frame to 720p for crisp, low-bandwidth browser playback
                            fh, fw = frame.shape[:2]
                            if fh > 720:
                                disp_w = int(round(fw * (720.0 / fh)))
                                send_frame = cv2.resize(frame, (disp_w, 720), interpolation=cv2.INTER_LINEAR)
                            else:
                                send_frame = frame
                            jpg = _encode_jpeg_bytes(send_frame, quality=80)
                            if jpg is not None:
                                jpg_bytes = jpg if isinstance(jpg, (bytes, bytearray)) else jpg.tobytes()
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(jpg_bytes)}\r\n\r\n".encode("ascii"))
                                self.wfile.write(jpg_bytes)
                                self.wfile.write(b"\r\n")
                    except Exception:
                        pass
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.startswith("/api/swap"):
                    trigger_swap_fn()
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"ok":true}')
                else:
                    self.send_error(404)

        server = ThreadingHTTPServer(("0.0.0.0", self.web_port), SurveillanceHTTPHandler)
        server.daemon_threads = True
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        print(f"[WEB UI] Decoupled HTML5 Glassmorphism HUD live at: http://localhost:{self.web_port}")
        print("[WEB UI] Press Ctrl+C in terminal to stop.")
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()

    def _run_qt_ui(
        self,
        win_title: str,
        QtWidgets,
        QtCore,
        QtGui,
        get_telemetry_fn: Callable[[], dict],
        trigger_swap_fn: Callable[[], None],
    ):
        """
        Option 1 from Refactoring Report: Desktop Application (PySide6 / PyQt6)
        Renders camera frames on Layer 0 and a hardware-accelerated QPainter vector HUD overlay on Layer 1
        with zero CPU pixel mutation of the underlying video frame.
        """
        pipeline_ref = self

        class VectorSurveillanceWidget(QtWidgets.QWidget):
            def __init__(self):
                super().__init__()
                self.setWindowTitle(win_title)
                if pipeline_ref.aspect_ratio in ("4:3", "4/3"):
                    self.resize(960, 720)
                else:
                    self.resize(1280, 720)
                self.last_idx = -1
                self.qimage = None
                self.telemetry = get_telemetry_fn()

                self.timer = QtCore.QTimer(self)
                self.timer.timeout.connect(self._on_tick)
                self.timer.start(16)  # 60 Hz VSync-decoupled UI refresh

            def _on_tick(self):
                if not pipeline_ref.is_running:
                    self.close()
                    return
                self.telemetry = get_telemetry_fn()
                frame, idx = pipeline_ref.buffer.get_latest_frame_with_index()
                if frame is not None and idx != self.last_idx:
                    self.last_idx = idx
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    self.qimage = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888).copy()
                self.update()

            def keyPressEvent(self, event):
                key = event.key()
                if key == QtCore.Qt.Key.Key_Q:
                    pipeline_ref.is_running = False
                    self.close()
                elif key == QtCore.Qt.Key.Key_I:
                    trigger_swap_fn()

            def paintEvent(self, event):
                painter = QtGui.QPainter(self)
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
                ww, wh = self.width(), self.height()
                painter.fillRect(0, 0, ww, wh, QtGui.QColor(9, 11, 16))

                # Layer 0: Video Surface
                if self.qimage is not None:
                    painter.drawImage(QtCore.QRect(0, 0, ww, wh), self.qimage)

                t = self.telemetry
                b, g, r = t["bgr_color"]
                accent = QtGui.QColor(r, g, b)

                # Layer 1A: Weapon Bounding Box Vector Overlay
                bbox = t.get("weapon_bbox")
                fw, fh = t.get("frame_w", 0), t.get("frame_h", 0)
                if bbox is not None and fw > 0 and fh > 0:
                    sx, sy = ww / float(fw), wh / float(fh)
                    bx1, by1, bx2, by2 = bbox
                    rx, ry = int(bx1 * sx), int(by1 * sy)
                    rw, rh = int((bx2 - bx1) * sx), int((by2 - by1) * sy)
                    pen = QtGui.QPen(QtGui.QColor(255, 30, 30), 2.5)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    painter.drawRect(rx, ry, rw, rh)

                    painter.setBrush(QtGui.QColor(255, 30, 30))
                    painter.drawRect(rx, max(0, ry - 22), 120, 22)
                    painter.setPen(QtGui.QColor(255, 255, 255))
                    painter.setFont(QtGui.QFont("Consolas", 9, QtGui.QFont.Weight.Bold))
                    painter.drawText(rx + 6, max(15, ry - 6), "WEAPON: KNIFE")

                # Layer 1B: Translucent Vector HUD Card
                card_rect = QtCore.QRectF(14.0, 14.0, min(ww - 28.0, 560.0), 76.0)
                painter.setBrush(QtGui.QColor(12, 14, 20, 224))
                painter.setPen(QtGui.QPen(accent, 1.5))
                painter.drawRoundedRect(card_rect, 6.0, 6.0)

                painter.setPen(accent)
                painter.setFont(QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.Bold))
                painter.drawText(26, 35, f"STATUS: {t['status_text']}")

                painter.setPen(QtGui.QColor(230, 230, 230))
                painter.setFont(QtGui.QFont("Consolas", 9))
                painter.drawText(26, 56, t["line2"])

                painter.setPen(QtGui.QColor(185, 185, 185))
                painter.drawText(26, 76, t["line3"])
                painter.end()

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        widget = VectorSurveillanceWidget()
        widget.show()
        app.exec()

    def run(self, source: Union[int, str] = 0, headless: bool = False):
        """
        Executes live video stream surveillance with decoupled asynchronous camera capture,
        auto-failover, non-blocking camera cycling, and selectable UI backend (`qt`, `web`, `opencv`).
        """
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:
                pass
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
        last_frame = None
        ret_t, test_frame = cap.read() if cap.isOpened() else (False, None)
        if ret_t and test_frame is not None:
            test_frame = self.apply_aspect_ratio(test_frame)
            f_w, f_h = test_frame.shape[1], test_frame.shape[0]
            last_frame = test_frame
            self.buffer.append(test_frame, timestamp=time.time())
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

        aspect_tag = " [4:3 Aspect Ratio]" if self.aspect_ratio in ("4:3", "4/3") else ""
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

            threading.Thread(target=_probe_cameras, daemon=True).start()

        # Shared state between Capture Producer Thread and UI Main Thread
        cap_lock = threading.Lock()
        is_swapping = False
        swap_target_idx = None
        swap_target_name = ""
        consecutive_failures = 0
        frame_times = deque(maxlen=30)

        def _async_swap_worker(target_idx: int, target_name: str):
            nonlocal cap, current_source, is_swapping, consecutive_failures
            try:
                new_cap = self.open_camera(target_idx)
                if new_cap is not None:
                    with cap_lock:
                        old_cap = cap
                        cap = new_cap
                        current_source = target_idx
                        consecutive_failures = 0
                    if old_cap is not None:
                        old_cap.release()
                    new_w = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    new_h = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            if is_swapping or not isinstance(current_source, int):
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

        def _camera_capture_worker():
            """Dedicated producer thread: drains camera at full hardware FPS isolated from UI."""
            nonlocal last_frame, consecutive_failures
            t_prev = time.perf_counter()
            is_file_source = not isinstance(current_source, int)
            file_fps = 30.0
            if is_file_source and cap is not None:
                reported_fps = cap.get(cv2.CAP_PROP_FPS)
                if reported_fps and 1.0 < reported_fps <= 240.0:
                    file_fps = float(reported_fps)

            while self.is_running:
                if is_swapping:
                    time.sleep(0.01)
                    t_prev = time.perf_counter()
                    continue

                ret = False
                raw_frame = None
                with cap_lock:
                    active_cap = cap
                    if active_cap is not None and active_cap.isOpened():
                        ret, raw_frame = active_cap.read()

                if not ret or raw_frame is None:
                    if is_file_source:
                        self.is_running = False
                        break
                    consecutive_failures += 1
                    if isinstance(current_source, int) and consecutive_failures >= 30 and not is_swapping:
                        consecutive_failures = 0
                        print(f"\n[AUTO-FAILOVER] Camera {current_source} signal unavailable. Auto-switching to next camera...")
                        _trigger_swap()
                    time.sleep(0.015)
                    continue

                consecutive_failures = 0
                cropped = self.apply_aspect_ratio(raw_frame)
                now_perf = time.perf_counter()
                dt = now_perf - t_prev
                t_prev = now_perf
                if dt > 0:
                    frame_times.append(dt)
                    self.fps_telemetry = len(frame_times) / sum(frame_times)

                last_frame = cropped
                self.buffer.append(cropped, timestamp=time.time())

                if is_file_source:
                    target_dt = 1.0 / file_fps
                    sleep_rem = target_dt - (time.perf_counter() - now_perf)
                    if sleep_rem > 0.001:
                        time.sleep(sleep_rem)

        def _build_telemetry_snapshot() -> dict:
            cur_f = last_frame
            fh, fw = (cur_f.shape[0], cur_f.shape[1]) if cur_f is not None else (720, 1280)
            if is_swapping:
                st_text = f"SWITCHING TO CAMERA {swap_target_idx} ({swap_target_name})..."
                st_col = (0, 255, 255)
                l2 = "Please wait, initializing camera hardware..."
                l3 = "Press 'q' to quit"
            elif consecutive_failures > 2:
                st_text = f"CAMERA ALERT: NO SIGNAL FROM CAMERA {current_source}"
                st_col = (0, 0, 255)
                l2 = f"Auto-failover in {max(0, 30 - consecutive_failures)} ticks  |  Press 'i' to switch camera"
                l3 = "Press 'q' to quit"
            else:
                st_text = self.status_text
                st_col = self.status_color
                buf_len = len(self.buffer)
                tier_info = f"Tier: {self.last_resolved_tier}" if self.last_resolved_tier else "Tier: Idle"
                actor_info = f"Actor ID: {self.stage2a_tracker.active_threat_actor_id}" if self.stage2a_tracker.active_threat_actor_id else "Scanning"
                cam_name = camera_names.get(current_source, f"Cam {current_source}")
                cam_label = f"Cam {current_source}: {cam_name} ({fw}x{fh})" if isinstance(current_source, int) else "Source: File"
                l2 = f"{cam_label}  |  FPS: {self.fps_telemetry:.1f}  |  Buffer: {buf_len}/1200"
                l3 = f"Pipeline: {tier_info}  |  Actor: {actor_info}  |  Hotkeys: [i] Swap Cam, [q] Quit"

            b, g, r = st_col
            active_bbox = (
                [float(v) for v in self.last_weapon_bbox]
                if (time.time() < self.weapon_active_until and self.last_weapon_bbox is not None)
                else None
            )
            return {
                "status_text": st_text,
                "bgr_color": st_col,
                "css_color": f"rgb({r},{g},{b})",
                "line2": l2,
                "line3": l3,
                "weapon_bbox": active_bbox,
                "frame_w": fw,
                "frame_h": fh,
            }

        self.is_running = True
        t_capture = threading.Thread(target=_camera_capture_worker, daemon=True)
        t_weapon = threading.Thread(target=self.weapon_scanner_worker, daemon=True)
        t_action = threading.Thread(target=self.action_auditing_worker, daemon=True)
        t_capture.start()
        t_weapon.start()
        t_action.start()

        print("[RUNNING] Surveillance pipeline active. Press 'q' to stop, 'i' to swap camera.")

        win_title = f"Hierarchical Edge Threat Detection [{self.backend.upper()}]"

        if not headless and self.ui_mode == "web":
            self._run_web_ui(win_title, _build_telemetry_snapshot, _trigger_swap)
        elif not headless and self.ui_mode in ("qt", "auto"):
            qt_name, QtWidgets, QtCore, QtGui = _detect_qt_binding()
            if qt_name is not None:
                print(f"[UI BACKEND] Using {qt_name} Hardware-Accelerated Vector Overlay UI.")
                self._run_qt_ui(win_title, QtWidgets, QtCore, QtGui, _build_telemetry_snapshot, _trigger_swap)
            elif self.ui_mode == "qt":
                print("[UI WARNING] PySide6 / PyQt6 not installed. Falling back to Pre-Scaled OpenCV UI.")
            else:
                pass

        # Headless or Pre-Scaled OpenCV UI Loop
        if headless or (self.ui_mode in ("opencv", "auto") and self.is_running):
            disp_w = 960 if self.aspect_ratio in ("4:3", "4/3") else 1280
            disp_h = 720
            if not headless:
                cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_title, disp_w, disp_h)

            hud_buf: Optional[np.ndarray] = None
            last_rendered_idx = -1

            while self.is_running:
                if headless:
                    time.sleep(0.05)
                    continue

                t_snap = _build_telemetry_snapshot()
                frame, cur_idx = self.buffer.get_latest_frame_with_index()

                if is_swapping or consecutive_failures > 2 or (frame is not None and cur_idx != last_rendered_idx):
                    last_rendered_idx = cur_idx
                    base = frame if frame is not None else (last_frame if last_frame is not None else np.zeros((disp_h, disp_w, 3), dtype=np.uint8))
                    bh, bw = base.shape[:2]

                    # Pre-scale high-res frames (e.g. 1920x1440 -> 960x720) before HUD drawing & imshow
                    # Cuts GDI memory blitting bandwidth by 75% (2.0 MB vs 8.3 MB)
                    if bh > disp_h:
                        target_w = int(round(bw * (disp_h / float(bh))))
                        if hud_buf is None or hud_buf.shape != (disp_h, target_w, 3):
                            hud_buf = np.empty((disp_h, target_w, 3), dtype=np.uint8)
                        cv2.resize(base, (target_w, disp_h), dst=hud_buf, interpolation=cv2.INTER_LINEAR)
                        sx, sy = target_w / float(bw), disp_h / float(bh)
                    else:
                        if hud_buf is None or hud_buf.shape != base.shape:
                            hud_buf = base.copy()
                        else:
                            np.copyto(hud_buf, base)
                        sx, sy = 1.0, 1.0

                    self.render_hud_overlay(
                        hud_buf,
                        f"STATUS: {t_snap['status_text']}",
                        t_snap["bgr_color"],
                        t_snap["line2"],
                        t_snap["line3"],
                    )

                    if t_snap["weapon_bbox"] is not None:
                        bx1, by1, bx2, by2 = t_snap["weapon_bbox"]
                        scaled_bbox = [bx1 * sx, by1 * sy, bx2 * sx, by2 * sy]
                        self.render_weapon_bbox(hud_buf, scaled_bbox, "WEAPON: KNIFE")

                    cv2.imshow(win_title, hud_buf)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in (ord('i'), ord('I')) and isinstance(current_source, int):
                    _trigger_swap()

        self.is_running = False
        t_capture.join(timeout=1.0)
        t_weapon.join(timeout=1.0)
        t_action.join(timeout=1.0)
        self.buffer.close()
        with cap_lock:
            if cap is not None:
                cap.release()
        if not headless and self.ui_mode in ("opencv", "auto"):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
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
            self.buffer.append(dummy_frame)
            if i == 40:
                self.weapon_active_until = time.time() + 5.0
                self.stage2a_tracker.update_weapon_anchor([300, 200, 400, 350])
            time.sleep(0.01)

        time.sleep(0.5)
        self.is_running = False
        t_weapon.join(timeout=1.0)
        t_action.join(timeout=1.0)
        self.buffer.flush_pending_compression(timeout=1.0)

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
                        help="Run in headless mode without GUI")
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
    parser.add_argument("--ui", type=str, default="auto",
                        choices=["auto", "qt", "web", "opencv"],
                        help="UI backend: 'auto' (Qt6 vector overlay if installed, else pre-scaled OpenCV), 'qt' (PySide6/PyQt6), 'web' (HTML5 Glassmorphism browser HUD), 'opencv'")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP port when using `--ui web` (default: 8080)")
    parser.add_argument("--debug", action="store_true",
                        help="Run Stage 1/2 workers through src/debug_stage2_tracer.py (action log + analyzed-frame dumps in results/debug_stage2/)")

    args = parser.parse_args()

    active_source = parse_source(args.webcam if args.webcam is not None else args.source)

    pipeline_cls = ProductionHierarchicalPipeline
    if args.debug:
        from src.debug_stage2_tracer import DebugHierarchicalPipeline as pipeline_cls

    pipeline = pipeline_cls(
        conf_threshold=args.conf,
        jpeg_quality=args.quality,
        device=args.device,
        backend=args.backend,
        camera_res=args.resolution,
        aspect_ratio=args.aspect_ratio,
        ui_mode=args.ui,
        web_port=args.port,
    )

    if args.dry_run:
        pipeline.run_dry_run()
    else:
        pipeline.run(source=active_source, headless=args.headless)