"""
src/debug_stage2_tracer.py

Stage 2 Debug Tracer for the Production Pipeline (v6.20).

Purpose
-------
Answers "what does the pipeline actually do after Stage 1 sees a knife, and what frames does
Stage 2 really analyze?" without touching any production module.

Activation (the only hook lives in `inference_production_pipeline.py`):
    python src/inference_production_pipeline.py --debug [usual flags]
    run_webcam_tensorrt.bat --debug
    run_webcam_pytorch.bat  --debug

How it works
------------
`DebugHierarchicalPipeline` subclasses `ProductionHierarchicalPipeline` and overrides ONLY:
  * `weapon_scanner_worker`        (Stage 1 loop, 3 FPS)
  * `action_auditing_worker`       (Stage 2 loop)
  * `_populate_window_skeletons`   (Stage 2a pose extraction for a 50-frame window)
Each override is an instrumented copy of the v6.20 production logic: the same thresholds, the same
cursor rules, the same calls into Stage 1 / 2a / 2b, the same status updates. Nothing in the
decision path is altered; the tracer only observes and records.

Two deliberate, clearly-labelled differences from production:
  1. Exceptions inside a worker loop are logged with a full traceback and the loop continues.
     (In production an exception silently kills that daemon thread and Stage 2 stays dead
     for the rest of the session.) Every such event is logged as `WORKER_EXCEPTION`.
  2. Optional diagnostic PROBE per clip (never used for the decision): evaluates all three
     cascade tiers on (a) the exact production input and (b) a "training-style" heatmap built
     from raw pixel skeletons with a single normalization, like `src/dataset.py` does.

Output: results/debug_stage2/session_<YYYYmmdd_HHMMSS>/
  trace.log             human-readable timeline of every action
  events.jsonl          one JSON object per event (machine-readable)
  stage1_detections/    every frame where Stage 1 reported a weapon (annotated)
  frames/               every frame Stage 2 analyzed (skeleton, actor box, weapon box, source tag)
  clips/clip_NNNN_fA-B/ contact_sheet.jpg, heatmap_compare.png, skeleton_raw.npy,
                        coords_normed.npy, clip.json
  summary.json          end-of-session statistics

Environment switches (optional):
  HTD_DEBUG_DIR=<path>          output root (default: results/debug_stage2)
  HTD_DEBUG_SAVE_FRAMES=0       do not write frames/ (contact sheets still written)
  HTD_DEBUG_PROBE=0             disable the per-clip all-tier probe
  HTD_DEBUG_NEAR_MISS=0         disable the Stage 1 near-miss score (sub-threshold) logging
  HTD_DEBUG_NEAR_MISS_CONF=0.10 lowest Stage 1 score recorded as a near-miss
"""

import os
import sys
import json
import time
import queue
import threading
import traceback
from collections import OrderedDict, Counter
from typing import Optional, List, Dict, Any, Sequence

import cv2
import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.inference_production_pipeline import ProductionHierarchicalPipeline
from src.pipeline_buffer import FrameItem
from src.skeleton_utils import interpolate_missing_joints, smooth_kinematics
from src.stage2b_staircase import rasterize_heatmap_from_coords


# COCO-17 skeleton edges (same keypoint order as YOLO26s-Pose)
_COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]

# Frame source tags used in logs and images
_SRC_DESC = {
    "C": "cached (pose already extracted by an earlier overlapping window)",
    "E": "extracted (YOLO26s-Pose run on this frame)",
    "I": "interpolated (cold-start strided fill, no pose run on this frame)",
    "N": "no image (frame could not be decoded -> zero skeleton)",
}
_SRC_COLOR = {"C": (200, 200, 200), "E": (0, 255, 0), "I": (0, 200, 255), "N": (0, 0, 255)}


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _thread_label() -> str:
    name = threading.current_thread().name
    if "weapon_scanner" in name:
        return "STAGE1"
    if "action_auditing" in name:
        return "STAGE2"
    if name == "MainThread":
        return "MAIN"
    return name[:12]


def _fmt_p(p: Optional[float]) -> str:
    return "  -  " if p is None else f"{p:.3f}"


def _fmt_box(b: Optional[Sequence[float]]) -> str:
    if b is None:
        return "None"
    return "[" + ",".join(f"{float(v):.0f}" for v in b) + "]"


# ----------------------------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------------------------

def _put_label(img: np.ndarray, text: str, org, color=(255, 255, 255), scale: float = 0.45):
    """Single-pass text on a dark box (readable on any background, any OpenCV font backend)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base + 1), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, 1, cv2.LINE_AA)
    return th + base + 4


def _annotate_frame(
    frame: np.ndarray,
    max_w: int,
    kpts: Optional[np.ndarray] = None,
    actor_bbox: Optional[Sequence[float]] = None,
    weapon_bbox: Optional[Sequence[float]] = None,
    weapon_boxes: Optional[List[Sequence[float]]] = None,
    lines: Optional[List[str]] = None,
    accent=(0, 255, 0),
) -> np.ndarray:
    """Downscales first (cheap), then draws skeleton / boxes / text in scaled coordinates."""
    h, w = frame.shape[:2]
    s = min(1.0, max_w / float(w)) if w > 0 else 1.0
    if s < 1.0:
        img = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    else:
        img = frame.copy()

    def sc(v):
        return int(round(float(v) * s))

    if actor_bbox is not None:
        x1, y1, x2, y2 = actor_bbox
        cv2.rectangle(img, (sc(x1), sc(y1)), (sc(x2), sc(y2)), (255, 200, 0), 2)

    for b in (weapon_boxes or []):
        x1, y1, x2, y2 = b
        cv2.rectangle(img, (sc(x1), sc(y1)), (sc(x2), sc(y2)), (0, 0, 180), 1)
    if weapon_bbox is not None:
        x1, y1, x2, y2 = weapon_bbox
        cv2.rectangle(img, (sc(x1), sc(y1)), (sc(x2), sc(y2)), (0, 0, 255), 2)

    if kpts is not None and np.any(kpts):
        pts = [(sc(x), sc(y)) if (abs(x) + abs(y)) > 1e-6 else None for x, y in kpts]
        for a, b in _COCO_EDGES:
            if pts[a] is not None and pts[b] is not None:
                cv2.line(img, pts[a], pts[b], accent, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if p is not None:
                col = (0, 0, 255) if i in (9, 10) else (255, 255, 255)  # wrists in red
                cv2.circle(img, p, 3, col, -1, cv2.LINE_AA)

    if lines:
        y = 16
        for text in lines:
            y += _put_label(img, text, (6, y), accent) - 2
    return img


def _heatmap_projection(hm: torch.Tensor, title: str, size: int = 280) -> np.ndarray:
    """Max-projection over joints and time of a (1,17,T,56,56) heatmap -> colored image."""
    proj = hm[0].amax(dim=(0, 1)).detach().float().cpu().numpy()  # (56, 56)
    mx = float(proj.max())
    norm = (proj / mx * 255.0).astype(np.uint8) if mx > 0 else np.zeros_like(proj, dtype=np.uint8)
    img = cv2.applyColorMap(cv2.resize(norm, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
    if mx <= 0:
        img[:] = 30
    canvas = np.full((size + 24, size, 3), 25, dtype=np.uint8)
    canvas[24:] = img
    cv2.putText(canvas, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _heatmap_stats(hm: torch.Tensor) -> Dict[str, Any]:
    """
    Per-frame body footprint inside the 56x56 PoseConv3D canvas.
    A healthy torso-normalized body spans roughly 15-40 cells tall; a value near the size of a
    single Gaussian blob (~5-8 cells) means all 17 joints collapsed onto the same spot.
    """
    v = hm[0].amax(dim=0).detach().float().cpu()  # (T, H, W)
    T, H, W = v.shape
    peak = v.amax(dim=(1, 2))
    active = peak > 1e-3
    out = {"active_frames": int(active.sum()), "total_frames": int(T), "max_value": float(v.max())}
    if not bool(active.any()):
        out.update({"mean_body_w_cells": 0.0, "mean_body_h_cells": 0.0})
        return out
    mask = v >= (0.1 * peak.clamp_min(1e-6))[:, None, None]
    rows = mask.any(dim=2)
    cols = mask.any(dim=1)

    def extent(b: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.arange(n)
        first = torch.where(b, idx, torch.full_like(idx, n)).amin(dim=1)
        last = torch.where(b, idx, torch.full_like(idx, -1)).amax(dim=1)
        return (last - first + 1).clamp_min(0).float()

    out["mean_body_h_cells"] = float(extent(rows, H)[active].mean())
    out["mean_body_w_cells"] = float(extent(cols, W)[active].mean())
    return out


def _infer_actor_strategy(pre_id, pre_has_bbox, pre_has_weapon, n_persons, track_ids, chosen_id) -> str:
    """Mirrors the branch order of Stage2aPoseTracker.extract_actor_pose (read-only inference)."""
    if n_persons == 0:
        return "NO_PERSON -> zero skeleton"
    if pre_id is not None and track_ids is not None:
        if pre_id in track_ids:
            return f"S1 keep ByteTrack ID {pre_id}"
        if pre_has_bbox:
            if chosen_id != pre_id and chosen_id is not None and chosen_id != (track_ids[0] if track_ids else None):
                return f"S1b IoU re-lock {pre_id}->{chosen_id}"
            return f"S1b ID {pre_id} lost (IoU re-lock / fallback person[0] -> {chosen_id})"
        return f"S1 ID {pre_id} lost, no bbox -> person[0]"
    if pre_has_weapon:
        return f"S2 nearest-to-weapon -> ID {chosen_id}"
    return f"S3 largest box -> ID {chosen_id}"


def _simulate_cascade(p1, p2, p3, eng) -> str:
    """Applies the Stage 2b gating rule to a set of probabilities (used only for probe reporting)."""
    if p1 is None:
        return "n/a"
    if p1 < eng.p_low1:
        return f"T1 discard (p1<{eng.p_low1:.2f})"
    if p1 >= eng.p_high1:
        return f"T1 ALARM (p1>={eng.p_high1:.2f})"
    if p2 < eng.p_low2:
        return f"T2 discard (p2<{eng.p_low2:.2f})"
    if p2 >= eng.p_high2:
        return f"T2 ALARM (p2>={eng.p_high2:.2f})"
    return f"T3 {'ALARM' if p3 >= eng.tau3 else 'clear'} (p3 vs tau {eng.tau3:.2f})"


# ----------------------------------------------------------------------------------------------
# Trace writer (log files + asynchronous image writer so disk I/O never stalls the pipeline)
# ----------------------------------------------------------------------------------------------

class DebugTraceWriter:
    def __init__(self, root_dir: Optional[str] = None, save_frames: bool = True):
        root = root_dir or os.environ.get("HTD_DEBUG_DIR") or os.path.join(REPO_DIR, "results", "debug_stage2")
        os.makedirs(root, exist_ok=True)
        gi = os.path.join(root, ".gitignore")
        if not os.path.exists(gi):
            with open(gi, "w", encoding="utf-8") as f:
                f.write("# Debug sessions are large local artifacts; keep them out of git.\n*\n!.gitignore\n")

        self.session_dir = os.path.join(root, "session_" + time.strftime("%Y%m%d_%H%M%S"))
        self.det_dir = os.path.join(self.session_dir, "stage1_detections")
        self.frames_dir = os.path.join(self.session_dir, "frames")
        self.clips_dir = os.path.join(self.session_dir, "clips")
        for d in (self.session_dir, self.det_dir, self.clips_dir):
            os.makedirs(d, exist_ok=True)
        self.save_frames = save_frames
        if save_frames:
            os.makedirs(self.frames_dir, exist_ok=True)

        self.t0 = time.time()
        self._lock = threading.Lock()
        self._log_f = open(os.path.join(self.session_dir, "trace.log"), "w", encoding="utf-8", buffering=1)
        self._jsonl_f = open(os.path.join(self.session_dir, "events.jsonl"), "w", encoding="utf-8", buffering=1)
        self._throttle: Dict[str, float] = {}
        self.closed = False

        # Statistics for summary.json
        self.counters: Counter = Counter()
        self.stats: Dict[str, list] = {
            "frame_age_ms_oldest": [], "frame_age_ms_newest": [], "pose_ms_per_window": [],
            "clip_total_ms": [], "valid_joint_ratio": [], "p1_prod": [], "p1_train_style": [],
            "hm_body_h_prod": [], "hm_body_h_train": [],
        }
        self.tier_hist: Counter = Counter()
        self.decision_hist: Counter = Counter()

        # Async image writer
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=3000)
        self._thumbs: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._saved_frames: set = set()
        self._writer = threading.Thread(target=self._writer_loop, name="debug-writer", daemon=True)
        self._writer.start()

    # ---------------- text / json logging ----------------
    def event(self, kind: str, msg: str = "", console: bool = False, **fields):
        now = time.time()
        rel = now - self.t0
        stamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
        thread = _thread_label()
        line = f"{stamp} +{rel:8.3f}s [{thread:<6}] {kind:<18} {msg}"
        rec = {"t": round(now, 4), "rel_s": round(rel, 4), "thread": thread, "event": kind, "msg": msg}
        rec.update(_jsonable(fields))
        with self._lock:
            if self.closed:
                return
            self.counters["event:" + kind] += 1
            self._log_f.write(line + "\n")
            self._jsonl_f.write(json.dumps(rec, default=str) + "\n")
        if console:
            print(f"[DBG] {kind}: {msg}", flush=True)

    def throttled(self, key: str, interval_s: float, kind: str, msg: str = "", **fields):
        now = time.time()
        if now - self._throttle.get(key, 0.0) >= interval_s:
            self._throttle[key] = now
            self.event(kind, msg, **fields)

    def exception(self, where: str, exc: BaseException):
        tb = traceback.format_exc()
        self.counters["exceptions"] += 1
        self.event(
            "WORKER_EXCEPTION",
            f"{where}: {type(exc).__name__}: {exc}  "
            f"(in PRODUCTION this exception would have killed the '{where}' thread permanently)",
            console=True, where=where, traceback=tb,
        )
        with self._lock:
            if not self.closed:
                self._log_f.write(tb + "\n")
        print(tb, flush=True)

    # ---------------- async image jobs ----------------
    def _put(self, job: tuple):
        try:
            self._q.put_nowait(job)
        except queue.Full:
            self.counters["dropped_image_jobs"] += 1

    def queue_detection_image(self, n: int, frame_idx: int, frame: np.ndarray, det: Dict, lines: List[str], trigger: bool):
        self._put(("det", n, frame_idx, frame, det, lines, trigger))

    def queue_frame_image(self, frame_idx: int, frame: Optional[np.ndarray], info: Dict):
        if frame_idx in self._saved_frames:
            return
        self._saved_frames.add(frame_idx)
        if len(self._saved_frames) > 20000:
            self._saved_frames.clear()
        self._put(("frame", frame_idx, frame, info))

    def queue_clip_dump(self, clip_dir: str, indices: List[int], tags: List[str], header: List[str],
                        arrays: Dict[str, np.ndarray], meta: Dict, hm_imgs: List[np.ndarray]):
        self._put(("clip", clip_dir, indices, tags, header, arrays, meta, hm_imgs))

    def _writer_loop(self):
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                break
            try:
                kind = job[0]
                if kind == "det":
                    _, n, frame_idx, frame, det, lines, trigger = job
                    img = _annotate_frame(
                        frame, 960, weapon_bbox=det.get("best_box"), weapon_boxes=det.get("boxes"),
                        lines=lines, accent=(0, 0, 255) if trigger else (0, 165, 255),
                    )
                    tag = "TRIGGER_" if trigger else ""
                    cv2.imwrite(os.path.join(self.det_dir, f"{tag}det_{n:05d}_f{frame_idx:07d}.jpg"), img)
                elif kind == "frame":
                    _, frame_idx, frame, info = job
                    if frame is None:
                        img = np.zeros((360, 480, 3), dtype=np.uint8)
                        cv2.putText(img, f"f{frame_idx}: NO IMAGE", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        img = _annotate_frame(
                            frame, 640, kpts=info.get("kpts"), actor_bbox=info.get("actor_bbox"),
                            weapon_bbox=info.get("weapon_bbox"), lines=info.get("lines"),
                            accent=_SRC_COLOR.get(info.get("src", "E"), (0, 255, 0)),
                        )
                    if self.save_frames:
                        cv2.imwrite(os.path.join(self.frames_dir, f"f{frame_idx:07d}.jpg"), img)
                    h, w = img.shape[:2]
                    tw = 192
                    th = max(1, int(round(h * tw / float(w))))
                    self._thumbs[frame_idx] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                    while len(self._thumbs) > 600:
                        self._thumbs.popitem(last=False)
                elif kind == "clip":
                    _, clip_dir, indices, tags, header, arrays, meta, hm_imgs = job
                    os.makedirs(clip_dir, exist_ok=True)
                    for name, arr in arrays.items():
                        np.save(os.path.join(clip_dir, f"{name}.npy"), arr)
                    with open(os.path.join(clip_dir, "clip.json"), "w", encoding="utf-8") as f:
                        json.dump(_jsonable(meta), f, indent=2, default=str)
                    if hm_imgs:
                        cv2.imwrite(os.path.join(clip_dir, "heatmap_compare.png"), np.hstack(hm_imgs))
                    cv2.imwrite(os.path.join(clip_dir, "contact_sheet.jpg"), self._contact_sheet(indices, tags, header))
            except Exception as e:  # never let the writer die silently
                self.counters["writer_errors"] += 1
                try:
                    self.event("WRITER_ERROR", f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
                except Exception:
                    pass
            finally:
                self._q.task_done()

    def _contact_sheet(self, indices: List[int], tags: List[str], header: List[str]) -> np.ndarray:
        cols = 10
        sample = next((self._thumbs[i] for i in indices if i in self._thumbs), None)
        th, tw = (sample.shape[0], sample.shape[1]) if sample is not None else (144, 192)
        rows = (len(indices) + cols - 1) // cols
        head_h = 22 * len(header) + 8
        sheet = np.full((head_h + rows * (th + 2), cols * (tw + 2), 3), 20, dtype=np.uint8)
        for k, text in enumerate(header):
            cv2.putText(sheet, text, (6, 18 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for n, (fi, tag) in enumerate(zip(indices, tags)):
            r, c = divmod(n, cols)
            y, x = head_h + r * (th + 2), c * (tw + 2)
            thumb = self._thumbs.get(fi)
            if thumb is None:
                thumb = np.full((th, tw, 3), 60, dtype=np.uint8)
                cv2.putText(thumb, "not saved", (8, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if thumb.shape[:2] != (th, tw):
                thumb = cv2.resize(thumb, (tw, th))
            sheet[y:y + th, x:x + tw] = thumb
            _put_label(sheet, f"f{fi} {tag}", (x + 4, y + th - 6), _SRC_COLOR.get(tag, (255, 255, 255)), 0.42)
            cv2.rectangle(sheet, (x, y), (x + tw - 1, y + th - 1), _SRC_COLOR.get(tag, (255, 255, 255)), 2)
        return sheet

    def close(self, summary: Dict):
        try:
            self._q.join()
        except Exception:
            pass
        self._put_none()
        with open(os.path.join(self.session_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, default=str)
        with self._lock:
            self.closed = True
            self._log_f.close()
            self._jsonl_f.close()

    def _put_none(self):
        try:
            self._q.put(None, timeout=1.0)
            self._writer.join(timeout=2.0)
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------
# Instrumented pipeline
# ----------------------------------------------------------------------------------------------

class DebugHierarchicalPipeline(ProductionHierarchicalPipeline):
    """Production pipeline with instrumented Stage 1 / Stage 2 workers (decision logic unchanged)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dbg = DebugTraceWriter(save_frames=_env_flag("HTD_DEBUG_SAVE_FRAMES", True))
        self._dbg_probe = _env_flag("HTD_DEBUG_PROBE", True)
        self._dbg_near_miss = _env_flag("HTD_DEBUG_NEAR_MISS", True)
        try:
            self._dbg_near_miss_conf = float(os.environ.get("HTD_DEBUG_NEAR_MISS_CONF", "0.10"))
        except ValueError:
            self._dbg_near_miss_conf = 0.10
        self._dbg_near_miss_conf = min(self._dbg_near_miss_conf, self.conf_threshold)

        self._dbg_tls = threading.local()
        self._dbg_frame_src: "OrderedDict[int, str]" = OrderedDict()
        self._dbg_det_count = 0
        self._dbg_clip_count = 0
        self._dbg_episode = 0
        self._dbg_episode_clips = 0
        self._dbg_last_det_time = 0.0
        self._dbg_last_det_idx = -1
        self._dbg_weapon_seen_prev = False
        self._dbg_prev_track_id = None
        self._dbg_last_status = None
        self._dbg_install_pose_probe()

        eng = self.stage2b_engine
        T = self._dbg
        T.event("SESSION_START", f"Debug session folder: {T.session_dir}", console=True,
                session_dir=T.session_dir)
        T.event(
            "CONFIG",
            f"backend={self.backend} device={self.device} stage1_conf={self.conf_threshold:.2f} "
            f"clip_length={self.clip_length} clip_stride={self.clip_stride} max_audit_lag={self.max_audit_lag} "
            f"weapon_hold=3.0s alert_hold=5.0s | T1={eng.tier1_key} [{eng.p_low1:.2f},{eng.p_high1:.2f}] "
            f"T2={eng.tier2_key} [{eng.p_low2:.2f},{eng.p_high2:.2f}] T3={eng.tier3_key} tau={eng.tau3:.2f} | "
            f"probe={'on' if self._dbg_probe else 'off'} near_miss={'on' if self._dbg_near_miss else 'off'} "
            f"save_frames={'on' if T.save_frames else 'off'}",
            console=True,
            tiers={k: {"key": key, "requires_heatmap": bool(eng.models[key]["requires_heatmap"]),
                       "ref_tau": float(eng.models[key]["tau"])}
                   for k, key in (("t1", eng.tier1_key), ("t2", eng.tier2_key), ("t3", eng.tier3_key))},
        )

    # ------------------------------------------------------------------ helpers
    def _dbg_install_pose_probe(self):
        """Wraps pose_model.track on THIS instance only, to see how many people YOLO found per frame."""
        tracker = self.stage2a_tracker
        try:
            orig_track = tracker.pose_model.track
        except Exception as e:
            self._dbg.event("POSE_PROBE_OFF", f"cannot wrap pose_model.track: {e}")
            return
        tls = self._dbg_tls

        def traced_track(*a, **kw):
            t0 = time.perf_counter()
            res = orig_track(*a, **kw)
            info = {"ms": (time.perf_counter() - t0) * 1000.0, "n_persons": 0, "track_ids": None,
                    "boxes": [], "person_conf": [], "kpt_conf": None}
            try:
                r = res[0]
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                    info["n_persons"] = int(len(r.boxes))
                    info["boxes"] = r.boxes.xyxy.cpu().numpy().tolist()
                    info["person_conf"] = [round(float(c), 3) for c in r.boxes.conf.cpu().numpy().tolist()]
                    if getattr(r.boxes, "id", None) is not None:
                        info["track_ids"] = r.boxes.id.cpu().numpy().astype(int).tolist()
                    if getattr(r, "keypoints", None) is not None and getattr(r.keypoints, "conf", None) is not None:
                        info["kpt_conf"] = r.keypoints.conf.cpu().numpy()
            except Exception as e:
                info["probe_error"] = str(e)
            tls.pose = info
            return res

        try:
            tracker.pose_model.track = traced_track
        except Exception as e:
            self._dbg.event("POSE_PROBE_OFF", f"cannot wrap pose_model.track: {e}")

    def _dbg_status(self, reason: str):
        """Logs status-text transitions (the status itself is set by the production logic copy)."""
        if self.status_text != self._dbg_last_status:
            self._dbg_last_status = self.status_text
            self._dbg.event("STATUS", f"'{self.status_text}'  <- {reason}")

    def _dbg_set_src(self, frame_idx: int, tag: str):
        self._dbg_frame_src[frame_idx] = tag
        while len(self._dbg_frame_src) > 5000:
            self._dbg_frame_src.popitem(last=False)

    # ------------------------------------------------------------------ Stage 1 (instrumented copy)
    def weapon_scanner_worker(self):
        """Thread 2: instrumented copy of the v6.20 Stage 1 loop (3 FPS)."""
        T = self._dbg
        T.event("THREAD_START", "weapon_scanner_worker (Stage 1, 3 FPS)")
        target_delay = 1.0 / 3.0
        last_scanned_idx = -1

        while self.is_running:
            t_start = time.perf_counter()
            try:
                frame, frame_idx = self.buffer.get_latest_frame_with_index()

                if frame is None:
                    T.throttled("s1_noframe", 2.0, "S1_NO_FRAME", "buffer empty - camera not delivering frames yet")
                elif frame_idx == last_scanned_idx:
                    T.throttled("s1_dup", 2.0, "S1_SAME_FRAME",
                                f"latest frame f{frame_idx} already scanned (camera slower than 3 FPS?)")
                else:
                    last_scanned_idx = frame_idx
                    t_lock = time.perf_counter()
                    with self.gpu_lock:
                        lock_wait_ms = (time.perf_counter() - t_lock) * 1000.0
                        if self._dbg_near_miss:
                            raw = self.stage1_detector.predict(frame, custom_conf=self._dbg_near_miss_conf,
                                                               imgsz=self.rect_imgsz)
                        else:
                            raw = None
                            det_res = self.stage1_detector.predict(frame, imgsz=self.rect_imgsz)

                    if raw is not None:
                        # Rebuild the exact production result (conf >= threshold) from the low-conf call
                        keep = [i for i, s in enumerate(raw["scores"]) if s >= self.conf_threshold]
                        boxes = [raw["boxes"][i] for i in keep]
                        scores = [raw["scores"][i] for i in keep]
                        det_res = {"has_threat": len(boxes) > 0, "boxes": boxes, "scores": scores,
                                   "best_box": boxes[0] if boxes else None, "latency_ms": raw["latency_ms"]}
                        near = [s for s in raw["scores"] if s < self.conf_threshold]
                        best_near = max(near) if near else None
                    else:
                        best_near = None

                    now = time.time()
                    remain = self.weapon_active_until - now
                    top = det_res["scores"][0] if det_res["scores"] else None
                    near_txt = f" best_sub_threshold={best_near:.3f}" if best_near is not None else ""
                    T.event(
                        "S1_SCAN",
                        f"f{frame_idx} weapon={'YES' if det_res['has_threat'] else 'no '} "
                        f"n={len(det_res['boxes'])} top={_fmt_p(top)}{near_txt} box={_fmt_box(det_res['best_box'])} "
                        f"infer={det_res['latency_ms']:.1f}ms gpu_lock_wait={lock_wait_ms:.1f}ms "
                        f"window_remaining={max(0.0, remain):.2f}s",
                        frame_idx=frame_idx, has_threat=det_res["has_threat"], scores=det_res["scores"],
                        best_box=det_res["best_box"], best_sub_threshold=best_near,
                        latency_ms=det_res["latency_ms"], lock_wait_ms=lock_wait_ms,
                    )

                    if det_res["has_threat"]:
                        was_inactive = now >= self.weapon_active_until
                        self.weapon_active_until = now + 3.0
                        self.last_weapon_bbox = det_res["best_box"]
                        self.stage2a_tracker.update_weapon_anchor(det_res["best_box"])
                        self._dbg_last_det_time = now
                        self._dbg_last_det_idx = frame_idx
                        self._dbg_det_count += 1
                        T.counters["s1_detections"] += 1

                        # Snap auditing cursor to current live window when transitioning from idle
                        if was_inactive:
                            latest_idx = self.buffer.get_latest_index()
                            old_cursor = self.last_analyzed_idx
                            if latest_idx >= 0:
                                self.last_analyzed_idx = max(0, latest_idx - self.clip_length + 1)
                            T.counters["s1_triggers"] += 1
                            T.event(
                                "KNIFE_TRIGGER",
                                f"Stage 1 weapon at f{frame_idx} conf={top:.3f} box={_fmt_box(det_res['best_box'])} "
                                f"-> Stage 2 ARMED for 3.0 s | cursor {old_cursor} -> {self.last_analyzed_idx} "
                                f"(live f{latest_idx}, scan was {latest_idx - frame_idx} frames behind live) "
                                f"| imgsz={self.rect_imgsz} frame={frame.shape[1]}x{frame.shape[0]}",
                                console=True, frame_idx=frame_idx, conf=top, box=det_res["best_box"],
                                cursor_from=old_cursor, cursor_to=self.last_analyzed_idx, latest_idx=latest_idx,
                            )
                        elif not self._dbg_weapon_seen_prev:
                            T.event("WEAPON_REACQUIRED",
                                    f"f{frame_idx} conf={top:.3f} - Stage 2 window extended to +3.0 s",
                                    console=True)

                        lines = [f"f{frame_idx} WEAPON conf={top:.3f}" + (" TRIGGER" if was_inactive else ""),
                                 f"det #{self._dbg_det_count}  t+{now - T.t0:.2f}s"]
                        T.queue_detection_image(self._dbg_det_count, frame_idx, frame, det_res, lines, was_inactive)

                        if not self.is_violent_alert and (was_inactive or self.last_resolved_tier is None):
                            conf_pct = det_res["scores"][0] * 100.0 if det_res["scores"] else 0.0
                            self.status_text = f"WEAPON VERIFIED ({conf_pct:.1f}%)"
                            self.status_color = (0, 165, 255)
                            self._dbg_status("Stage 1 detection")
                        self._dbg_weapon_seen_prev = True
                    else:
                        if self._dbg_weapon_seen_prev:
                            T.event(
                                "WEAPON_LOST",
                                f"f{frame_idx} Stage 1 no longer sees the weapon"
                                f"{near_txt} - Stage 2 keeps running for {max(0.0, remain):.2f}s more unless re-detected",
                                console=True, best_sub_threshold=best_near,
                            )
                        self._dbg_weapon_seen_prev = False
            except Exception as e:
                T.exception("weapon_scanner_worker", e)
                time.sleep(0.1)

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.005, target_delay - elapsed))
        T.event("THREAD_END", "weapon_scanner_worker")

    # ------------------------------------------------------------------ Stage 2a (instrumented copy)
    def _populate_window_skeletons(self, window: Sequence[FrameItem]):
        """Instrumented copy of the v6.20 pose population (same strided cold-start rule)."""
        T = self._dbg
        missing_indices = [i for i, item in enumerate(window) if item.skeleton is None]
        for i, item in enumerate(window):
            if item.skeleton is not None and item.index not in self._dbg_frame_src:
                self._dbg_set_src(item.index, "C")
        self._dbg_tls.window_stats = {"missing": len(missing_indices), "extracted": 0, "interpolated": 0,
                                      "no_image": 0, "pose_ms": 0.0, "max_persons": 0, "strategies": Counter(),
                                      "cold_start": False}
        st = self._dbg_tls.window_stats
        if not missing_indices:
            return

        use_strided_cold_start = len(missing_indices) > (self.clip_stride + 5)
        st["cold_start"] = use_strided_cold_start

        def _extract(item: FrameItem):
            raw_frame = item.get_frame()
            if raw_frame is None:
                item.skeleton = np.zeros((17, 2), dtype=np.float32)
                self._dbg_set_src(item.index, "N")
                st["no_image"] += 1
                T.event("S2A_NO_IMAGE", f"f{item.index} could not be decoded -> zero skeleton")
                return
            trk = self.stage2a_tracker
            pre_id, pre_bbox, pre_weapon = trk.active_threat_actor_id, trk.last_actor_bbox, trk.last_weapon_bbox
            self._dbg_tls.pose = None
            t0 = time.perf_counter()
            with self.gpu_lock:
                kpts, bbox, track_id = self.stage2a_tracker.extract_actor_pose(raw_frame, imgsz=self.rect_imgsz)
            dt = (time.perf_counter() - t0) * 1000.0
            item.skeleton = kpts
            item.actor_bbox = bbox
            item.track_id = track_id
            self._dbg_set_src(item.index, "E")
            st["extracted"] += 1
            st["pose_ms"] += dt

            pinfo = self._dbg_tls.pose or {}
            n_p = int(pinfo.get("n_persons", 0))
            st["max_persons"] = max(st["max_persons"], n_p)
            strategy = _infer_actor_strategy(pre_id, pre_bbox is not None, pre_weapon is not None, n_p,
                                             pinfo.get("track_ids"), track_id)
            st["strategies"][strategy.split(" ")[0]] += 1
            valid = int(np.sum(np.sum(np.abs(kpts), axis=-1) > 1e-6)) if kpts is not None else 0
            kconf = None
            if pinfo.get("kpt_conf") is not None and bbox is not None and pinfo.get("boxes"):
                try:
                    sel = int(np.argmin([np.abs(np.array(b) - np.array(bbox)).sum() for b in pinfo["boxes"]]))
                    kconf = float(np.mean(pinfo["kpt_conf"][sel]))
                except Exception:
                    kconf = None
            T.event(
                "S2A_POSE",
                f"f{item.index} persons={n_p} ids={pinfo.get('track_ids')} -> actor ID {track_id} "
                f"box={_fmt_box(bbox)} valid_joints={valid}/17"
                f"{'' if kconf is None else f' kpt_conf={kconf:.2f}'} | {strategy} | {dt:.1f}ms",
                frame_idx=item.index, n_persons=n_p, track_ids=pinfo.get("track_ids"), actor_id=track_id,
                actor_box=bbox, valid_joints=valid, kpt_conf_mean=kconf, strategy=strategy, ms=dt,
                person_conf=pinfo.get("person_conf"),
            )
            if n_p == 0:
                T.counters["frames_no_person"] += 1
            if self._dbg_prev_track_id is not None and track_id is not None and track_id != self._dbg_prev_track_id:
                T.event("ACTOR_SWITCH", f"f{item.index} actor ID {self._dbg_prev_track_id} -> {track_id} ({strategy})",
                        console=True)
                T.counters["actor_switches"] += 1
            if track_id is not None:
                self._dbg_prev_track_id = track_id

        if not use_strided_cold_start:
            for idx in missing_indices:
                if not self.is_running:
                    return
                _extract(window[idx])
        else:
            key_positions = set(missing_indices[::2]) | {missing_indices[0], missing_indices[-1]}
            T.event("S2A_COLD_START",
                    f"{len(missing_indices)} frames without pose -> strided: run pose on {len(key_positions)}, "
                    f"interpolate {len(missing_indices) - len(key_positions)}")
            for idx in sorted(key_positions):
                if not self.is_running:
                    return
                _extract(window[idx])
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
                self._dbg_set_src(item.index, "I")
                st["interpolated"] += 1

    # ------------------------------------------------------------------ Stage 2b probe
    def _dbg_probe_tiers(self, graph_tensor: torch.Tensor, raw_seq: np.ndarray) -> Dict[str, Any]:
        """Evaluates every tier on the production input and on a training-style heatmap (diagnostic only)."""
        eng = self.stage2b_engine
        out: Dict[str, Any] = {}
        hm_prod = rasterize_heatmap_from_coords(graph_tensor, device=eng.device, normalize=False)
        smoothed_px = smooth_kinematics(interpolate_missing_joints(raw_seq), sigma=1.0)
        hm_train = rasterize_heatmap_from_coords(torch.from_numpy(np.asarray(smoothed_px, dtype=np.float32)),
                                                 device=eng.device)
        out["hm_prod"] = hm_prod
        out["hm_train"] = hm_train
        out["hm_stats_prod"] = _heatmap_stats(hm_prod)
        out["hm_stats_train"] = _heatmap_stats(hm_train)
        if not self._dbg_probe:
            return out

        coords_dev = graph_tensor.to(eng.device)
        if coords_dev.dim() == 3:
            coords_dev = coords_dev.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
        probs = {}
        for name, key in (("p1", eng.tier1_key), ("p2", eng.tier2_key), ("p3", eng.tier3_key)):
            item = eng.models[key]
            if item["requires_heatmap"]:
                probs[name] = {"input": "heatmap", "prod": eng._eval_model(key, hm_prod),
                               "train_style": eng._eval_model(key, hm_train)}
            else:
                p = eng._eval_model(key, coords_dev)
                probs[name] = {"input": "coords", "prod": p, "train_style": p}
        out["probs"] = probs
        out["sim_prod"] = _simulate_cascade(probs["p1"]["prod"], probs["p2"]["prod"], probs["p3"]["prod"], eng)
        out["sim_train"] = _simulate_cascade(probs["p1"]["train_style"], probs["p2"]["train_style"],
                                             probs["p3"]["train_style"], eng)
        return out

    # ------------------------------------------------------------------ Stage 2 (instrumented copy)
    def action_auditing_worker(self):
        """Thread 3: instrumented copy of the v6.20 sliding-window Stage 2 loop."""
        T = self._dbg
        T.event("THREAD_START", "action_auditing_worker (Stage 2a pose + Stage 2b cascade)")
        was_active = False

        while self.is_running:
            try:
                now = time.time()
                is_active = (now < self.weapon_active_until)

                if not is_active:
                    if was_active:
                        since = now - self._dbg_last_det_time if self._dbg_last_det_time else float("nan")
                        T.event(
                            "STAGE2_STOP",
                            f"episode #{self._dbg_episode} ended: no Stage 1 weapon for 3.0 s "
                            f"(last detection f{self._dbg_last_det_idx}, {since:.2f}s ago) after "
                            f"{self._dbg_episode_clips} clip(s) started",
                            console=True, episode=self._dbg_episode, clips=self._dbg_episode_clips,
                        )
                        if self._dbg_episode_clips == 0:
                            T.event("STAGE2_NO_CLIP",
                                    "Stage 2 was armed but NOT A SINGLE CLIP was evaluated in this episode "
                                    "(armed time too short for pose extraction, or exception)", console=True)
                            T.counters["episodes_without_clip"] += 1

                    if self.is_violent_alert and now > self.alert_until:
                        self.is_violent_alert = False
                        self.status_text = "SCANNING (IDLE)"
                        self.status_color = (0, 255, 0)
                        self._dbg_status("alert hold (5 s) expired")
                    elif not self.is_violent_alert:
                        self.status_text = "SCANNING (IDLE)"
                        self.status_color = (0, 255, 0)
                        self._dbg_status("no active weapon window")

                    self.pending_audit_frames = 0
                    self.last_resolved_tier = None
                    if was_active:
                        self.stage2a_tracker.reset_tracking()
                        self._dbg_prev_track_id = None
                        T.event("S2A_RESET", "actor tracking lock + weapon anchor cleared")
                        was_active = False

                    latest_idx = self.buffer.get_latest_index()
                    if latest_idx >= 0:
                        self.last_analyzed_idx = max(0, latest_idx - self.clip_length + 1)

                    time.sleep(0.04)
                    continue

                if not was_active:
                    self._dbg_episode += 1
                    self._dbg_episode_clips = 0
                    T.counters["episodes"] += 1
                    T.event("STAGE2_START",
                            f"episode #{self._dbg_episode}: weapon window open for "
                            f"{self.weapon_active_until - now:.2f}s, latest frame f{self.buffer.get_latest_index()}",
                            console=True, episode=self._dbg_episode)

                latest_idx = self.buffer.get_latest_index()
                if latest_idx >= self.clip_length:
                    live_window_start = latest_idx - self.clip_length + 1
                    lag = live_window_start - self.last_analyzed_idx
                    if not was_active or lag > self.max_audit_lag:
                        old = self.last_analyzed_idx
                        self.last_analyzed_idx = max(0, live_window_start)
                        reason = "episode start" if not was_active else f"audit lag {lag} > max {self.max_audit_lag} (frames skipped)"
                        T.event("CURSOR_SNAP", f"cursor {old} -> {self.last_analyzed_idx} ({reason})",
                                cursor_from=old, cursor_to=self.last_analyzed_idx, lag=lag)
                        if was_active:
                            T.counters["cursor_lag_snaps"] += 1
                            T.counters["frames_skipped_by_snap"] += max(0, lag)
                else:
                    T.throttled("s2_short_buf", 1.0, "S2_WAIT_BUFFER",
                                f"only {latest_idx + 1} frames ingested (< {self.clip_length})")
                was_active = True

                window, pending = self.buffer.get_window_by_index(self.last_analyzed_idx, self.clip_length)
                if len(window) < self.clip_length:
                    T.throttled("s2_wait", 0.5, "S2_WAIT_WINDOW",
                                f"window starting f{self.last_analyzed_idx} not complete yet "
                                f"(latest f{self.buffer.get_latest_index()})")
                    time.sleep(0.015)
                    continue

                self.last_analyzed_idx = window[0].index + self.clip_stride
                self.pending_audit_frames = pending

                # ---------------- one clip ----------------
                self._dbg_clip_count += 1
                self._dbg_episode_clips += 1
                clip_no = self._dbg_clip_count
                t_clip = time.perf_counter()
                t_wall = time.time()
                first, last = window[0].index, window[-1].index
                ages = [(t_wall - it.timestamp) * 1000.0 for it in window]
                live_idx = self.buffer.get_latest_index()
                T.event(
                    "CLIP_BEGIN",
                    f"#{clip_no} frames f{first}..f{last} | frame age {ages[-1]:.0f}..{ages[0]:.0f} ms | "
                    f"live f{live_idx} ({live_idx - last} frames ahead) | pending={pending} | "
                    f"weapon window {self.weapon_active_until - t_wall:.2f}s left",
                    clip=clip_no, first=first, last=last, live_idx=live_idx, pending=pending,
                    age_newest_ms=ages[-1], age_oldest_ms=ages[0],
                )

                # Step 1: Extract/Cache skeletons
                self._populate_window_skeletons(window)
                if not self.is_running:
                    break
                st = getattr(self._dbg_tls, "window_stats", {}) or {}
                tags = [self._dbg_frame_src.get(it.index, "?") for it in window]

                # Step 2: Kinematic preprocessing & Torso-Scale Normalization
                raw_seq = np.array([item.skeleton for item in window], dtype=np.float32)
                normed_coords, graph_tensor = self.stage2a_tracker.preprocess_sliding_window(raw_seq)

                # Step 3: Evaluate Progressive Staircase Cascade (Champion 2)
                with self.gpu_lock:
                    cascade_res = self.stage2b_engine.evaluate_clip(graph_tensor)
                    probe = self._dbg_probe_tiers(graph_tensor, raw_seq)

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
                    self._dbg_status(f"clip #{clip_no} cascade ALARM")
                else:
                    if not self.is_violent_alert:
                        detail_str = f"Tier {cascade_res['resolved_tier']} Cleared ({cascade_res['confidence']*100:.1f}%)"
                        self.status_text = f"WEAPON DRAWN - NORMAL MOTION ({actor_tag} | {detail_str})"
                        self.status_color = (0, 165, 255)
                        self._dbg_status(f"clip #{clip_no} cascade cleared")

                # ---------------- record ----------------
                self._dbg_record_clip(clip_no, window, tags, ages, st, raw_seq, normed_coords,
                                      cascade_res, probe, t_clip, live_idx, pending)
            except Exception as e:
                T.exception("action_auditing_worker", e)
                time.sleep(0.1)
        if was_active:
            T.event("STAGE2_STOP", f"episode #{self._dbg_episode} still armed at shutdown after "
                                   f"{self._dbg_episode_clips} clip(s)", episode=self._dbg_episode,
                    clips=self._dbg_episode_clips)
        T.event("THREAD_END", "action_auditing_worker")

    def _dbg_record_clip(self, clip_no, window, tags, ages, st, raw_seq, normed_coords,
                         cascade_res, probe, t_clip, live_idx, pending):
        T = self._dbg
        first, last = window[0].index, window[-1].index
        total_ms = (time.perf_counter() - t_clip) * 1000.0

        valid_per_frame = (np.sum(np.abs(raw_seq), axis=-1) > 1e-6).sum(axis=-1)  # (50,)
        valid_ratio = float(valid_per_frame.sum()) / float(raw_seq.shape[0] * raw_seq.shape[1])
        empty_frames = int((valid_per_frame == 0).sum())
        nv = normed_coords[np.abs(normed_coords).sum(-1) > 1e-6]
        nrange = (float(nv.min()), float(nv.max())) if nv.size else (0.0, 0.0)
        tag_counts = Counter(tags)
        src_txt = " ".join(f"{k}{tag_counts.get(k, 0)}" for k in ("E", "I", "C", "N"))
        ids = sorted({it.track_id for it in window if it.track_id is not None})

        hs_p, hs_t = probe["hm_stats_prod"], probe["hm_stats_train"]
        probs = probe.get("probs")
        r = cascade_res
        decision = "ALARM" if r["is_threat"] else "clear"
        T.tier_hist[f"tier{r['resolved_tier']}"] += 1
        T.decision_hist[decision] += 1
        T.counters["clips"] += 1

        stats = T.stats
        stats["frame_age_ms_oldest"].append(ages[0])
        stats["frame_age_ms_newest"].append(ages[-1])
        stats["pose_ms_per_window"].append(st.get("pose_ms", 0.0))
        stats["clip_total_ms"].append(total_ms)
        stats["valid_joint_ratio"].append(valid_ratio)
        stats["hm_body_h_prod"].append(hs_p.get("mean_body_h_cells", 0.0))
        stats["hm_body_h_train"].append(hs_t.get("mean_body_h_cells", 0.0))
        if probs:
            stats["p1_prod"].append(probs["p1"]["prod"])
            stats["p1_train_style"].append(probs["p1"]["train_style"])

        cascade_txt = (f"T{r['resolved_tier']} {decision.upper()} conf={r['confidence']:.3f} "
                       f"(p1={_fmt_p(r['tier1_prob'])} p2={_fmt_p(r['tier2_prob'])} p3={_fmt_p(r['tier3_prob'])}) "
                       f"{r['decision_reason']}")
        probe_txt = ""
        if probs:
            probe_txt = (f" | PROBE prod p1/p2/p3={_fmt_p(probs['p1']['prod'])}/{_fmt_p(probs['p2']['prod'])}/"
                         f"{_fmt_p(probs['p3']['prod'])} -> {probe['sim_prod']} ; train-style "
                         f"{_fmt_p(probs['p1']['train_style'])}/{_fmt_p(probs['p2']['train_style'])}/"
                         f"{_fmt_p(probs['p3']['train_style'])} -> {probe['sim_train']}")
        msg = (f"#{clip_no} f{first}..f{last} | pose {src_txt} ({st.get('pose_ms', 0.0):.0f}ms, "
               f"max persons {st.get('max_persons', 0)}, actor IDs {ids or 'none'}) | joints {valid_ratio*100:.0f}% valid, "
               f"{empty_frames} empty frames, normed range [{nrange[0]:.2f},{nrange[1]:.2f}] | "
               f"heatmap body h/w cells prod={hs_p.get('mean_body_h_cells', 0):.1f}/{hs_p.get('mean_body_w_cells', 0):.1f} "
               f"train-style={hs_t.get('mean_body_h_cells', 0):.1f}/{hs_t.get('mean_body_w_cells', 0):.1f} | "
               f"{cascade_txt}{probe_txt} | clip {total_ms:.0f}ms")
        T.event("CLIP_RESULT", msg, console=True, clip=clip_no, first=first, last=last,
                cascade={k: v for k, v in r.items()}, hm_stats_prod=hs_p, hm_stats_train=hs_t,
                probe=probs, sim_prod=probe.get("sim_prod"), sim_train=probe.get("sim_train"),
                valid_joint_ratio=valid_ratio, empty_frames=empty_frames, sources=dict(tag_counts),
                actor_ids=ids, total_ms=total_ms)

        if empty_frames == len(window):
            T.event("CLIP_WARNING", f"#{clip_no}: every frame has an EMPTY skeleton (no person found) - "
                                    f"Stage 2b is classifying zeros", console=True)
        elif valid_ratio < 0.5:
            T.event("CLIP_WARNING", f"#{clip_no}: only {valid_ratio*100:.0f}% of joints valid", console=True)
        if hs_t.get("mean_body_h_cells", 0) > 0 and hs_p.get("mean_body_h_cells", 0) < 0.5 * hs_t["mean_body_h_cells"]:
            T.event("CLIP_WARNING",
                    f"#{clip_no}: production heatmap body is {hs_p['mean_body_h_cells']:.1f} cells tall vs "
                    f"{hs_t['mean_body_h_cells']:.1f} for training-style input -> skeleton collapsed in the "
                    f"PoseConv3D input (see clips/.../heatmap_compare.png)", console=True)
            T.counters["clips_heatmap_collapsed"] += 1

        # ---- images / arrays (async) ----
        weapon_box = self.last_weapon_bbox
        for it, tag, age in zip(window, tags, ages):
            if it.index in T._saved_frames:
                continue
            valid = int(np.sum(np.sum(np.abs(it.skeleton), axis=-1) > 1e-6)) if it.skeleton is not None else 0
            info = {
                "kpts": it.skeleton, "actor_bbox": it.actor_bbox, "weapon_bbox": weapon_box, "src": tag,
                "lines": [f"f{it.index} [{tag}] age {age:.0f}ms clip#{clip_no}",
                          f"actor ID {it.track_id}  joints {valid}/17"],
            }
            T.queue_frame_image(it.index, it.get_frame(), info)

        clip_dir = os.path.join(T.clips_dir, f"clip_{clip_no:04d}_f{first:07d}-{last:07d}")
        header = [
            f"clip #{clip_no}  f{first}..f{last}  age {ages[-1]:.0f}..{ages[0]:.0f} ms  live f{live_idx}  pending {pending}",
            f"pose: {src_txt}  (E=extracted green, I=interpolated orange, C=cached grey, N=no image red)",
            f"cascade: {cascade_txt}"[:150],
        ]
        if probs:
            header.append(f"probe: prod -> {probe['sim_prod']} | train-style -> {probe['sim_train']}"[:150])
        hm_imgs = [
            _heatmap_projection(probe["hm_prod"], f"PRODUCTION input  h={hs_p.get('mean_body_h_cells', 0):.1f}"),
            _heatmap_projection(probe["hm_train"], f"TRAINING-style    h={hs_t.get('mean_body_h_cells', 0):.1f}"),
        ]
        meta = {
            "clip": clip_no, "frames": [it.index for it in window], "sources": tags,
            "timestamps": [it.timestamp for it in window], "age_ms": ages,
            "track_ids": [it.track_id for it in window], "actor_boxes": [it.actor_bbox for it in window],
            "weapon_bbox": weapon_box, "live_idx": live_idx, "pending": pending,
            "window_stats": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()},
            "cascade": dict(cascade_res), "probe": probs, "sim_prod": probe.get("sim_prod"),
            "sim_train": probe.get("sim_train"), "hm_stats_prod": hs_p, "hm_stats_train": hs_t,
            "valid_joint_ratio": valid_ratio, "empty_frames": empty_frames, "total_ms": total_ms,
        }
        T.queue_clip_dump(clip_dir, [it.index for it in window], tags, header,
                          {"skeleton_raw": raw_seq, "coords_normed": np.asarray(normed_coords, dtype=np.float32)},
                          meta, hm_imgs)

    # ------------------------------------------------------------------ session wrap-up
    def _dbg_summary(self) -> Dict[str, Any]:
        T = self._dbg

        def m(xs):
            return float(np.mean(xs)) if xs else None

        s = {
            "session_dir": T.session_dir,
            "duration_s": time.time() - T.t0,
            "stage1": {"detections": T.counters["s1_detections"], "triggers": T.counters["s1_triggers"]},
            "stage2": {
                "episodes": T.counters["episodes"],
                "episodes_without_any_clip": T.counters["episodes_without_clip"],
                "clips": T.counters["clips"],
                "tier_histogram": dict(T.tier_hist),
                "decision_histogram": dict(T.decision_hist),
                "cursor_lag_snaps": T.counters["cursor_lag_snaps"],
                "frames_skipped_by_snap": T.counters["frames_skipped_by_snap"],
                "frames_no_person": T.counters["frames_no_person"],
                "actor_switches": T.counters["actor_switches"],
                "clips_heatmap_collapsed": T.counters["clips_heatmap_collapsed"],
                "mean_frame_age_ms_newest": m(T.stats["frame_age_ms_newest"]),
                "mean_frame_age_ms_oldest": m(T.stats["frame_age_ms_oldest"]),
                "mean_pose_ms_per_window": m(T.stats["pose_ms_per_window"]),
                "mean_clip_total_ms": m(T.stats["clip_total_ms"]),
                "mean_valid_joint_ratio": m(T.stats["valid_joint_ratio"]),
                "mean_heatmap_body_h_cells_prod": m(T.stats["hm_body_h_prod"]),
                "mean_heatmap_body_h_cells_train_style": m(T.stats["hm_body_h_train"]),
                "mean_p1_prod": m(T.stats["p1_prod"]),
                "mean_p1_train_style": m(T.stats["p1_train_style"]),
                "max_p1_prod": max(T.stats["p1_prod"]) if T.stats["p1_prod"] else None,
                "max_p1_train_style": max(T.stats["p1_train_style"]) if T.stats["p1_train_style"] else None,
            },
            "worker_exceptions": T.counters["exceptions"],
            "dropped_image_jobs": T.counters["dropped_image_jobs"],
            "writer_errors": T.counters["writer_errors"],
        }
        return s

    def _dbg_close(self):
        T = self._dbg
        if T.closed:
            return
        summary = self._dbg_summary()
        T.event("SESSION_END", json.dumps(_jsonable(summary)), summary=summary)
        T.close(summary)
        st2 = summary["stage2"]
        print("\n" + "=" * 80)
        print("  STAGE 2 DEBUG SUMMARY")
        print(f"  Stage 1: {summary['stage1']['detections']} weapon detections, {summary['stage1']['triggers']} trigger(s)")
        print(f"  Stage 2: {st2['episodes']} episode(s), {st2['clips']} clip(s), "
              f"{st2['episodes_without_any_clip']} episode(s) with NO clip | tiers {st2['tier_histogram']} | "
              f"decisions {st2['decision_histogram']}")
        print(f"  Heatmap body height (cells of 56): production={st2['mean_heatmap_body_h_cells_prod']} "
              f"training-style={st2['mean_heatmap_body_h_cells_train_style']}")
        print(f"  Mean p1: production={st2['mean_p1_prod']} training-style={st2['mean_p1_train_style']}")
        print(f"  Worker exceptions: {summary['worker_exceptions']} | dropped image jobs: {summary['dropped_image_jobs']}")
        print(f"  Logs + frames: {T.session_dir}")
        print("=" * 80 + "\n", flush=True)

    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        finally:
            self._dbg_close()

    def run_dry_run(self, *args, **kwargs):
        try:
            return super().run_dry_run(*args, **kwargs)
        finally:
            self._dbg_close()
