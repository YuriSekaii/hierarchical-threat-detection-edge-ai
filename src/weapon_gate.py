"""
Weapon plausibility gates between Stage 1 and Stage 2.

1. WeaponMotionGate (Stage 1 -> arming)
   A real threat is a weapon that moves with a person. A box that sits in the same place
   scan after scan (an object on a table, a rod on the floor, a printed picture) is treated
   as a false alarm: it never arms Stage 2 and never extends an active window.

   Verdicts for each Stage 1 detection, comparing it with the previous detection:
     "first"  - no previous detection within max_gap_s. Not enough evidence yet:
                does not arm an idle pipeline (extends an already active window).
     "moving" - box centre moved >= min_move * box diagonal. Arms / extends Stage 2.
     "still"  - small move, but not yet static for static_hits detections. Extends only.
     "static" - the last static_hits detections all stayed within min_move of each other.
                Ignored: no arming, no extension.

   Stage 2 always analyzes the 50 frames *before* the arming moment, so waiting for the
   second (moving) detection costs no motion history.

2. weapon_in_hand (Stage 2 -> per clip)
   The weapon must be held. For every Stage 1 detection that falls inside the clip, the
   distance from the threat actor's nearest wrist to the weapon box is measured in torso
   lengths (scale invariant). If no detection is within max_dist of a wrist, the clip is
   dropped before the action cascade.

Both Stage 1 boxes and Stage 2a keypoints are in full-frame pixel coordinates.
"""

import threading
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

# COCO-17 keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


def _centre_and_diag(box: Sequence[float]) -> Tuple[np.ndarray, float]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5]), max(1.0, float(np.hypot(x2 - x1, y2 - y1)))


class WeaponMotionGate:
    """Thread-safe history of Stage 1 detections + static-object classification."""

    def __init__(
        self,
        min_move: float = 0.15,
        static_hits: int = 3,
        max_gap_s: float = 1.5,
        history_len: int = 64,
    ):
        self.min_move = float(min_move)
        self.static_hits = max(2, int(static_hits))
        self.max_gap_s = float(max_gap_s)
        self._hist: Deque[Tuple[int, float, List[float], str]] = deque(maxlen=history_len)
        self._lock = threading.Lock()

    def classify(self, frame_idx: int, t: float, box: Sequence[float]) -> Tuple[str, Optional[float]]:
        """Classifies a new detection and records it. Returns (verdict, move vs previous in box diagonals)."""
        box = [float(v) for v in box[:4]]
        c, diag = _centre_and_diag(box)
        with self._lock:
            prev = self._hist[-1] if self._hist else None
            if prev is None or (t - prev[1]) > self.max_gap_s:
                verdict, move = "first", None
            else:
                move = float(np.linalg.norm(c - _centre_and_diag(prev[2])[0]) / diag)
                if move >= self.min_move:
                    verdict = "moving"
                else:
                    run = list(self._hist)[-(self.static_hits - 1):]
                    times = [h[1] for h in run] + [t]
                    contiguous = len(run) == self.static_hits - 1 and all(
                        (b - a) <= self.max_gap_s for a, b in zip(times, times[1:])
                    )
                    stationary = contiguous and all(
                        np.linalg.norm(c - _centre_and_diag(h[2])[0]) / diag < self.min_move for h in run
                    )
                    verdict = "static" if stationary else "still"
            self._hist.append((int(frame_idx), float(t), box, verdict))
        return verdict, move

    def detections_between(self, first_idx: int, last_idx: int) -> List[Tuple[int, List[float], str]]:
        """Detections whose frame index lies in [first_idx, last_idx] (static ones excluded)."""
        with self._lock:
            return [(f, b, v) for f, _, b, v in self._hist if first_idx <= f <= last_idx and v != "static"]

    def nearest_detection(self, frame_idx: int, max_frames: int) -> Optional[Tuple[int, List[float], str]]:
        """Detection closest in frame index to frame_idx (static ones included), or None if none within max_frames."""
        with self._lock:
            best = min(self._hist, key=lambda h: abs(h[0] - frame_idx), default=None)
        if best is None or abs(best[0] - frame_idx) > max_frames:
            return None
        return best[0], best[2], best[3]

    def reset(self):
        with self._lock:
            self._hist.clear()


def _point_to_box(p: np.ndarray, box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    dx = max(x1 - p[0], 0.0, p[0] - x2)
    dy = max(y1 - p[1], 0.0, p[1] - y2)
    return float(np.hypot(dx, dy))


def _valid(k: np.ndarray) -> bool:
    return bool(np.abs(k).sum() > 1e-6)


def _torso_length(kpts: np.ndarray) -> Optional[float]:
    sh = [kpts[i] for i in (L_SHOULDER, R_SHOULDER) if _valid(kpts[i])]
    hp = [kpts[i] for i in (L_HIP, R_HIP) if _valid(kpts[i])]
    if sh and hp:
        t = float(np.linalg.norm(np.mean(sh, axis=0) - np.mean(hp, axis=0)))
        if t > 10.0:
            return t
    pts = kpts[np.abs(kpts).sum(-1) > 1e-6]
    if len(pts) >= 2:
        d = float(np.linalg.norm(pts.max(0) - pts.min(0))) * 0.5
        if d > 10.0:
            return d
    return None


def wrist_to_box_distance(kpts: Optional[np.ndarray], box: Sequence[float]) -> Optional[float]:
    """Nearest wrist to weapon box distance in torso lengths. None if it cannot be measured."""
    if kpts is None:
        return None
    kpts = np.asarray(kpts, dtype=np.float32)
    torso = _torso_length(kpts)
    wrists = [kpts[i] for i in (L_WRIST, R_WRIST) if _valid(kpts[i])]
    if torso is None or not wrists:
        return None
    return min(_point_to_box(w, box) for w in wrists) / torso


def weapon_in_hand(
    frame_kpts: dict,
    detections: Sequence[Tuple[int, Sequence[float], str]],
    max_dist: float = 0.5,
) -> Tuple[str, Optional[float], int]:
    """
    frame_kpts: {frame_idx: (17, 2) keypoints of the threat actor}
    detections: [(frame_idx, box, verdict)] inside the clip
    Returns (verdict, closest distance in torso lengths, detections checked):
      "in_hand"   - at least one detection within max_dist of a wrist -> analyze
      "far"       - weapon measured, but never near a wrist          -> drop clip
      "no_person" - weapon seen only on frames without a person       -> drop clip
      "unknown"   - no detection in the clip, or wrists not visible   -> analyze
    """
    best = None
    measured = 0
    no_person = 0
    for f, box, _ in detections:
        k = frame_kpts.get(f)
        if k is None:
            continue
        if not _valid(np.asarray(k)):
            no_person += 1
            continue
        d = wrist_to_box_distance(k, box)
        if d is None:
            continue
        measured += 1
        best = d if best is None else min(best, d)
    if measured:
        return ("in_hand" if best <= max_dist else "far"), best, measured
    if no_person and no_person == len([d for d in detections if d[0] in frame_kpts]):
        return "no_person", None, no_person
    return "unknown", None, 0
