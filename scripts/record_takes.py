"""
Lab take recorder for the skeleton simulator data (calibration, check takes, test takes).

Opens the camera exactly like the production pipeline (MSMF -> DSHOW -> ANY, MJPG, 30 fps, 2560x1440),
walks through a fixed take plan, and saves every take as a near-lossless MJPG .avi plus one row in
takes.csv. The full sensor frame is saved; the dashed lines in the preview show the 4:3 crop the
pipeline actually analyzes.

Keys (camera window):
    SPACE  start / stop recording the current take
    N / P  next / previous take in the plan
    R      redo: discard the last saved take and record it again
    F      free take (anything not in the plan)
    I      switch to the next camera
    Q      quit

Output: data/lab_recordings/<date>/<category>/<NNN>_<label>_<HHMMSS>.avi and data/lab_recordings/<date>/takes.csv
Re-running on the same day resumes at the first unfinished take.

Test without a camera:
    python scripts/record_takes.py --source video.mp4 --auto-test 3
"""

import argparse
import csv
import datetime as dt
import os
import queue
import shutil
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (category, label, repeats, instruction, hand, height, speed, direction)
PLAN = [
    ("calib", "floor_markers", 1, "Nobody in view. Tape crosses on the floor visible. Record ~5 s.", "", "", "", ""),
    ("calib", "checkerboard", 1, "Optional. Move the checkerboard across the whole view, near/far/tilted, ~60 s.",
     "", "", "", ""),
    ("check", "thrust_low", 3, "Thrust at HIP level, right hand, normal speed.", "right", "low", "normal", "across"),
    ("check", "thrust_chest", 3, "Thrust at CHEST level, right hand, normal speed.", "right", "chest", "normal",
     "across"),
    ("check", "thrust_left_hand", 3, "Thrust with the LEFT hand, chest level, normal speed.", "left", "chest",
     "normal", "across"),
    ("check", "thrust_toward_camera", 3, "Thrust TOWARD the camera, right hand, chest level.", "right", "chest",
     "normal", "toward"),
    ("check", "thrust_slow", 2, "SLOW thrust, right hand, chest level.", "right", "chest", "slow", "across"),
    ("check", "thrust_fast", 2, "FAST thrust, right hand, chest level.", "right", "chest", "fast", "across"),
    ("test", "test_thrust_low", 5, "TEST: thrust at hip level (your demo style).", "right", "low", "normal", "across"),
    ("test", "test_thrust_high", 5, "TEST: thrust like the training videos (shoulder level, straight arm).",
     "right", "high", "normal", "across"),
    ("test", "test_walk_knife", 5, "TEST: walk through the view holding the knife low. No attack.", "right", "low",
     "", "walk"),
    ("normal", "walk_knife_low", 2, "Optional. Walk holding the knife low, arms swinging.", "right", "low", "",
     "walk"),
    ("normal", "point_with_knife", 2, "Optional. Point / gesture with the knife hand.", "right", "", "", ""),
    ("normal", "put_down_pick_up", 2, "Optional. Put the knife down, pick it up again.", "right", "", "", ""),
    ("normal", "hand_over", 2, "Optional. Hand the knife over to someone / out of view.", "right", "", "", ""),
]

_ATTACKS = [("thrust_chest", "Straight thrust at chest height"), ("thrust_low", "Thrust at hip height"),
            ("overhand_stab", "Overhand stab: raise the knife, drive it down"), ("cut_down", "Big chop from above the head"),
            ("slash", "Side-to-side slash"), ("walking_thrust", "Walk in and thrust without stopping")]
_FACING = [("front", "Face the camera"), ("left", "Left side to the camera"), ("back", "Back to the camera"),
           ("right", "Right side to the camera")]
_NORMALS = [("walk", "Walk around, arms swinging"), ("turn", "Turn around on the spot, both ways"),
            ("gesture", "Talk with your hands, big gestures"), ("bend_pick_up", "Bend down, pick something up, stand"),
            ("reach", "Reach up, forward and sideways"), ("stretch", "Stretch, arm circles, wave"),
            ("hand_to_head", "Touch your head, face and neck"), ("point", "Point with the knife, slowly"),
            ("inspect_knife", "Look at the knife, turn it in your hand"), ("fidget", "Fidget, hands to belt and pockets"),
            ("hurry", "Walk fast, change direction")]
# Samples for the motion generator. test_* takes stay out of training; each test attack take holds exactly one attack,
# so its label comes from the take itself.
PLAN_SAMPLES = (
    [("train_attack", f"{a}_{f}", 1, f"{d}. {fd}. About 5 times, pause between.", "right", "", "", f)
     for a, d in _ATTACKS for f, fd in _FACING]
    + [("train_attack", f"{a}_lefthand", 1, f"LEFT hand. {d}. Any direction. About 5 times.", "left", "", "", "")
       for a, d in _ATTACKS]
    + [("train_normal", n, 2, f"{d}. Hold the knife. About 20 s. No strikes.", "right", "", "", "") for n, d in _NORMALS]
    + [("test_attack", f"test_{a}", 5, f"TEST, ONE attack only, any direction. {d}.", "right", "", "", "")
       for a, d in _ATTACKS]
    + [("test_normal", "test_normal_mixed", 2, "TEST, about 3 minutes of mixed normal activity. No strikes.", "right", "", "", "")]
)
PLANS = {"lab": PLAN, "samples": PLAN_SAMPLES}

CSV_FIELDS = ["file", "status", "category", "label", "repeat", "hand", "height", "speed", "direction",
              "started", "duration_s", "frames", "fps", "dropped", "width", "height_px", "camera", "note"]


def open_camera(idx: int) -> Optional[cv2.VideoCapture]:
    """Same negotiation as ProductionHierarchicalPipeline.open_camera (1440p)."""
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        try:
            c = cv2.VideoCapture(idx, backend)
            if c.isOpened():
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                c.set(cv2.CAP_PROP_FPS, 30)
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
                for _ in range(3):
                    r, f = c.read()
                    if r and f is not None and f.size > 0:
                        return c
                c.release()
        except Exception:
            pass
    return None


class Source:
    """Background reader that always holds the newest frame and feeds the writer while recording."""

    def __init__(self, cap: cv2.VideoCapture, paced_fps: Optional[float] = None):
        self.cap = cap
        self.paced_fps = paced_fps  # video files: play back in real time
        self.lock = threading.Lock()
        self.frame = None
        self.frame_no = 0
        self.ended = False
        self.sink: Optional["TakeWriter"] = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            t0 = time.perf_counter()
            ok, f = self.cap.read()
            if not ok or f is None:
                self.ended = True
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f
                self.frame_no += 1
                sink = self.sink
            if sink is not None:
                sink.push(f)
            if self.paced_fps:
                time.sleep(max(0.0, 1.0 / self.paced_fps - (time.perf_counter() - t0)))

    def latest(self):
        with self.lock:
            return self.frame, self.frame_no

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


class TakeWriter:
    """Encodes frames on its own thread so the camera never waits for the encoder."""

    def __init__(self, path: str, size, fps: float = 30.0):
        self.path = path
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {path}")
        self.q: "queue.Queue" = queue.Queue(maxsize=90)
        self.frames = 0
        self.dropped = 0
        self.t_start = time.time()
        self.t_last = self.t_start
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def push(self, frame):
        try:
            self.q.put_nowait(frame)
            self.t_last = time.time()
        except queue.Full:
            self.dropped += 1

    def _loop(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            self.writer.write(f)
            self.frames += 1

    def finish(self):
        self.q.put(None)
        self.thread.join()
        self.writer.release()
        dur = max(1e-6, self.t_last - self.t_start)
        return dur, self.frames, (self.frames - 1) / dur if self.frames > 1 else 0.0, self.dropped


class Session:
    def __init__(self, out_root: str, plan=PLAN):
        self.day_dir = os.path.join(out_root, dt.date.today().isoformat())
        os.makedirs(self.day_dir, exist_ok=True)
        self.csv_path = os.path.join(self.day_dir, "takes.csv")
        self.slots = [(cat, label, rep, instr, hand, height, speed, direction)
                      for cat, label, n, instr, hand, height, speed, direction in plan for rep in range(1, n + 1)]
        self.rows = self._read_rows()
        self.pos = self._first_unfinished()

    def _read_rows(self):
        if not os.path.exists(self.csv_path):
            return []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def kept(self):
        discarded = {r["file"] for r in self.rows if r["status"] == "discarded"}
        return [r for r in self.rows if r["status"] == "ok" and r["file"] not in discarded]

    def done(self, label: str, rep: int) -> bool:
        return any(r["label"] == label and r["repeat"] == str(rep) for r in self.kept())

    def _first_unfinished(self) -> int:
        for i, s in enumerate(self.slots):
            if not self.done(s[1], s[2]):
                return i
        return len(self.slots) - 1

    def next_number(self) -> int:
        return len(self.rows) + 1

    def append(self, row: dict):
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        self.rows.append({k: str(v) for k, v in row.items()})


def draw_overlay(frame, sess: Session, recording: Optional[TakeWriter], free: bool, msg: str, cam: str):
    h, w = frame.shape[:2]
    s = min(1280 / w, 720 / h)
    img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ih, iw = img.shape[:2]
    # 4:3 crop the pipeline analyzes
    cw = int(ih * 4 / 3)
    if cw < iw:
        x0 = (iw - cw) // 2
        for x in (x0, x0 + cw):
            for y in range(0, ih, 20):
                cv2.line(img, (x, y), (x, min(ih, y + 10)), (0, 255, 255), 1)

    cat, label, rep, instr, *_ = sess.slots[sess.pos]
    n_label = sum(1 for sl in sess.slots if sl[1] == label)
    head = "FREE TAKE (not in plan)" if free else f"[{cat}] {label}  ({rep}/{n_label})   take {sess.pos + 1}/{len(sess.slots)}"
    sub = "Anything you want to keep. SPACE to start/stop." if free else instr
    done = sum(1 for sl in sess.slots if sess.done(sl[1], sl[2]))

    cv2.rectangle(img, (0, 0), (iw, 70), (0, 0, 0), -1)
    cv2.putText(img, head, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    if recording is not None:
        el = time.time() - recording.t_start
        cv2.rectangle(img, (0, 0), (iw - 1, ih - 1), (0, 0, 255), 6)
        cv2.circle(img, (iw - 180, 100), 10, (0, 0, 255), -1)
        cv2.putText(img, f"REC {el:4.1f}s", (iw - 160, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(img, "READY - SPACE to record", (iw - 300, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    cv2.LINE_AA)

    cv2.rectangle(img, (0, ih - 52), (iw, ih), (0, 0, 0), -1)
    cv2.putText(img, msg.replace("\\", "/"), (10, ih - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"SPACE rec/stop | N next | P prev | R redo last | F free take | I camera | Q quit   "
                     f"done {done}/{len(sess.slots)} | {cam} {w}x{h}",
                (10, ih - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Lab take recorder")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0, same as the pipeline launcher)")
    ap.add_argument("--source", type=str, default=None, help="video file instead of a camera (testing)")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "data", "lab_recordings"))
    ap.add_argument("--auto-test", type=int, default=0, help="headless test: record this many plan takes, 2 s each")
    ap.add_argument("--plan", choices=list(PLANS), default="lab", help="take plan (samples: record_samples.bat)")
    args = ap.parse_args()

    cam_idx = args.camera
    if args.source:
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src = Source(cap, paced_fps=fps)
        cam_name = os.path.basename(args.source)
    else:
        cap = open_camera(cam_idx)
        if cap is None:
            print(f"[ERROR] Camera {cam_idx} could not be opened. Close the detection pipeline first, "
                  f"or try another index with --camera 1")
            return 1
        src = Source(cap)
        cam_name = f"camera {cam_idx}"

    sess = Session(args.out, PLANS[args.plan])
    print(f"[REC] Saving to {sess.day_dir}")
    print(f"[REC] {sum(1 for sl in sess.slots if sess.done(sl[1], sl[2]))}/{len(sess.slots)} planned takes already done")

    writer: Optional[TakeWriter] = None
    free = False
    msg = f"Saving to {sess.day_dir}"
    rec_meta = None
    last_saved: Optional[dict] = None
    win = "Lab take recorder"
    headless = args.auto_test > 0
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    def start():
        nonlocal writer, rec_meta
        frame, _ = src.latest()
        if frame is None:
            return "No frame yet"
        cat, label, rep, _, hand, height, speed, direction = sess.slots[sess.pos]
        if free:
            cat, label, rep, hand, height, speed, direction = "free", "free", 0, "", "", "", ""
        folder = os.path.join(sess.day_dir, cat)
        os.makedirs(folder, exist_ok=True)
        name = f"{sess.next_number():03d}_{label}_{dt.datetime.now():%H%M%S}.avi"
        h, w = frame.shape[:2]
        writer = TakeWriter(os.path.join(folder, name), (w, h))
        rec_meta = dict(file=os.path.relpath(writer.path, sess.day_dir), category=cat, label=label, repeat=rep,
                        hand=hand, height=height, speed=speed, direction=direction,
                        started=dt.datetime.now().isoformat(timespec="seconds"), width=w, height_px=h,
                        camera=cam_name, note="")
        src.sink = writer
        return f"Recording {name}"

    def stop():
        nonlocal writer, free, last_saved
        src.sink = None
        dur, frames, fps, dropped = writer.finish()
        row = dict(rec_meta, status="ok", duration_s=f"{dur:.2f}", frames=frames, fps=f"{fps:.1f}", dropped=dropped)
        sess.append(row)
        last_saved = row
        writer = None
        warn = f"  WARNING: {dropped} frames dropped" if dropped else ""
        out = f"Saved {row['file']}  {dur:.1f}s {frames} frames {fps:.1f} fps{warn}"
        print("[REC]", out)
        if not free and sess.pos < len(sess.slots) - 1:
            sess.pos += 1
        free = False
        return out

    def redo():
        nonlocal last_saved
        if last_saved is None:
            return "Nothing to redo in this run"
        src_path = os.path.join(sess.day_dir, last_saved["file"])
        dst_dir = os.path.join(sess.day_dir, "_discarded")
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(dst_dir, os.path.basename(src_path)))
        sess.append(dict(last_saved, status="discarded", note="redo"))
        for i, sl in enumerate(sess.slots):
            if sl[1] == last_saved["label"] and str(sl[2]) == str(last_saved["repeat"]):
                sess.pos = i
                break
        out = f"Discarded {last_saved['file']} (moved to _discarded). Record it again."
        last_saved = None
        return out

    try:
        if headless:
            for _ in range(args.auto_test):
                while src.latest()[0] is None:
                    time.sleep(0.05)
                print("[REC]", start())
                time.sleep(2.0)
                stop()
            return 0

        while True:
            frame, _ = src.latest()
            if frame is not None:
                cv2.imshow(win, draw_overlay(frame, sess, writer, free, msg, cam_name))
            k = cv2.waitKey(15) & 0xFF
            if k == 255:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            key = chr(k).lower() if k < 128 else ""
            if k == 32:
                msg = stop() if writer is not None else start()
            elif writer is not None:
                msg = "Stop the recording first (SPACE)"
            elif key == "q":
                break
            elif key == "n":
                sess.pos = min(len(sess.slots) - 1, sess.pos + 1)
                free = False
                msg = "Next take"
            elif key == "p":
                sess.pos = max(0, sess.pos - 1)
                free = False
                msg = "Previous take"
            elif key == "r":
                msg = redo()
            elif key == "f":
                free = not free
                msg = "Free take mode" if free else "Back to the plan"
            elif key == "i" and not args.source:
                src.close()
                for step in range(1, 6):
                    cand = (cam_idx + step) % 6
                    c = open_camera(cand)
                    if c is not None:
                        cam_idx, cam_name = cand, f"camera {cand}"
                        break
                else:
                    c = open_camera(cam_idx)
                if c is None:
                    print("[ERROR] No camera could be opened.")
                    break
                src = Source(c)
                msg = f"Switched to {cam_name}"
    finally:
        if writer is not None:
            stop()
        src.close()
        if not headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
