"""
Skeletons of recorded takes, the way the live pipeline sees them: 4:3 centre crop, YOLO26s-pose at 384x640
(grey bars on the sides, as the TensorRT engine runs it) with ByteTrack, largest person first.

  python scripts/extract_take_skeletons.py data/lab_recordings/2026-11-05 --out results/samples/kp

Writes <out>/<take>.npy: (frames, 17, 3) x, y in crop pixels and confidence; zeros where nobody was found.
Existing files are kept.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ultralytics import YOLO  # noqa: E402

from scripts.record_takes import crop_4_3  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract(path, model, device):
    cap = cv2.VideoCapture(path)
    model.predictor = None  # new tracker state for each take
    out, actor = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(crop_4_3(frame), persist=True, imgsz=(384, 640), verbose=False, device=device,
                        half=True, tracker="bytetrack.yaml")[0]
        k = np.zeros((17, 3), np.float32)
        if len(r.boxes):
            ids = r.boxes.id.cpu().numpy().astype(int).tolist() if r.boxes.id is not None else []
            if actor in ids:
                i = ids.index(actor)
            else:  # same fallback as Stage2aPoseTracker: largest box
                i = int(((r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])).argmax())
                actor = ids[i] if ids else None
            k[:, :2] = r.keypoints.xy[i].cpu().numpy()
            k[:, 2] = r.keypoints.conf[i].cpu().numpy()
        out.append(k)
    cap.release()
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(os.path.join(ROOT, "weights", "yolo26s-pose.pt"), task="pose")
    with open(os.path.join(args.day, "takes.csv"), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    for r in rows:
        name = os.path.splitext(os.path.basename(r["file"]))[0]
        dst = os.path.join(args.out, name + ".npy")
        if os.path.exists(dst):
            continue
        k = extract(os.path.join(args.day, r["file"].replace("\\", "/")), model, args.device)
        np.save(dst, k)
        print(f"{name}: {len(k)} frames, person in {(k[..., 2].sum(-1) > 0).mean():.0%}", flush=True)


if __name__ == "__main__":
    main()
