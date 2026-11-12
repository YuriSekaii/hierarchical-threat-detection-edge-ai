"""Knife boxes for the Nov 5 takes, in the same 4:3 crop pixels as results/samples/kp.
Writes results/samples/knife/<take>.npy: (frames, 5) x1, y1, x2, y2, confidence; zeros where no knife was found."""
import csv, os, sys
import cv2, numpy as np

REPO = sys.argv[1]
sys.path.insert(0, REPO)
from ultralytics import YOLO
from scripts.record_takes import crop_4_3

day = os.path.join(REPO, "data", "lab_recordings", "2026-11-05")
out = os.path.join(REPO, "results", "samples", "knife")
os.makedirs(out, exist_ok=True)
model = YOLO(os.path.join(REPO, "weights", "yolo_weapon_distilled.pt"))
print("classes:", model.names, flush=True)
for r in csv.DictReader(open(os.path.join(day, "takes.csv"), newline="", encoding="utf-8")):
    name = os.path.splitext(os.path.basename(r["file"].replace("\\", "/")))[0]
    dst = os.path.join(out, name + ".npy")
    if r["status"] != "ok" or os.path.exists(dst):
        continue
    cap = cv2.VideoCapture(os.path.join(day, r["file"].replace("\\", "/")))
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b = model.predict(crop_4_3(frame), imgsz=(384, 640), conf=0.25, verbose=False, device="cuda:0", half=True)[0].boxes
        k = np.zeros(5, np.float32)
        if len(b):
            i = int(b.conf.argmax())
            k[:4], k[4] = b.xyxy[i].cpu().numpy(), float(b.conf[i])
        rows.append(k)
    cap.release()
    np.save(dst, np.stack(rows))
    print(f"{name}: {len(rows)} frames, knife in {(np.stack(rows)[:, 4] > 0).mean():.0%}", flush=True)
