"""Nov 5 takes 006-009 were performed with a phone, not a knife. Every frame where Stage 1 still reports a knife is a
false detection: saved as a YOLO negative (image + empty label file) for Stage 1 retraining.
  python extract_phone_negatives.py <REPO> [min_conf=0.25] [every=5]
Writes data/stage1_negatives/phone_2026-11-05/{images,labels} and prints counts per take and class."""
import glob, os, sys
from collections import Counter
import cv2
from ultralytics import YOLO

REPO = sys.argv[1]
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 5  # shortcut: every 5th flagged frame, neighbours are near-duplicates
sys.path.insert(0, REPO)
from scripts.record_takes import crop_4_3

src = os.path.join(REPO, "data", "lab_recordings", "2026-11-05", "train_attack")
out = os.path.join(REPO, "data", "stage1_negatives", "phone_2026-11-05")
for d in ("images", "labels"):
    os.makedirs(os.path.join(out, d), exist_ok=True)
model = YOLO(os.path.join(REPO, "weights", "yolo_weapon_distilled.pt"))
for num in ("006", "007", "008", "009"):
    path = glob.glob(os.path.join(src, num + "_*.avi"))[0]
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    i = flagged = saved = 0
    cls, confs = Counter(), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = crop_4_3(frame)
        b = model.predict(img, imgsz=(384, 640), conf=CONF, verbose=False, device="cuda:0", half=True)[0].boxes
        if len(b):
            j = int(b.conf.argmax())
            cls[model.names[int(b.cls[j])]] += 1; confs.append(float(b.conf[j]))
            if flagged % EVERY == 0:
                stem = f"{name}_f{i:05d}"
                cv2.imwrite(os.path.join(out, "images", stem + ".jpg"), img)
                open(os.path.join(out, "labels", stem + ".txt"), "w").close()  # empty = no weapon in this image
                saved += 1
            flagged += 1
        i += 1
    cap.release()
    hi = sum(c >= 0.5 for c in confs)
    print(f"{name}: {i} frames, flagged {flagged} ({flagged / max(i, 1):.0%}), at conf>=0.5 {hi} ({hi / max(i, 1):.0%}), classes {dict(cls)}, saved {saved}", flush=True)
assert os.listdir(os.path.join(out, "images")), "no flagged frames saved"
