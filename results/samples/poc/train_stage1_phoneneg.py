"""Fine-tune the shipped Stage 1 detector with the phone false detections as negatives (see extract_phone_negatives.py).
Never overwrites weights/yolo_weapon_distilled.pt: the result goes to weights/yolo_weapon_distilled_phoneneg.pt.
  python train_stage1_phoneneg.py <REPO> build|train|check [weights]
build: data/stage1_negatives/ft = backup Stage 1 set (D:/Backup/Weapon_Detection_Split, label class 0 -> 1 'knife' of the
       shipped two-class model) + phone negatives (takes 006, 007, 009 train; take 008 val, so the check on 008 is held out).
check: share of frames with a detection at the live confidence (0.45) on phone take 008 and on knife takes."""
import glob, os, shutil, sys

REPO, MODE = sys.argv[1], sys.argv[2]
SRC = r"D:\Backup\Weapon_Detection_Split"
NEG = os.path.join(REPO, "data", "stage1_negatives", "phone_2026-11-05")
FT = os.path.join(REPO, "data", "stage1_negatives", "ft")
BASE = os.path.join(REPO, "weights", "yolo_weapon_distilled.pt")
OUT = os.path.join(REPO, "weights", "yolo_weapon_distilled_phoneneg.pt")

if MODE == "build":
    for split in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(FT, d, split), exist_ok=True)
        for img in glob.glob(os.path.join(SRC, "images", split, "*")):
            shutil.copy2(img, os.path.join(FT, "images", split))
            lab = os.path.join(SRC, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt")
            rows = [l.split() for l in open(lab)] if os.path.exists(lab) else []
            assert all(r[0] == "0" for r in rows), lab
            with open(os.path.join(FT, "labels", split, os.path.basename(lab)), "w") as f:
                f.writelines("1 " + " ".join(r[1:]) + "\n" for r in rows)  # shipped model: 0 hand_grip_knife, 1 knife
    for img in glob.glob(os.path.join(NEG, "images", "*.jpg")):
        split = "val" if os.path.basename(img).startswith("008") else "train"
        shutil.copy2(img, os.path.join(FT, "images", split))
        open(os.path.join(FT, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt"), "w").close()
    # Lab knife positives: without them the phone negatives teach "lab camera = no knife" (first run: knife takes 0-11 %).
    # shortcut: boxes are the shipped detector's own detections at conf >= 0.5, not hand labels; hand-label if this is kept
    import cv2, numpy as np
    sys.path.insert(0, REPO)
    from scripts.record_takes import crop_4_3
    CHECK = ("002", "005", "010", "014", "018", "022")  # kept out so the check on them is on unseen takes
    pos = 0
    for npy in sorted(glob.glob(os.path.join(REPO, "results", "samples", "knife", "*.npy"))):
        name = os.path.splitext(os.path.basename(npy))[0]
        if name[:3] in CHECK + ("001", "006", "007", "008", "009"):
            continue
        box = np.load(npy)
        cap = cv2.VideoCapture(os.path.join(REPO, "data", "lab_recordings", "2026-11-05", "train_attack", name + ".avi"))
        for i in np.flatnonzero(box[:, 4] >= 0.5)[::15]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            img = crop_4_3(frame)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = box[i, :4]
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, (name, i, box[i], w, h)
            cv2.imwrite(os.path.join(FT, "images", "train", f"{name}_f{i:05d}.jpg"), img)
            with open(os.path.join(FT, "labels", "train", f"{name}_f{i:05d}.txt"), "w") as f:
                f.write(f"1 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            pos += 1
        cap.release()
    print("lab knife positives:", pos)
    with open(os.path.join(FT, "data.yaml"), "w") as f:
        f.write(f"path: {FT}\ntrain: images/train\nval: images/val\nnames:\n  0: hand_grip_knife\n  1: knife\n")
    print({s: len(os.listdir(os.path.join(FT, "images", s))) for s in ("train", "val")})

elif MODE == "train":
    from ultralytics import YOLO
    assert not os.path.exists(OUT), OUT + " exists; move it away first"
    # shortcut: plain fine-tune of the student at a low rate, no teacher; redo the distillation if this is kept
    r = YOLO(BASE).train(data=os.path.join(FT, "data.yaml"), epochs=40, imgsz=640, batch=16, lr0=0.001, lrf=0.1, warmup_epochs=1,
                         device=int(os.environ.get("DEV", 0)), project=os.path.join(REPO, "results", "stage1_phoneneg"), name="ft", exist_ok=True, workers=0)  # workers>0 hangs on Windows: this script has no __main__ guard
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), OUT)
    print("saved", OUT)

elif MODE == "check":
    import cv2
    from ultralytics import YOLO
    sys.path.insert(0, REPO)
    from scripts.record_takes import crop_4_3
    model = YOLO(sys.argv[3])
    src = os.path.join(REPO, "data", "lab_recordings", "2026-11-05", "train_attack")
    for num, what in (("008", "phone, not in the fine-tune"), ("006", "phone"), ("007", "phone"), ("009", "phone"),
                      ("002", "knife"), ("005", "knife"), ("010", "knife"), ("014", "knife"), ("018", "knife"), ("022", "knife")):
        cap = cv2.VideoCapture(glob.glob(os.path.join(src, num + "_*.avi"))[0])
        n = hit = i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % 3 == 0:
                b = model.predict(crop_4_3(frame), imgsz=(384, 640), conf=0.45, verbose=False, device=int(os.environ.get("DEV", 0)), half=True)[0].boxes
                n += 1; hit += len(b) > 0
            i += 1
        cap.release()
        print(f"take {num} ({what}): detection in {hit / n:.0%} of {n} frames", flush=True)
