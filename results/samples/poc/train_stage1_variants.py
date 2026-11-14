"""Train Stage 1 knife-detector variants building on data/stage1_negatives/ft.

Usage: python train_stage1_variants.py <REPO> <variant> [build]

Variants:
  freeze    fine-tune weights/yolo_weapon_distilled.pt on ft (freeze=10, epochs=20, lr0=0.0005)
  phonecls  build ft_phonecls (add class-2 phone boxes) then train from D:/Backup/yolo26s.pt (3-class, epochs=80)
  scratch2  train D:/Backup/yolo26s.pt on ft (2-class, phone frames as background, epochs=80)
  big       same as phonecls data but from D:/Backup/yolo26x.pt (epochs=60, batch=8)
  cropcls   build crops/{train,val}/{knife,other} and train a classifier (epochs=60, imgsz=128, batch=64)

Optional third arg `build` stops after dataset construction, before training.
"""
import glob, os, random, shutil, sys

REPO, VARIANT = sys.argv[1], sys.argv[2]
BUILD_ONLY = len(sys.argv) > 3 and sys.argv[3] == "build"

SRC = r"D:\Backup\Weapon_Detection_Split"
FT = os.path.join(REPO, "data", "stage1_negatives", "ft")
FT_PHONECLS = os.path.join(REPO, "data", "stage1_negatives", "ft_phonecls")
CROPS = os.path.join(REPO, "data", "stage1_negatives", "crops")
BASE = os.path.join(REPO, "weights", "yolo_weapon_distilled.pt")
YOLO26S = r"D:/Backup/yolo26s.pt"
YOLO26X = r"D:/Backup/yolo26x.pt"
PROJECT = os.path.join(REPO, "results", "stage1_phoneneg")
DEVICE = int(os.environ.get("DEV", 0))
CHECK_TAKES = ("002", "005", "010", "014", "018", "022")
PHONE_PREFIXES = ("006_", "007_", "008_", "009_")


def is_phone_image(name):
    return name.startswith(PHONE_PREFIXES)


def counts_per_split_class(img_dir, lab_dir):
    """Return {class_id: count} for labels in lab_dir, plus total image count."""
    counts = {}
    total = 0
    if os.path.isdir(img_dir):
        total = len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if os.path.isdir(lab_dir):
        for f in os.listdir(lab_dir):
            if not f.endswith(".txt"):
                continue
            with open(os.path.join(lab_dir, f)) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    cls = int(line.split()[0])
                    counts[cls] = counts.get(cls, 0) + 1
    return total, counts


def print_counts(tag, base_dir):
    for split in ("train", "val"):
        img_dir = os.path.join(base_dir, "images", split)
        lab_dir = os.path.join(base_dir, "labels", split)
        total, counts = counts_per_split_class(img_dir, lab_dir)
        cls_str = ", ".join(f"class {k}: {v}" for k, v in sorted(counts.items()))
        print(f"[{tag}] {split}: {total} images, {cls_str}", flush=True)


def build_ft_if_missing():
    """Build data/stage1_negatives/ft following train_stage1_phoneneg.py build recipe."""
    if os.path.exists(FT) and os.path.exists(os.path.join(FT, "data.yaml")):
        print(f"[build] {FT} already exists, reusing", flush=True)
        print_counts("ft", FT)
        return
    from scripts.record_takes import crop_4_3
    import cv2, numpy as np
    sys.path.insert(0, REPO)
    NEG = os.path.join(REPO, "data", "stage1_negatives", "phone_2026-11-05")
    for split in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(FT, d, split), exist_ok=True)
        for img in glob.glob(os.path.join(SRC, "images", split, "*")):
            shutil.copy2(img, os.path.join(FT, "images", split))
            lab = os.path.join(SRC, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt")
            rows = [l.split() for l in open(lab)] if os.path.exists(lab) else []
            assert all(r[0] == "0" for r in rows), lab
            with open(os.path.join(FT, "labels", split, os.path.basename(lab)), "w") as f:
                f.writelines("1 " + " ".join(r[1:]) + "\n" for r in rows)
    for img in glob.glob(os.path.join(NEG, "images", "*.jpg")):
        split = "val" if os.path.basename(img).startswith("008") else "train"
        shutil.copy2(img, os.path.join(FT, "images", split))
        open(os.path.join(FT, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt"), "w").close()
    pos = 0
    for npy in sorted(glob.glob(os.path.join(REPO, "results", "samples", "knife", "*.npy"))):
        name = os.path.splitext(os.path.basename(npy))[0]
        if name[:3] in CHECK_TAKES + ("001", "006", "007", "008", "009"):
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
    print("lab knife positives:", pos, flush=True)
    with open(os.path.join(FT, "data.yaml"), "w") as f:
        f.write(f"path: {FT}\ntrain: images/train\nval: images/val\nnames:\n  0: hand_grip_knife\n  1: knife\n")
    print_counts("ft", FT)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aarea = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    barea = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (aarea + barea - inter)


def xywhn_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def enlarge_box(x1, y1, x2, y2, factor=1.5):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def build_phonecls():
    """Build data/stage1_negatives/ft_phonecls: copy of ft with class-2 phone labels on phone frames."""
    if os.path.exists(FT_PHONECLS) and os.path.exists(os.path.join(FT_PHONECLS, "data.yaml")):
        print(f"[build] {FT_PHONECLS} already exists, reusing", flush=True)
        print_counts("ft_phonecls", FT_PHONECLS)
        return
    build_ft_if_missing()
    for split in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(FT_PHONECLS, d, split), exist_ok=True)
    for split in ("train", "val"):
        for fname in os.listdir(os.path.join(FT, "images", split)):
            src_img = os.path.join(FT, "images", split, fname)
            dst_img = os.path.join(FT_PHONECLS, "images", split, fname)
            shutil.copy2(src_img, dst_img)
            src_lab = os.path.join(FT, "labels", split, os.path.splitext(fname)[0] + ".txt")
            dst_lab = os.path.join(FT_PHONECLS, "labels", split, os.path.splitext(fname)[0] + ".txt")
            shutil.copy2(src_lab, dst_lab)
    from ultralytics import YOLO
    model = YOLO(BASE)
    phone_imgs = []
    for split in ("train", "val"):
        for fname in os.listdir(os.path.join(FT_PHONECLS, "images", split)):
            if is_phone_image(fname):
                phone_imgs.append((split, fname))
    print(f"[build] phone frames to label: {len(phone_imgs)}", flush=True)
    labelled = 0
    empty = 0
    for split, fname in phone_imgs:
        img_path = os.path.join(FT_PHONECLS, "images", split, fname)
        results = model.predict(img_path, imgsz=(384, 640), conf=0.20, verbose=False, device=DEVICE)[0]
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            best = int(confs.argmax())
            b = boxes.xyxy[best].cpu().numpy()
            import cv2
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = b
            x1 = max(0, min(x1, w)); y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w)); y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                open(os.path.join(FT_PHONECLS, "labels", split, os.path.splitext(fname)[0] + ".txt"), "w").close()
                empty += 1
                continue
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            with open(os.path.join(FT_PHONECLS, "labels", split, os.path.splitext(fname)[0] + ".txt"), "w") as f:
                f.write(f"2 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            labelled += 1
        else:
            open(os.path.join(FT_PHONECLS, "labels", split, os.path.splitext(fname)[0] + ".txt"), "w").close()
            empty += 1
    print(f"[build] phonecls: labelled={labelled} empty={empty}", flush=True)
    with open(os.path.join(FT_PHONECLS, "data.yaml"), "w") as f:
        f.write(f"path: {FT_PHONECLS}\ntrain: images/train\nval: images/val\nnames:\n  0: hand_grip_knife\n  1: knife\n  2: phone\n")
    print_counts("ft_phonecls", FT_PHONECLS)


def build_cropcls():
    """Build data/stage1_negatives/crops/{train,val}/{knife,other}."""
    build_ft_if_missing()
    build_phonecls()
    for split in ("train", "val"):
        for cls in ("knife", "other"):
            os.makedirs(os.path.join(CROPS, split, cls), exist_ok=True)
    from ultralytics import YOLO
    import cv2
    model = YOLO(BASE)
    knife_count = {"train": 0, "val": 0}
    other_count = {"train": 0, "val": 0}
    rng = random.Random(42)
    for split in ("train", "val"):
        img_dir = os.path.join(FT, "images", split)
        lab_dir = os.path.join(FT, "labels", split)
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            lab_path = os.path.join(lab_dir, os.path.splitext(fname)[0] + ".txt")
            labelled_boxes = []
            if os.path.exists(lab_path):
                with open(lab_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls_id = int(parts[0])
                        if cls_id == 1:
                            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            x1, y1, x2, y2 = xywhn_to_xyxy(cx, cy, bw, bh, w, h)
                            labelled_boxes.append((x1, y1, x2, y2))
            # Knife crops: every labelled knife box enlarged 50%, clipped
            for (x1, y1, x2, y2) in labelled_boxes:
                ex1, ey1, ex2, ey2 = enlarge_box(x1, y1, x2, y2, 2.0)
                ex1 = max(0, min(ex1, w)); ey1 = max(0, min(ey1, h))
                ex2 = max(0, min(ex2, w)); ey2 = max(0, min(ey2, h))
                if ex2 <= ex1 or ey2 <= ey1:
                    continue
                crop = img[int(ey1):int(ey2), int(ex1):int(ex2)]
                if crop.size == 0:
                    continue
                out = os.path.join(CROPS, split, "knife", f"{fname}_{knife_count[split]:05d}.jpg")
                cv2.imwrite(out, crop)
                knife_count[split] += 1
            # Other crops: phone boxes from ft_phonecls
            phone_lab = os.path.join(FT_PHONECLS, "labels", split, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(phone_lab):
                with open(phone_lab) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if int(parts[0]) != 2:
                            continue
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1, y1, x2, y2 = enlarge_box(*xywhn_to_xyxy(cx, cy, bw, bh, w, h), 2.0)
                        x1 = max(0, min(x1, w)); y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w)); y2 = max(0, min(y2, h))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop = img[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size == 0:
                            continue
                        out = os.path.join(CROPS, split, "other", f"{fname}_{other_count[split]:05d}.jpg")
                        cv2.imwrite(out, crop)
                        other_count[split] += 1
            # False detections: up to 2 boxes at conf>=0.10 with IoU<0.1 to every labelled box
            results = model.predict(img_path, imgsz=(384, 640), conf=0.10, verbose=False, device=DEVICE)[0]
            boxes = results.boxes
            fd_boxes = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                order = confs.argsort()[::-1]
                for idx in order:
                    b = xyxy[idx]
                    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    ok = True
                    for lb in labelled_boxes:
                        if iou_xyxy((bx1, by1, bx2, by2), lb) >= 0.10:
                            ok = False
                            break
                    if ok:
                        fd_boxes.append((bx1, by1, bx2, by2))
                    if len(fd_boxes) >= 2:
                        break
            for (bx1, by1, bx2, by2) in fd_boxes:
                bx1, by1, bx2, by2 = enlarge_box(bx1, by1, bx2, by2, 2.0)
                bx1 = max(0, min(bx1, w)); by1 = max(0, min(by1, h))
                bx2 = max(0, min(bx2, w)); by2 = max(0, min(by2, h))
                if bx2 <= bx1 or by2 <= by1:
                    continue
                crop = img[int(by1):int(by2), int(bx1):int(bx2)]
                if crop.size == 0:
                    continue
                out = os.path.join(CROPS, split, "other", f"{fname}_{other_count[split]:05d}.jpg")
                cv2.imwrite(out, crop)
                other_count[split] += 1
            # One random crop per image of similar size away from labelled boxes
            if labelled_boxes:
                sizes = [(lb[2] - lb[0], lb[3] - lb[1]) for lb in labelled_boxes]
                med_w = sorted(s[0] for s in sizes)[len(sizes) // 2]
                med_h = sorted(s[1] for s in sizes)[len(sizes) // 2]
            else:
                med_w, med_h = 64, 64
            rw, rh = int(med_w * 0.8), int(med_h * 0.8)
            rw = max(8, min(rw, w)); rh = max(8, min(rh, h))
            placed = False
            for _ in range(100):
                rx = rng.randint(0, max(0, w - rw))
                ry = rng.randint(0, max(0, h - rh))
                cand = (float(rx), float(ry), float(rx + rw), float(ry + rh))
                if all(iou_xyxy(cand, lb) < 0.10 for lb in labelled_boxes):
                    crop = img[ry:ry + rh, rx:rx + rw]
                    if crop.size == 0:
                        continue
                    out = os.path.join(CROPS, split, "other", f"{fname}_{other_count[split]:05d}.jpg")
                    cv2.imwrite(out, crop)
                    other_count[split] += 1
                    placed = True
                    break
    print(f"[build] cropcls train: knife={knife_count['train']} other={other_count['train']}", flush=True)
    print(f"[build] cropcls val:   knife={knife_count['val']}   other={other_count['val']}", flush=True)


def train_freeze():
    from ultralytics import YOLO
    out = os.path.join(REPO, "weights", "stage1_freeze.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    r = YOLO(BASE).train(data=os.path.join(FT, "data.yaml"), epochs=20, imgsz=640, batch=16, lr0=0.0005,
                         freeze=10, device=DEVICE, project=PROJECT, name="freeze", exist_ok=True, workers=0)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
    print("saved", out, flush=True)
    # Print final val metrics
    metrics = r.box
    print(f"[freeze] val metrics: mAP50={metrics.map50:.4f} mAP50-95={metrics.map:.4f}", flush=True)


def train_phonecls():
    from ultralytics import YOLO
    out = os.path.join(REPO, "weights", "stage1_phonecls.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    r = YOLO(YOLO26S).train(data=os.path.join(FT_PHONECLS, "data.yaml"), epochs=80, imgsz=640, batch=16,
                            device=DEVICE, project=PROJECT, name="phonecls", exist_ok=True, workers=0)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
    print("saved", out, flush=True)
    metrics = r.box
    print(f"[phonecls] val metrics: mAP50={metrics.map50:.4f} mAP50-95={metrics.map:.4f}", flush=True)


def train_phonecls512():
    """phonecls at the deployed scale: a 4:3 frame in the 384x640 engine is 512 px wide, so train at 512, not 640."""
    from ultralytics import YOLO
    out = os.path.join(REPO, "weights", "stage1_phonecls512.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    r = YOLO(YOLO26S).train(data=os.path.join(FT_PHONECLS, "data.yaml"), epochs=80, imgsz=512, batch=16,
                            device=DEVICE, project=PROJECT, name="phonecls512", exist_ok=True, workers=0)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
    print("saved", out, flush=True)


def train_phonecls_ft():
    """phone class added to the shipped detector itself (keeps its distilled knife weights), at the deployed scale."""
    from ultralytics import YOLO
    out = os.path.join(REPO, "weights", "stage1_phonecls_ft.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    r = YOLO(BASE).train(data=os.path.join(FT_PHONECLS, "data.yaml"), epochs=40, imgsz=512, batch=16, lr0=0.001,
                         device=DEVICE, project=PROJECT, name="phonecls_ft", exist_ok=True, workers=0)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
    print("saved", out, flush=True)


def train_cropcls2():
    """Crop classifier, second version: the first one rejects many lab knives (knife frames 68 % -> 51 %), and its lab knife
    crops were only the detector's confident boxes (>= 0.5). Adds the low-confidence lab knife boxes (below 0.5) of the
    training takes as knife crops and trains at 224 px instead of 128.
    shortcut: those boxes are detector output on knife takes, not hand labels; a few may not be the knife."""
    import cv2, numpy as np
    from ultralytics import YOLO
    sys.path.insert(0, REPO)
    from scripts.record_takes import crop_4_3
    out = os.path.join(REPO, "weights", "stage1_cropcls2.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    crops2 = CROPS + "2"
    if not os.path.exists(crops2):
        shutil.copytree(CROPS, crops2)
        n = 0
        for npy in sorted(glob.glob(os.path.join(REPO, "results", "samples", "knife", "*.npy"))):
            name = os.path.splitext(os.path.basename(npy))[0]
            if name[:3] in CHECK_TAKES + ("001", "006", "007", "008", "009"):
                continue
            box = np.load(npy)
            cap = cv2.VideoCapture(os.path.join(REPO, "data", "lab_recordings", "2026-11-05", "train_attack", name + ".avi"))
            for i in np.flatnonzero((box[:, 4] > 0) & (box[:, 4] < 0.5))[::10]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    continue
                img = crop_4_3(frame)
                h, w = img.shape[:2]
                x1, y1, x2, y2 = enlarge_box(*box[i, :4], 2.0)
                crop = img[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))]
                if crop.size:
                    cv2.imwrite(os.path.join(crops2, "train", "knife", f"{name}_low_f{i:05d}.jpg"), crop)
                    n += 1
            cap.release()
        print("[build] cropcls2: low-confidence lab knife crops added:", n, flush=True)
    r = YOLO("yolo26s-cls.yaml").load(YOLO26S).train(data=crops2, epochs=60, imgsz=224, batch=64, device=DEVICE,
                                                     project=PROJECT, name="cropcls2", exist_ok=True, workers=0)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
    print("saved", out, "top1", getattr(r, "top1", None), flush=True)


def train_scratch2():
    from ultralytics import YOLO
    out = os.path.join(REPO, "weights", "stage1_scratch2.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    r = YOLO(YOLO26S).train(data=os.path.join(FT, "data.yaml"), epochs=80, imgsz=640, batch=16,
                            device=DEVICE, project=PROJECT, name="scratch2", exist_ok=True, workers=0)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
    print("saved", out, flush=True)
    metrics = r.box
    print(f"[scratch2] val metrics: mAP50={metrics.map50:.4f} mAP50-95={metrics.map:.4f}", flush=True)


def train_big():
    from ultralytics import YOLO
    out = os.path.join(REPO, "weights", "stage1_big.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    r = YOLO(YOLO26X).train(data=os.path.join(FT_PHONECLS, "data.yaml"), epochs=60, imgsz=640, batch=8,
                            device=DEVICE, project=PROJECT, name="big", exist_ok=True, workers=0)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
    print("saved", out, flush=True)
    metrics = r.box
    print(f"[big] val metrics: mAP50={metrics.map50:.4f} mAP50-95={metrics.map:.4f}", flush=True)


def train_cropcls():
    from ultralytics import YOLO
    out = os.path.join(REPO, "weights", "stage1_cropcls.pt")
    assert not os.path.exists(out), out + " exists; move it away first"
    try:
        model = YOLO("yolo26s-cls.yaml").load(YOLO26S)
    except Exception:
        model = YOLO("yolo26n-cls.yaml")
    r = model.train(data=CROPS, epochs=60, imgsz=128, batch=64,
                    device=DEVICE, project=PROJECT, name="cropcls", exist_ok=True, workers=0)
    # Save best.pt
    best = os.path.join(str(r.save_dir), "weights", "best.pt")
    shutil.copy2(best, out)
    print("saved", out, flush=True)
    # Print class names and val top-1 accuracy
    names = model.names
    print(f"[cropcls] class names: {names}", flush=True)
    metrics = r
    top1 = getattr(metrics, "top1", None)
    if top1 is None:
        top1 = getattr(metrics, "accuracy", None)
    if top1 is None:
        top1 = metrics.results_dict.get("metrics/accuracy_top1", float("nan")) if hasattr(metrics, "results_dict") else float("nan")
    print(f"[cropcls] val top-1 accuracy: {top1}", flush=True)


if __name__ == "__main__":
    build_ft_if_missing()
    if VARIANT == "freeze":
        if not BUILD_ONLY:
            train_freeze()
    elif VARIANT == "phonecls":
        build_phonecls()
        if not BUILD_ONLY:
            train_phonecls()
    elif VARIANT in ("phonecls512", "phonecls_ft"):
        build_phonecls()
        if not BUILD_ONLY:
            {"phonecls512": train_phonecls512, "phonecls_ft": train_phonecls_ft}[VARIANT]()
    elif VARIANT == "cropcls2":
        train_cropcls2()
    elif VARIANT == "scratch2":
        if not BUILD_ONLY:
            train_scratch2()
    elif VARIANT == "big":
        build_phonecls()
        if not BUILD_ONLY:
            train_big()
    elif VARIANT == "cropcls":
        build_cropcls()
        if not BUILD_ONLY:
            train_cropcls()
    else:
        print(f"unknown variant: {VARIANT}", flush=True)
        sys.exit(1)
