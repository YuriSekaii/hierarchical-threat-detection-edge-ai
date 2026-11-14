"""Stage 1 with a real teacher: fine-tune YOLO26x, then distil it into YOLO26s with a distillation loss.
The original train_yolo_distill.py loads the teacher but never uses it; here the teacher's outputs enter the loss.

  python train_stage1_kd.py <REPO> build              # data/stage1_negatives/lab1114 (ft_phonecls + Nov 14 lab frames)
  python train_stage1_kd.py <REPO> teacher            # YOLO26x, original recipe  -> weights/stage1_kd_teacher.pt
  python train_stage1_kd.py <REPO> student [kd]       # YOLO26s, original recipe + KD -> weights/stage1_kd_student.pt
  python train_stage1_kd.py <REPO> student 0          # same without KD (control) -> weights/stage1_kd_control.pt
DEV=<gpu> selects the GPU. Never writes weights/yolo_weapon_distilled.pt.

Data: classes 0 hand_grip_knife (unused), 1 knife, 2 phone (also pen, wallet, remote: the benchmark filters by this name).
shortcut: lab boxes are the shipped detector's own detections (knife >= 0.5, other objects >= 0.25), not hand labels,
so lab knives the shipped detector misses are not in the set; hand-label lab frames if this is kept.
Nov 14 test takes 025-036 are never read. Takes in VAL go to validation only.
"""
import glob, os, shutil, sys

REPO, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
DEV = int(os.environ.get("DEV", 0))
BASE = os.path.join(REPO, "data", "stage1_negatives", "ft_phonecls")
SET = os.path.join(REPO, "data", "stage1_negatives", "lab1114")
DAY = os.path.join(REPO, "data", "lab_recordings", "2026-11-14")
PROJECT = os.path.join(REPO, "results", "stage1_kd")
TEACHER = os.path.join(REPO, "weights", "stage1_kd_teacher.pt")
VAL = ("020", "041", "050", "056")  # inspect_knife, thrust_low_knife_right, phone2_strike, remote_strike
RECIPE = dict(imgsz=640, degrees=15.0, mosaic=1.0, mixup=0.1, fliplr=0.5, workers=4, plots=False, exist_ok=True)


def build():
    import cv2
    from ultralytics import YOLO
    sys.path.insert(0, REPO)
    from scripts.record_takes import crop_4_3
    model = YOLO(os.path.join(REPO, "weights", "yolo_weapon_distilled.pt"))
    for s in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(SET, d, s), exist_ok=True)
    count = {}
    for path in sorted([p for d in ("train_attack", "train_normal", "confuser") for p in glob.glob(os.path.join(DAY, d, "*.avi"))]):
        folder, name = os.path.basename(os.path.dirname(path)), os.path.splitext(os.path.basename(path))[0]
        if folder.startswith("test") or "test" in name:
            continue
        other = folder == "confuser"
        split = "val" if name[:3] in VAL else "train"
        cap, i, n = cv2.VideoCapture(path), -1, 0
        while True:
            ok, frame = cap.read()
            i += 1
            if not ok:
                break
            if i % 15:
                continue
            img = crop_4_3(frame)
            h, w = img.shape[:2]
            b = model.predict(img, imgsz=(384, 640), conf=0.25, verbose=False, device=DEV, half=True)[0].boxes
            if len(b) == 0 and not (other and i % 60 == 0):
                continue  # knife takes: undetected frames are skipped, an empty label would call a held knife background
            if len(b) and not other and float(b.conf.max()) < 0.5:
                continue
            stem = f"n14_{name}_f{i:05d}"
            cv2.imwrite(os.path.join(SET, "images", split, stem + ".jpg"), img)
            with open(os.path.join(SET, "labels", split, stem + ".txt"), "w") as f:
                if len(b):
                    x1, y1, x2, y2 = b.xyxy[int(b.conf.argmax())].tolist()
                    f.write(f"{2 if other else 1} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            n += 1
        cap.release()
        count[name] = n
        print(name, split, n, flush=True)
    for s in ("train", "val"):
        imgs = glob.glob(os.path.join(BASE, "images", s, "*")) + glob.glob(os.path.join(SET, "images", s, "*"))
        with open(os.path.join(SET, s + ".txt"), "w") as f:
            f.write("\n".join(p.replace("\\", "/") for p in sorted(imgs)) + "\n")
        print(s, len(imgs), "images")
    with open(os.path.join(SET, "data.yaml"), "w") as f:
        f.write(f"path: {SET}\ntrain: train.txt\nval: val.txt\nnames:\n  0: hand_grip_knife\n  1: knife\n  2: phone\n")
    assert sum(count.values()) > 0


class KD:
    """Wraps the detection loss: adds the teacher's class scores and box distances as soft targets on every anchor."""

    def __init__(self, crit, teacher, weight):
        self.crit, self.teacher, self.w = crit, teacher, weight

    def __getattr__(self, k):  # trainer calls criterion.update()
        return getattr(self.crit, k)

    def __call__(self, preds, batch):
        import torch
        import torch.nn.functional as F
        loss, items = self.crit(preds, batch)
        s = self.crit.one2many.parse_output(preds)
        with torch.no_grad():
            t = self.crit.one2many.parse_output(self.teacher(batch["img"]))
        kd = 0.0
        for head, g in (("one2many", self.crit.o2m), ("one2one", self.crit.o2o)):
            ss, ts = s[head]["scores"].float(), t[head]["scores"].float().sigmoid()
            assert ss.shape == ts.shape and s[head]["boxes"].shape == t[head]["boxes"].shape, (ss.shape, ts.shape)
            norm = ts.sum().clamp(min=1.0)
            cls = F.binary_cross_entropy_with_logits(ss, ts, reduction="sum") / norm
            wgt = ts.amax(1, keepdim=True)  # box error counts where the teacher sees an object
            box = ((s[head]["boxes"].float() - t[head]["boxes"].float()).abs() * wgt).sum() / (4 * wgt.sum().clamp(min=1.0))
            kd = kd + g * (cls + box)
        return loss + self.w * kd * batch["img"].shape[0], items


def train_teacher():
    from ultralytics import YOLO
    assert not os.path.exists(TEACHER), TEACHER + " exists"
    r = YOLO(r"D:/Backup/yolo26x.pt").train(data=os.path.join(SET, "data.yaml"), epochs=200, patience=20, batch=12, nbs=64,
                                           device=DEV, project=PROJECT, name="teacher", **RECIPE)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), TEACHER)


def train_student(kd):
    from ultralytics import YOLO
    name = "student" if kd else "control"
    out = os.path.join(REPO, "weights", f"stage1_kd_{name}.pt")
    assert not os.path.exists(out), out + " exists"
    model = YOLO(r"D:/Backup/yolo26s.pt")
    if kd:
        teacher = YOLO(TEACHER).model.to(f"cuda:{DEV}").eval()
        for p in teacher.parameters():
            p.requires_grad = False

        def hook(trainer):  # after the EMA copy is made, so the teacher is not saved inside the student checkpoint
            trainer.model.criterion = KD(trainer.model.init_criterion(), teacher, kd)
        model.add_callback("on_train_start", hook)
    r = model.train(data=os.path.join(SET, "data.yaml"), epochs=250, patience=30, batch=32, device=DEV, project=PROJECT,
                    name=name, **RECIPE)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)


if __name__ == "__main__":
    if MODE == "build":
        build()
    elif MODE == "teacher":
        train_teacher()
    else:
        train_student(float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
