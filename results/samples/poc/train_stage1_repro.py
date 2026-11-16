"""Stage 1 student trained the way the shipped detector was: Ultralytics 8.4.6, where optimizer=auto on a small set
means MuSGD(lr 0.002, muon 0.5, sgd 0.5). 8.4.9 (in .venv) picks AdamW(0.002) or MuSGD(0.01) instead.

  .venv/Scripts/python.exe -m pip install --no-deps --target .ul846 ultralytics==8.4.6      # once
  DEV=0 python train_stage1_repro.py <REPO> orig   # original 752 images, original arguments -> weights/stage1_repro846_orig.pt
  DEV=1 python train_stage1_repro.py <REPO> lab    # lab1114 set, same optimizer forced     -> weights/stage1_repro846_lab.pt
  python train_stage1_repro.py <REPO> teacher      # YOLO26x on the lab set, same optimizer -> weights/stage1_repro846_teacher.pt
  python train_stage1_repro.py <REPO> kd           # as lab, plus the KD loss of train_stage1_kd.py -> weights/stage1_repro846_kd.pt
"""
import os, shutil, sys

REPO, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
sys.path.insert(0, os.path.join(REPO, ".ul846"))

if __name__ == "__main__":
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer
    assert ultralytics.__version__ == "8.4.6", ultralytics.__version__
    out = os.path.join(REPO, "weights", f"stage1_repro846_{MODE}.pt")
    assert not os.path.exists(out), out + " exists"
    extra, lab = {}, MODE != "orig"
    if lab:
        # the lab set has more than 10000 iterations, where auto switches to lr 0.01 and muon 0.1 / sgd 1.0: keep the small-set values
        build = BaseTrainer.build_optimizer
        BaseTrainer.build_optimizer = lambda self, iterations=1e5, **kw: build(self, iterations=min(iterations, 9999), **kw)
        extra = dict(optimizer="MuSGD", lr0=0.002, momentum=0.9, warmup_bias_lr=0.0)
    teacher_pt = os.path.join(REPO, "weights", "stage1_repro846_teacher.pt")
    size = dict(epochs=200, patience=20, batch=12, nbs=64) if MODE == "teacher" else dict(epochs=250, patience=30, batch=32)
    model = YOLO(r"D:/Backup/yolo26x.pt" if MODE == "teacher" else r"D:/Backup/yolo26s.pt")
    if MODE == "kd":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_stage1_kd import KD
        teacher = YOLO(teacher_pt).model.to(f"cuda:{os.environ.get('DEV', 0)}").eval()
        for q in teacher.parameters():
            q.requires_grad = False
        model.add_callback("on_train_start", lambda t: setattr(t.model, "criterion", KD(t.model.init_criterion(), teacher, 1.0)))
    r = model.train(
        data=os.path.join(REPO, "data", "stage1_negatives", "lab1114" if lab else "orig", "data.yaml"), imgsz=640,
        workers=4, device=int(os.environ.get("DEV", 0)), degrees=15.0, mosaic=1.0, mixup=0.1, plots=False, exist_ok=True,
        project=os.path.join(REPO, "results", "stage1_kd"), name=f"repro846_{MODE}", **size, **extra)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
