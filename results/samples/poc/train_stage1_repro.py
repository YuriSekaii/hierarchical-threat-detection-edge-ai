"""Stage 1 student trained the way the shipped detector was: Ultralytics 8.4.6, where optimizer=auto on a small set
means MuSGD(lr 0.002, muon 0.5, sgd 0.5). 8.4.9 (in .venv) picks AdamW(0.002) or MuSGD(0.01) instead.

  .venv/Scripts/python.exe -m pip install --no-deps --target .ul846 ultralytics==8.4.6      # once
  DEV=0 python train_stage1_repro.py <REPO> orig   # original 752 images, original arguments -> weights/stage1_repro846_orig.pt
  DEV=1 python train_stage1_repro.py <REPO> lab    # lab1114 set, same optimizer forced     -> weights/stage1_repro846_lab.pt
  python train_stage1_repro.py <REPO> teacher      # YOLO26x on the lab set, same optimizer -> weights/stage1_repro846_teacher.pt
  python train_stage1_repro.py <REPO> kd           # as lab, plus the KD loss of train_stage1_kd.py -> weights/stage1_repro846_kd.pt
  python train_stage1_repro.py <REPO> clean        # as orig, on data/stage1_negatives/orig_clean (analyse_stage1_misses.py trainneg)
  PATIENCE=100 python train_stage1_repro.py <REPO> clean   # same with early-stopping patience 100 -> ..._clean_p100.pt
  PATIENCE=100 python train_stage1_repro.py <REPO> v2      # as orig, on the relabelled set orig_v2 (relabel_stage1.py)
  PATIENCE=100 python train_stage1_repro.py <REPO> labv2   # as lab, on lab_v2 = lab frames + orig_v2
  EPOCHS=250 PATIENCE=250 python train_stage1_repro.py <REPO> teachergt   # YOLO26x teacher on lab_gt, 250 epochs like the students
  TEACHER=weights/x.pt SUFFIX=_x PATIENCE=250 python train_stage1_repro.py <REPO> kdgt   # lab_gt + distillation from TEACHER
  PATIENCE=250 python train_stage1_repro.py <REPO> labgt   # as lab, on lab_gt = person-checked lab frames (label_lab_stage1.py trainset) + orig_v2
"""
import os, shutil, sys

REPO, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
sys.path.insert(0, os.path.join(REPO, ".ul846"))

if __name__ == "__main__":
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer
    assert ultralytics.__version__ == "8.4.6", ultralytics.__version__
    PAT = os.environ.get("PATIENCE")  # the validation set is small: patience 30 can stop on a chance peak (clean stopped at 47, best 17)
    TAG = MODE + (f"_p{PAT}" if PAT else "") + os.environ.get("SUFFIX", "")
    out = os.path.join(REPO, "weights", f"stage1_repro846_{TAG}.pt")
    assert not os.path.exists(out), out + " exists"
    extra, lab = {}, MODE not in ("orig", "clean", "v2")
    if lab:
        # the lab set has more than 10000 iterations, where auto switches to lr 0.01 and muon 0.1 / sgd 1.0: keep the small-set values
        build = BaseTrainer.build_optimizer
        BaseTrainer.build_optimizer = lambda self, iterations=1e5, **kw: build(self, iterations=min(iterations, 9999), **kw)
        extra = dict(optimizer="MuSGD", lr0=0.002, momentum=0.9, warmup_bias_lr=0.0)
    teacher_pt = os.path.join(REPO, os.environ.get("TEACHER", "weights/stage1_repro846_teacher.pt"))
    size = dict(epochs=200, patience=20, batch=12, nbs=64) if MODE.startswith("teacher") else dict(epochs=250, patience=30, batch=32)
    if PAT:
        size["patience"] = int(PAT)
    if os.environ.get("EPOCHS"):  # the teacher recipe is 200; EPOCHS=250 matches the students
        size["epochs"] = int(os.environ["EPOCHS"])
    model = YOLO(r"D:/Backup/yolo26x.pt" if MODE.startswith("teacher") else r"D:/Backup/yolo26s.pt")
    if MODE.startswith("kd"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_stage1_kd import KD
        teacher = YOLO(teacher_pt).model.to(f"cuda:{os.environ.get('DEV', 0)}").eval()
        for q in teacher.parameters():
            q.requires_grad = False
        model.add_callback("on_train_start", lambda t: setattr(t.model, "criterion", KD(t.model.init_criterion(), teacher, 1.0)))
    r = model.train(
        data=os.path.join(REPO, "data", "stage1_negatives", {"orig": "orig", "clean": "orig_clean", "v2": "orig_v2", "labv2": "lab_v2", "labgt": "lab_gt", "teachergt": "lab_gt", "kdgt": "lab_gt"}.get(MODE, "lab1114"), "data.yaml"), imgsz=640,
        workers=4, device=int(os.environ.get("DEV", 0)), degrees=15.0, mosaic=1.0, mixup=0.1, plots=False, exist_ok=True,
        project=os.path.join(REPO, "results", "stage1_kd"), name=f"repro846_{TAG}", **size, **extra)
    shutil.copy2(os.path.join(str(r.save_dir), "weights", "best.pt"), out)
