"""Stage 1 knowledge distillation on the phone-negative datasets.

This is a copy of the original Stage 1 distillation recipe (train_yolo_distill.py: train a
yolo26x teacher for 200 epochs, then distil into a yolo26s student for 250 epochs with the
custom KnowledgeDistillationTrainer). Only the data, paths, run names and workers were
changed; the trainer class, hyper-parameters, epochs, imgsz and loss setup are kept as-is.

Usage: python train_stage1_distill_phone.py <REPO> <dataset>
  <dataset> is `ft_phonecls` (3 classes, phone as class 2) or `ft` (2 classes, phone frames
  as background); the data yaml is <REPO>/data/stage1_negatives/<dataset>/data.yaml.
  The distilled student is copied to <REPO>/weights/stage1_distill_<dataset>.pt.
"""
import os
import shutil
import sys
import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from copy import deepcopy
from pathlib import Path


# ==========================================
# CUSTOM TRAINER CLASS
# ==========================================
class KnowledgeDistillationTrainer(DetectionTrainer):
    """
    Custom Trainer that loads and freezes the Teacher model during student training.
    Inherits from Ultralytics standard DetectionTrainer.
    """
    def __init__(self, overrides=None, _callbacks=None):
        self.teacher_file = overrides.pop('teacher', None) if overrides else None
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.teacher_model = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if self.teacher_file and os.path.exists(self.teacher_file):
            print(f"\n[DISTILLATION] Loading Teacher weights from: {self.teacher_file}")
            self.teacher_model = YOLO(self.teacher_file).model
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            self.teacher_model.half()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            print("[DISTILLATION] Teacher loaded successfully and frozen.\n")
        else:
            print(f"[WARNING] Teacher file '{self.teacher_file}' not found. Training normally.")
        return model


# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python train_stage1_distill_phone.py <REPO> <dataset>")
        sys.exit(1)

    REPO = os.path.abspath(sys.argv[1])
    DATASET = sys.argv[2]
    if DATASET not in ("ft_phonecls", "ft"):
        print(f"unknown dataset: {DATASET} (expected ft_phonecls or ft)")
        sys.exit(1)

    # ---------------------------------------------------------
    # 1. CONFIGURATION
    # ---------------------------------------------------------
    # The original trainer loads the teacher and freezes it but never uses it in the loss, so the shipped 'distilled'
    # student is a plain yolo26s run with these settings. The teacher stage is therefore skipped here.
    TRAIN_TEACHER = False
    TRAIN_STUDENT = True    # Set to True to run Student Distillation (yolo26s.pt)

    DEVICE = int(os.environ.get("DEV", 0))

    TEACHER_WEIGHTS = r"D:/Backup/yolo26x.pt"
    STUDENT_WEIGHTS = r"D:/Backup/yolo26s.pt"
    assert os.path.exists(TEACHER_WEIGHTS), TEACHER_WEIGHTS + " not found"
    assert os.path.exists(STUDENT_WEIGHTS), STUDENT_WEIGHTS + " not found"

    DATA_YAML = os.path.join(REPO, "data", "stage1_negatives", DATASET, "data.yaml")
    assert os.path.exists(DATA_YAML), DATA_YAML + " not found (do not rebuild it here)"

    YOLO_PROJECT_DIR = os.path.join(REPO, "results", "stage1_phoneneg")
    TEACHER_RUN_NAME = f"distill_{DATASET}_teacher"
    STUDENT_RUN_NAME = f"distill_{DATASET}_student"

    OUT = os.path.join(REPO, "weights", f"stage1_distill_{DATASET}.pt")
    assert not os.path.exists(OUT), OUT + " exists; move it away first"
    assert OUT != os.path.join(REPO, "weights", "yolo_weapon_distilled.pt")

    # ==========================================
    # DYNAMIC BATCH & SCALE SETUP
    # ==========================================
    # 1. Define Batch Sizes based on Model Scale
    BATCH_CONFIG = {
        'n': 64,  # Nano
        's': 32,  # Small
        'm': 24,  # Medium
        'l': 16,  # Large
        'x': 12   # X-Large
    }

    # 2. Extract Names and Scales
    TEACHER_NAME = Path(TEACHER_WEIGHTS).stem
    STUDENT_NAME = Path(STUDENT_WEIGHTS).stem

    # Helper to get scale char
    def get_scale_and_batch(model_name):
        scale = model_name[-1].lower()
        if scale not in BATCH_CONFIG:
            print(f"[WARNING] Could not determine scale for {model_name}. Defaulting to 'x'.")
            scale = 'x'
        return scale, BATCH_CONFIG[scale]

    TEACHER_SCALE, TEACHER_BATCH = get_scale_and_batch(TEACHER_NAME)
    STUDENT_SCALE, STUDENT_BATCH = get_scale_and_batch(STUDENT_NAME)

    print(f"--- CONFIGURATION ---")
    print(f"Teacher: {TEACHER_NAME} [{TEACHER_SCALE.upper()}] | Batch: {TEACHER_BATCH} | Run: {TEACHER_RUN_NAME}")
    print(f"Student: {STUDENT_NAME} [{STUDENT_SCALE.upper()}] | Batch: {STUDENT_BATCH} | Run: {STUDENT_RUN_NAME}")
    print(f"--- DATASET: {DATASET} ({DATA_YAML}) ---")
    print(f"---------------------")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    yaml_full = DATA_YAML

    # ---------------------------------------------------------
    # STAGE 1: TRAIN TEACHER (STANDARD Architecture - NON P2)
    # ---------------------------------------------------------
    path_to_best_teacher = os.path.join(YOLO_PROJECT_DIR, TEACHER_RUN_NAME, 'weights', 'best.pt')

    if TRAIN_TEACHER:
        if os.path.exists(path_to_best_teacher):
            print(f"\n[STAGE 1] Existing Teacher found at {path_to_best_teacher}. Reusing trained Teacher weights.")
        else:
            print(f"\n[STAGE 1] Initialize STANDARD Architecture (Non-P2)...")
            teacher_model = YOLO(TEACHER_WEIGHTS)

            print(f"[STAGE 1] Starting Teacher Training (Batch: {TEACHER_BATCH})...")
            teacher_model.train(
                data=yaml_full,
                epochs=200,
                patience=20,
                imgsz=640,
                batch=TEACHER_BATCH,
                nbs=64,
                workers=0,
                device=DEVICE,
                degrees=15.0,
                mosaic=1.0,
                mixup=0.1,
                fliplr=0.5,
                project=YOLO_PROJECT_DIR,
                name=TEACHER_RUN_NAME,
                exist_ok=True,
                plots=True
            )
            print(f"[STAGE 1] Complete. Teacher saved at {path_to_best_teacher}")
    else:
        print(f"[STAGE 1] Skipped. Using existing Teacher at {path_to_best_teacher}")

    # ---------------------------------------------------------
    # STAGE 2: TRAIN STUDENT with DISTILLATION
    # ---------------------------------------------------------
    if TRAIN_STUDENT:
        if not os.path.exists(path_to_best_teacher):
            print(f"[NOTE] No teacher at {path_to_best_teacher}; it has no effect on the loss, training the student alone.")

        print(f"\n[STAGE 2] Starting Student Distillation (Batch: {STUDENT_BATCH})...")

        args = dict(
            model=STUDENT_WEIGHTS,
            data=yaml_full,
            epochs=250,
            patience=30,
            imgsz=640,
            batch=STUDENT_BATCH,
            workers=0,
            device=DEVICE,
            degrees=15.0,
            mosaic=1.0,
            mixup=0.1,
            project=YOLO_PROJECT_DIR,
            name=STUDENT_RUN_NAME,
            exist_ok=True,
            plots=True,
            teacher=path_to_best_teacher
        )

        trainer = KnowledgeDistillationTrainer(overrides=args)

        print(f"[DISTILLATION] Initializing Custom Trainer Loop for {STUDENT_NAME}...")
        trainer.train()

        shutil.copy2(trainer.best, OUT)
        print(f"[STAGE 2] Distilled student saved at {OUT}")
