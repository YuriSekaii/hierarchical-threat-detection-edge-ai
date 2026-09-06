import os
import shutil
import random
import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from copy import deepcopy
from pathlib import Path

# ==========================================
# RESEARCH CONTROLLER
# ==========================================
TRAIN_TEACHER = True    # Set to True to train Teacher (yolo26x.pt) or reuse if already trained
TRAIN_STUDENT = True    # Set to True to run Student Distillation (yolo26s.pt)

# Configs
GPU_ID = 0             
WORKERS = 8            
TEACHER_WEIGHTS = "yolo26x.pt" 
STUDENT_WEIGHTS = "yolo26s.pt" 

# Fixed Workers for Student
STUDENT_WORKERS = 4 

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

# 3. Define Run Names (Matches production champion run in results/weapon_training_results)
TEACHER_RUN_NAME = f"teacher_run_{TEACHER_NAME}"
STUDENT_RUN_NAME = f"distilled_{STUDENT_NAME}_from_{TEACHER_NAME}"

print(f"--- CONFIGURATION ---")
print(f"Teacher: {TEACHER_NAME} [{TEACHER_SCALE.upper()}] | Batch: {TEACHER_BATCH} | Run: {TEACHER_RUN_NAME}")
print(f"Student: {STUDENT_NAME} [{STUDENT_SCALE.upper()}] | Batch: {STUDENT_BATCH} | Run: {STUDENT_RUN_NAME}")
print(f"---------------------")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    # ---------------------------------------------------------
    # 1. DATASET SETUP
    # ---------------------------------------------------------
    REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    WEAPON_DATA_DIR = os.path.join(REPO_DIR, 'raw_data') 
    SPLIT_DATA_DIR = os.path.join(REPO_DIR, 'Weapon_Detection_Split')
    YOLO_PROJECT_DIR = os.path.join(REPO_DIR, 'runs', 'weapon_training')
    YAML_FILE_NAME = 'weapon_data.yaml'

    def setup_yolo_dataset(base_dir, output_dir, ratio=0.8):
        if not os.path.exists(base_dir):
            alt = os.path.join(base_dir, 'Weapon_Detection')
            if os.path.exists(alt): base_dir = alt
        if os.path.exists(output_dir) and len(os.listdir(os.path.join(output_dir, 'images', 'train'))) > 0:
            return os.path.abspath(output_dir)
        if not os.path.exists(base_dir) or not os.path.exists(os.path.join(base_dir, 'images')):
            print(f"\n[NOTE] Raw dataset directory '{base_dir}' not found.")
            print("       Pre-trained distilled weights are available at 'weights/yolo_weapon_distilled.pt'.")
            print("       To retrain, place your images and labels under 'raw_data/images' and 'raw_data/labels'.")
            return None
        print("Processing data...")
        for s in ['train', 'val']:
            os.makedirs(os.path.join(output_dir, 'images', s), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels', s), exist_ok=True)
        img_src = os.path.join(base_dir, 'images')
        labels_src = os.path.join(base_dir, 'labels')
        imgs = sorted([f for f in os.listdir(img_src) if f.endswith(('.jpg', '.png', '.jpeg'))])
        random.shuffle(imgs)
        idx = int(len(imgs) * ratio)
        for i, f in enumerate(imgs):
            split = 'train' if i < idx else 'val'
            shutil.copy(os.path.join(img_src, f), os.path.join(output_dir, 'images', split, f))
            lbl = os.path.splitext(f)[0] + '.txt'
            if os.path.exists(os.path.join(labels_src, lbl)):
                shutil.copy(os.path.join(labels_src, lbl), os.path.join(output_dir, 'labels', split, lbl))
        return os.path.abspath(output_dir)

    abs_path = setup_yolo_dataset(WEAPON_DATA_DIR, SPLIT_DATA_DIR)
    yaml_full = os.path.join(SPLIT_DATA_DIR, YAML_FILE_NAME)
    if abs_path is not None:
        os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
        yaml_data = {'path': abs_path, 'train': 'images/train', 'val': 'images/val', 'names': {0: 'knife'}}
        with open(yaml_full, 'w') as f: yaml.dump(yaml_data, f, sort_keys=False)

    # ---------------------------------------------------------
    # STAGE 1: TRAIN TEACHER (STANDARD Architecture - NON P2)
    # ---------------------------------------------------------
    path_to_best_teacher = os.path.join(YOLO_PROJECT_DIR, TEACHER_RUN_NAME, 'weights', 'best.pt')

    if TRAIN_TEACHER:
        if os.path.exists(path_to_best_teacher):
            print(f"\n[STAGE 1] Existing Teacher found at {path_to_best_teacher}. Reusing trained Teacher weights.")
        else:
            if abs_path is None:
                print("[ERROR] Cannot train Teacher without dataset. Exiting.")
                exit()
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
                workers=WORKERS,
                device=GPU_ID,
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
        if abs_path is None:
            exit(0)
        if not os.path.exists(path_to_best_teacher):
            print(f"[ERROR] Teacher weights not found at {path_to_best_teacher}. Train Teacher first or provide weights!")
            exit()

        print(f"\n[STAGE 2] Starting Student Distillation (Batch: {STUDENT_BATCH})...")
        
        args = dict(
            model=STUDENT_WEIGHTS,
            data=yaml_full,
            epochs=250,
            patience=30,
            imgsz=640,
            batch=STUDENT_BATCH,
            workers=STUDENT_WORKERS,
            device=GPU_ID,
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

