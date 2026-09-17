"""
scripts/benchmark_yolo_baseline_v4.00.py

Runs the Stage 1 Baseline Detector (Distilled YOLO26s) on the diagnostic evaluation set
across two production operating points (Conf = 0.45 and Conf = 0.20).
Logs False Positives on bare hands/fists, clothing seams/shadows, and True Positives on weapons.
"""

import os
import time
import json
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data" / "stage1_diagnostic_frames"
WEIGHTS_PATH = REPO_DIR / "weights" / "yolo_weapon_distilled.pt"
OUTPUT_PATH = REPO_DIR / "results" / "benchmark_yolo_baseline_v4.00.json"

def run_evaluation(model, conf_thresh=0.45, iou_thresh=0.20):
    categories = ["threat_weapons", "bare_hands_fists", "clothing_seams_shadows", "civilian_objects"]
    results = {cat: {"total": 0, "detected": 0, "detections": []} for cat in categories}

    latencies = []

    for cat in categories:
        cat_dir = DATA_DIR / cat
        if not cat_dir.exists():
            continue
        images = sorted(list(cat_dir.glob("*.jpg")))
        results[cat]["total"] = len(images)

        for img_path in images:
            img = cv2.imread(str(img_path))
            t0 = time.perf_counter()
            preds = model(img, conf=conf_thresh, iou=iou_thresh, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            boxes = preds[0].boxes
            det_count = len(boxes) if boxes is not None else 0
            if det_count > 0:
                results[cat]["detected"] += 1
                det_info = []
                for box in boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    xyxy = [round(float(c), 2) for c in box.xyxy[0]]
                    det_info.append({"conf": conf, "cls": cls_id, "xyxy": xyxy})
                results[cat]["detections"].append({"file": img_path.name, "boxes": det_info})

    # Metrics
    tp = results["threat_weapons"]["detected"]
    fn = results["threat_weapons"]["total"] - tp
    fp_fists = results["bare_hands_fists"]["detected"]
    fp_seams = results["clothing_seams_shadows"]["detected"]
    fp_civ = results["civilian_objects"]["detected"]
    fp_total = fp_fists + fp_seams + fp_civ

    tn_total = (results["bare_hands_fists"]["total"] - fp_fists) + \
               (results["clothing_seams_shadows"]["total"] - fp_seams) + \
               (results["civilian_objects"]["total"] - fp_civ)

    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "conf_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "tp": tp,
        "fn": fn,
        "fp_bare_fists": fp_fists,
        "fp_clothing_seams": fp_seams,
        "fp_civilian": fp_civ,
        "fp_total": fp_total,
        "tn_total": tn_total,
        "category_breakdown": results
    }

def main():
    print(f"=== STAGE 1 BASELINE BENCHMARK (v4.00: YOLO26s) ===")
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Weights not found: {WEIGHTS_PATH}")
        return

    print(f"Loading weights: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    # Warmup
    dummy = torch.zeros((1, 3, 640, 640), device="cuda:0" if torch.cuda.is_available() else "cpu")
    for _ in range(5):
        _ = model(dummy, verbose=False)

    print("\n--- Running at Production Setting (Conf=0.45, IoU=0.20) ---")
    res_45 = run_evaluation(model, conf_thresh=0.45, iou_thresh=0.20)
    print(f"Conf 0.45 -> Recall: {res_45['recall']*100:.1f}%, Precision: {res_45['precision']*100:.1f}%, F1: {res_45['f1_score']:.4f}")
    print(f"  Threats Detected (TP): {res_45['tp']}/{res_45['category_breakdown']['threat_weapons']['total']} | Missed (FN): {res_45['fn']}")
    print(f"  FP on Bare Fists: {res_45['fp_bare_fists']}/{res_45['category_breakdown']['bare_hands_fists']['total']}")
    print(f"  FP on Seams/Shadows: {res_45['fp_clothing_seams']}/{res_45['category_breakdown']['clothing_seams_shadows']['total']}")
    print(f"  FP Civilian: {res_45['fp_civilian']}/{res_45['category_breakdown']['civilian_objects']['total']}")
    print(f"  Avg Latency: {res_45['avg_latency_ms']} ms")

    print("\n--- Running at High-Sensitivity Setting (Conf=0.20, IoU=0.20) ---")
    res_20 = run_evaluation(model, conf_thresh=0.20, iou_thresh=0.20)
    print(f"Conf 0.20 -> Recall: {res_20['recall']*100:.1f}%, Precision: {res_20['precision']*100:.1f}%, F1: {res_20['f1_score']:.4f}")
    print(f"  Threats Detected (TP): {res_20['tp']}/{res_20['category_breakdown']['threat_weapons']['total']} | Missed (FN): {res_20['fn']}")
    print(f"  FP on Bare Fists: {res_20['fp_bare_fists']}/{res_20['category_breakdown']['bare_hands_fists']['total']}")
    print(f"  FP on Seams/Shadows: {res_20['fp_clothing_seams']}/{res_20['category_breakdown']['clothing_seams_shadows']['total']}")
    print(f"  FP Civilian: {res_20['fp_civilian']}/{res_20['category_breakdown']['civilian_objects']['total']}")
    print(f"  Avg Latency: {res_20['avg_latency_ms']} ms")

    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"benchmark_v4.00_yolo": {"conf_0.45": res_45, "conf_0.20": res_20}}, f, indent=2)
    print(f"\n[SAVED] Benchmark saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
