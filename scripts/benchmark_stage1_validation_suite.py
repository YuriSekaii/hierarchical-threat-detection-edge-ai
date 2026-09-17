"""
scripts/benchmark_stage1_validation_suite.py

Standardized Stage 1 Threat Object & Weapon Detection Benchmark.
Evaluates on the master 77-video validation suite from data/Validate/:
- With_Coat: 16 violent assault videos (Label 1)
- Without_Coat: 17 violent assault videos (Label 1)
- OOD: 44 civilian movement videos (Label 0)

Computes standard:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix: TP, TN, FP, FN
- Average Latency (ms) & Frame Rate (FPS)
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset

OUTPUT_JSON = os.path.join(REPO_DIR, "results", "benchmark_stage1_validation_suite.json")


def load_validation_videos():
    """ Resolves the exact 77 video sources used across the benchmark history. """
    train_dir, val_root = resolve_data_paths()
    val_items = []

    for cat in ["With_Coat", "Without_Coat", "OOD"]:
        cat_dir = os.path.join(val_root, cat)
        ds = ActionDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, mode="test")
        unique_xmls = sorted(list(set(ds[i][2] for i in range(len(ds)))))

        for xml_path in unique_xmls:
            xml_name = os.path.basename(xml_path)
            vid_name = os.path.splitext(xml_name)[0] + ".mp4"
            parent = os.path.dirname(os.path.dirname(xml_path))
            vid_path = os.path.join(parent, vid_name)

            if not os.path.exists(vid_path):
                # Fallback check
                matches = [f for f in os.listdir(parent) if f.lower() == vid_name.lower()]
                if matches:
                    vid_path = os.path.join(parent, matches[0])

            if os.path.exists(vid_path):
                val_items.append({
                    "video_path": vid_path,
                    "video_name": vid_name,
                    "category": cat,
                    "label": 0 if cat == "OOD" else 1
                })
            else:
                print(f"[WARN] Video not found for: {xml_path}")

    return val_items


def extract_keyframe(video_path, sample_ratio=0.50):
    """ Extracts a representative action frame at sample_ratio of the video. """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    target_idx = max(0, min(total_frames - 1, int(total_frames * sample_ratio)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        return frame
    return None


def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def benchmark_yolo(val_items, weights_path="weights/yolo_weapon_distilled.pt", conf=0.25):
    print(f"\n=======================================================")
    print(f"BENCHMARKING: Distilled YOLO26s (Conf={conf})")
    print(f"=======================================================")
    model = YOLO(weights_path)
    y_true = [item["label"] for item in val_items]
    y_pred = []
    latencies = []

    for idx, item in enumerate(val_items):
        frame = extract_keyframe(item["video_path"], sample_ratio=0.50)
        if frame is None:
            y_pred.append(0)
            continue

        t0 = time.perf_counter()
        results = model.predict(source=frame, conf=conf, imgsz=(384, 640), device="cuda:0", verbose=False)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        has_weapon = False
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            has_weapon = True

        y_pred.append(1 if has_weapon else 0)
        status = "ALERT" if has_weapon else "CLEAR"
        gt = "THREAT" if item["label"] == 1 else "NORMAL"
        match = "MATCH" if (has_weapon == (item["label"] == 1)) else "ERROR"
        if (idx + 1) % 15 == 0 or (idx + 1) == len(val_items):
            print(f"  [{idx+1:02d}/{len(val_items)}] {item['category']} | {item['video_name']} -> {status} (GT: {gt}) [{match}] ({lat:.1f}ms)")

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["avg_latency_ms"] = round(float(np.mean(latencies)), 2)
    metrics["fps"] = round(1000.0 / np.mean(latencies), 1)
    return metrics, y_pred


def benchmark_sa2va(val_items):
    print(f"\n=======================================================")
    print(f"BENCHMARKING: ByteDance Sa2VA-Qwen2.5-VL-7B")
    print(f"=======================================================")
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen2Config
    from models.sa2va.modeling_sa2va_qwen import Sa2VAChatModelQwen

    device = "cuda:0"
    model_path = "ByteDance/Sa2VA-Qwen2_5-VL-7B"

    print("Loading Sa2VA weights into bfloat16...")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg.vocab_size = 151670
    cfg.text_config = Qwen2Config(**cfg.text_config)

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    proc.image_processor.min_pixels = 256 * 28 * 28
    proc.image_processor.max_pixels = 1024 * 28 * 28

    model = Sa2VAChatModelQwen.from_pretrained(
        model_path,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    ).eval()
    print("Sa2VA loaded cleanly.")

    prompt = "Please locate and segment any knife, blade, dagger, or handheld weapon in this image."
    y_true = [item["label"] for item in val_items]
    y_pred = []
    latencies = []

    for idx, item in enumerate(val_items):
        frame = extract_keyframe(item["video_path"], sample_ratio=0.50)
        if frame is None:
            y_pred.append(0)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        t0 = time.perf_counter()
        with torch.inference_mode():
            res = model.predict_forward(image=pil_img, text=prompt, processor=proc)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        # Topological solidity & scale verification
        masks = res.get("prediction_masks", [])
        has_valid_threat = False

        for m in masks:
            m_2d = m[0] if m.ndim == 3 else m
            binary = (m_2d > 0.5).astype(np.uint8)
            active = np.where(binary > 0)
            n_pts = len(active[0])
            h, w = m_2d.shape
            pct = n_pts / float(h * w) if (h * w) > 0 else 0.0

            if n_pts >= 200 and pct <= 0.03:
                y_min, y_max = np.min(active[0]), np.max(active[0])
                x_min, x_max = np.min(active[1]), np.max(active[1])
                box_area = (x_max - x_min + 1) * (y_max - y_min + 1)
                solidity = n_pts / float(box_area) if box_area > 0 else 0.0
                if solidity >= 0.15:
                    has_valid_threat = True
                    break

        y_pred.append(1 if has_valid_threat else 0)
        status = "ALERT" if has_valid_threat else "CLEAR"
        gt = "THREAT" if item["label"] == 1 else "NORMAL"
        match = "MATCH" if (has_valid_threat == (item["label"] == 1)) else "ERROR"
        print(f"  [{idx+1:02d}/{len(val_items)}] {item['category']} | {item['video_name']} -> {status} (GT: {gt}) [{match}] ({lat:.0f}ms)")

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["avg_latency_ms"] = round(float(np.mean(latencies)), 2)
    metrics["fps"] = round(1000.0 / np.mean(latencies), 2)
    return metrics, y_pred


def main():
    print("=" * 80)
    print("STAGE 1 VALIDATION SUITE BENCHMARK: YOLO26s vs Sa2VA-Qwen2.5-VL-7B")
    print("=" * 80)

    val_items = load_validation_videos()
    print(f"Loaded {len(val_items)} standardized validation videos:")
    print(f"  - Positive (Weapons Present): {sum(1 for v in val_items if v['label'] == 1)} videos")
    print(f"  - Negative (Civilian / OOD): {sum(1 for v in val_items if v['label'] == 0)} videos")

    # 1. Benchmark YOLO
    yolo_res, yolo_preds = benchmark_yolo(val_items, conf=0.25)
    print("\nYOLO26s Results:", yolo_res)

    # 2. Benchmark Sa2VA
    sa2va_res, sa2va_preds = benchmark_sa2va(val_items)
    print("\nSa2VA Results:", sa2va_res)

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    report = {
        "total_videos": len(val_items),
        "positive_videos": sum(1 for v in val_items if v['label'] == 1),
        "negative_videos": sum(1 for v in val_items if v['label'] == 0),
        "distilled_yolo26s": yolo_res,
        "sa2va_qwen2_5_vl_7b": sa2va_res
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved master results to: {OUTPUT_JSON}")

    # Head-to-Head Table
    print("\n" + "=" * 85)
    print("HEAD-TO-HEAD BENCHMARK: MASTER VALIDATION SUITE (77 VIDEOS)")
    print("=" * 85)
    row_fmt = "{:<32} | {:<24} | {:<24}"
    print(row_fmt.format("Metric", "Distilled YOLO26s (v4.00)", "Sa2VA-Qwen2.5-VL-7B (v4.10)"))
    print("-" * 85)
    print(row_fmt.format("F1-Score", f"{yolo_res['f1_score']:.4f}", f"{sa2va_res['f1_score']:.4f}"))
    print(row_fmt.format("Accuracy", f"{yolo_res['accuracy']*100:.2f}%", f"{sa2va_res['accuracy']*100:.2f}%"))
    print(row_fmt.format("Recall (Threat Safety)", f"{yolo_res['recall']*100:.2f}% ({yolo_res['tp']}/33)", f"{sa2va_res['recall']*100:.2f}% ({sa2va_res['tp']}/33)"))
    print(row_fmt.format("Missed Weapons (FN)", f"{yolo_res['fn']} missed", f"{sa2va_res['fn']} missed"))
    print(row_fmt.format("Precision", f"{yolo_res['precision']*100:.2f}%", f"{sa2va_res['precision']*100:.2f}%"))
    print(row_fmt.format("False Alarms (FP / 44 OOD)", f"{yolo_res['fp']} false alarms", f"{sa2va_res['fp']} false alarms"))
    print(row_fmt.format("Confusion Matrix (TP/TN/FP/FN)", f"{yolo_res['tp']}/{yolo_res['tn']}/{yolo_res['fp']}/{yolo_res['fn']}", f"{sa2va_res['tp']}/{sa2va_res['tn']}/{sa2va_res['fp']}/{sa2va_res['fn']}"))
    print(row_fmt.format("Inference Latency", f"{yolo_res['avg_latency_ms']:.2f} ms", f"{sa2va_res['avg_latency_ms']:.2f} ms"))
    print(row_fmt.format("Throughput (FPS)", f"{yolo_res['fps']:.1f} FPS", f"{sa2va_res['fps']:.2f} FPS"))
    print("=" * 85)


if __name__ == "__main__":
    main()
