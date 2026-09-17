"""
scripts/benchmark_sa2va_v4.10.py

Comprehensive Live Evaluation for Version 4.10: ByteDance Sa2VA-Qwen2.5-VL-7B.
Evaluates the candidate architecture across:
1. Threat Weapons (Recall on real armed assaults)
2. Bare Hands & Clenched Fists (Suppression rate of empty fist co-occurrence bias)
3. Clothing Seams, Zippers & Shadows (Rejection rate of 1D planar edge bias)
4. Civilian Activities (Everyday motion rejection)
Outputs precision, recall, F1, latency, VRAM, and a side-by-side comparison with v4.00 YOLO baseline.
"""

import os
import sys
import time
import json
from pathlib import Path
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.inference_sa2va_v4_10 import Sa2VAWeaponDetector

DATA_DIR = REPO_DIR / "data" / "stage1_diagnostic_frames"
OUTPUT_JSON = REPO_DIR / "results" / "benchmark_sa2va_v4.10.json"
YOLO_BASELINE_JSON = REPO_DIR / "results" / "benchmark_yolo_baseline_v4.00.json"

def main():
    print("=" * 80)
    print("STAGE 1 CANDIDATE BENCHMARK: ByteDance Sa2VA-Qwen2.5-VL-7B (Version 4.10)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU required.")
        return

    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    detector = Sa2VAWeaponDetector(device="cuda:0")
    print("Loading Sa2VA model weights into bfloat16...")
    t_load_start = time.time()
    detector.load_model()
    t_load = time.time() - t_load_start
    print(f"Model loaded in {t_load:.2f}s.")

    initial_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
    print(f"Initial VRAM Footprint: {initial_vram:.2f} GB / 24.0 GB")

    categories = ["threat_weapons", "bare_hands_fists", "clothing_seams_shadows", "civilian_objects"]
    results = {cat: {"total": 0, "detected": 0, "details": []} for cat in categories}
    latencies = []

    # Run evaluation across diagnostic frames
    print("\n--- Running Evaluation across Diagnostic Frames ---")
    for cat in categories:
        cat_dir = DATA_DIR / cat
        if not cat_dir.exists():
            print(f"[WARN] Missing category folder: {cat_dir}")
            continue

        images = sorted(list(cat_dir.glob("*.jpg")))
        results[cat]["total"] = len(images)
        print(f"\nEvaluating category: {cat} ({len(images)} images)...")

        for img_path in images:
            res = detector.detect(img_path)
            latencies.append(res["latency_ms"])

            is_alert = res["alert"]
            if is_alert:
                results[cat]["detected"] += 1

            results[cat]["details"].append({
                "file": img_path.name,
                "alert": is_alert,
                "status": res["status"],
                "has_seg": res["has_seg_token"],
                "threat_count": res["threat_count"],
                "latency_ms": res["latency_ms"],
                "narrative": res["narrative"]
            })

            status_icon = "[THREAT ALERT]" if is_alert else "[CLEARED]"
            print(f"  {img_path.name}: {status_icon} ({res['latency_ms']:.1f} ms) | {res['narrative'][:80]}")

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Compute Core Performance Metrics
    tp = results["threat_weapons"]["detected"]
    threat_total = results["threat_weapons"]["total"]
    fn = threat_total - tp

    fp_fists = results["bare_hands_fists"]["detected"]
    total_fists = results["bare_hands_fists"]["total"]

    fp_seams = results["clothing_seams_shadows"]["detected"]
    total_seams = results["clothing_seams_shadows"]["total"]

    fp_civ = results["civilian_objects"]["detected"]
    total_civ = results["civilian_objects"]["total"]

    fp_total = fp_fists + fp_seams + fp_civ
    total_negatives = total_fists + total_seams + total_civ
    tn_total = total_negatives - fp_total

    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn_total) / (threat_total + total_negatives) if (threat_total + total_negatives) > 0 else 0.0

    fist_suppression_rate = (total_fists - fp_fists) / total_fists if total_fists > 0 else 1.0
    seam_rejection_rate = (total_seams - fp_seams) / total_seams if total_seams > 0 else 1.0

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    summary = {
        "model": "ByteDance/Sa2VA-Qwen2_5-VL-7B",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "threat_recall_fraction": f"{tp}/{threat_total}",
        "bare_fist_suppression_rate": round(fist_suppression_rate, 4),
        "bare_fist_fp": f"{fp_fists}/{total_fists}",
        "clothing_seam_rejection_rate": round(seam_rejection_rate, 4),
        "clothing_seam_fp": f"{fp_seams}/{total_seams}",
        "civilian_fp": f"{fp_civ}/{total_civ}",
        "total_fp": fp_total,
        "total_fn": fn,
        "avg_latency_ms": round(avg_latency, 2),
        "effective_fps": round(fps, 2),
        "initial_vram_gb": round(initial_vram, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "category_breakdown": results
    }

    # Save to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"benchmark_sa2va_v4.10": summary}, f, indent=2)
    print(f"\n[SAVED] Benchmark saved to {OUTPUT_JSON}")

    # Head-to-Head Comparison Table
    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD COMPARISON: Baseline v4.00 (YOLO26s) vs Candidate v4.10 (Sa2VA-Qwen2.5-VL-7B)")
    print("=" * 90)

    yolo_data = {}
    if YOLO_BASELINE_JSON.exists():
        with open(YOLO_BASELINE_JSON, "r") as f:
            yolo_raw = json.load(f)
            yolo_data = yolo_raw.get("benchmark_v4.00_yolo", {})

    yolo_prod = yolo_data.get("conf_0.45", {})
    yolo_sens = yolo_data.get("conf_0.20", {})

    headers = ["Metric / Dimension", "Baseline YOLO26s (Conf=0.45)", "Baseline YOLO26s (Conf=0.20)", "Sa2VA-Qwen2.5-VL-7B (v4.10)"]
    row_fmt = "{:<35} | {:<28} | {:<28} | {:<28}"
    print(row_fmt.format(*headers))
    print("-" * 125)
    print(row_fmt.format("F1-Score", f"{yolo_prod.get('f1_score', 0):.4f}", f"{yolo_sens.get('f1_score', 0):.4f}", f"{f1:.4f}"))
    print(row_fmt.format("Threat Detection Recall", f"{yolo_prod.get('recall', 0)*100:.1f}% ({yolo_prod.get('tp', 0)}/{threat_total})", f"{yolo_sens.get('recall', 0)*100:.1f}% ({yolo_sens.get('tp', 0)}/{threat_total})", f"{recall*100:.1f}% ({tp}/{threat_total})"))
    print(row_fmt.format("Missed Attacks (FN)", f"{yolo_prod.get('fn', 0)} missed", f"{yolo_sens.get('fn', 0)} missed", f"{fn} missed"))
    print(row_fmt.format("Threat Detection Precision", f"{yolo_prod.get('precision', 0)*100:.1f}%", f"{yolo_sens.get('precision', 0)*100:.1f}%", f"{precision*100:.1f}%"))
    print(row_fmt.format("Bare Fist False Alarms", f"{yolo_prod.get('fp_bare_fists', 0)}/{total_fists} FP", f"{yolo_sens.get('fp_bare_fists', 0)}/{total_fists} FP", f"{fp_fists}/{total_fists} FP"))
    print(row_fmt.format("Bare Fist Suppression Rate", f"{(total_fists - yolo_prod.get('fp_bare_fists', 0))/total_fists*100:.1f}%", f"{(total_fists - yolo_sens.get('fp_bare_fists', 0))/total_fists*100:.1f}%", f"{fist_suppression_rate*100:.1f}%"))
    print(row_fmt.format("Clothing Seam False Alarms", f"{yolo_prod.get('fp_clothing_seams', 0)}/{total_seams} FP", f"{yolo_sens.get('fp_clothing_seams', 0)}/{total_seams} FP", f"{fp_seams}/{total_seams} FP"))
    print(row_fmt.format("Clothing Seam Rejection Rate", f"{(total_seams - yolo_prod.get('fp_clothing_seams', 0))/total_seams*100:.1f}%", f"{(total_seams - yolo_sens.get('fp_clothing_seams', 0))/total_seams*100:.1f}%", f"{seam_rejection_rate*100:.1f}%"))
    print(row_fmt.format("Total False Alarms (FP / 23)", f"{yolo_prod.get('fp_total', 0)} FP", f"{yolo_sens.get('fp_total', 0)} FP", f"{fp_total} FP"))
    print(row_fmt.format("Inference Latency (RTX 3090)", f"{yolo_prod.get('avg_latency_ms', 0):.2f} ms", f"{yolo_sens.get('avg_latency_ms', 0):.2f} ms", f"{avg_latency:.2f} ms"))
    print(row_fmt.format("VRAM Footprint", "< 1.0 GB", "< 1.0 GB", f"{initial_vram:.2f} GB (Peak: {peak_vram:.2f} GB)"))
    print("=" * 125)

if __name__ == "__main__":
    main()
