"""
scripts/test_sa2va_single_frame.py

Smoke test for Sa2VA-Qwen2.5-VL-7B using Sa2VAWeaponDetector.
"""

import os
import sys
from pathlib import Path
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from src.inference_sa2va_v4_10 import Sa2VAWeaponDetector

def main():
    print("=== Sa2VA-Qwen2.5-VL-7B Single Frame Smoke Test ===")
    detector = Sa2VAWeaponDetector(device="cuda:0")

    print("[1/2] Initializing and Loading Model...")
    detector.load_model()

    # Find a test image
    test_img_dir = REPO_DIR / "data" / "stage1_diagnostic_frames" / "threat_weapons"
    test_images = sorted(list(test_img_dir.glob("*.jpg")))
    if not test_images:
        print("[ERROR] No test images found!")
        return

    test_image_path = test_images[0]
    print(f"\n[2/2] Running inference on: {test_image_path.name}...")
    res = detector.detect(test_image_path)

    print(f"\n--- Inference Result ---")
    print(f"Alert Status: {res['status']}")
    print(f"Threat Alert Active: {res['alert']}")
    print(f"Has [SEG] Token: {res['has_seg_token']}")
    print(f"Threat Count: {res['threat_count']}")
    print(f"Inference Latency: {res['latency_ms']} ms")
    print(f"Model Narrative: {res['narrative']}")

    if res['threats']:
        for t in res['threats']:
            print(f"  Threat Box: {t['bbox']}, Solidity: {t['solidity']}, Pixels: {t['pixel_area']}")

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak VRAM during run: {peak_vram:.2f} GB / 24.0 GB")
    print("\n[SUCCESS] Single frame smoke test completed successfully!")

if __name__ == "__main__":
    main()
