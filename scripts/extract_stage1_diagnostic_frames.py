"""
scripts/extract_stage1_diagnostic_frames.py

Extracts categorized diagnostic evaluation frames to test Stage 1 failure modes:
1. threat_weapons: Confirmed handheld knives / blades during attacks.
2. bare_hands_fists: Empty clenched fists, bare hands, martial arts strikes (Jab.mp4).
3. clothing_seams_shadows: Linear clothing seams, fabric folds, shadows, and past false detections.
4. civilian_objects: Normal civilian activities.
"""

import os
import cv2
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_DIR / "data" / "stage1_diagnostic_frames"

def extract_frames_from_video(video_path, output_subfolder, frame_indices, prefix="frame"):
    if not video_path.exists():
        print(f"[WARN] Video file not found: {video_path}")
        return 0

    dest_dir = OUTPUT_DIR / output_subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    saved_count = 0
    target_set = set(frame_indices)
    current_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_set:
            out_name = f"{prefix}_{video_path.stem}_idx{current_idx:04d}.jpg"
            out_path = dest_dir / out_name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        current_idx += 1

    cap.release()
    print(f"[EXTRACT] Saved {saved_count} frames from {video_path.name} -> {output_subfolder}/")
    return saved_count


def main():
    print(f"--- EXTRACTING STAGE 1 DIAGNOSTIC TEST FRAMES ---")
    print(f"Destination: {OUTPUT_DIR}")

    # 1. Threat Weapons (Armed Attacks with Knife/Blade)
    v_thrust = REPO_DIR / "data" / "Validate" / "Without_Coat" / "Thrust" / "Thrust_1.mp4"
    extract_frames_from_video(v_thrust, "threat_weapons", [15, 25, 35, 45, 55, 65, 75, 85, 95, 105], prefix="threat_thrust")

    v_cutdown = REPO_DIR / "data" / "Validate" / "With_Coat" / "Cut_Down" / "Cut_Down_1.mp4"
    extract_frames_from_video(v_cutdown, "threat_weapons", [10, 20, 30, 40, 50, 60, 70, 80], prefix="threat_cutdown")

    # 2. Bare Hands & Clenched Fists (Targeting Hand Co-occurrence Bias)
    v_jab = REPO_DIR / "data" / "Validate" / "OOD" / "Jab.mp4"
    extract_frames_from_video(v_jab, "bare_hands_fists", [10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160], prefix="probe_fist")

    v_chop = REPO_DIR / "data" / "Validate" / "OOD" / "Chop_Wood.mp4"
    extract_frames_from_video(v_chop, "bare_hands_fists", [20, 40, 60, 80, 100], prefix="probe_chop")

    # 3. Clothing Seams, Zippers, and Shadows (Targeting 1D Linear Edge Bias)
    v_seam1 = REPO_DIR / "data" / "False Detected Clip" / "video" / "clip_1772789899168.mp4"
    extract_frames_from_video(v_seam1, "clothing_seams_shadows", [10, 25, 40, 55, 70], prefix="probe_seam_falseclip1")

    v_seam2 = REPO_DIR / "data" / "False Detected Clip" / "video" / "clip_1772789911504.mp4"
    extract_frames_from_video(v_seam2, "clothing_seams_shadows", [10, 25, 40, 55, 70], prefix="probe_seam_falseclip2")

    v_walk = REPO_DIR / "data" / "Validate" / "OOD" / "clip_1772692490348.mp4"
    extract_frames_from_video(v_walk, "clothing_seams_shadows", [15, 30, 45, 60], prefix="probe_seam_walk")

    # 4. Civilian Non-Threat Actions
    v_strike = REPO_DIR / "data" / "Validate" / "OOD" / "Volleyball_Strike.mp4"
    extract_frames_from_video(v_strike, "civilian_objects", [20, 50, 80, 110], prefix="civilian_volleyball")

    print("[SUCCESS] All diagnostic frame sets extracted successfully.")


if __name__ == "__main__":
    main()
