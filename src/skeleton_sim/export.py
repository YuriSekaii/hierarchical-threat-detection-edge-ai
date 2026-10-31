"""Writes simulated skeletons in the training format (CVAT XML, COCO-17 pixels) and draws previews."""

import os
import cv2
import numpy as np

from src.skeleton_sim.body import BONES

COLORS = {"R": (60, 60, 230), "L": (230, 140, 40), "C": (200, 200, 200)}


def write_xml(path, kpts, video_name):
    """Same layout as data/<class>/annotations_yolo26s-pose/*.xml, readable by ActionDataset._parse_xml."""
    lines = ["<?xml version='1.0' encoding='utf-8'?>", f'<root video_name="{video_name}">']
    for i, fr in enumerate(kpts):
        pts = ";".join(f"{x:.2f},{y:.2f}" for x, y in fr[:, :2])
        lines.append(f'  <image id="{i}">\n    <points points="{pts}" />\n  </image>')
    lines.append("</root>\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _bone_color(a, b):
    right = {6, 8, 10, 12, 14, 16, 2, 4}
    left = {5, 7, 9, 11, 13, 15, 1, 3}
    if a in right and b in right:
        return COLORS["R"]
    if a in left and b in left:
        return COLORS["L"]
    return COLORS["C"]


def draw_skeleton(img, kp, scale=1.0, offset=(0, 0), knife_wrist=None, thick=2):
    p = [(int(x * scale + offset[0]), int(y * scale + offset[1])) for x, y in kp[:, :2]]
    for a, b in BONES:
        cv2.line(img, p[a], p[b], _bone_color(a, b), thick, cv2.LINE_AA)
    for i, q in enumerate(p):
        cv2.circle(img, q, max(2, thick), (255, 255, 255), -1, cv2.LINE_AA)
    if knife_wrist is not None:
        cv2.circle(img, p[knife_wrist], 4 + 2 * thick, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def fit_tile(kpts, tw, th, pad=0.08):
    """Scale/offset that fits the whole clip's skeleton into a tw x th tile."""
    lo, hi = kpts[..., :2].reshape(-1, 2).min(0), kpts[..., :2].reshape(-1, 2).max(0)
    span = np.maximum(hi - lo, 1)
    s = min(tw * (1 - 2 * pad) / span[0], th * (1 - 2 * pad) / span[1])
    off = (tw - span[0] * s) / 2 - lo[0] * s, (th - span[1] * s) / 2 - lo[1] * s
    return s, off


def contact_sheet(path, clips, n_frames=8, tw=150, th=190):
    """One row per clip: n_frames evenly spaced frames, plus a label."""
    rows = []
    for kpts, label, wrist in clips:
        s, off = fit_tile(kpts, tw, th)
        row = np.full((th + 22, tw * n_frames, 3), 24, np.uint8)
        for i, f in enumerate(np.linspace(0, len(kpts) - 1, n_frames).astype(int)):
            tile = np.full((th, tw, 3), 24, np.uint8)
            draw_skeleton(tile, kpts[f], s, off, knife_wrist=wrist, thick=1)
            cv2.putText(tile, f"f{f}", (3, th - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            row[22:, i * tw:(i + 1) * tw] = tile
        cv2.putText(row, label[:110], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        rows.append(row)
    img = np.concatenate(rows, 0)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def grid_video(path, clips, cols=4, tw=320, th=300, fps=30):
    """Animated grid of clips (each loops) for eyeballing the motion."""
    n = len(clips)
    rows = (n + cols - 1) // cols
    T = max(len(k) for k, _, _ in clips)
    fits = [fit_tile(k, tw, th - 18) for k, _, _ in clips]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cols * tw, rows * th))
    for t in range(T):
        img = np.full((rows * th, cols * tw, 3), 20, np.uint8)
        for i, ((kpts, label, wrist), (s, off)) in enumerate(zip(clips, fits)):
            r, c = divmod(i, cols)
            tile = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            draw_skeleton(tile[18:], kpts[t % len(kpts)], s, off, knife_wrist=wrist)
            cv2.putText(tile, label[:44], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (60, 60, 60), 1)
        vw.write(img)
    vw.release()
