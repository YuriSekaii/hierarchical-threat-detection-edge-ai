"""
Checks a simulated dataset against real data, with the current Stage 2 cascade (PyTorch weights).

1. Motion statistics (torso units, after the same preprocessing as the pipeline) of the real
   training classes, the lab segments (2026-10-31) and each simulated / augmented folder:
   knife-arm reach and range, wrist speed, highest wrist point, stance width, frame-to-frame noise.
2. Cascade scores per folder: each clip is cut into 50-frame windows (stride 10) like the live
   pipeline and its best window counts. Reports alarm rate and median T1/T2/T3 probabilities,
   and for simulated attacks the alarm rate per camera position (front / side / back, low / high).

Run it before retraining for the baseline, and after retraining to see what the new models learned.
  python scripts/validate_simulation.py --sim data/Simulated --max-per-folder 150
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)

from src.dataset import ActionDataset  # noqa: E402
from src.skeleton_utils import interpolate_missing_joints, normalize_skeleton_clip, smooth_kinematics  # noqa: E402

ANN = "annotations_yolo26s-pose"


def prep(raw):
    """Identical to Stage2aPoseTracker.preprocess_sliding_window."""
    arr = smooth_kinematics(interpolate_missing_joints(np.asarray(raw, np.float32)[..., :2]), sigma=1.0)
    return normalize_skeleton_clip(arr)


def motion_stats(seq):
    c = prep(seq)
    arms = []
    for sh, wr in ((5, 9), (6, 10)):
        ext = np.linalg.norm(c[:, wr] - c[:, sh], axis=-1)
        spd = np.linalg.norm(np.diff(c[:, wr], axis=0), axis=-1)
        arms.append((ext.max(), np.ptp(ext), np.percentile(spd, 98), (c[:, wr, 1] - c[:, sh, 1]).min()))
    knife = max(arms, key=lambda t: t[1])
    stance = np.linalg.norm(c[:, 15] - c[:, 16], axis=-1).max()
    noise = np.median(np.abs(np.diff(c, 2, axis=0)))
    return (*knife, stance, noise)


STAT_NAMES = ["reach", "reach_range", "wrist_speed", "wrist_top", "stance", "noise"]


class Scorer:
    def __init__(self):
        from src.stage2b_staircase import Stage2bStaircaseEngine, rasterize_heatmap_from_coords
        self.eng = Stage2bStaircaseEngine(backend="pytorch")
        self.raster = rasterize_heatmap_from_coords

    @torch.no_grad()
    def __call__(self, seq):
        """Best 50-frame window of a clip: (p1, p2, p3, alarm)."""
        eng, best = self.eng, None
        for s in range(0, max(1, len(seq) - 49), 10):
            w = seq[s:s + 50]
            if len(w) < 50:
                break
            g = torch.from_numpy(prep(w)).float()
            hm = self.raster(g, device=eng.device, normalize=False)
            co = g.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(eng.device)
            p = [eng._eval_model(k, hm if eng.models[k]["requires_heatmap"] else co)
                 for k in (eng.tier1_key, eng.tier2_key, eng.tier3_key)]
            cur = (*p, float(eng.evaluate_clip(g)["is_threat"]))
            best = cur if best is None else tuple(max(x, y) for x, y in zip(best, cur))
        return best


def load_xml_dir(folder, limit, rng):
    files = sorted(glob.glob(os.path.join(folder, ANN, "*.xml")))
    if limit and len(files) > limit:
        files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
    parser = ActionDataset.__new__(ActionDataset)
    parser.num_keypoints, parser.num_coords = 17, 2
    return {os.path.basename(f): s for f in files if (s := parser._parse_xml(f)) is not None and len(s) >= 50}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default=os.path.join(REPO_DIR, "data", "Simulated"))
    ap.add_argument("--lab", default=os.path.join(REPO_DIR, "data", "lab_recordings", "2026-10-31", "segments"))
    ap.add_argument("--max-per-folder", type=int, default=150)
    ap.add_argument("--no-score", action="store_true", help="motion statistics only")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    groups = {}
    for cls in ["Thrust", "Stab", "Cut-Down"]:
        ds = ActionDataset(REPO_DIR, [os.path.join("data", cls)], clip_length=50, mode="test")
        groups["REAL " + cls] = {str(i): s for i, s in enumerate(ds.skeleton_data)}
    if os.path.isdir(args.lab):
        for cat in ("attack", "normal"):
            segs = {os.path.basename(f): np.load(f)["kpts"] for f in sorted(glob.glob(os.path.join(args.lab, f"*_{cat}_*.npz")))}
            groups[f"LAB {cat}"] = {k: v for k, v in segs.items() if len(v) >= 50}
    for folder in sorted(glob.glob(os.path.join(args.sim, "*", ANN))):
        name = os.path.basename(os.path.dirname(folder))
        groups[name] = load_xml_dir(os.path.dirname(folder), args.max_per_folder, rng)

    print("\n== Motion statistics (median, torso units) ==")
    print(f"{'group':22s} {'n':>5s} " + " ".join(f"{n:>11s}" for n in STAT_NAMES))
    for g, clips in groups.items():
        st = np.array([motion_stats(s) for s in clips.values()])
        print(f"{g:22s} {len(st):5d} " + " ".join(f"{v:11.3f}" for v in np.median(st, 0)))

    if args.no_score:
        return
    scorer = Scorer()
    print("\n== Current cascade, best window per clip ==")
    print(f"{'group':22s} {'n':>5s} {'alarm':>7s} {'p1 med':>7s} {'p1>=.20':>8s} {'p2 med':>7s} {'p3 med':>7s}")
    scores = {}
    for g, clips in groups.items():
        sc = {k: scorer(s) for k, s in clips.items()}
        scores[g] = sc
        P = np.array(list(sc.values()))
        print(f"{g:22s} {len(P):5d} {100 * P[:, 3].mean():6.1f}% {np.median(P[:, 0]):7.3f} "
              f"{100 * (P[:, 0] >= 0.2).mean():7.1f}% {np.median(P[:, 1]):7.3f} {np.median(P[:, 2]):7.3f}")

    meta_path = os.path.join(args.sim, "metadata.csv")
    if os.path.exists(meta_path):
        print("\n== Simulated attacks: alarm rate by camera position ==")
        bins = {}
        with open(meta_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["source"] != "sim" or "OOD" in r["folder"]:
                    continue
                s = scores.get(r["folder"], {}).get(os.path.basename(r["file"]))
                if s is None:
                    continue
                az = float(r["cam_azimuth"]) % 360
                view = "front" if az < 45 or az >= 315 else ("back" if 135 <= az < 225 else "side")
                hgt = "high" if float(r["cam_height"]) >= 2.2 else "low"
                bins.setdefault(f"{view:5s} {hgt}", []).append(s[3])
        for k in sorted(bins):
            print(f"  {k}: n={len(bins[k]):4d} alarm {100 * np.mean(bins[k]):5.1f}%")


if __name__ == "__main__":
    main()
