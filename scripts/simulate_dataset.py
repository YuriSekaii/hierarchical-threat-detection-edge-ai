"""
Generates a simulated skeleton dataset for the Stage 2 action models.

Output (default data/Simulated/), same layout as the real training data so ActionDataset can read it:
  Sim_Thrust/ Sim_Stab/ Sim_Cut-Down/ Sim_Slash/ Sim_Multi-Slash/ Sim_Swing/ Sim_Combo/
      annotations_yolo26s-pose/*.xml   COCO-17 pixels in a 1920x1440 frame
      skeleton3d/*.npz                 3D joints, clean 2D, camera (to re-film or debug)
  Sim_OOD_Normal/                      walking, idle, turning, pointing, ... ("OOD" in the name -> label 0)
  Aug_Thrust/ Aug_Stab/ Aug_Cut-Down/  perturbed copies of the real training clips
  metadata.csv                         one row per clip: action, variant, hand, speed, camera, ...
  previews/                            contact sheets (.jpg) and animated grids (.mp4) per folder

Examples:
  python scripts/simulate_dataset.py                       # full default set
  python scripts/simulate_dataset.py --per-action 20 --normals 10 --real-aug 2 --out results/sim_test
"""

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)

from src.dataset import ActionDataset  # noqa: E402
from src.skeleton_sim import actions as A  # noqa: E402
from src.skeleton_sim.augment2d import knife_arm, perturb  # noqa: E402
from src.skeleton_sim.export import contact_sheet, grid_video, write_xml  # noqa: E402
from src.skeleton_sim.scene import build_motion, film  # noqa: E402

ANN = "annotations_yolo26s-pose"
REAL_CLASSES = ["Thrust", "Stab", "Cut-Down"]


def _sim_job(job):
    """One simulated clip. job = (index, action, folder, seed, out)."""
    idx, action, folder, seed, out = job
    rng = np.random.default_rng(seed)
    for _ in range(10):
        joints, facing, info = build_motion(rng, action)
        shot = film(joints, facing, rng)
        if shot is not None:
            break
    else:
        return None
    kp, uv, occ, cam = shot
    name = f"sim_{action}_{idx:05d}"
    write_xml(os.path.join(out, folder, ANN, name + ".xml"), kp, name + ".mp4")
    os.makedirs(os.path.join(out, folder, "skeleton3d"), exist_ok=True)
    np.savez_compressed(os.path.join(out, folder, "skeleton3d", name + ".npz"), joints3d=joints.astype(np.float32),
                        uv_clean=uv, kpts=kp, occluded=occ, **{"cam_" + k: v for k, v in cam.meta().items()})
    row = dict(file=f"{folder}/{ANN}/{name}.xml", folder=folder, source="sim", seed=seed, **info,
               **{"cam_" + k: round(float(v), 3) for k, v in cam.meta().items()})
    return row


def _real_jobs(out, copies, rng):
    rows, previews = [], {}
    for cls in REAL_CLASSES:
        ds = ActionDataset(REPO_DIR, [os.path.join("data", cls)], clip_length=50, mode="test")
        folder = "Aug_" + cls
        seq_by_file = {}
        for s in ds.samples:
            seq_by_file.setdefault(s["source_file"], ds.skeleton_data[s["video_idx"]])
        for f, seq in seq_by_file.items():
            base = os.path.splitext(os.path.basename(f))[0].replace(" ", "_")
            for c in range(copies):
                kp, desc = perturb(seq, rng)
                name = f"aug_{base}_{c:02d}"
                write_xml(os.path.join(out, folder, ANN, name + ".xml"), kp, name + ".mp4")
                rows.append(dict(file=f"{folder}/{ANN}/{name}.xml", folder=folder, source="real_aug",
                                 action=cls, variant=desc, frames=len(kp), hand=knife_arm(kp)))
                if len(previews.setdefault(folder, [])) < 14:
                    if c == 0:
                        previews[folder].append((np.asarray(seq), f"ORIGINAL {base}", None))
                    previews[folder].append((kp, f"{name} {desc}", None))
    return rows, previews


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO_DIR, "data", "Simulated"))
    ap.add_argument("--per-action", type=int, default=200, help="clips per attack type (%d types)" % len(A.ATTACKS))
    ap.add_argument("--normals", type=int, default=150, help="clips per normal activity (%d types)" % len(A.NORMALS))
    ap.add_argument("--real-aug", type=int, default=10, help="perturbed copies of each real training clip")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "metadata.csv")):
        sys.exit(f"[SIM] {out} already has a dataset. Pick another --out or delete it first.")
    os.makedirs(os.path.join(out, "previews"), exist_ok=True)
    t0 = time.time()

    plan = [(a, f) for a, (_, f) in A.ATTACKS.items() for _ in range(args.per_action)]
    plan += [(a, "Sim_OOD_Normal") for a in A.NORMALS for _ in range(args.normals)]
    jobs = [(i, a, f, args.seed * 1_000_003 + i, out) for i, (a, f) in enumerate(plan)]  # stable seed per clip
    print(f"[SIM] {len(jobs)} simulated clips with {args.workers} workers -> {out}")
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_sim_job, jobs, chunksize=8)):
            if r:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"      {i + 1}/{len(jobs)}  {time.time() - t0:.0f} s")

    previews = {}
    if args.real_aug > 0:
        print(f"[SIM] perturbing real training clips x{args.real_aug}")
        r2, previews = _real_jobs(out, args.real_aug, np.random.default_rng(args.seed))
        rows += r2

    rows.sort(key=lambda r: r["file"])
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # previews: one sheet + one animated grid per simulated folder, mixing actions
    rng = np.random.default_rng(0)
    by_folder = {}
    for r in rows:
        if r["source"] == "sim":
            by_folder.setdefault(r["folder"], []).append(r)
    for folder, rs in by_folder.items():
        pick = [rs[i] for i in rng.choice(len(rs), min(16, len(rs)), replace=False)]
        clips = []
        for r in pick:
            z = np.load(os.path.join(out, folder, "skeleton3d", os.path.basename(r["file"])[:-4] + ".npz"))
            wr = 10 if r["hand"] == "R" else 9
            clips.append((z["kpts"], f"{r['variant']} | {r['hand']} | cam az{r['cam_azimuth']:.0f} "
                                     f"h{r['cam_height']:.1f} d{r['cam_distance']:.1f}", wr))
        contact_sheet(os.path.join(out, "previews", folder + ".jpg"), clips[:12])
        grid_video(os.path.join(out, "previews", folder + ".mp4"), clips)
    for folder, clips in previews.items():
        contact_sheet(os.path.join(out, "previews", folder + ".jpg"), clips)

    counts = {}
    for r in rows:
        counts[r["folder"]] = counts.get(r["folder"], 0) + 1
    print("[SIM] done in %.0f s:" % (time.time() - t0))
    for k in sorted(counts):
        print(f"      {k:18s} {counts[k]:5d} clips")
    print(f"      metadata: {os.path.join(out, 'metadata.csv')}")
    print(f"      previews: {os.path.join(out, 'previews')}")


if __name__ == "__main__":
    main()
