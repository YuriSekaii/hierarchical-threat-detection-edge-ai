"""Proof of concept: score every Validate clip with the original models, two ways.
suite = how the 1.0000 benchmark scores (stride 25, dataset heatmaps, max over the video, then cascade)
live  = how the live pipeline scores (stride 10, per-window preprocessing, per-window cascade verdict)
"""
import csv, glob, os, sys
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np, torch

REPO = sys.argv[1]
sys.path.insert(0, REPO)
from src.stage2b_staircase import Stage2bStaircaseEngine, rasterize_heatmap_from_coords
from src.dataset import ActionDataset, PoseConv3DHeatmapDataset
from src.skeleton_utils import normalize_skeleton_clip, interpolate_missing_joints, smooth_kinematics

dev = torch.device("cuda:0")
eng = Stage2bStaircaseEngine(device=dev)
VAL = os.path.join(REPO, "data", "Validate")


def raw_xml(path):  # raw pixel keypoints, no smoothing (what the live buffer holds)
    imgs = ET.parse(path).getroot().findall(".//image")
    K = np.zeros((len(imgs), 17, 2), np.float32)
    for i, im in enumerate(imgs):
        p = im.find("points")
        pts = p.get("points", "").split(";") if p is not None else []
        if len(pts) >= 17:
            K[i] = [[float(v) for v in q.split(",")] for q in pts[:17]]
    return K


def live_windows(K, stride=10):
    out = []
    for s in range(0, len(K) - 49, stride):
        a = smooth_kinematics(interpolate_missing_joints(K[s:s + 50].copy()), sigma=1.0)
        t = torch.from_numpy(normalize_skeleton_clip(a)).float()
        hm = rasterize_heatmap_from_coords(t, device=dev, normalize=False)
        c = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(dev)
        p = [eng._eval_model(k, hm if eng.models[k]["requires_heatmap"] else c)
             for k in (eng.tier1_key, eng.tier2_key, eng.tier3_key)]
        out.append((s, *p, eng.evaluate_clip(t)["verdict"]))
    return out


rows, wrows = [], []
folders = [("With_Coat", a) for a in ("Thrust", "Stab", "Cut_Down")] + \
          [("Without_Coat", a) for a in ("Thrust", "Stab", "Cut_Down")] + [("OOD", "")]
for cat, act in folders:
    d = os.path.join(VAL, cat, act) if act else os.path.join(VAL, cat)
    cds = ActionDataset(root_dir=".", action_folders=[d], clip_length=50, mode="test")
    hds = PoseConv3DHeatmapDataset(root_dir=".", action_folders=[d], clip_length=50, heatmap_size=56, mode="test")
    vc, vh = defaultdict(list), defaultdict(list)
    for i in range(len(cds)):
        c, _, src = cds[i]; vc[src].append(c)
        h, _, src = hds[i]; vh[src].append(h)
    for src in sorted(vc):
        r = eng.evaluate_video(torch.stack(vc[src]), torch.stack(vh[src]))
        K = raw_xml(src)
        lw = live_windows(K)
        for s, p1, p2, p3, v in lw:
            wrows.append([cat, act, os.path.basename(src), s, f"{p1:.4f}", f"{p2:.4f}", f"{p3:.4f}", v])
        rows.append([cat, act or "OOD", os.path.basename(src), len(K), r["verdict"], r["resolved_tier"],
                     len(lw), sum(w[4] for w in lw), int(any(w[4] for w in lw)),
                     max((w[1] for w in lw), default=-1), max((w[2] for w in lw), default=-1),
                     max((w[3] for w in lw), default=-1)])

out = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(out, "poc_videos.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["cat", "act", "file", "frames", "suite", "suite_tier", "n_win", "win_alarm",
                              "live", "p1max", "p2max", "p3max"]] + rows)
with open(os.path.join(out, "poc_windows.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["cat", "act", "file", "start", "p1", "p2", "p3", "verdict"]] + wrows)

print(f"\n{'folder':<24}{'clips':>6}{'suite':>8}{'live':>8}{'<50 fr':>8}{'win alarmed':>14}")
g = defaultdict(list)
for r in rows:
    g[(r[0], r[1])].append(r)
for k, v in g.items():
    nw, na = sum(r[6] for r in v), sum(r[7] for r in v)
    print(f"{k[0] + '/' + k[1]:<24}{len(v):>6}{sum(r[4] for r in v):>8}{sum(r[8] for r in v):>8}"
          f"{sum(r[3] < 50 for r in v):>8}{f'{na}/{nw}':>14}")
