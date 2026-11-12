"""Compare what the model is fed: Validate (training camera) vs Oct 31 lab (new camera)."""
import csv, glob, os, sys
import numpy as np, cv2

REPO = sys.argv[1]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import train_stage2_sim as T
from src.skeleton_utils import normalize_skeleton_clip

def vid(p):
    c = cv2.VideoCapture(p)
    r = (c.get(cv2.CAP_PROP_FPS), int(c.get(3)), int(c.get(4)), int(c.get(cv2.CAP_PROP_FRAME_COUNT))); c.release(); return r

for name, pat in [("Validate thrust", "data/Validate/Without_Coat/Thrust/*.mp4"), ("Validate OOD", "data/Validate/OOD/*.mp4"),
                  ("train Thrust", "data/Thrust/**/*.mp4"), ("lab Oct31 raw", "data/lab_recordings/2026-10-31/raw/*.avi")]:
    fs = sorted(glob.glob(os.path.join(REPO, pat), recursive=True))[:4]
    print(f"{name:<16}", [f"{f:.1f}fps {w}x{h} {n}fr" for f, w, h, n in map(vid, fs)])

def stats(seqs):
    """seqs: list of (T,17,2) smoothed pixel sequences. Returns medians of shape and motion measures."""
    o = {k: [] for k in ("torso_px", "sh_w/torso", "hip_w/torso", "leg/torso", "head_above", "wrist_spd", "wrist_reach", "valid")}
    for K in seqs:
        K = np.asarray(K, np.float32)
        ok = np.abs(K).sum(-1) > 0
        o["valid"].append(ok.mean())
        ms, mh = K[:, [5, 6]].mean(1), K[:, [11, 12]].mean(1)
        torso = np.linalg.norm(ms - mh, axis=-1) + 1e-6
        o["torso_px"].append(np.median(torso))
        o["sh_w/torso"].append(np.median(np.linalg.norm(K[:, 5] - K[:, 6], axis=-1) / torso))
        o["hip_w/torso"].append(np.median(np.linalg.norm(K[:, 11] - K[:, 12], axis=-1) / torso))
        o["leg/torso"].append(np.median(np.linalg.norm(mh - K[:, [15, 16]].mean(1), axis=-1) / torso))
        o["head_above"].append(np.median((ms[:, 1] - K[:, 0, 1]) / torso))
        N = np.concatenate([normalize_skeleton_clip(K[s:s + 50]) for s in range(0, max(1, len(K) - 49), 50)])
        w = N[:, [9, 10]]
        spd = np.linalg.norm(np.diff(w, axis=0), axis=-1).max(1)
        o["wrist_spd"].append(np.quantile(spd, 0.95) if len(spd) else 0)
        o["wrist_reach"].append(np.quantile(np.linalg.norm(w - N[:, [5, 6]], axis=-1).max(1), 0.95))
    return {k: float(np.median(v)) for k, v in o.items()}

sets = {}
for a in ("Thrust", "Stab", "Cut_Down"):
    sets["Validate " + a] = [T.parse_xml(f) for f in glob.glob(os.path.join(REPO, "data/Validate/*", a, T.ANN, "*.xml"))]
sets["Validate OOD"] = [s for s in (T.parse_xml(f) for f in glob.glob(os.path.join(REPO, "data/Validate/OOD", T.ANN, "*.xml"))) if s is not None]
LAB = T.LAB
tk = {}
for r in csv.DictReader(open(os.path.join(LAB, "segments", "segments.csv"), encoding="utf-8")):
    if r["take"] not in tk:
        tk[r["take"]] = np.load(os.path.join(LAB, "analysis", r["take"] + "_analysis.npz"))["kpts"][..., :2]
    f0, f1 = int(float(r["start_s"]) * 30), int(float(r["end_s"]) * 30)
    lab = r["label"]
    key = "lab thrust" if "thrust" in lab else "lab overhand/cut/stab" if r["category"] == "attack" else "lab normal"
    sets.setdefault(key, []).append(T.live_prep(tk[r["take"]][f0:max(f1, f0 + 50)]))
print(f"\n{'set':<24}{'n':>4}" + "".join(f"{k:>13}" for k in ("frames", "torso_px", "sh_w/torso", "hip_w/torso", "leg/torso", "head_above", "wrist_spd", "wrist_reach", "valid")))
for k, v in sets.items():
    v = [s for s in v if s is not None and len(s) >= 10]
    st = stats(v)
    print(f"{k:<24}{len(v):>4}{np.median([len(s) for s in v]):>13.0f}" + "".join(f"{st[c]:>13.3f}" for c in ("torso_px", "sh_w/torso", "hip_w/torso", "leg/torso", "head_above", "wrist_spd", "wrist_reach", "valid")))
npz = np.load(os.path.join(LAB, "analysis", "002_checkerboard_142846_analysis.npz"))
print("\nlab npz keys:", {k: npz[k].shape for k in npz.files})
