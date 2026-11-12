"""Original models on the Nov 5 attack takes, by facing direction, against Oct 31 lab normal windows."""
import glob, os, sys
from collections import defaultdict
import numpy as np, torch
from sklearn.metrics import roc_auc_score

REPO = sys.argv[1]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import train_stage2_sim as T

cas = T.Cascade("old", torch.device("cuda:0"))
_, takes = T.lab_windows()
rep = T.take_replay(takes)
Zn = cas.probs(np.stack([r[4] for r in rep if r[2] == "normal"]), raw=True)
g = defaultdict(list)
for f in sorted(glob.glob(os.path.join(REPO, "results", "samples", "kp", "*.npy"))):
    name = os.path.basename(f)[:-4]
    K = np.load(f)[..., :2]
    ok = np.abs(K).sum((1, 2)) > 0
    W = [T.live_prep(K[s:s + 50]) for s in range(0, len(K) - 49, 10) if ok[s:s + 50].mean() > 0.9]
    if not W or "floor" in name:
        continue
    d = next((x for x in ("front", "left", "back", "right", "lefthand") if f"_{x}_" in name + "_"), "?")
    a = next((x for x in ("walking_thrust", "thrust", "overhand_stab", "cut_down", "slash") if x in name), "?")
    Z = cas.probs(np.stack(W), raw=True)
    g[("dir", d)].append(Z); g[("act", a)].append(Z)
print(f"lab normal windows: {len(Zn)};  columns: share of windows over each model's own threshold (z>0), then AUC vs lab normal")
print(f"{'group':<24}{'win':>6}{'T1>0':>8}{'T2>0':>8}{'T3>0':>8}{'cascade':>9}{'AUC T1':>8}{'AUC T2':>8}{'AUC T3':>8}{'AUC blend':>10}")
rows = [("lab normal (Oct 31)", Zn)] + [(f"{k[0]} {k[1]}", np.concatenate(v)) for k, v in sorted(g.items())]
for n, Z in rows:
    A = cas.alarm(1 / (1 + np.exp(-Z)))
    y = np.r_[np.ones(len(Z)), np.zeros(len(Zn))]
    aucs = ["   -   "] * 4 if Z is Zn else [f"{roc_auc_score(y, np.r_[s, t]):.3f}" for s, t in
                                         [(Z[:, i], Zn[:, i]) for i in range(3)] + [(Z.mean(1), Zn.mean(1))]]
    print(f"{n:<24}{len(Z):>6}" + "".join(f"{(Z[:, i] > 0).mean():>8.1%}" for i in range(3)) + f"{A.mean():>9.1%}" + "".join(f"{a:>8}" for a in aucs[:3]) + f"{aucs[3]:>10}")
