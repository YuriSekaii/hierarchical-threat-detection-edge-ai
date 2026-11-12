"""Original models on the new camera (Oct 31 lab takes), per tier, without the tuned cascade.
Bounds are set on take 001 (normal only) and tested on take 002, so no bound is chosen on the test take."""
import os, sys
import numpy as np, torch
from sklearn.metrics import roc_auc_score

REPO = sys.argv[1]
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import train_stage2_sim as T

dev = torch.device("cuda:0")
cas = T.Cascade("old", dev)
_, takes = T.lab_windows()
rep = T.take_replay(takes)
Z = cas.probs(np.stack([w for *_, w in rep]), raw=True)          # (N, 3) RMD - tau per tier
take = np.array([r[0] for r in rep]); lab = np.array([r[2] for r in rep]); aid = np.array([str(r[3]) for r in rep])
t1 = np.array([t.startswith("001") for t in take]); t2 = ~t1
print({t: int((take == t).sum()) for t in takes}, "windows;  take 002:", {l: int((lab[t2] == l).sum()) for l in set(lab)})

S = {"T1 PoseConv3D": Z[:, 0], "T2 distilled": Z[:, 1], "T3 ST-GCN": Z[:, 2], "blend of 3": Z.mean(1)}
A = cas.alarm(1 / (1 + np.exp(-Z)))
att = t2 & (lab == "attack"); nor = t2 & (lab == "normal"); ids = sorted(set(aid[att]))
print(f"\ntuned cascade as shipped: {len({i for i in aid[att & A]})}/{len(ids)} attacks, "
      f"{int(A[nor].sum())}/{int(nor.sum())} normal windows alarmed")
print(f"\n{'score':<16}{'AUC win':>8}{'  bound = take-001 max':>23}{'  bound = take-001 98%':>23}{'  take-002 own 98%':>19}")
for n, s in S.items():
    auc = roc_auc_score(np.r_[np.ones(att.sum()), np.zeros(nor.sum())], np.r_[s[att], s[nor]])
    cells = []
    for b in (s[t1].max(), np.quantile(s[t1], 0.98), np.quantile(s[nor], 0.98)):
        hit = len({i for i in aid[att & (s > b)]})
        cells.append(f"{hit:>2}/{len(ids)} att, {int((s[nor] > b).sum()):>3} FA win")
    print(f"{n:<16}{auc:>8.3f}   " + "   ".join(cells))
print(f"(normal windows in take 002: {int(nor.sum())}, about {nor.sum() * 10 / 30 / 60:.1f} min)")
# saturation check: how many attack windows sit at p1 < 0.20 (tier 1 discard)
P = 1 / (1 + np.exp(-Z))
print(f"\nattack windows discarded by tier 1 (p1<0.20): {int((P[att, 0] < 0.2).sum())}/{int(att.sum())};"
      f" median p1/p2/p3 on attack windows: {np.median(P[att], 0).round(3)}; on normal: {np.median(P[nor], 0).round(3)}")
print("median raw z (RMD - tau) attack:", np.median(Z[att], 0).round(1), " normal:", np.median(Z[nor], 0).round(1),
      " | Validate-like scale: tau crossing is z=0")
