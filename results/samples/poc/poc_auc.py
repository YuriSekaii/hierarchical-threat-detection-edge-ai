import csv, sys, math, numpy as np
from sklearn.metrics import roc_auc_score
d = sys.argv[1]
W = list(csv.DictReader(open(d + "/poc_windows.csv")))
lg = lambda p: math.log(min(max(p,1e-6),1-1e-6)/(1-min(max(p,1e-6),1-1e-6)))
clips = {}
for w in W:
    k = (w["cat"], w["act"], w["file"])
    p = [float(w["p1"]), float(w["p2"]), float(w["p3"])]
    s = {"T1 PoseConv3D": p[0], "T2 distilled": p[1], "T3 ST-GCN": p[2], "blend of 3": sum(map(lg, p))/3, "max of 3": max(p), "min of 3": min(p)}
    c = clips.setdefault(k, {})
    for n, v in s.items(): c[n] = max(c.get(n, -1e9), v)
keys = list(clips); y = np.array([k[0] != "OOD" for k in keys]); act = np.array([k[1] for k in keys])
print(f"clips: {y.sum()} attack, {(~y).sum()} normal (5 normal clips under 50 frames have no window)")
print(f"{'score':<16}{'AUC':>7}{'caught @0 FA':>14}{'@<=2 FA':>9}   thrust/stab/cut @0 FA")
for n in ["T1 PoseConv3D", "T2 distilled", "T3 ST-GCN", "blend of 3", "max of 3", "min of 3"]:
    s = np.array([clips[k][n] for k in keys]); neg = np.sort(s[~y])[::-1]
    c0 = s > neg[0]; c2 = s > neg[2]
    print(f"{n:<16}{roc_auc_score(y, s):>7.3f}{c0[y].sum():>10}/104{c2[y].sum():>6}/104   " + " / ".join(f"{c0[y & (act==a)].sum()}/{(y & (act==a)).sum()}" for a in ("Thrust","Stab","Cut_Down")))
