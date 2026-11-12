"""The user's design, tested on the lab camera: "attack" = close to a trained action, everything else is rejected.
The alarm bound comes from attack recordings only (it keeps 90 % of the strikes in the held-out Nov 5 takes).
No take 002 data and no normal gesture data is used for training or for the bound.

labels   binary : attack / not (pauses, walking)            fine : attack type x facing direction x hand, + not
scores   output : the network's own attack probability      rmd  : nearest attack class vs the "not" class (current design)
         maha   : distance to the nearest attack class only  knn  : distance to the 5 nearest training attack windows
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import train_stage2_sim as T
import generate_from_samples as S
import generate_from_edits as GE

dev = torch.device("cuda:0")
LIVE = os.path.join(REPO, "data", "live_clips_2026-10-31")
takes = [GE.prepare(t) for t in S.load_takes()]
names = {}
tr, held = [], []
for t in takes:
    key = (t["cls"], t["dir"], t["hand"])
    for s, i in S.label_windows(GE.relabelled(t)):
        w = T.live_prep(t["xy"][s:s + 50])
        if t["num"] in S.HELD_OUT:
            held.append((w, i, t["num"]))
        else:
            tr.append((w, names.setdefault(key, len(names) + 1) if i >= 0 else 0))
segs, lab_takes = T.lab_windows()
t001 = [T.live_prep(lab_takes[k][s:s + 50]) for k in lab_takes if k.startswith("001") for s in range(0, len(lab_takes[k]) - 49, 10)]
excl = {"s1031": (28, 34), "s110035": (29, 35)}
live = [w for sess, clip, w in T.live_clips(LIVE) if sess in excl and not excl[sess][0] <= int(clip.split("_")[1]) <= excl[sess][1]]
seg002 = [(n, c, ws) for n, c, ws in segs if int(n[:3]) >= 4 and ws]
X = np.stack([w for w, _ in tr] + t001 + live).astype(np.float32)
yf = np.array([c for _, c in tr] + [0] * (len(t001) + len(live)))
print(f"train windows {len(X)}: {int((yf > 0).sum())} attack in {len(names)} type/direction classes, {int((yf == 0).sum())} not-attack")


def fit(y, seed):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    k = int(y.max()) + 1
    m = T.STGCNModel(num_classes=1 if k == 2 else k).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    cnt = np.bincount(y, minlength=k)
    w = torch.from_numpy(1.0 / cnt[y]).double()
    w[yt == 0] *= max(1, k - 1)  # "not attack" gets half of every batch
    steps = 40 * (len(X) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    m.train()
    for _ in range(steps):
        idx = torch.multinomial(w, 64, replacement=True)
        n, v = T.normalize_batch(Xt[idx].to(dev))
        n, v = T.augment(n, v, gen)
        if k > 2:  # mirroring swaps left and right, which is part of the fine label: undo it by not flipping
            n, v = T.normalize_batch(Xt[idx].to(dev))
        out, _ = m(T.coords_input(n), return_features=True)
        loss = (F.binary_cross_entropy_with_logits(out.reshape(-1), (yt[idx] > 0).float().to(dev)) if k == 2
                else F.cross_entropy(out, yt[idx].to(dev)))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def feats(m, W):
    O, Fs = [], []
    for i in range(0, len(W), 256):
        n, _ = T.normalize_batch(torch.from_numpy(np.stack(W[i:i + 256])).float().to(dev))
        o, f = m(T.coords_input(n), return_features=True)
        O.append(o.float().cpu()); Fs.append(f.float().cpu())
    return torch.cat(O), torch.cat(Fs).double().numpy()


def scorers(m, y):
    o, f = feats(m, list(X))
    cls = [c for c in np.unique(y) if c > 0]
    mu = {c: f[y == c].mean(0) for c in np.unique(y)}
    cov = sum(((f[y == c] - mu[c]).T @ (f[y == c] - mu[c])) for c in mu) / len(f)
    P = np.linalg.inv(cov + 1e-3 * np.trace(cov) / len(cov) * np.eye(len(cov)))
    d = lambda g, c: np.einsum("ni,ij,nj->n", g - mu[c], P, g - mu[c])
    fa = f[y > 0] / np.linalg.norm(f[y > 0], axis=1, keepdims=True)

    def sc(W):
        o, g = feats(m, W)
        prob = torch.sigmoid(o.reshape(-1)).numpy() if o.shape[1] == 1 else 1 - torch.softmax(o, 1)[:, 0].numpy()
        dmin = np.min([d(g, c) for c in cls], 0)
        sim = (g / np.linalg.norm(g, axis=1, keepdims=True)) @ fa.T
        return {"output": prob, "rmd": d(g, 0) - dmin, "maha": -dmin, "knn": np.sort(sim, 1)[:, -5:].mean(1)}
    return sc


def pair(p, n=2):  # best run of n windows in a row (single window if the segment has only one)
    p = np.asarray(p)
    return p.max() if len(p) < n else np.min(np.lib.stride_tricks.sliding_window_view(p, n), 1).max()

walk_cls = {c for (cls, d, h), c in names.items() if any(t["cls"] == cls and t["dir"] == d and t["hand"] == h and "walking" in t["kind"] for t in takes)}
print("walking-thrust classes:", sorted(walk_cls), " windows:", int(np.isin(yf, list(walk_cls)).sum()))
one = sum(len(ws) < 2 for n, c, ws in seg002)
print(f"take 002 segments with a single window: {one}")
print(f"\n{'train':<14}{'rule':<8}{'seed':>5}{'AUC seg':>9}{'attacks':>9}{'normal seg':>12}")
rows, Xall, yall = {}, X, yf
for name, keepm in (("all takes", np.ones(len(yf), bool)), ("no walk-thrust", ~np.isin(yf, list(walk_cls)))):
    X = Xall[keepm]; y = (yall[keepm] > 0).astype(np.int64)
    hk = [h for h in held if not (name != "all takes" and any(t["num"] == h[2] and "walking" in t["kind"] for t in takes))]
    for seed in (1, 2, 3):
        sc = scorers(fit(y, seed), y)
        H = sc([w for w, _, _ in hk])["output"]
        S2 = [(n, c, sc(ws)["output"]) for n, c, ws in seg002]
        for rule, n in (("1 win", 1), ("2 in row", 2), ("3 in row", 3)):
            by = {}
            for (_, i, num), q in zip(hk, H):
                if i >= 0:
                    by.setdefault((num, i), []).append(q)
            b = np.quantile([pair(v, n) for v in by.values()], 0.10)
            a = np.array([pair(s, n) for _, c, s in S2 if c == "attack"]); nn = np.array([pair(s, n) for _, c, s in S2 if c != "attack"])
            r = (roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(nn))], np.r_[a, nn]), (a >= b).sum(), (nn >= b).sum())
            rows.setdefault((name, rule), []).append(r)
            print(f"{name:<14}{rule:<8}{seed:>5}{r[0]:>9.3f}{f'{r[1]}/31':>9}{f'{r[2]}/24':>12}", flush=True)
            if seed == 1 and rule == "2 in row":
                print("      normal over bound:", [m[4:] for (m, c, s) in S2 if c != "attack" and pair(s, n) >= b])
                print("      attacks missed:   ", [m[4:] for (m, c, s) in S2 if c == "attack" and pair(s, n) < b])
print("\nmean of 3 seeds")
for (name, rule), v in rows.items():
    v = np.array(v, float).mean(0)
    print(f"{name:<14}{rule:<8}  AUC {v[0]:.3f}   attacks {v[1]:.1f}/31   normal segments {v[2]:.1f}/24")
