"""The project's own Stage 2 recipe on the lab takes: three models, each scored by the RMD out-of-distribution score.
  t3 ST-GCN, t1 PoseConv3D, t2 PoseConv3D distilled from t3 (relational KD) - same architectures, same loss as
  scripts/train_stage2_sim.py (attack-type cross-entropy + attack/normal BCE + batch-hard triplet), same augmentation,
  RMD reference built as rmd_reference() does (background = all training windows, one Gaussian per attack type).
Data: precise labels (attack = window holds the whole forward hand movement, class = attack type; not = no hand trip),
  Nov 5 training takes + lab take 001.  Bound per model: keeps 90 % of held-out Nov 5 strikes.  Test: take 002 segments.
Nothing is written to weights/.
"""
import os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import train_stage2_sim as T
import generate_from_samples as S
import generate_from_edits as GE
from src.distillation_losses import RelationalKDLoss

dev = torch.device("cuda:0")
STEPS, BS = 700, 48
takes = [GE.prepare(t) for t in S.load_takes()]
tr, ytr, held, hg = [], [], [], []
for t in takes:
    for s in range(0, len(t["xy"]) - 49, 10):
        if t["missing"][s:s + 50].sum() > 10:
            continue
        e = s + 50
        w = T.live_prep(t["xy"][s:e])
        if t["num"] in S.HELD_OUT:
            ov = max(((min(e, b) - max(s, a), i) for i, (a, _, b) in enumerate(t["strikes"])), default=(0, -1))
            if ov[0] >= 15:
                held.append(w); hg.append((t["num"], ov[1]))
        elif any(s <= a and land < e for a, land, _ in t["strikes"]):
            tr.append(w); ytr.append(t["cls"])
        elif not any(min(e, b + 5) - max(s, a - 5) > 0 for a, _, b in t["any_trips"]):
            tr.append(w); ytr.append(0)
segs, lab_takes = T.lab_windows()
t001 = [T.live_prep(lab_takes[k][s:s + 50]) for k in lab_takes if k.startswith("001") for s in range(0, len(lab_takes[k]) - 49, 10)]
seg002 = [(n[4:], c, np.stack(ws)) for n, c, ws in segs if int(n[:3]) >= 4 and ws]
X = np.stack(tr + t001).astype(np.float32); Y = np.array(ytr + [0] * len(t001)); held = np.stack(held).astype(np.float32)
NC = int(Y.max()) + 1
print(f"train windows: {np.bincount(Y).tolist()} (class 0 = not attack, 1-4 = thrust, stab, cut-down, slash)", flush=True)


def fit(kind, seed, teacher=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, head = T.build(kind).to(dev), nn.Linear(256, NC).to(dev)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.08)
    rkd = RelationalKDLoss(w_dist=1.0, w_angle=2.0).to(dev) if teacher is not None else None
    pools = [np.flatnonzero(Y == c) for c in range(NC)]
    per = [BS // 2] + [BS // 2 // (NC - 1)] * (NC - 1)  # half of each batch is "not attack"
    Xt, Yt = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)
    m.train()
    for _ in range(STEPS):
        idx = torch.from_numpy(np.concatenate([rng.choice(p, k) for p, k in zip(pools, per)])).to(dev)
        n, v = T.augment(*T.normalize_batch(Xt[idx]), gen)
        y = Yt[idx]
        logit, f = m(T.model_input(kind, n, v), return_features=True)
        loss = F.cross_entropy(head(f), y, label_smoothing=0.05) + 0.5 * F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), (y > 0).float()) + 0.5 * T.batch_hard_triplet(f, y)
        if teacher is not None:
            with torch.no_grad():
                _, ft = teacher(T.coords_input(n), return_features=True)
            loss = loss + 20.0 * rkd(f, ft)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def run(m, kind, W):
    L, Fs = [], []
    for i in range(0, len(W), 96):
        n, v = T.normalize_batch(torch.from_numpy(np.asarray(W[i:i + 96], np.float32)).to(dev))
        l, f = m(T.model_input(kind, n, v), return_features=True)
        L.append(l.float().reshape(-1).cpu()); Fs.append(f.float().cpu())
    return torch.cat(L), torch.cat(Fs)


def reference(f, y, eps=1e-2):
    mu_0 = f.mean(0, keepdim=True); d0 = f - mu_0
    eye = torch.eye(f.shape[1]); mus, inv = [], []
    for c in range(1, NC):
        fc = f[y == c]; mus.append(fc.mean(0)); dc = fc - mus[-1]
        inv.append(torch.linalg.pinv(dc.T @ dc / (len(fc) - 1) + eps * eye))
    return dict(mu_0=mu_0, mu_c=torch.stack(mus), inv_Sigma_0=torch.linalg.pinv(d0.T @ d0 / (len(f) - 1) + eps * eye), inv_Sigma_c=torch.stack(inv))


def per_strike(s):
    by = {}
    for g, q in zip(hg, s):
        by[g] = max(by.get(g, -1e18), q)
    return np.array(list(by.values()))


from src.ood_metrics import compute_relative_mahalanobis_distance as crmd
auc = lambda p, n: roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])
m = fit("t3", 1)
ftr = run(m, "t3", X)[1]; yt = torch.from_numpy(Y)
half = np.arange(len(X)) % 2 == 0          # fit the reference on half of the training windows, check on the other half
for label, sel in (("background = all training windows (as the project does)", torch.from_numpy(half)),
                   ("background = not-attack training windows only", torch.from_numpy(half & (Y == 0)))):
    ref = reference(ftr[torch.from_numpy(half)], yt[torch.from_numpy(half)])
    f0 = ftr[sel]; ref["mu_0"] = f0.mean(0, keepdim=True); d0 = f0 - ref["mu_0"]
    ref["inv_Sigma_0"] = torch.linalg.pinv(d0.T @ d0 / (len(f0) - 1) + 1e-2 * torch.eye(256))
    def parts(W):
        s, m0, mc = crmd(run(m, "t3", W)[1], ref["mu_0"], ref["mu_c"], ref["inv_Sigma_0"], ref["inv_Sigma_c"])
        return s.numpy(), np.asarray(m0), np.asarray(mc)
    tr_s = parts(X[~half]); ya = Y[~half] > 0
    A = [parts(w) for _, c, w in seg002 if c == "attack"]; N = [parts(w) for _, c, w in seg002 if c != "attack"]
    H = parts(held)
    print(f"\n{label}")
    print(f"  unseen half of training windows, attack vs not:  AUC {auc(tr_s[0][ya], tr_s[0][~ya]):.3f}")
    print(f"  held-out Nov 5 strikes vs take 002 normal windows: AUC {auc(H[0], np.concatenate([n[0] for n in N])):.3f}")
    print(f"  take 002 attack vs normal segments (best window):  AUC {auc([a[0].max() for a in A], [n[0].max() for n in N]):.3f}")
    med = lambda L, i: float(np.median(np.concatenate([np.ravel(x[i]) for x in L])))
    print(f"  median distance to nearest attack type / to background:   Nov 5 train attack {np.median(tr_s[2][ya]):.0f} / {np.median(tr_s[1][ya]):.0f}"
          f"   held-out Nov 5 strikes {np.median(H[2]):.0f} / {np.median(H[1]):.0f}   take 002 attack {med(A, 2):.0f} / {med(A, 1):.0f}   take 002 normal {med(N, 2):.0f} / {med(N, 1):.0f}")
o = lambda W: run(m, "t3", W)[0].numpy()
print(f"\nplain output, same model: take 002 attack vs normal segments AUC {auc([o(w).max() for _, c, w in seg002 if c == 'attack'], [o(w).max() for _, c, w in seg002 if c != 'attack']):.3f}")
