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


names = ["T3 ST-GCN, RMD", "T1 PoseConv3D, RMD", "T2 distilled, RMD", "blend of 3, RMD", "all 3 must agree, RMD", "blend of 3, output", "T3 ST-GCN, output"]
rows, fa = {n: [] for n in names}, {n: {} for n in names}
for seed in (1, 2, 3):
    t3 = fit("t3", seed); t1 = fit("t1", seed); t2 = fit("t2", seed, teacher=t3)
    Z = {}  # per model: (held-out, per-segment) window scores, shifted so 0 = that model's own bound
    for key, kind, m in (("t3", "t3", t3), ("t1", "t1", t1), ("t2", "t2", t2)):
        ref = reference(run(m, kind, X)[1], torch.from_numpy(Y))
        for typ in ("rmd", "out"):
            sc = (lambda W: T.rmd(run(m, kind, W)[1], ref).numpy()) if typ == "rmd" else (lambda W: run(m, kind, W)[0].numpy())
            h = sc(held); b = np.quantile(per_strike(h), 0.10)
            Z[(key, typ)] = (h - b, [sc(w) - b for _, _, w in seg002])
    combos = {"T3 ST-GCN, RMD": [("t3", "rmd")], "T1 PoseConv3D, RMD": [("t1", "rmd")], "T2 distilled, RMD": [("t2", "rmd")],
              "blend of 3, RMD": [("t3", "rmd"), ("t1", "rmd"), ("t2", "rmd")], "all 3 must agree, RMD": "min",
              "blend of 3, output": [("t3", "out"), ("t1", "out"), ("t2", "out")], "T3 ST-GCN, output": [("t3", "out")]}
    for name, keys in combos.items():
        if keys == "min":
            ks = [("t3", "rmd"), ("t1", "rmd"), ("t2", "rmd")]
            sd = [Z[k][0].std() for k in ks]
            h = np.min([Z[k][0] / s for k, s in zip(ks, sd)], 0); sg = [np.min([Z[k][1][i] / s for k, s in zip(ks, sd)], 0) for i in range(len(seg002))]
        else:
            sd = [Z[k][0].std() for k in keys]
            h = np.mean([Z[k][0] / s for k, s in zip(keys, sd)], 0); sg = [np.mean([Z[k][1][i] / s for k, s in zip(keys, sd)], 0) for i in range(len(seg002))]
        b = np.quantile(per_strike(h), 0.10)  # the combination gets its own attack-only bound
        a = np.array([s.max() for (n, c, _), s in zip(seg002, sg) if c == "attack"]); nn_ = np.array([s.max() for (n, c, _), s in zip(seg002, sg) if c != "attack"])
        rows[name].append((roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(nn_))], np.r_[a, nn_]), (a >= b).sum(), (nn_ >= b).sum(), (a > nn_.max()).sum()))
        for (n, c, _), s in zip(seg002, sg):
            if c != "attack" and s.max() >= b:
                fa[name][n] = fa[name].get(n, 0) + 1
    print(f"seed {seed}: " + " | ".join(f"{n.split(',')[0][:10]}{'/out' if 'output' in n else ''} {rows[n][-1][0]:.2f} {rows[n][-1][1]}/{rows[n][-1][2]}" for n in names), flush=True)
print(f"\n{'score':<24}{'AUC (sd)':>15}{'attacks /31':>13}{'normal seg /24':>16}{'attacks above all normals':>27}   most alarmed normal movements (of 3 seeds)")
for n in names:
    v = np.array(rows[n], float)
    top = ", ".join(f"{k} {c}" for k, c in sorted(fa[n].items(), key=lambda kv: -kv[1])[:5])
    print(f"{n:<24}{f'{v[:, 0].mean():.3f} ({v[:, 0].std():.3f})':>15}{v[:, 1].mean():>13.1f}{v[:, 2].mean():>16.1f}{v[:, 3].mean():>27.1f}   {top}")
