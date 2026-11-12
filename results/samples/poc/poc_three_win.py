"""Three-model recipe (ST-GCN, PoseConv3D, distilled PoseConv3D) at window 50 and 30, plain output and RMD,
with (a) the attack-only bound and (b) the best bound that could be tuned on take 002 itself (optimistic ceiling)."""
import csv, os, sys
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
rows_csv = list(csv.DictReader(open(os.path.join(T.LAB, "segments", "segments.csv"), encoding="utf-8")))
lab = {}
for r in rows_csv:
    if r["take"] not in lab:
        lab[r["take"]] = np.load(os.path.join(T.LAB, "analysis", r["take"] + "_analysis.npz"))["kpts"][..., :2]
k1 = [k for k in lab if k.startswith("001")][0]


def data(L, stride):
    tr, ytr, held, hg = [], [], [], []
    for t in takes:
        for s in range(0, len(t["xy"]) - L + 1, stride):
            e = s + L
            if t["missing"][s:e].sum() > L // 5:
                continue
            hit = [i for i, (a, land, _) in enumerate(t["strikes"]) if (s <= a or e - 1 - land <= stride) and s <= land < e and e - 1 - land < L // 2 + stride]
            w = T.live_prep(t["xy"][s:e])
            if t["num"] in S.HELD_OUT:
                for i in hit:
                    held.append(w); hg.append((t["num"], i))
            elif hit:
                tr.append(w); ytr.append(t["cls"])
            elif not any(min(e, b + 5) - max(s, a - 5) > 0 for a, _, b in t["any_trips"]):
                tr.append(w); ytr.append(0)
    t001 = [T.live_prep(lab[k1][s:s + L]) for s in range(0, len(lab[k1]) - L + 1, stride)]
    segs = []
    for r in rows_csv:
        K = lab[r["take"]]
        f0, f1 = int(round(float(r["start_s"]) * 30)), int(round(float(r["end_s"]) * 30))
        if f1 - f0 < 50:
            pad = 50 - (f1 - f0); f0 = max(0, f0 - pad // 2 - pad % 2); f1 = min(len(K), f0 + 50); f0 = f1 - 50
        if int(r["id"]) >= 4:
            segs.append((r["label"], r["category"], np.stack([T.live_prep(K[s:s + L]) for s in range(f0, f1 - L + 1, stride)])))
    return np.stack(tr + t001).astype(np.float32), np.array(ytr + [0] * len(t001)), np.stack(held).astype(np.float32), hg, segs


def fit(kind, seed, X, Y, teacher=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed); rng = np.random.default_rng(seed)
    NC = int(Y.max()) + 1
    m, head = T.build(kind).to(dev), nn.Linear(256, NC).to(dev)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.08)
    rkd = RelationalKDLoss(w_dist=1.0, w_angle=2.0).to(dev) if teacher is not None else None
    pools = [np.flatnonzero(Y == c) for c in range(NC)]; per = [BS // 2] + [BS // 2 // (NC - 1)] * (NC - 1)
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
    L_, Fs = [], []
    for i in range(0, len(W), 96):
        n, v = T.normalize_batch(torch.from_numpy(np.asarray(W[i:i + 96], np.float32)).to(dev))
        l, f = m(T.model_input(kind, n, v), return_features=True)
        L_.append(l.float().reshape(-1).cpu()); Fs.append(f.float().cpu())
    return torch.cat(L_), torch.cat(Fs)


def reference(f, y, eps=1e-2):
    f0 = f[y == 0]; mu_0 = f0.mean(0, keepdim=True); d0 = f0 - mu_0; eye = torch.eye(f.shape[1]); mus, inv = [], []
    for c in range(1, int(y.max()) + 1):
        fc = f[y == c]; mus.append(fc.mean(0)); dc = fc - mus[-1]
        inv.append(torch.linalg.pinv(dc.T @ dc / (len(fc) - 1) + eps * eye))
    return dict(mu_0=mu_0, mu_c=torch.stack(mus), inv_Sigma_0=torch.linalg.pinv(d0.T @ d0 / (len(f0) - 1) + eps * eye), inv_Sigma_c=torch.stack(inv))


print(f"{'window':>7}{'score':>22}{'AUC':>8} | attack-only bound: {'att/31':>7}{'norm/24':>8} | tuned on take 002: {'att @0 FA':>10}{'@<=2 FA':>9}{'@<=4 FA':>9}", flush=True)
for L, stride in ((50, 10), (30, 10)):
    X, Y, held, hg, segs = data(L, stride)
    acc = {}
    for seed in (1, 2, 3):
        try:
            t3 = fit("t3", seed, X, Y); t1 = fit("t1", seed, X, Y); t2 = fit("t2", seed, X, Y, teacher=t3)
        except Exception as ex:
            print(f"window {L}: training failed ({type(ex).__name__}: {str(ex)[:120]})", flush=True); break
        Z = {}
        for key, m in (("t3", t3), ("t1", t1), ("t2", t2)):
            ref = reference(run(m, key, X)[1], torch.from_numpy(Y))
            for typ in ("rmd", "out"):
                sc = (lambda W, m=m, key=key, ref=ref: T.rmd(run(m, key, W)[1], ref).numpy()) if typ == "rmd" else (lambda W, m=m, key=key: run(m, key, W)[0].numpy())
                h = sc(held); Z[(key, typ)] = (h, [sc(w) for _, _, w in segs])
        for name, keys in (("ST-GCN, output", [("t3", "out")]), ("3 blended, output", [(k, "out") for k in ("t3", "t1", "t2")]),
                           ("ST-GCN, RMD", [("t3", "rmd")]), ("3 blended, RMD", [(k, "rmd") for k in ("t3", "t1", "t2")])):
            sd = [Z[k][0].std() for k in keys]; mu = [np.median(Z[k][0]) for k in keys]
            h = np.mean([(Z[k][0] - u) / s for k, s, u in zip(keys, sd, mu)], 0)
            sg = [np.mean([(Z[k][1][i] - u) / s for k, s, u in zip(keys, sd, mu)], 0).max() for i in range(len(segs))]
            by = {}
            for g, q in zip(hg, h):
                by[g] = max(by.get(g, -1e18), q)
            b = np.quantile(list(by.values()), 0.10)
            a = np.array([s for (n, c, _), s in zip(segs, sg) if c == "attack"]); nn_ = np.sort(np.array([s for (n, c, _), s in zip(segs, sg) if c != "attack"]))[::-1]
            acc.setdefault(name, []).append((roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(nn_))], np.r_[a, nn_]), (a >= b).sum(), (nn_ >= b).sum(),
                                             (a > nn_[0]).sum(), (a > nn_[2]).sum(), (a > nn_[4]).sum()))
    for name, v in acc.items():
        v = np.array(v, float).mean(0)
        print(f"{L:>7}{name:>22}{v[0]:>8.3f} | {'':<19}{v[1]:>7.1f}{v[2]:>8.1f} | {'':<19}{v[3]:>10.1f}{v[4]:>9.1f}{v[5]:>9.1f}", flush=True)
