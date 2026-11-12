"""Window length test: 50 vs 30 vs 20 frames, precise labels, body joints only, same small temporal CNN.
attack = the window holds the whole forward hand movement (start of trip to landing); strikes longer than the window
are labelled by the window that ends at the landing.  not = no hand trip at all.  Stride 10 (and 5 for short windows).
Train: Nov 5 training takes + lab take 001.  Bound: keeps 90 % of held-out Nov 5 strikes.  Test: take 002 segments.
"""
import csv, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import train_stage2_sim as T
import generate_from_samples as S
import generate_from_edits as GE

dev = torch.device("cuda:0")
FLIP = T.FLIP


def feats(K):
    K = T.live_prep(K)
    torso = np.linalg.norm(K[:, [5, 6]].mean(1) - K[:, [11, 12]].mean(1), axis=-1)
    return ((K - K.mean(1, keepdims=True)) / max(float(np.median(torso)), 10.0)).reshape(len(K), 34).astype(np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(34, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                               nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
                               nn.Conv1d(96, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.f = nn.Linear(128, 1)

    def forward(self, x):
        return self.f(self.c(x.transpose(1, 2)).amax(2)).reshape(-1)


def fit(X, y, seed):
    torch.manual_seed(seed)
    m = Net().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).float().to(dev)
    w = torch.where(yt > 0, 0.5 / yt.mean(), 0.5 / (1 - yt.mean())).double()
    steps = 60 * (len(X) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    m.train()
    for _ in range(steps):
        idx = torch.multinomial(w, 64, replacement=True)
        x = Xt[idx]
        j = x.reshape(64, -1, 17, 2)[:, :, FLIP] * torch.tensor([-1.0, 1.0], device=dev)
        x = torch.where((torch.rand(64, device=dev) < 0.5).view(-1, 1, 1), j.reshape(x.shape), x)
        x = x * (1 + (torch.rand(64, 1, 1, device=dev) - 0.5) * 0.2) + torch.randn_like(x) * 0.02
        loss = F.binary_cross_entropy_with_logits(m(x), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def score(m, X):
    return torch.sigmoid(m(torch.from_numpy(np.asarray(X, np.float32)).to(dev))).cpu().numpy()


takes = [GE.prepare(t) for t in S.load_takes()]
lab = {}
rows_csv = list(csv.DictReader(open(os.path.join(T.LAB, "segments", "segments.csv"), encoding="utf-8")))
for r in rows_csv:
    if r["take"] not in lab:
        lab[r["take"]] = np.load(os.path.join(T.LAB, "analysis", r["take"] + "_analysis.npz"))["kpts"][..., :2]
k1 = [k for k in lab if k.startswith("001")][0]

print(f"{'window':>7}{'stride':>7}{'attack win':>11}{'not win':>9}{'AUC (sd)':>15}{'attacks /31':>13}{'normal seg /24':>16}   most alarmed normal movements")
for L, stride in ((50, 10), (30, 10), (30, 5), (20, 5)):
    tr, ytr, held, hg = [], [], [], []
    for t in takes:
        for s in range(0, len(t["xy"]) - L + 1, stride):
            e = s + L
            if t["missing"][s:e].sum() > L // 5:
                continue
            hit = [i for i, (a, land, _) in enumerate(t["strikes"]) if (s <= a or e - 1 - land <= stride) and s <= land < e and e - 1 - land < L // 2 + stride]
            if t["num"] in S.HELD_OUT:
                for i in hit:
                    held.append(feats(t["xy"][s:e])); hg.append((t["num"], i))
            elif hit:
                tr.append(feats(t["xy"][s:e])); ytr.append(1)
            elif not any(min(e, b + 5) - max(s, a - 5) > 0 for a, _, b in t["any_trips"]):
                tr.append(feats(t["xy"][s:e])); ytr.append(0)
    t001 = [feats(lab[k1][s:s + L]) for s in range(0, len(lab[k1]) - L + 1, stride)]
    segs = []
    for r in rows_csv:
        K = lab[r["take"]]
        f0, f1 = int(round(float(r["start_s"]) * 30)), int(round(float(r["end_s"]) * 30))
        if f1 - f0 < 50:  # same widening as before, so every window length scores the same stretch of video
            pad = 50 - (f1 - f0); f0 = max(0, f0 - pad // 2 - pad % 2); f1 = min(len(K), f0 + 50); f0 = f1 - 50
        if int(r["id"]) >= 4:
            segs.append((r["label"], r["category"], np.stack([feats(K[s:s + L]) for s in range(f0, f1 - L + 1, stride)])))
    X = np.stack(tr + t001); y = np.array(ytr + [0] * len(t001)); held = np.stack(held)
    out, fa = [], {}
    for seed in range(1, 6):
        m = fit(X, y, seed)
        by = {}
        for g, q in zip(hg, score(m, held)):
            by[g] = max(by.get(g, 0), q)
        b = np.quantile(list(by.values()), 0.10)
        S2 = [(n, c, score(m, w).max()) for n, c, w in segs]
        a = np.array([p for _, c, p in S2 if c == "attack"]); nn_ = np.array([p for _, c, p in S2 if c != "attack"])
        out.append((roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(nn_))], np.r_[a, nn_]), (a >= b).sum(), (nn_ >= b).sum()))
        for n, c, p in S2:
            if c != "attack" and p >= b:
                fa[n] = fa.get(n, 0) + 1
    v = np.array(out, float)
    top = ", ".join(f"{k} {n}" for k, n in sorted(fa.items(), key=lambda kv: -kv[1])[:5])
    print(f"{L:>7}{stride:>7}{int(y.sum()):>11}{int((1 - y).sum()):>9}{f'{v[:, 0].mean():.3f} ({v[:, 0].std():.3f})':>15}{v[:, 1].mean():>13.1f}{v[:, 2].mean():>16.1f}   {top}", flush=True)
