"""Does the knife box help Stage 2 tell a strike from a stretch?  Same small temporal CNN, same windows, two inputs:
  body       : 17 joints, body-normalized (34 numbers per frame)
  body+knife : the same, plus the Stage 1 knife box: centre relative to the body, centre relative to the nearer wrist,
               box width and height, and a "knife seen" flag (7 numbers per frame; zeros when not seen)
Labels: precise ("trim+"): attack = window holds the whole forward hand movement; not = no hand trip at all.
Train: Nov 5 training takes + lab take 001.  Bound: keeps 90 % of held-out Nov 5 strikes.  Test: take 002 segments.
"""
import csv, os, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import train_stage2_sim as T
import generate_from_samples as S
import generate_from_edits as GE

dev = torch.device("cuda:0")
FLIP = T.FLIP


def fill(kn, gap=10):
    """Knife boxes with gaps of up to `gap` frames filled linearly; column 4 becomes seen (1) / not seen (0)."""
    kn = kn.astype(np.float32).copy()
    seen = kn[:, 4] > 0
    idx = np.flatnonzero(seen)
    out = np.zeros((len(kn), 5), np.float32)
    if len(idx) == 0:
        return out
    for c in range(4):
        out[:, c] = np.interp(np.arange(len(kn)), idx, kn[idx, c])
    near = np.abs(np.arange(len(kn))[:, None] - idx[None]).min(1) if len(idx) < 4000 else None
    if near is None:
        d = np.full(len(kn), 10 ** 6); last = -10 ** 6
        for i in range(len(kn)):
            last = i if seen[i] else last; d[i] = i - last
        nxt = 10 ** 6
        for i in range(len(kn) - 1, -1, -1):
            nxt = i if seen[i] else nxt; d[i] = min(d[i], nxt - i)
        near = d
    out[:, 4] = near <= gap // 2
    out[out[:, 4] == 0, :4] = 0
    return out


def feats(K, kn):
    """K (50,17,2) smoothed pixels, kn (50,5) filled boxes -> (50, 41)."""
    c = K.mean(1, keepdims=True)
    torso = np.linalg.norm(K[:, [5, 6]].mean(1) - K[:, [11, 12]].mean(1), axis=-1)
    s = max(float(np.median(torso)), 10.0)
    n = (K - c) / s
    kc = np.stack([(kn[:, 0] + kn[:, 2]) / 2, (kn[:, 1] + kn[:, 3]) / 2], -1)
    rel = (kc - c[:, 0]) / s
    dw = (kc[:, None] - K[:, [9, 10]]) / s
    near = dw[np.arange(50), np.linalg.norm(dw, axis=-1).argmin(1)]
    size = np.stack([kn[:, 2] - kn[:, 0], kn[:, 3] - kn[:, 1]], -1) / s
    seen = kn[:, 4:5]
    return np.concatenate([n.reshape(50, 34), rel * seen, near * seen, size * seen, seen], 1).astype(np.float32)


def mirror(x):  # (B,50,41): flip x, swap left/right joints, flip knife x offsets
    j = x[:, :, :34].reshape(len(x), 50, 17, 2)[:, :, FLIP] * torch.tensor([-1.0, 1.0], device=x.device)
    k = x[:, :, 34:].clone(); k[:, :, [0, 2]] *= -1
    return torch.cat([j.reshape(len(x), 50, 34), k], 2)


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(d, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                               nn.Conv1d(64, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2),
                               nn.Conv1d(96, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.f = nn.Linear(128, 1)

    def forward(self, x):
        return self.f(self.c(x.transpose(1, 2)).amax(2)).reshape(-1)


def fit(X, y, cols, seed):
    torch.manual_seed(seed)
    m = Net(len(cols)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    Xt, yt = torch.from_numpy(X).to(dev), torch.from_numpy(y).float().to(dev)
    w = torch.where(yt > 0, 0.5 / yt.mean(), 0.5 / (1 - yt.mean())).double()
    steps = 60 * (len(X) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    m.train()
    for _ in range(steps):
        idx = torch.multinomial(w, 64, replacement=True)
        x = Xt[idx]
        fl = torch.rand(64, device=dev) < 0.5
        x = torch.where(fl.view(-1, 1, 1), mirror(x), x)
        x = x * (1 + (torch.rand(64, 1, 1, device=dev) - 0.5) * 0.2)
        x = x + torch.randn_like(x) * 0.02 * (x != 0)
        loss = F.binary_cross_entropy_with_logits(m(x[:, :, cols]), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def score(m, X, cols):
    return torch.sigmoid(m(torch.from_numpy(np.asarray(X, np.float32)).to(dev)[:, :, cols])).cpu().numpy()


# ---- Nov 5 takes
takes = [GE.prepare(t) for t in S.load_takes()]
tr, ytr, held, hg, share = [], [], [], [], []
for t in takes:
    kn = fill(np.load(os.path.join(REPO, "results", "samples", "knife", t["name"] + ".npy"))[t["offset"]:t["offset"] + len(t["xy"])])
    share.append(kn[:, 4].mean())
    for s in range(0, len(t["xy"]) - 49, 10):
        if t["missing"][s:s + 50].sum() > 10:
            continue
        e = s + 50
        f = feats(T.live_prep(t["xy"][s:e]), kn[s:e])
        hit = [i for i, (a, land, _) in enumerate(t["strikes"]) if s <= a and land < e]
        if t["num"] in S.HELD_OUT:
            ov = max(((min(e, b) - max(s, a), i) for i, (a, _, b) in enumerate(t["strikes"])), default=(0, -1))
            if ov[0] >= 15:
                held.append(f); hg.append((t["num"], ov[1]))
        elif hit:
            tr.append(f); ytr.append(1)
        elif not any(min(e, b + 5) - max(s, a - 5) > 0 for a, _, b in t["any_trips"]):
            tr.append(f); ytr.append(0)
# ---- Oct 31 lab takes
npz = {}
segs = []
for r in csv.DictReader(open(os.path.join(T.LAB, "segments", "segments.csv"), encoding="utf-8")):
    if r["take"] not in npz:
        z = np.load(os.path.join(T.LAB, "analysis", r["take"] + "_analysis.npz"))
        npz[r["take"]] = (z["kpts"][..., :2], fill(z["knife"]))
    K, kn = npz[r["take"]]
    f0, f1 = int(round(float(r["start_s"]) * 30)), int(round(float(r["end_s"]) * 30))
    if f1 - f0 < 50:
        pad = 50 - (f1 - f0); f0 = max(0, f0 - pad // 2 - pad % 2); f1 = min(len(K), f0 + 50); f0 = f1 - 50
    ws = [feats(T.live_prep(K[s:s + 50]), kn[s:s + 50]) for s in range(f0, f1 - 49, 10)]
    if int(r["id"]) >= 4 and ws:
        segs.append((r["label"], r["category"], np.stack(ws)))
k1 = [k for k in npz if k.startswith("001")][0]
t001 = [feats(T.live_prep(npz[k1][0][s:s + 50]), npz[k1][1][s:s + 50]) for s in range(0, len(npz[k1][0]) - 49, 10)]
X = np.stack(tr + t001); y = np.array(ytr + [0] * len(t001)); held = np.stack(held)
seen = lambda A: float(A[:, :, 40].mean())
dist = lambda A: float(np.median(np.linalg.norm(A[:, :, 36:38], axis=-1)[A[:, :, 40] > 0]))
A2 = np.concatenate([w for _, c, w in segs if c == "attack"]); N2 = np.concatenate([w for _, c, w in segs if c != "attack"])
print(f"train: {int(y.sum())} attack, {int((1 - y).sum())} not-attack ({len(t001)} from take 001)")
print("knife seen (share of frames) / median distance knife-to-nearer-wrist in torso lengths:")
for nm, A in (("Nov 5 attack", X[y == 1]), ("Nov 5 not-attack", X[:len(tr)][np.array(ytr) == 0]), ("take 001", np.stack(t001)),
              ("take 002 attack", A2), ("take 002 normal", N2)):
    print(f"   {nm:<18}{seen(A):>6.0%}   {dist(A):.2f}")

ARMS = {"body": list(range(34)), "body+knife": list(range(41)), "arm+knife only": [2 * j + c for j in (5, 6, 7, 8, 9, 10) for c in (0, 1)] + list(range(34, 41))}
rows, fa, miss = {}, {}, {}
print()
for arm, cols in ARMS.items():
    for seed in range(1, 6):
        m = fit(X, y, cols, seed)
        by = {}
        for g, q in zip(hg, score(m, held, cols)):
            by[g] = max(by.get(g, 0), q)
        b = np.quantile(list(by.values()), 0.10)
        S2 = [(n, c, score(m, w, cols).max()) for n, c, w in segs]
        a = np.array([p for _, c, p in S2 if c == "attack"]); nn_ = np.array([p for _, c, p in S2 if c != "attack"])
        r = (roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(nn_))], np.r_[a, nn_]), (a >= b).sum(), (nn_ >= b).sum(),
             (a > nn_.max()).sum())
        rows.setdefault(arm, []).append(r)
        fa.setdefault(arm, Counter()).update(n for n, c, p in S2 if c != "attack" and p >= b)
        miss.setdefault(arm, Counter()).update(n for n, c, p in S2 if c == "attack" and p < b)
        print(f"{arm:<16} seed {seed}: AUC {r[0]:.3f}  attacks {r[1]}/31  normal segments {r[2]}/24", flush=True)
print("\nmean (sd) of 5 seeds")
for arm, v in rows.items():
    v = np.array(v, float)
    print(f"{arm:<16} AUC {v[:, 0].mean():.3f} ({v[:, 0].std():.3f})   attacks {v[:, 1].mean():.1f}/31   normal segments {v[:, 2].mean():.1f}/24"
          f"   attacks above every normal segment {v[:, 3].mean():.1f}/31")
for arm in rows:
    print(f"\n{arm} - normal alarmed (of 5 seeds): {dict(fa[arm].most_common())}")
    print(f"{arm} - attacks missed (of 5 seeds): {dict(miss[arm].most_common())}")
