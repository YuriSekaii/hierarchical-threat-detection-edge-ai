"""Does precise attack labelling help? Plain ST-GCN on real Nov 5 lab takes, own output, bound from attacks only.

old     : a window is "attack" if it overlaps a strike by >= 15 frames; "not" = standing pauses only
trim    : "attack" only if the window holds the whole forward hand movement (start of the trip to landing)
trim+   : trim, and every window that touches no hand trip at all (walking in, walking between strikes) is "not"
Windows that hold only part of a strike, or only the pull-back, are left out of training.
Bound for every arm: keeps 90 % of the held-out Nov 5 strikes (best window per strike). Test: take 002 segments.
"""
import os, sys
from collections import Counter
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = sys.argv[1]
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import train_stage2_sim as T
import generate_from_samples as S
import generate_from_edits as GE

dev = torch.device("cuda:0")
takes = [GE.prepare(t) for t in S.load_takes()]


def windows(t, mode):
    out = []
    for s in range(0, len(t["xy"]) - 49, 10):
        if t["missing"][s:s + 50].sum() > 10:
            continue
        e = s + 50
        if mode == "old":
            ov = max((min(e, b) - max(s, a) for a, _, b in t["strikes"]), default=0)
            if ov >= 15:
                out.append((s, 1))
            elif ov <= 0 and t["pause"][s:e].mean() >= 0.8:
                out.append((s, 0))
            continue
        if any(s <= a and land < e for a, land, _ in t["strikes"]):
            out.append((s, 1))
        elif not any(min(e, b + 5) - max(s, a - 5) > 0 for a, _, b in t["any_trips"]):
            if mode == "trim+" or t["pause"][s:e].mean() >= 0.8:
                out.append((s, 0))
    return out


segs, lab_takes = T.lab_windows()
t001 = [T.live_prep(lab_takes[k][s:s + 50]) for k in lab_takes if k.startswith("001") for s in range(0, len(lab_takes[k]) - 49, 10)]
excl = {"s1031": (28, 34), "s110035": (29, 35)}
live = [w for sess, clip, w in T.live_clips(os.path.join(REPO, "data", "live_clips_2026-10-31"))
        if sess in excl and not excl[sess][0] <= int(clip.split("_")[1]) <= excl[sess][1]]
seg002 = [(n, c, ws) for n, c, ws in segs if int(n[:3]) >= 4 and ws]
held = [(T.live_prep(t["xy"][s:s + 50]), (t["num"], i)) for t in takes if t["num"] in S.HELD_OUT
        for s, i in S.label_windows(GE.relabelled(t)) if i >= 0]
fwd = [land - a for t in takes for a, land, _ in t["strikes"]]
print(f"forward hand movement lasts {np.median(fwd):.0f} frames (median), {np.percentile(fwd, 90):.0f} at the 90th percentile; window = 50")


def fit(X, y, seed):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    m = T.STGCNModel(num_classes=1).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).float()
    w = torch.where(yt > 0, 0.5 / yt.mean(), 0.5 / (1 - yt.mean())).double()
    steps = 40 * (len(X) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    m.train()
    for _ in range(steps):
        idx = torch.multinomial(w, 64, replacement=True)
        n, v = T.augment(*T.normalize_batch(Xt[idx].to(dev)), gen)
        out, _ = m(T.coords_input(n), return_features=True)
        loss = F.binary_cross_entropy_with_logits(out.reshape(-1), yt[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def score(m, W):
    out = []
    for i in range(0, len(W), 256):
        n, _ = T.normalize_batch(torch.from_numpy(np.stack(W[i:i + 256])).float().to(dev))
        out.append(torch.sigmoid(m(T.coords_input(n), return_features=True)[0].reshape(-1)).cpu().numpy())
    return np.concatenate(out)


rows, fa, miss = {}, {}, {}
for mode in ("old", "trim", "trim+"):
    W, y = [], []
    for t in takes:
        if t["num"] not in S.HELD_OUT:
            for s, l in windows(t, mode):
                W.append(T.live_prep(t["xy"][s:s + 50])); y.append(l)
    X = np.stack(W + t001 + live).astype(np.float32); y = np.array(y + [0] * (len(t001) + len(live)))
    print(f"\n{mode}: {int(y.sum())} attack windows, {int((1 - y).sum())} not-attack ({len(t001) + len(live)} of them from take 001 and live clips)")
    for seed in (1, 2, 3):
        m = fit(X, y, seed)
        by = {}
        for (_, g), q in zip(held, score(m, [w for w, _ in held])):
            by[g] = max(by.get(g, 0), q)
        b = np.quantile(list(by.values()), 0.10)
        S2 = [(n, c, score(m, ws).max()) for n, c, ws in seg002]
        a = np.array([p for _, c, p in S2 if c == "attack"]); nn = np.array([p for _, c, p in S2 if c != "attack"])
        r = (roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(nn))], np.r_[a, nn]), (a >= b).sum(), (nn >= b).sum(), b)
        rows.setdefault(mode, []).append(r)
        fa.setdefault(mode, Counter()).update(n[4:] for n, c, p in S2 if c != "attack" and p >= b)
        miss.setdefault(mode, Counter()).update(n[4:] for n, c, p in S2 if c == "attack" and p < b)
        print(f"  seed {seed}: AUC {r[0]:.3f}  attacks {r[1]}/31  normal segments {r[2]}/24  (bound {b:.3f})", flush=True)
print("\nmean of 3 seeds")
for mode, v in rows.items():
    v = np.array(v, float).mean(0)
    print(f"{mode:<6} AUC {v[0]:.3f}   attacks {v[1]:.1f}/31   normal segments {v[2]:.1f}/24")
for mode in rows:
    print(f"\n{mode} - normal segments alarmed (times in 3 seeds): {dict(fa[mode].most_common())}")
    print(f"{mode} - attacks missed (times in 3 seeds): {dict(miss[mode].most_common())}")
