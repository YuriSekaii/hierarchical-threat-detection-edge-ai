"""Proof of concept on the lab camera: one plain ST-GCN, real windows only, its own output, threshold 0.5.
No simulator, no cascade, no RMD, no tuned bound. Nothing is saved to weights/.

Arms
  no_oct31 : train = Nov 5 training takes (strikes = attack, pauses + walk-in = normal) + lab take 001 + live normal clips.
             test  = all take 002 segments + Nov 5 held-out takes.
  fold A/B : the same, plus half of the take 002 segments (even / odd id). test = the other half + Nov 5 held-out.
             Take 002 is otherwise a test take; using half of it here is only to answer "does it work once the
             model has seen this camera's normal activity".
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

# ---- data
takes = [GE.prepare(t) for t in S.load_takes()]
nov = {"train": [], "held": []}
for t in takes:
    part = "held" if t["num"] in S.HELD_OUT else "train"
    for s, i in S.label_windows(GE.relabelled(t)):
        nov[part].append((T.live_prep(t["xy"][s:s + 50]), int(i >= 0), f"n{t['num']}_{i}" if i >= 0 else f"n{t['num']}_pause"))
segs, lab_takes = T.lab_windows()
t001 = [T.live_prep(lab_takes[k][s:s + 50]) for k in lab_takes if k.startswith("001") for s in range(0, len(lab_takes[k]) - 49, 10)]
excl = {"s1031": (28, 34), "s110035": (29, 35)}  # the attack tests inside these live sessions, as the training script excludes them
live = [w for sess, clip, w in T.live_clips(LIVE) if sess in excl and not excl[sess][0] <= int(clip.split("_")[1]) <= excl[sess][1]]
seg002 = [(name, cat, ws) for name, cat, ws in segs if int(name[:3]) >= 4 and ws]
print(f"Nov 5 train: {sum(y for _, y, _ in nov['train'])} attack / {sum(1 - y for _, y, _ in nov['train'])} pause windows;"
      f" held-out: {sum(y for _, y, _ in nov['held'])} / {sum(1 - y for _, y, _ in nov['held'])};"
      f" take 001: {len(t001)}; live normal clips: {len(live)};"
      f" take 002 segments: {sum(c == 'attack' for _, c, _ in seg002)} attack, {sum(c != 'attack' for _, c, _ in seg002)} normal")


def fit(X, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    m = T.build("t3").to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
    X = torch.from_numpy(np.stack(X)).float(); y = torch.tensor(y).float()
    w = torch.where(y > 0, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # balance the classes
    steps = 40 * max(1, len(X) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    m.train()
    for _ in range(steps):
        idx = torch.multinomial(w, 64, replacement=True)
        n, v = T.normalize_batch(X[idx].to(dev))
        n, v = T.augment(n, v, gen)
        out = m(T.coords_input(n))
        out = out[0] if isinstance(out, tuple) else out
        loss = F.binary_cross_entropy_with_logits(out.reshape(-1), y[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def score(m, X):
    out = []
    for i in range(0, len(X), 256):
        n, _ = T.normalize_batch(torch.from_numpy(np.stack(X[i:i + 256])).float().to(dev))
        o = m(T.coords_input(n)); o = o[0] if isinstance(o, tuple) else o
        out.append(torch.sigmoid(o.reshape(-1)).cpu().numpy())
    return np.concatenate(out)


def run(arm, seed):
    base = nov["train"]
    X = [w for w, _, _ in base] + t001 + live
    y = [l for _, l, _ in base] + [0] * (len(t001) + len(live))
    folds = [None] if arm == "no_oct31" else [0, 1]
    res = dict(att=[], norm_seg=[], norm_win=[], held=[], held_pause=[])
    for f in folds:
        tr = [] if f is None else [s for s in seg002 if int(s[0][:3]) % 2 == f]
        te = seg002 if f is None else [s for s in seg002 if int(s[0][:3]) % 2 != f]
        m = fit(X + [w for _, _, ws in tr for w in ws], y + [int(c == "attack") for _, c, ws in tr for _ in ws], seed)
        for name, cat, ws in te:
            p = score(m, ws)
            (res["att"] if cat == "attack" else res["norm_seg"]).append((name, float(p.max())))
            if cat != "attack":
                res["norm_win"] += p.tolist()
        if f in (None, 0):  # held-out Nov 5 takes: per strike, best window
            p = score(m, [w for w, _, _ in nov["held"]])
            by = {}
            for (_, l, g), q in zip(nov["held"], p):
                if l:
                    by[g] = max(by.get(g, 0), q)
                else:
                    res["held_pause"].append(q)
            res["held"] = list(by.values())
    return res


print(f"\n{'arm':<10}{'seed':>5}{'take002 AUC':>12}{'attacks @0.5':>14}{'normal seg @0.5':>17}{'normal win @0.5':>17}{'Nov5 held strikes':>19}{'held pauses @0.5':>18}")
keep = {}
for arm in ("no_oct31", "folds"):
    for seed in (1, 2, 3):
        r = run(arm, seed)
        a = np.array([p for _, p in r["att"]]); n = np.array([p for _, p in r["norm_seg"]]); nw = np.array(r["norm_win"])
        h = np.array(r["held"]); hp = np.array(r["held_pause"])
        auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(n))], np.r_[a, n])
        print(f"{arm:<10}{seed:>5}{auc:>12.3f}{f'{(a >= .5).sum()}/{len(a)}':>14}{f'{(n >= .5).sum()}/{len(n)}':>17}"
              f"{f'{(nw >= .5).sum()}/{len(nw)}':>17}{f'{(h >= .5).sum()}/{len(h)}':>19}{f'{(hp >= .5).sum()}/{len(hp)}':>18}", flush=True)
        keep[(arm, seed)] = r
r = keep[("folds", 1)]
print("\nfolds, seed 1 - missed attacks:", [n for n, p in r["att"] if p < .5])
print("folds, seed 1 - false-alarm normal segments:", [(n, round(p, 2)) for n, p in r["norm_seg"] if p >= .5])
r = keep[("no_oct31", 1)]
print("no_oct31, seed 1 - false-alarm normal segments:", [(n, round(p, 2)) for n, p in r["norm_seg"] if p >= .5])
