"""Can models trained only on the ORIGINAL (old-camera) dataset detect attacks in the NEW lab
recordings, if the original dataset is fitted into the 3D body model and re-filmed by the lab
camera from all directions?

Fits every original window alone into the body (own camera, the saved shared), caches the fit,
re-renders each kept window through the shared lab camera turned 0..315 degrees in 45 degree steps,
converts back to pixel windows, and trains the poc_relabel3.py three-model set on the raw original
windows, the refilmed yaw-0 windows and the refilmed 8-direction windows. Tests on take 002 segments
and Nov 5 non-held-out windows; no lab data is used for training.

Usage: python results/samples/poc/poc_rotate3.py <REPO> [--seeds 1,2,3]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--actions", action="store_true", help="same-action benchmark: classify lab cut-down / stab / thrust, no normal data in the score")
ap.add_argument("--selftest", action="store_true", help="train, then score the very windows trained on: checks the training code itself")
ap.add_argument("--seeds", default="1,2,3")
args = ap.parse_args()

REPO = args.repo
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
os.chdir(REPO)

import train_stage2_sim as T  # noqa: E402
import generate_from_samples as S  # noqa: E402
import generate_from_edits as GE  # noqa: E402
from src.distillation_losses import RelationalKDLoss  # noqa: E402
from src.skeleton_sim import body_model as B  # noqa: E402
from src.skeleton_sim import motion_edit as E  # noqa: E402

dev = torch.device(os.environ.get("HTD_DEV", "cuda:0"))
STEPS, BS = 700, 48
WINDOW, STRIDE = 50, 10
seeds = [int(x) for x in args.seeds.split(",")]
NC = len(T.CLASSES)
OUT = os.path.join(REPO, "results", "train_sim")
BM = os.path.join(REPO, "results", "samples", "body_model")
CACHE = os.path.join(BM, "fit_original_windows.npz")

COMBOS = {
    "RMD blend": [("t3", "rmd"), ("t1", "rmd"), ("t2", "rmd")],
    "output blend": [("t3", "out"), ("t1", "out"), ("t2", "out")],
}


# ---------------------------------------------------------------- original data
def load_original():
    z = np.load(os.path.join(OUT, "windows.npz"))
    src, grp, cls = z["source"], z["group"], z["cls"]
    ood = np.char.startswith(grp, "real_normal/Validate_OOD")
    fd = np.char.startswith(grp, "real_normal/False_Detected")
    mask = ((src == "real") & (cls > 0)) | (src == "validate") | \
        ((src == "real_normal") & (cls == 0) & (ood | fd))
    X, y = z["X"][mask].astype(np.float32), cls[mask].astype(np.int64)
    bodies, roots, torsos = [], [], []
    for x in X:
        b, r, t = E.to_body(x)
        bodies.append(b.astype(np.float32))
        roots.append(r.astype(np.float32))
        torsos.append(float(t))
    return X, y, bodies, roots, torsos


# ---------------------------------------------------------------- fitting
def load_shared():
    z = np.load(os.path.join(BM, "fit_train_s0.05_t13_b0.1.npz"))
    return dict(log_scale=torch.tensor(z["log_scale"], device=dev),
                face=torch.tensor(z["face"], device=dev),
                tilt=torch.tensor(float(z["tilt"]), device=dev),
                log_dist=torch.tensor(float(np.log(z["dist"])), device=dev))


def fit_all(bodies):
    shared = load_shared()
    body = B.Body(os.path.join(BM, "Rajagopal2016.osim"))
    if os.path.exists(CACHE):
        cz = np.load(CACHE)
        n = len(cz["logk"])
        assert n == len(bodies), "fit cache is from another window selection: delete it"
        return body, shared, cz["u"].reshape(n, WINDOW, -1), cz["yaw"].reshape(n, WINDOW), cz["lean"].reshape(n, WINDOW, 2), cz["logk"], cz["err"]
    parts = []
    for c0 in range(0, len(bodies), 300):
        chunk = bodies[c0:c0 + 300]
        f, err, _ = B.fit_takes(body, chunk,
                                [np.ones((WINDOW, 17), np.float32)] * len(chunk),
                                [0.0] * len(chunk),
                                shared=shared, device=dev, own_camera=True)
        parts.append((f, err))
    us = np.concatenate([p[0].u.detach().cpu().numpy() for p in parts])
    yaw = np.concatenate([p[0].yaw.detach().cpu().numpy() for p in parts])
    lean = np.concatenate([p[0].lean.detach().cpu().numpy() for p in parts])
    logk = np.concatenate([p[0].logk.detach().cpu().numpy() for p in parts])
    err = np.concatenate([p[1] for p in parts])
    os.makedirs(BM, exist_ok=True)
    np.savez(CACHE, u=us, yaw=yaw, lean=lean, logk=logk, err=err)
    n = len(logk)
    return body, shared, us.reshape(n, WINDOW, -1), yaw.reshape(n, WINDOW), lean.reshape(n, WINDOW, 2), logk, err


def to_pixels(body, root, torso):
    # E.to_body has no stored inverse: body coords are (pixel - hip) / torso length, so pixels are
    # body * torso + hip using the recorded hip path and torso length. If those were not recorded,
    # fall back to 120 pixels per torso length centred at (320, 240).
    return body * float(torso) + root[:, None, :]


def render_windows(body, shared, us, yaw, lean, logk, err, roots, torsos, labels, keep, turns):
    out, out_y = [], []
    for i in range(len(us)):
        if not keep[i]:
            continue
        f = B.Fit(body, np.array([WINDOW]), np.zeros(1), shared, dev)
        with torch.no_grad():
            f.u.copy_(torch.from_numpy(us[i].astype(np.float32)))
            f.yaw.copy_(torch.from_numpy(yaw[i].astype(np.float32)))
            f.lean.copy_(torch.from_numpy(lean[i].astype(np.float32)))
            f.logk.fill_(float(logk[i]))
        f.cam = None  # the shared lab camera (tilt 13 deg) is used
        for turn in turns:
            with torch.no_grad():
                img = f.render(body, yaw_turn=float(turn)).cpu().numpy()  # (50, 17, 2) torso lengths
            out.append(T.live_prep(to_pixels(img, roots[i], torsos[i])))
            out_y.append(labels[i])
    return np.stack(out).astype(np.float32), np.array(out_y, np.int64)


# ---------------------------------------------------------------- test data
def build_segs002():
    segs002 = []
    seg_i = 0
    npz = {}
    with open(os.path.join(T.LAB, "segments", "segments.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["take"] not in npz:
                z = np.load(os.path.join(T.LAB, "analysis", r["take"] + "_analysis.npz"))
                npz[r["take"]] = z["kpts"][..., :2]
            K = npz[r["take"]]
            f0 = int(round(float(r["start_s"]) * 30)); f1 = int(round(float(r["end_s"]) * 30))
            if f1 - f0 < WINDOW:
                pad = WINDOW - (f1 - f0)
                f0 = max(0, f0 - pad // 2 - pad % 2); f1 = min(len(K), f0 + WINDOW); f0 = f1 - WINDOW
            starts = list(range(f0, f1 - WINDOW + 1, STRIDE))
            ws = [T.live_prep(K[s:s + WINDOW]) for s in starts]
            if not ws:
                continue
            if int(r["id"]) >= 4:
                segs002.append((r["label"], r["category"], seg_i, starts, np.stack(ws), f0, f1))
                seg_i += 1
    all_indices = list(range(len(segs002)))
    labels_all = np.array([1 if s[1] == "attack" else 0 for s in segs002])
    return segs002, all_indices, labels_all


def build_nov5():
    takes = [GE.prepare(t) for t in S.load_takes()]
    tr, held, hg = [], [], []
    for t in takes:
        for s in range(0, len(t["xy"]) - WINDOW + 1, STRIDE):
            if t["missing"][s:s + WINDOW].sum() > 10:
                continue
            e = s + WINDOW
            w = T.live_prep(t["xy"][s:e])
            if t["num"] in S.HELD_OUT:
                ov = max(((min(e, b) - max(s, a), i) for i, (a, _, b) in enumerate(t["strikes"])), default=(0, -1))
                if ov[0] >= 15:
                    held.append(w); hg.append((t["num"], ov[1]))
            elif any(s <= a and land < e for a, land, _ in t["strikes"]):
                tr.append((w, t["cls"]))
            elif not any(min(e, b + 5) - max(s, a - 5) > 0 for a, _, b in t["any_trips"]):
                tr.append((w, 0))
    Xn = np.stack([w for w, _ in tr]).astype(np.float32)
    Yn = np.array([c for _, c in tr], np.int64)
    held = np.stack(held).astype(np.float32)
    return Xn, Yn, held, hg


# ---------------------------------------------------------------- three-model recipe (from poc_relabel3.py)
def fit(X, Y, kind, seed, teacher=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, head = T.build(kind).to(dev), nn.Linear(256, NC).to(dev)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.08)
    rkd = RelationalKDLoss(w_dist=1.0, w_angle=2.0).to(dev) if teacher is not None else None
    pools = [np.flatnonzero(Y == c) for c in range(NC)]
    pools = [pools[0]] + [p for p in pools[1:] if len(p)]
    per = [BS // 2] + [BS // 2 // (len(pools) - 1)] * (len(pools) - 1)
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
    m.head = head.eval()  # kept for the same-action benchmark
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
    f0 = f[y == 0]; mu_0 = f0.mean(0, keepdim=True); d0 = f0 - mu_0  # background = not-attack windows only
    eye = torch.eye(f.shape[1]); mus, inv = [], []
    for c in range(1, NC):
        fc = f[y == c]
        if len(fc) < 2:
            continue
        mus.append(fc.mean(0)); dc = fc - mus[-1]
        inv.append(torch.linalg.pinv(dc.T @ dc / (len(fc) - 1) + eps * eye))
    return dict(mu_0=mu_0, mu_c=torch.stack(mus), inv_Sigma_0=torch.linalg.pinv(d0.T @ d0 / (len(f0) - 1) + eps * eye), inv_Sigma_c=torch.stack(inv))


def per_strike(s):
    by = {}
    for g, q in zip(hg, s):
        by[g] = max(by.get(g, -1e18), q)
    return np.array(list(by.values()))


def model_set(X, Y, seed):
    t3 = fit(X, Y, "t3", seed)
    t1 = fit(X, Y, "t1", seed)
    t2 = fit(X, Y, "t2", seed, teacher=t3)
    return t3, t1, t2


def model_scores(models, X, Y):
    """Z[(kind, typ)] = (held scores shifted by that model's own bound,
    per-segment window scores shifted by that model's own bound) over all segs002."""
    t3, t1, t2 = models
    Z = {}
    for key, kind, m in (("t3", "t3", t3), ("t1", "t1", t1), ("t2", "t2", t2)):
        ref = reference(run(m, kind, X)[1], torch.from_numpy(Y))
        sc = (lambda W: T.rmd(run(m, kind, W)[1], ref).numpy())
        out = (lambda W: run(m, kind, W)[0].numpy())
        for typ, fn in (("rmd", sc), ("out", out)):
            h = fn(held); b = np.quantile(per_strike(h), 0.10)
            Z[(key, typ)] = (h - b, [fn(w) - b for _, _, _, _, w, _, _ in segs002])
    return Z


def blend(Z, keys, indices):
    sd = [Z[k][0].std() for k in keys]
    comb_held = np.mean([Z[k][0] / s for k, s in zip(keys, sd)], 0)
    b = np.quantile(per_strike(comb_held), 0.10)  # the combination gets its own attack-only bound
    seg_max = [float(np.mean([Z[k][1][i] / s for k, s in zip(keys, sd)], 0).max()) for i in indices]
    return b, seg_max


def best_ops(scores, labels):
    """(attacks caught with 0 normals alarmed, attacks caught with <=2 normals alarmed)
    using the best threshold on the pooled scores."""
    att = scores[labels == 1]; nor = scores[labels == 0]
    cands = [float(n + 1e-9) for n in nor]
    if len(nor):
        cands.append(float(nor.max() + 1e-6))
    out = []
    for k in (0, 2):
        best = 0
        for thr in cands:
            if int((nor >= thr).sum()) <= k:
                best = max(best, int((att >= thr).sum()))
        out.append(best)
    return out


def nov5_auc(models, Xtr, Ytr, keys, Z):
    """Blended score over Nov 5 non-held-out windows, labelled attack (cls > 0) vs not-attack."""
    t3, t1, t2 = models
    raw = {}
    for key, kind, m in (("t3", "t3", t3), ("t1", "t1", t1), ("t2", "t2", t2)):
        ref = reference(run(m, kind, Xtr)[1], torch.from_numpy(Ytr))
        l, f = run(m, kind, Xn)
        raw[(key, "rmd")] = T.rmd(f, ref).numpy()
        raw[(key, "out")] = l.float().reshape(-1).cpu().numpy()
    sd = [Z[k][0].std() for k in keys]
    comb = np.mean([raw[k] / s for k, s in zip(keys, sd)], 0)
    return roc_auc_score(Yn > 0, comb)


# ---------------------------------------------------------------- run
Xo, yo, bodies, roots, torsos = load_original()
body, shared, us, yaw, lean, logk, err = fit_all(bodies)
med = float(np.median(err))
limit = float(np.percentile(err, 95))
keep = err <= limit
print(f"original windows: {len(err)}; median fit error {med:.3f}; 95th pct {limit:.3f}; "
      f"{int(keep.sum())} kept for refilming", flush=True)
turns = np.deg2rad(np.arange(8) * 45.0)
X0, y0 = render_windows(body, shared, us, yaw, lean, logk, err, roots, torsos, yo, keep, np.array([0.0]))
X8, y8 = render_windows(body, shared, us, yaw, lean, logk, err, roots, torsos, yo, keep, turns)

sets = {"original": (Xo, yo), "refilmed yaw 0": (X0, y0), "refilmed 8 directions": (X8, y8)}
for name, (Xtr, Ytr) in sets.items():
    print(f"train {name}: {np.bincount(Ytr, minlength=NC).tolist()} windows per class (class 0 = normal)", flush=True)

segs002, all_indices, labels_all = build_segs002()
n_att = int((labels_all == 1).sum())
n_nor = int((labels_all == 0).sum())
print(f"take 002 segments: {n_att} attack, {n_nor} normal (of {len(segs002)})", flush=True)
Xn, Yn, held, hg = build_nov5()
print(f"nov5 non-held-out test windows: {len(Xn)} ({int((Yn > 0).sum())} attack, {int((Yn == 0).sum())} normal); "
      f"held-out nov5 strikes: {len(held)}", flush=True)

if args.selftest:
    from sklearn.metrics import f1_score
    tests = dict(sets); tests["lab (Nov 5 training takes)"] = (Xn, Yn)
    for name, (Xtr, Ytr) in tests.items():
        for seed in seeds:
            models = model_set(Xtr, Ytr, seed)
            L, P = [], []
            with torch.no_grad():
                for kind, m in zip(("t3", "t1", "t2"), models):
                    l, f = run(m, kind, Xtr)
                    L.append(l.numpy()); P.append(torch.softmax(m.head(f.to(dev)), 1).cpu().numpy())
            pred = np.mean(L, 0) >= 0  # the models' own 0.5 point, nothing tuned
            y = Ytr > 0
            each = " ".join(f"{f1_score(y, l >= 0):.3f}" for l in L)
            print(f"seed {seed} {name}: own-data F1 {f1_score(y, pred):.4f} (ST-GCN, PoseConv3D, distilled: {each})  attacks caught {int((pred & y).sum())}/{int(y.sum())}  "
                  f"normal alarmed {int((pred & ~y).sum())}/{int((~y).sum())}  AUC {roc_auc_score(y, np.mean(L, 0)):.4f}  class accuracy {float((np.mean(P, 0).argmax(1) == Ytr).mean()):.3f}", flush=True)
    sys.exit(0)

if args.actions:
    # Same-action benchmark: the three trained classes (1 thrust, 2 stab, 3 cut-down) on the lab camera, attacks only.
    def cls3(label):
        L = label.lower()
        if "kick" in L or "combo" in L or "then" in L or "_and_" in L: return 0
        if "thrust" in L or "lunge" in L: return 1
        if "stab" in L: return 2
        if "cut_down" in L or "overhand" in L: return 3
        return 0  # slash and others are not one of the three classes
    n5 = []  # (window, class, facing, take) for every Nov 5 take: no lab data is trained here, so held-out takes count too
    for t in [GE.prepare(t) for t in S.load_takes()]:
        if t["cls"] not in (1, 2, 3):
            continue
        for s_ in range(0, len(t["xy"]) - WINDOW + 1, STRIDE):
            if t["missing"][s_:s_ + WINDOW].sum() <= 10 and any(s_ <= a and land < s_ + WINDOW for a, land, _ in t["strikes"]):
                n5.append((T.live_prep(t["xy"][s_:s_ + WINDOW]), t["cls"], str(t.get("dir") or "front/other"), t["num"]))
    W5 = np.stack([w for w, *_ in n5]).astype(np.float32); C5 = np.array([c for _, c, _, _ in n5]); D5 = np.array([d for *_, d, _ in n5])
    s2 = [(s[0], cls3(s[0]), s[4]) for s in segs002 if s[1] == "attack" and cls3(s[0])]
    print(f"Nov 5 strike windows by class (thrust, stab, cut-down): {[int((C5 == c).sum()) for c in (1, 2, 3)]}, facings {sorted(set(D5))}")
    print(f"take 002 attack segments by class: {[sum(c == k for _, c, _ in s2) for k in (1, 2, 3)]} (slash, kick and combos left out)", flush=True)

    @torch.no_grad()
    def probs(models, W):
        """Mean over the three models of: class shares among thrust/stab/cut-down, and the attack output (sigmoid)."""
        P, A = [], []
        for kind, m in zip(("t3", "t1", "t2"), models):
            l, f = run(m, kind, W)
            P.append(torch.softmax(m.head(f.to(dev))[:, 1:4], 1).cpu().numpy()); A.append(torch.sigmoid(l).numpy())
        return np.mean(P, 0), np.mean(A, 0)

    NAMES = ("thrust", "stab", "cut-down")
    res = {name: [] for name in sets}
    conf = {name: np.zeros((3, 3)) for name in sets}
    for name, (Xtr, Ytr) in sets.items():
        for seed in seeds:
            models = model_set(Xtr, Ytr, seed)
            P, A = probs(models, W5); pred = P.argmax(1) + 1
            r = [float((pred == C5).mean())] + [float((pred[C5 == c] == c).mean()) for c in (1, 2, 3)] + [float((A >= 0.5).mean())]
            r += [float((pred[D5 == d] == C5[D5 == d]).mean()) for d in sorted(set(D5))]
            for c, q in zip(C5, pred): conf[name][c - 1, q - 1] += 1
            ps = [(c, probs(models, w)) for _, c, w in s2]
            r += [float(np.mean([pa[0].mean(0).argmax() + 1 == c for c, pa in ps])), float(np.mean([pa[1].max() >= 0.5 for _, pa in ps]))]
            res[name].append(r)
            print(f"seed {seed} {name}: Nov 5 class accuracy {r[0]:.3f} (thrust {r[1]:.3f} stab {r[2]:.3f} cut-down {r[3]:.3f}), seen as attack {r[4]:.3f} | take 002 class accuracy {r[-2]:.3f}, seen as attack {r[-1]:.3f}", flush=True)
    print(f"\nmeans over {len(seeds)} seeds (chance for class accuracy = 0.333):")
    for name in sets:
        v = np.mean(res[name], 0)
        print(f"{name:<22} Nov 5: accuracy {v[0]:.3f}  thrust {v[1]:.3f}  stab {v[2]:.3f}  cut-down {v[3]:.3f}  seen as attack {v[4]:.3f} | by facing " +
              "  ".join(f"{d} {x:.3f}" for d, x in zip(sorted(set(D5)), v[5:-2])) + f" | take 002: accuracy {v[-2]:.3f}  seen as attack {v[-1]:.3f}")
        print("   Nov 5 true class (rows) against predicted (columns), share of row: " + "; ".join(
            f"{NAMES[i]} -> " + " ".join(f"{x:.2f}" for x in conf[name][i] / max(conf[name][i].sum(), 1)) for i in range(3)))
    sys.exit(0)

rows = {(name, c): [] for name in sets for c in COMBOS}
for name, (Xtr, Ytr) in sets.items():
    for seed in seeds:
        models = model_set(Xtr, Ytr, seed)
        Z = model_scores(models, Xtr, Ytr)
        line = []
        for cname, keys in COMBOS.items():
            b, seg_max = blend(Z, keys, all_indices)
            scores = np.array(seg_max)
            auc = roc_auc_score(labels_all, scores)
            attacks = int((scores[labels_all == 1] >= b).sum())
            normals = int((scores[labels_all == 0] >= b).sum())
            at0, at2 = best_ops(scores, labels_all)
            n5 = nov5_auc(models, Xtr, Ytr, keys, Z)
            line.append(f"{cname.split()[0]} t002 auc {auc:.3f} att {attacks}/{n_att} fa {normals}/{n_nor} "
                        f"att@0 {at0} att@<=2 {at2} | nov5 auc {n5:.3f}")
            rows[(name, cname)].append((auc, attacks, normals, at0, at2, n5))
        print(f"seed {seed} {name}: " + " | ".join(line), flush=True)

print(f"\nmeans over {len(seeds)} seeds:", flush=True)
for name, (Xtr, Ytr) in sets.items():
    for cname in COMBOS:
        v = np.array(rows[(name, cname)], float)
        print(f"{name:<20} {cname:<12} t002: AUC {v[:, 0].mean():.3f} ({v[:, 0].std():.3f}) "
              f"att {v[:, 1].mean():.1f}/{n_att} fa {v[:, 2].mean():.1f}/{n_nor} "
              f"att@0 {v[:, 3].mean():.1f} att@<=2 {v[:, 4].mean():.1f} | nov5 AUC {v[:, 5].mean():.3f} ({v[:, 5].std():.3f})", flush=True)
