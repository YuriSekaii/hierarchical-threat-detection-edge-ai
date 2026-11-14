"""Combine the three-model recipe of poc_three_bg.py (ST-GCN, PoseConv3D, PoseConv3D distilled;
same losses, augmentation, STEPS, batch size; RMD reference from not-attack windows only; blended
'output' and blended RMD score) with the 5-fold cross-validation of poc_relabel.py on take 002.
fit() takes the training arrays as arguments (no module globals).
Usage: python results/samples/poc/poc_relabel3.py <REPO> [--extra] [--sim N] [--seeds 1,2,3]
"""
import argparse, csv, glob, os, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--extra", action="store_true")
ap.add_argument("--sim", type=int, default=None)
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--violent", default="", help="comma-separated take 002 normal labels to count as attacks (class swing), e.g. stretch_arms,knife_swing_low")
ap.add_argument("--cache", default="exp_S_gen", help="results/train_sim/<cache>/windows.npz used by --sim; exp_F* caches are body-filtered")
args = ap.parse_args()

REPO = args.repo
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import train_stage2_sim as T
import generate_from_samples as S
import generate_from_edits as GE
from src.distillation_losses import RelationalKDLoss

dev = torch.device("cuda:0")
STEPS, BS = 700, 48
WINDOW, STRIDE = 50, 10
seeds = [int(x) for x in args.seeds.split(",")]
EXTRAS = args.extra
NSIM = args.sim

COMBOS = {
    "RMD blend": [("t3", "rmd"), ("t1", "rmd"), ("t2", "rmd")],
    "output blend": [("t3", "out"), ("t1", "out"), ("t2", "out")],
}


def raw_xml(path):  # raw pixel keypoints, no smoothing (what the live buffer holds)
    import xml.etree.ElementTree as ET
    imgs = ET.parse(path).getroot().findall(".//image")
    K = np.zeros((len(imgs), 17, 2), np.float32)
    for i, im in enumerate(imgs):
        p = im.find("points")
        pts = p.get("points", "").split(";") if p is not None else []
        if len(pts) >= 17:
            K[i] = [[float(v) for v in q.split(",")] for q in pts[:17]]
    return K


def action_class(name):
    L = name.lower()
    if "thrust" in L or "lunge" in L: return 1
    if "stab" in L: return 2
    if "cut" in L or "overhand" in L: return 3
    if "slash" in L: return 4
    return 1


def seg_class(label):
    # take-002 attack-window class mapped from the segment label by keyword into
    # poc_three_bg.py's classes (1 thrust, 2 stab, 3 cut-down, 4 slash):
    # thrust/lunge -> thrust, stab -> stab, cut_down/overhand -> cut-down,
    # slash -> slash, anything else -> thrust (closest existing class)
    L = label.lower()
    if label in args.violent.split(","): return 5  # swing
    if "thrust" in L or "lunge" in L: return 1
    if "stab" in L: return 2
    if "cut_down" in L or "overhand" in L: return 3
    if "slash" in L: return 4
    return 1


# ---------------- Nov 5 base set + held-out Nov 5 strikes (as poc_three_bg.py) ----------------
takes = [GE.prepare(t) for t in S.load_takes()]
tr, ytr, held, hg = [], [], [], []
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
            tr.append(w); ytr.append(t["cls"])
        elif not any(min(e, b + 5) - max(s, a - 5) > 0 for a, _, b in t["any_trips"]):
            tr.append(w); ytr.append(0)

segs, lab_takes = T.lab_windows()
t001 = [T.live_prep(lab_takes[k][s:s + WINDOW]) for k in lab_takes if k.startswith("001")
        for s in range(0, len(lab_takes[k]) - WINDOW + 1, STRIDE)]

# ---------------- take 002 segments with label/category/starts/frame range (as poc_relabel.py) ----------------
npz = {}
segs002 = []  # (label, category, seg_index, starts, windows (N,50,17,2), f0, f1)
seg_i = 0
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
            if r["label"] in args.violent.split(","):  # armed swing-like movement the user wants flagged
                r["category"] = "attack"
            segs002.append((r["label"], r["category"], seg_i, starts, np.stack(ws), f0, f1))
            seg_i += 1

X = np.stack(tr + t001).astype(np.float32)
Y = np.array(ytr + [0] * len(t001), np.int64)
held = np.stack(held).astype(np.float32)
NC = len(T.CLASSES)
print(f"train windows: {np.bincount(Y, minlength=NC).tolist()} (class 0 = not attack, 1-4 = thrust, stab, cut-down, slash)", flush=True)
n_att = sum(1 for s in segs002 if s[1] == "attack")
n_nor = len(segs002) - n_att
print(f"take 002 segments: {n_att} attack, {n_nor} normal (of {len(segs002)})", flush=True)

# ---------------- optional extra data / simulation data (added to every training set) ----------------
extra_X, extra_y = [], []
if EXTRAS:
    da = dn = 0
    for coat in ("With_Coat", "Without_Coat"):
        cdir = os.path.join(REPO, "data", "Validate", coat)
        if not os.path.isdir(cdir):
            continue
        for act in sorted(os.listdir(cdir)):
            d = os.path.join(cdir, act)
            if not os.path.isdir(d):
                continue
            for p in sorted(glob.glob(os.path.join(d, "annotations_yolo26s-pose", "*.xml"))):
                K = raw_xml(p)
                if len(K) < WINDOW:
                    continue
                for s in range(0, len(K) - WINDOW + 1, STRIDE):
                    extra_X.append(T.live_prep(K[s:s + WINDOW])); extra_y.append(action_class(act)); da += 1
    ood = os.path.join(REPO, "data", "Validate", "OOD")
    if os.path.isdir(ood):
        for p in sorted(glob.glob(os.path.join(ood, "annotations_yolo26s-pose", "*.xml"))):
            K = raw_xml(p)
            if len(K) < WINDOW:
                continue
            for s in range(0, len(K) - WINDOW + 1, STRIDE):
                extra_X.append(T.live_prep(K[s:s + WINDOW])); extra_y.append(0); dn += 1
    print(f"--extra: {da} attack, {dn} normal Validate windows added", flush=True)

sim_X, sim_y = [], []
if NSIM is not None:
    # exp_S_gen cache: physics-simulated (sim_*), generated from Nov 5 training takes (gen_*), original real clips
    z = np.load(os.path.join(T.OUT, args.cache, "windows.npz"))
    src, trn = z["source"], z["split"] == "train"
    r0 = np.random.default_rng(0)
    for name, cap in (("sim_attack", NSIM), ("sim_normal", NSIM), ("gen_attack", NSIM), ("gen_normal", NSIM), ("real", None), ("real_normal", None)):
        i = np.flatnonzero((src == name) & trn)
        if cap is not None and len(i) > cap:
            i = np.sort(r0.choice(i, cap, replace=False))
        sim_X += list(z["X"][i]); sim_y += z["cls"][i].tolist()
        print(f"--sim: {name} {len(i)} windows", flush=True)


def full_base():
    parts_x = [X]; parts_y = [Y]
    if EXTRAS and extra_X:
        parts_x.append(np.stack(extra_X)); parts_y.append(np.array(extra_y, np.int64))
    if NSIM is not None and sim_X:
        parts_x.append(np.stack(sim_X)); parts_y.append(np.array(sim_y, np.int64))
    return np.concatenate(parts_x), np.concatenate(parts_y)


# ---------------- refactored fit: training arrays are arguments, not module globals ----------------
def fit(X, Y, kind, seed, teacher=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, head = T.build(kind).to(dev), nn.Linear(256, NC).to(dev)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.08)
    rkd = RelationalKDLoss(w_dist=1.0, w_angle=2.0).to(dev) if teacher is not None else None
    pools = [np.flatnonzero(Y == c) for c in range(NC)]
    pools = [pools[0]] + [p for p in pools[1:] if len(p)]  # skip attack classes with no windows
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


folds = [[s for j, s in enumerate(segs002) if j % 5 == f] for f in range(5)]
all_indices = list(range(len(segs002)))
labels_all = np.array([1 if s[1] == "attack" else 0 for s in segs002])

base_rows = {c: [] for c in COMBOS}
cv_rows = {c: [] for c in COMBOS}
fa_nor, miss_att = Counter(), Counter()

for seed in seeds:
    Xb, Yb = full_base()
    models_b = model_set(Xb, Yb, seed)
    Z_b = model_scores(models_b, Xb, Yb)
    base_res = {}
    for cname, keys in COMBOS.items():
        b, seg_max = blend(Z_b, keys, all_indices)
        scores = np.array(seg_max)
        auc = roc_auc_score(labels_all, scores)
        attacks = int((scores[labels_all == 1] >= b).sum())
        normals = int((scores[labels_all == 0] >= b).sum())
        at0, at2 = best_ops(scores, labels_all)
        base_res[cname] = (auc, attacks, normals, at0, at2)
        base_rows[cname].append((auc, attacks, normals, at0, at2))

    cv_scores = {c: [] for c in COMBOS}
    cv_labels = {c: [] for c in COMBOS}
    cv_bounds = {c: [] for c in COMBOS}
    for f in range(5):
        held_f = folds[f]
        ranges = [(s[5], s[6]) for s in held_f]
        idx_held = [s[2] for s in held_f]
        tr_w, tr_l = [], []
        for j, s in enumerate(segs002):
            if j % 5 == f:
                continue
            for w, st in zip(s[4], s[3]):
                if any(st < f1 and st + WINDOW > f0 for f0, f1 in ranges):
                    continue
                tr_w.append(w); tr_l.append(seg_class(s[0]) if s[1] == "attack" else 0)
        Xf = np.concatenate([Xb, np.stack(tr_w)])
        Yf = np.concatenate([Yb, np.array(tr_l, np.int64)])
        models_f = model_set(Xf, Yf, seed)
        Z_f = model_scores(models_f, Xf, Yf)
        for cname, keys in COMBOS.items():
            b, seg_max = blend(Z_f, keys, idx_held)
            flabels = np.array([1 if segs002[i][1] == "attack" else 0 for i in idx_held])
            scores = np.array(seg_max)
            cv_scores[cname].extend((scores - b).tolist())  # shift by the fold's own bound so folds pool
            cv_labels[cname].extend(flabels.tolist())
            cv_bounds[cname].extend([0.0] * len(idx_held))
            if cname == "output blend":  # Counters at the attack-only bound, fold order == score order
                for i, sc_, lab in zip(idx_held, seg_max, flabels):
                    if lab == 0 and sc_ >= b:
                        fa_nor[segs002[i][0]] += 1
                    if lab == 1 and sc_ < b:
                        miss_att[segs002[i][0]] += 1

    cv_res = {}
    for cname in COMBOS:
        scores = np.array(cv_scores[cname]); labels = np.array(cv_labels[cname]); bounds = np.array(cv_bounds[cname])
        auc = roc_auc_score(labels, scores)
        attacks = int((scores[labels == 1] >= bounds[labels == 1]).sum())
        normals = int((scores[labels == 0] >= bounds[labels == 0]).sum())
        at0, at2 = best_ops(scores, labels)
        cv_res[cname] = (auc, attacks, normals, at0, at2)
        cv_rows[cname].append((auc, attacks, normals, at0, at2))

    line = []
    for cname in COMBOS:
        br = base_res[cname]; cr = cv_res[cname]
        line.append(f"{cname.split()[0]} base auc {br[0]:.3f} att {br[1]}/{n_att} fa {br[2]}/{n_nor} | cv auc {cr[0]:.3f} att {cr[1]}/{n_att} fa {cr[2]}/{n_nor}")
    print(f"seed {seed}: " + " | ".join(line), flush=True)

print(f"\nmeans over {len(seeds)} seeds:")
for cname in COMBOS:
    v = np.array(base_rows[cname], float); cv = np.array(cv_rows[cname], float)
    print(f"{cname:<12} base: AUC {v[:, 0].mean():.3f} ({v[:, 0].std():.3f}) att {v[:, 1].mean():.1f}/{n_att} fa {v[:, 2].mean():.1f}/{n_nor} att@0 {v[:, 3].mean():.1f} att@<=2 {v[:, 4].mean():.1f} | "
          f"cv: AUC {cv[:, 0].mean():.3f} ({cv[:, 0].std():.3f}) att {cv[:, 1].mean():.1f}/{n_att} fa {cv[:, 2].mean():.1f}/{n_nor} "
          f"att@0 {cv[:, 3].mean():.1f} att@<=2 {cv[:, 4].mean():.1f}")
print("normal labels alarmed at bound (CV output blend, 3 seeds):", dict(fa_nor.most_common()))
print("attack labels missed at bound (CV output blend, 3 seeds):", dict(miss_att.most_common()))
