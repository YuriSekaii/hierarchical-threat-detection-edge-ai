"""Train on Nov 5 + Oct 31 take 001 (+ take 002 in cross-validation), test fairly on left-out Oct 31 take 002 segments.
Same small temporal CNN and window code as poc_oct31neg.py; body columns only (cols=range(34)); knife input zeros
for the Oct 31 data.

Nov 5 windows: exactly as poc_oct31neg.py builds tr/ytr/held/hg (held-out Nov 5 strikes keep the attack-only bound).
Oct 31 take 001: all windows normal.  Oct 31 take 002 (segments.csv id>=4): windows labelled by segment category,
each window keeps its start frame and segment index.
CV: 5 folds, segment i (csv order among id>=4) -> fold i % 5.  Each fold trains on Nov 5 + take 001 + take 002 of the
other folds, dropping training windows whose 50-frame span overlaps any held segment's frame range.  Held segments are
scored (max over windows).  3 seeds.
Usage: python results/samples/poc/poc_relabel.py <REPO> [--extra]
"""
import csv, glob, os, sys
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
EXTRAS = len(sys.argv) > 2 and sys.argv[2] == "--extra"


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


# ---------------- Nov 5 takes ----------------
takes = [GE.prepare(t) for t in S.load_takes()]
tr, ytr, held, hg = [], [], [], []
for t in takes:
    kn = fill(np.load(os.path.join(REPO, "results", "samples", "knife", t["name"] + ".npy"))[t["offset"]:t["offset"] + len(t["xy"])])
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

# ---------------- Oct 31 take 001 and take 002 segments ----------------
npz = {}
segs = []  # (label, category, seg_index, starts, windows (N,50,41), f0, f1)
seg_i = 0
with open(os.path.join(T.LAB, "segments", "segments.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["take"] not in npz:
            z = np.load(os.path.join(T.LAB, "analysis", r["take"] + "_analysis.npz"))
            npz[r["take"]] = (z["kpts"][..., :2], fill(z["knife"]))
        K, _ = npz[r["take"]]
        f0 = int(round(float(r["start_s"]) * 30)); f1 = int(round(float(r["end_s"]) * 30))
        if f1 - f0 < 50:
            pad = 50 - (f1 - f0); f0 = max(0, f0 - pad // 2 - pad % 2); f1 = min(len(K), f0 + 50); f0 = f1 - 50
        starts = list(range(f0, f1 - 49, 10))
        ws = [feats(T.live_prep(K[s:s + 50]), np.zeros((50, 5), np.float32)) for s in starts]
        if not ws:
            continue
        if int(r["id"]) >= 4:
            segs.append((r["label"], r["category"], seg_i, starts, np.stack(ws), f0, f1))
            seg_i += 1

k1 = [k for k in npz if k.startswith("001")][0]
t001 = [feats(T.live_prep(npz[k1][0][s:s + 50]), np.zeros((50, 5), np.float32)) for s in range(0, len(npz[k1][0]) - 49, 10)]

X = np.stack(tr + t001); y = np.array(ytr + [0] * len(t001))
held = np.stack(held)
cols = list(range(34))

n_att = sum(1 for s in segs if s[1] == "attack")
n_nor = len(segs) - n_att
print(f"train: {int(y.sum())} attack, {int((1 - y).sum())} not-attack ({len(t001)} from take 001)")
print(f"take 002 segments: {n_att} attack, {n_nor} normal (of {len(segs)})")

def bound(m):
    by = {}
    for g, q in zip(hg, score(m, held, cols)):
        by[g] = max(by.get(g, 0), q)
    return np.quantile(list(by.values()), 0.10)

# ---------------- extra datasets (--extra) ----------------
extra = []  # (window, label)
if EXTRAS:
    for ds in ("Validate",):  # Ground_Truth_Dataset is single frames with knife boxes (Stage 1 data), no motion
        root = os.path.join(REPO, "data", ds)
        da = dn = 0
        for coat in ("With_Coat", "Without_Coat"):
            cdir = os.path.join(root, coat)
            if not os.path.isdir(cdir):
                continue
            for act in sorted(os.listdir(cdir)):
                d = os.path.join(cdir, act)
                if not os.path.isdir(d):
                    continue
                for p in sorted(glob.glob(os.path.join(d, "annotations_yolo26s-pose", "*.xml"))):
                    K = raw_xml(p)
                    if len(K) < 50:
                        continue
                    for s in range(0, len(K) - 49, 10):
                        extra.append((feats(T.live_prep(K[s:s + 50]), np.zeros((50, 5), np.float32)), 1))
                        da += 1
        ood = os.path.join(root, "OOD")
        if os.path.isdir(ood):
            for p in sorted(glob.glob(os.path.join(ood, "annotations_yolo26s-pose", "*.xml"))):
                K = raw_xml(p)
                if len(K) < 50:
                    continue
                for s in range(0, len(K) - 49, 10):
                    extra.append((feats(T.live_prep(K[s:s + 50]), np.zeros((50, 5), np.float32)), 0))
                    dn += 1
        print(f"{ds}: {da} attack, {dn} normal windows", flush=True)

extra_X = np.stack([e[0] for e in extra]) if extra else np.zeros((0, 50, 41), np.float32)
extra_y = np.array([e[1] for e in extra], np.float32)

folds = [[s for j, s in enumerate(segs) if j % 5 == f] for f in range(5)]

res = {k: [] for k in ("base_att_b", "base_nor_b", "base_att_05", "base_nor_05",
                       "cv_att_b", "cv_nor_b", "cv_att_05", "cv_nor_05", "cv_auc")}
if EXTRAS:
    for k in ("base2_att_b", "base2_nor_b", "base2_att_05", "base2_nor_05"):
        res[k] = []
fa_nor, miss_att = Counter(), Counter()

for seed in (1, 2, 3):
    # (a) baseline: no take 002 data
    m0 = fit(X, y, cols, seed); b0 = bound(m0)
    sc0 = np.array([score(m0, s[4], cols).max() for s in segs])
    lab = np.array([1 if s[1] == "attack" else 0 for s in segs])
    res["base_att_b"].append(int((sc0[lab == 1] >= b0).sum()))
    res["base_nor_b"].append(int((sc0[lab == 0] >= b0).sum()))
    res["base_att_05"].append(int((sc0[lab == 1] >= 0.5).sum()))
    res["base_nor_05"].append(int((sc0[lab == 0] >= 0.5).sum()))

    if EXTRAS:
        m2 = fit(np.concatenate([X, extra_X]), np.concatenate([y, extra_y]), cols, seed); b2 = bound(m2)
        sc2 = np.array([score(m2, s[4], cols).max() for s in segs])
        res["base2_att_b"].append(int((sc2[lab == 1] >= b2).sum()))
        res["base2_nor_b"].append(int((sc2[lab == 0] >= b2).sum()))
        res["base2_att_05"].append(int((sc2[lab == 1] >= 0.5).sum()))
        res["base2_nor_05"].append(int((sc2[lab == 0] >= 0.5).sum()))

    # (b) cross-validated model
    cv_scores, cv_labels, cv_bounds = [], [], []
    for f in range(5):
        held_f = folds[f]
        ranges = [(s[5], s[6]) for s in held_f]
        tr_w, tr_l = [], []
        for j, s in enumerate(segs):
            if j % 5 == f:
                continue
            for w, st in zip(s[4], s[3]):
                if any(st < f1 and st + 50 > f0 for f0, f1 in ranges):
                    continue
                tr_w.append(w); tr_l.append(1 if s[1] == "attack" else 0)
        if EXTRAS:
            tr_w = list(tr_w) + [e for e in extra_X]; tr_l = list(tr_l) + [int(v) for v in extra_y]
        Xf = np.concatenate([X, np.stack(tr_w)]); yf = np.r_[y, tr_l].astype(np.float32)  # Nov 5 + take 001 always in
        mf = fit(Xf, yf, cols, seed); bf = bound(mf)
        for s in held_f:
            q = score(mf, s[4], cols).max()
            cv_scores.append(q); cv_labels.append(1 if s[1] == "attack" else 0); cv_bounds.append(bf)
    cv_scores = np.array(cv_scores); cv_labels = np.array(cv_labels); cv_bounds = np.array(cv_bounds)
    res["cv_att_b"].append(int((cv_scores[cv_labels == 1] >= cv_bounds[cv_labels == 1]).sum()))
    res["cv_nor_b"].append(int((cv_scores[cv_labels == 0] >= cv_bounds[cv_labels == 0]).sum()))
    res["cv_att_05"].append(int((cv_scores[cv_labels == 1] >= 0.5).sum()))
    res["cv_nor_05"].append(int((cv_scores[cv_labels == 0] >= 0.5).sum()))
    res["cv_auc"].append(roc_auc_score(cv_labels, cv_scores))
    res.setdefault("base_auc", []).append(roc_auc_score(lab, sc0))
    for s, q in zip([s for f in range(5) for s in folds[f]], cv_scores):  # same fold order as cv_scores
        if s[1] != "attack" and q >= 0.5:
            fa_nor[s[0]] += 1
        if s[1] == "attack" and q < 0.5:
            miss_att[s[0]] += 1
    extra_txt = f"   base2 att {res['base2_att_b'][-1]}/{n_att} fa {res['base2_nor_b'][-1]}/{n_nor}" if EXTRAS else ""
    print(f"seed {seed}: base att {res['base_att_b'][-1]}/{n_att} at bound fa {res['base_nor_b'][-1]}/{n_nor} | "
          f"att05 {res['base_att_05'][-1]}/{n_att} fa05 {res['base_nor_05'][-1]}/{n_nor}{extra_txt} | "
          f"cv att {res['cv_att_b'][-1]}/{n_att} at bound fa {res['cv_nor_b'][-1]}/{n_nor} | "
          f"att05 {res['cv_att_05'][-1]}/{n_att} fa05 {res['cv_nor_05'][-1]}/{n_nor} auc {res['cv_auc'][-1]:.3f}", flush=True)

mean = lambda k: float(np.mean(res[k]))
print(f"\nno take 002 data:          attacks {mean('base_att_b'):.1f}/{n_att} at bound  "
      f"{mean('base_att_05'):.1f}/{n_att} at 0.5   normal alarmed {mean('base_nor_b'):.1f}/{n_nor} at bound "
      f"{mean('base_nor_05'):.1f}/{n_nor} at 0.5")
if EXTRAS:
    print(f"no take 002 + extra data:  attacks {mean('base2_att_b'):.1f}/{n_att} at bound  "
          f"{mean('base2_att_05'):.1f}/{n_att} at 0.5   normal alarmed {mean('base2_nor_b'):.1f}/{n_nor} at bound "
          f"{mean('base2_nor_05'):.1f}/{n_nor} at 0.5")
print(f"cross-validated:          attacks {mean('cv_att_b'):.1f}/{n_att} at bound  "
      f"{mean('cv_att_05'):.1f}/{n_att} at 0.5   normal alarmed {mean('cv_nor_b'):.1f}/{n_nor} at bound "
      f"{mean('cv_nor_05'):.1f}/{n_nor} at 0.5   AUC {mean('cv_auc'):.3f}")
print(f"baseline AUC {mean('base_auc'):.3f}")
print("normal labels alarmed at 0.5 (CV, over 3 seeds):", dict(fa_nor.most_common()))
print("attack labels missed at 0.5 (CV, over 3 seeds):", dict(miss_att.most_common()))
