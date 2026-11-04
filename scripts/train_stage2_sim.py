"""
Retrains the three Stage 2 cascade models on simulated skeletons only.

Same architectures and the same output format as the live models, so Stage2bStaircaseEngine loads
them unchanged: T1 PoseConv3D bc=12 (598k), T2 PoseConv3D bc=12 distilled from the new T3 with RKD
(as the live T2 was), T3 ST-GCN. Each model keeps a Relative Mahalanobis reference
(mu_0, mu_c, inv_Sigma_0, inv_Sigma_c, optimal_threshold).

Final recipe: training uses only simulated data from data/Simulated_v2, mix
real:0,sim:16,normal:12,real_normal:0. Mixing real attacks with only-simulated normals taught a
"looks real = attack" shortcut and scored worse on the lab test. The ST-GCN makes the decision; the
PoseConv3D tiers are calibrated as pass-through (never discard or alarm) because they threw away
82 % of the real lab attack windows.

Objective: 7-class cross-entropy (normal + 6 attack groups) + violent/normal BCE on the model's own
head + batch-hard triplet on the features; T2 adds 20 x RKD to the new T3. Random mirror / rotation /
scale / jitter on every batch. RMD background (mu_0) is fitted on attacks AND normal activity.

Lab take 002 and the known live thrust clips are never used for training or thresholds.

LIVE_DIR = data/live_clips_2026-10-31. Order matters; train needs an explicit --tag _v2simonly and
t2 needs t3 first:

  python scripts/train_stage2_sim.py prepare
  python scripts/train_stage2_sim.py train --model t3 --steps 6000 --tag _v2simonly
  python scripts/train_stage2_sim.py train --model t1 --steps 8000 --tag _v2simonly
  python scripts/train_stage2_sim.py train --model t2 --steps 8000 --tag _v2simonly
  python scripts/train_stage2_sim.py deploy --tag _v2simonly --live-dir LIVE_DIR \
      --exclude "s111247:30-40,s114942:34-42,s1031:28-34,s110035:29-35"
  python scripts/train_stage2_sim.py evaluate --version v2 --test-take 002 --live-dir LIVE_DIR

Outputs go to weights/*_sim_v2.* via the deploy stage (the live weights are not touched) and
results/train_sim/.

The sim3 blend (--stage2-models sim3, reports/MODIFICATION_REPORT_STAGE2_SIM3.md) averages three ST-GCNs. HTD_SIM_DIR and
HTD_TRAIN_OUT choose the dataset and the output folder. Train each with --steps 6000 --seed 1, then move the weights to
results/train_sim/experiment_weights/ and run `blend`:

  old:   dataset from commit e6bfc07, HTD_TRAIN_OUT=train_sim/exp_A_old, --tag _A_old_s1
  fixed: dataset from this commit,    HTD_TRAIN_OUT=train_sim/exp_H_all, --tag _H_all_s1
  mix:   same dataset, prepare --live-dir LIVE_DIR --real-normal-live s1031:28-34,s110035:29-35,
         HTD_TRAIN_OUT=train_sim/exp_H_mix, --tag _H_mixB_s1 --mix real:3,sim:12,normal:6,real_normal:7

  python scripts/train_stage2_sim.py blend --live-dir LIVE_DIR
"""

import argparse
import glob
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)

from models.poseconv3d import PoseConv3DModel  # noqa: E402
from models.stgcn import STGCNModel  # noqa: E402
from src.dataset import ActionDataset  # noqa: E402
from src.distillation_losses import RelationalKDLoss  # noqa: E402
from src.ood_metrics import compute_relative_mahalanobis_distance  # noqa: E402
from src.skeleton_utils import interpolate_missing_joints, normalize_skeleton_clip, smooth_kinematics  # noqa: E402

ANN = "annotations_yolo26s-pose"
OUT = os.path.join(REPO_DIR, "results", os.environ.get("HTD_TRAIN_OUT", "train_sim"))
SIM = os.path.join(REPO_DIR, "data", os.environ.get("HTD_SIM_DIR", "Simulated_v2"))
LAB = os.path.join(REPO_DIR, "data", "lab_recordings", "2026-10-31")
CLASSES = ["normal", "thrust", "stab", "cut_down", "slash", "swing", "combo"]
FOLDER_CLASS = {"Thrust": 1, "Stab": 2, "Cut-Down": 3, "Cut_Down": 3, "Slash": 4, "Multi-Slash": 4, "Swing": 5,
                "Combo": 6, "OOD_Normal": 0, "OOD": 0}
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
STRIDE = 10
WEIGHT_FILES = {"t1": "poseconv3d_tier3_sim_v1.pth", "t2": "poseconv3d_dt3_sim_v1.pth", "t3": "stgcn_sim_v1.pth"}
REF_FILES = {k: "reference_data_" + v.replace(".pth", ".pt") for k, v in WEIGHT_FILES.items()}


def folder_class(name):
    for k in sorted(FOLDER_CLASS, key=len, reverse=True):
        if name.endswith(k):
            return FOLDER_CLASS[k]
    raise KeyError(name)


# ------------------------------------------------------------------ data
def parse_xml(path):
    p = ActionDataset.__new__(ActionDataset)
    p.num_keypoints, p.num_coords = 17, 2
    return p._parse_xml(path)


def windows_of(seq, stride=STRIDE):
    """50-frame windows of an already interpolated + smoothed sequence (as ActionDataset slices them).
    A clip shorter than 50 frames gives one window padded with zeros, as the dataset loader does."""
    seq = np.asarray(seq, np.float32)
    if len(seq) <= 50:
        w = np.zeros((50, 17, 2), np.float32)
        w[:len(seq)] = seq
        return [w]
    starts = list(range(0, len(seq) - 50 + 1, stride))
    if starts[-1] != len(seq) - 50:
        starts.append(len(seq) - 50)
    return [seq[s:s + 50] for s in starts]


def live_prep(seq):
    """Lab / live keypoints: smoothed per 50-frame window like Stage2aPoseTracker."""
    return smooth_kinematics(interpolate_missing_joints(np.asarray(seq, np.float32)[..., :2]), sigma=1.0)


def prepare(args):
    """Collects every clip as raw (smoothed) windows + labels into one cache file."""
    os.makedirs(OUT, exist_ok=True)
    W, meta = [], []

    def add(seq, cls, split, group, source):
        for w in windows_of(seq):
            W.append(w)
            meta.append((cls, split, group, source))

    t0 = time.time()
    for cls_name in ["Thrust", "Stab", "Cut-Down"]:
        for f in sorted(glob.glob(os.path.join(REPO_DIR, "data", cls_name, ANN, "*.xml"))):
            s = parse_xml(f)
            if s is not None:
                add(s, folder_class(cls_name), "train", f"real/{cls_name}/{os.path.basename(f)}", "real")
    for d in sorted(glob.glob(os.path.join(SIM, "*", ANN))):
        folder = os.path.basename(os.path.dirname(d))
        cls = folder_class(folder)
        aug = folder.startswith("Aug_")
        for f in sorted(glob.glob(os.path.join(d, "*.xml"))):
            s = parse_xml(f)
            if s is None:
                continue
            base = os.path.basename(f)
            # perturbed real clips always train; simulated clips: 15 % held out for calibration, by clip
            split = "train" if aug or zlib.crc32(base.encode()) % 100 >= 15 else "calib"
            add(s, cls, split, f"{folder}/{base}", "aug" if aug else ("sim_normal" if cls == 0 else "sim_attack"))
        print(f"  {folder}: {len(W)} windows so far ({time.time() - t0:.0f} s)", flush=True)
    # real non-violent clips: every model so far has only seen real ATTACKS, so "looks real" meant "attack".
    # False Detected Clip trains; Validate/OOD is split in half by clip (train / calibration).
    for f in sorted(glob.glob(os.path.join(REPO_DIR, "data", "False Detected Clip", "*.xml"))):
        s = parse_xml(f)
        if s is not None:
            add(s, 0, "train", f"real_normal/False_Detected/{os.path.basename(f)}", "real_normal")
    for cat in ["With_Coat", "Without_Coat", "OOD"]:
        for f in sorted(glob.glob(os.path.join(REPO_DIR, "data", "Validate", cat, "**", "*.xml"), recursive=True)):
            s = parse_xml(f)
            if s is None:
                continue
            sub = os.path.basename(os.path.dirname(os.path.dirname(f))) if cat != "OOD" else "OOD"
            base = os.path.basename(f)
            if cat == "OOD" and zlib.crc32(base.encode()) % 2 == 0:
                add(s, 0, "train", f"real_normal/Validate_OOD/{base}", "real_normal")
            else:
                add(s, folder_class(sub), "calib", f"validate/{cat}/{sub}/{base}", "validate")
    # optional real normals from the deployment camera: first half of lab take 001 and the named live sessions,
    # minus each session's range of attack tests. The blend stage sets its bound on the other sessions.
    excl = dict(item.split(":") for item in filter(None, args.real_normal_live.split(",")))
    if excl:
        K = np.load(os.path.join(LAB, "analysis", "001_floor_markers_142701_analysis.npz"))["kpts"][..., :2]
        for s in range(0, len(K) // 2 - 49, STRIDE):
            W.append(live_prep(K[s:s + 50]))
            meta.append((0, "train", f"real_normal/lab001/{s}", "real_normal"))
        for sess, clip, w in live_clips(args.live_dir):
            if sess in excl and not int(excl[sess].split("-")[0]) <= int(clip.split("_")[1]) <= int(excl[sess].split("-")[1]):
                W.append(w)
                meta.append((0, "train", f"real_normal/{sess}/{clip}", "real_normal"))
    W = np.stack(W).astype(np.float32)
    cls, split, group, source = map(np.array, zip(*meta))
    np.savez(os.path.join(OUT, "windows.npz"), X=W, cls=cls.astype(np.int64), split=split, group=group, source=source)
    print(f"[PREPARE] {len(W)} windows ({(split == 'train').sum()} train, {(split == 'calib').sum()} calibration) "
          f"in {time.time() - t0:.0f} s -> {OUT}/windows.npz")
    for s in ("train", "calib"):
        m = split == s
        print(f"  {s}: " + ", ".join(f"{CLASSES[c]} {int(((cls == c) & m).sum())}" for c in range(7)))


def load_cache():
    z = np.load(os.path.join(OUT, "windows.npz"))
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ batch ops (GPU, identical to the pipeline)
def normalize_batch(x):
    """Batched normalize_skeleton_clip (centroid of visible joints, torso scale, same fallbacks). x: (B,T,17,2)."""
    valid = x.abs().sum(-1) > 1e-6
    vf = valid.float().unsqueeze(-1)
    cnt = vf.sum(2).clamp(min=1)
    centroid = (x * vf).sum(2) / cnt
    centered = x - centroid.unsqueeze(2)

    def mid(a, b):
        va, vb = valid[..., a].unsqueeze(-1), valid[..., b].unsqueeze(-1)
        both = (x[..., a, :] + x[..., b, :]) / 2
        return torch.where(va & vb, both, torch.where(va, x[..., a, :], x[..., b, :])), (va | vb).squeeze(-1)

    sh, has_sh = mid(5, 6)
    hip, has_hip = mid(11, 12)
    torso = (sh - hip).norm(dim=-1)
    big = 1e9
    lo = torch.where(valid.unsqueeze(-1), x, torch.full_like(x, big)).amin(2)
    hi = torch.where(valid.unsqueeze(-1), x, torch.full_like(x, -big)).amax(2)
    diag = (hi - lo).norm(dim=-1)
    fallback = torch.where(diag > 10.0, diag * 0.5, torch.full_like(diag, 100.0))
    scale = torch.where(has_sh & has_hip & (torso > 10.0), torso, fallback)
    out = centered / (scale + 1e-6).unsqueeze(-1).unsqueeze(-1)
    return out * vf, valid


def augment(n, valid, gen):
    """Mirror, rotation, anisotropic scale, jitter on normalized coords (B,T,17,2)."""
    B = n.shape[0]
    dev = n.device
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    n = torch.where(flip.view(B, 1, 1, 1), n[:, :, FLIP] * torch.tensor([-1.0, 1.0], device=dev), n)
    valid = torch.where(flip.view(B, 1, 1), valid[:, :, FLIP], valid)
    ang = (torch.rand(B, device=dev, generator=gen) - 0.5) * np.radians(24)
    c, s = torch.cos(ang), torch.sin(ang)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,2,2)
    sc = 1 + (torch.rand(B, 2, device=dev, generator=gen) - 0.5) * 0.2
    n = torch.einsum("btkc,bdc->btkd", n * sc.view(B, 1, 1, 2), R)
    # noise randomization: keypoint noise of random strength (0-3 % torso) and slow wobble, plus rare
    # one-joint jumps, so the noise style of a source (real camera vs simulator) carries no label
    lvl = torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 0.03
    n = n + torch.randn(n.shape, device=dev, generator=gen) * lvl
    wob = torch.randn((B, 6, 17, 2), device=dev, generator=gen)
    wob = F.interpolate(wob.permute(0, 2, 3, 1).reshape(B, 34, 6), size=n.shape[1], mode="linear",
                        align_corners=True).reshape(B, 17, 2, -1).permute(0, 3, 1, 2)
    n = n + wob * torch.rand(B, 1, 1, 1, device=dev, generator=gen) * 0.03
    jump = (torch.rand(n.shape[:3], device=dev, generator=gen) < 0.002).unsqueeze(-1)
    n = n + jump * torch.randn(n.shape, device=dev, generator=gen) * 0.15
    return n * valid.unsqueeze(-1).float(), valid


def heatmaps(n, valid, H=56, sigma=1.5):
    """(B,T,17,2) normalized -> (B,17,T,56,56), as rasterize_heatmap_from_coords(normalize=False)."""
    mx = ((n[..., 0] + 3.0) / 6.0).clamp(0, 1).permute(0, 2, 1)[..., None, None]
    my = ((n[..., 1] + 3.0) / 6.0).clamp(0, 1).permute(0, 2, 1)[..., None, None]
    g = torch.linspace(0.0, 1.0, H, device=n.device)
    s2 = 2.0 * (sigma / H) ** 2
    hm = torch.exp(-((g.view(1, 1, 1, 1, H) - mx) ** 2 + (g.view(1, 1, 1, H, 1) - my) ** 2) / s2)
    return hm * valid.permute(0, 2, 1)[..., None, None].float()


def coords_input(n):
    return n.permute(0, 3, 1, 2).unsqueeze(-1)  # (B,2,T,17,1)


def build(kind):
    if kind == "t3":
        return STGCNModel(num_classes=1)
    return PoseConv3DModel(num_classes=1, base_channels=12, feature_dim=256, dropout=0.2)


def model_input(kind, n, valid):
    return coords_input(n) if kind == "t3" else heatmaps(n, valid)


def batch_hard_triplet(f, y, margin=0.3):
    f = F.normalize(f, dim=-1)
    d = torch.cdist(f, f)
    same = y.unsqueeze(0) == y.unsqueeze(1)
    pos = (d * same.float()).max(1).values
    neg = (d + same.float() * 10).min(1).values
    return F.relu(pos - neg + margin).mean()


# ------------------------------------------------------------------ training
def train(args):
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    c = load_cache()
    tr = np.where(c["split"] == "train")[0]
    ca = np.where((c["split"] == "calib"))[0]
    X = torch.from_numpy(c["X"]).to(dev)
    Y = torch.from_numpy(c["cls"]).to(dev)
    # batch mix: real + perturbed real 30 %, simulated attacks 45 %, normal activity 25 %
    pools = {k: tr[np.isin(c["source"][tr], v)] for k, v in
             {"real": ["real", "aug"], "sim": ["sim_attack"], "normal": ["sim_normal"],
              "real_normal": ["real_normal"]}.items()}
    mix = {k: int(v) for k, v in (kv.split(":") for kv in args.mix.split(","))}
    mix = {k: v for k, v in mix.items() if v > 0}
    rng = np.random.default_rng(args.seed)

    model = build(args.model).to(dev)
    head = nn.Linear(256, len(CLASSES)).to(dev)
    teacher = None
    if args.model == "t2":
        teacher = build("t3").to(dev)
        teacher.load_state_dict(torch.load(weight_path("t3", args.tag), map_location=dev))
        teacher.eval()
        rkd = RelationalKDLoss(w_dist=1.0, w_angle=2.0).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.08)
    print(f"[TRAIN {args.model}] {sum(p.numel() for p in model.parameters()):,} params, {len(tr)} train windows "
          f"(per batch {mix}; pool sizes { {k: len(v) for k, v in pools.items()} }), "
          f"{args.steps} steps on {dev}", flush=True)

    def evaluate():
        model.eval()
        head.eval()
        correct, viol_ok, n_tot, loss_sum = 0, 0, 0, 0.0
        with torch.no_grad():
            for i in range(0, len(ca), 128):
                idx = torch.from_numpy(ca[i:i + 128]).to(dev)
                n, v = normalize_batch(X[idx])
                logit, f = model(model_input(args.model, n, v), return_features=True)
                out = head(f)
                y = Y[idx]
                loss_sum += F.cross_entropy(out, y, reduction="sum").item()
                p = out.argmax(1)
                correct += (p == y).sum().item()
                viol_ok += ((p > 0) == (y > 0)).sum().item()
                n_tot += len(idx)
        model.train()
        head.train()
        return loss_sum / n_tot, correct / n_tot, viol_ok / n_tot

    best, log = None, []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        idx = np.concatenate([rng.choice(pools[k], m) for k, m in mix.items()])
        idx = torch.from_numpy(idx).to(dev)
        n, v = normalize_batch(X[idx])
        n, v = augment(n, v, gen)
        y = Y[idx]
        logit, f = model(model_input(args.model, n, v), return_features=True)
        loss = F.cross_entropy(head(f), y, label_smoothing=0.05)
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(logit.squeeze(-1), (y > 0).float())
        loss = loss + 0.5 * batch_hard_triplet(f, y)
        if teacher is not None:
            with torch.no_grad():
                _, ft = teacher(coords_input(n), return_features=True)
            loss = loss + 20.0 * rkd(f, ft)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            vl, acc, vacc = evaluate()
            log.append(dict(step=step, train_loss=loss.item(), calib_ce=vl, calib_acc7=acc, calib_violent_acc=vacc))
            mark = ""
            if best is None or vl < best[0]:
                best = (vl, step, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
                mark = " *"
            print(f"  step {step:5d} loss {loss.item():.3f} | calib CE {vl:.3f} acc7 {acc:.3f} violent/normal {vacc:.3f}"
                  f" | {time.time() - t0:.0f} s{mark}", flush=True)
    path = weight_path(args.model, args.tag)
    torch.save(best[2], path)
    with open(os.path.join(OUT, f"train_log_{args.model}{args.tag}.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    print(f"[TRAIN {args.model}] best calib CE {best[0]:.3f} at step {best[1]} -> {path}")


# ------------------------------------------------------------------ features, RMD reference, cascade
def weight_path(kind, tag=""):
    return os.path.join(REPO_DIR, "weights", WEIGHT_FILES[kind].replace("_v1", "_v1" + tag))


def ref_path(kind, tag=""):
    return os.path.join(REPO_DIR, "weights", REF_FILES[kind].replace("_v1", "_v1" + tag))


def load_trained(kind, dev, tag=""):
    m = build(kind).to(dev)
    m.load_state_dict(torch.load(weight_path(kind, tag), map_location=dev, weights_only=True))
    return m.eval()


@torch.no_grad()
def features(model, kind, X, dev, bs=128):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs]
        x = torch.from_numpy(x).to(dev) if isinstance(x, np.ndarray) else x.to(dev)
        n, v = normalize_batch(x)
        out.append(model(model_input(kind, n, v), return_features=True)[1].float().cpu())
    return torch.cat(out)


def rmd_reference(f_train, y_train, eps):
    D = f_train.shape[1]
    mu_0 = f_train.mean(0, keepdim=True)
    d0 = f_train - mu_0
    S0 = d0.T @ d0 / (len(f_train) - 1)
    mus, Ss = [], []
    for c in range(1, len(CLASSES)):
        fc = f_train[y_train == c]
        mus.append(fc.mean(0))
        dc = fc - mus[-1]
        Ss.append(dc.T @ dc / (len(fc) - 1))
    eye = torch.eye(D)
    return dict(mu_0=mu_0, mu_c=torch.stack(mus), inv_Sigma_0=torch.linalg.pinv(S0 + eps * eye),
                inv_Sigma_c=torch.stack([torch.linalg.pinv(S + eps * eye) for S in Ss]))


def rmd(f, ref):
    s, _, _ = compute_relative_mahalanobis_distance(f, ref["mu_0"], ref["mu_c"], ref["inv_Sigma_0"], ref["inv_Sigma_c"])
    return s


def best_f1_threshold(scores, labels):
    best = (0.0, 0.0)
    for t in np.quantile(scores, np.linspace(0.01, 0.99, 400)):
        p = scores >= t
        tp, fp, fn = (p & labels).sum(), (p & ~labels).sum(), (~p & labels).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, float(t))
    return best


def calibrate(args):
    """Fits the RMD reference of each new model and picks shrinkage + threshold on the calibration split
    (clip level: a clip counts as an alarm if any of its windows does, like the live pipeline)."""
    dev = torch.device(args.device)
    c = load_cache()
    tr = c["split"] == "train"
    ca = c["split"] == "calib"
    groups = c["group"][ca]
    y_clip = {}
    for g, y in zip(groups, c["cls"][ca]):
        y_clip[g] = y > 0
    summary = {}
    for kind in args.models.split(","):
        m = load_trained(kind, dev, args.tag)
        ft = features(m, kind, c["X"][tr], dev)
        fc = features(m, kind, c["X"][ca], dev)
        yt = torch.from_numpy(c["cls"][tr])
        best = None
        for eps in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
            ref = rmd_reference(ft, yt, eps)
            s = rmd(fc, ref).numpy()
            clip_s = {}
            for g, v in zip(groups, s):
                clip_s[g] = max(clip_s.get(g, -1e18), v)
            keys = list(clip_s)
            f1, tau = best_f1_threshold(np.array([clip_s[k] for k in keys]), np.array([y_clip[k] for k in keys]))
            print(f"  {kind} eps {eps:<6} best clip F1 {f1:.3f} at tau {tau:.2f}", flush=True)
            if best is None or f1 > best[0]:
                best = (f1, eps, tau, ref)
        f1, eps, tau, ref = best
        torch.save({"version": f"{kind}_sim_v1", "mu_0": ref["mu_0"], "mu_c": ref["mu_c"],
                    "inv_Sigma_0": ref["inv_Sigma_0"], "inv_Sigma_c": ref["inv_Sigma_c"],
                    "optimal_epsilon": eps, "optimal_threshold": tau, "classes": CLASSES[1:],
                    "calibration_clip_f1": f1}, ref_path(kind, args.tag))
        summary[kind] = dict(eps=eps, tau=tau, calib_clip_f1=f1)
        print(f"[CALIBRATE] {kind}: eps {eps}, tau {tau:.2f}, calibration clip F1 {f1:.3f}", flush=True)
    with open(os.path.join(OUT, f"calibration{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


class Cascade:
    """Stage2bStaircaseEngine decision logic over batches of windows, for old or new weights."""

    def __init__(self, which, dev, bounds=(0.20, 0.99, 0.35, 0.95, 0.55), tag="", version="v1"):
        self.dev, self.b = dev, bounds
        self.models = []
        if which == "old":
            from models.ensemble_super import load_model_instance
            for key in ("pc3d_tier3", "pc3d_dt3", "stgcn"):
                it = load_model_instance(key, dev)
                self.models.append((it["model"], "t3" if key == "stgcn" else "t1",
                                    {k: it[k] for k in ("mu_0", "mu_c", "inv_Sigma_0", "inv_Sigma_c")}, it["tau"]))
        else:
            for kind in (SIM_ORDER if version != "v1" else ("t1", "t2", "t3")):
                if version != "v1":
                    rp = os.path.join(REPO_DIR, "weights", REF_FILES[kind].replace("_v1", "_" + version))
                    m = build(kind).to(dev)
                    m.load_state_dict(torch.load(os.path.join(REPO_DIR, "weights", WEIGHT_FILES[kind].replace(
                        "_v1", "_" + version)), map_location=dev, weights_only=True))
                    m.eval()
                else:
                    rp, m = ref_path(kind, tag), load_trained(kind, dev, tag)
                ref = torch.load(rp, map_location=dev, weights_only=False)
                self.models.append((m, kind,
                                    {k: ref[k].to(dev) for k in ("mu_0", "mu_c", "inv_Sigma_0", "inv_Sigma_c")},
                                    float(ref["optimal_threshold"])))

    @torch.no_grad()
    def probs(self, X, bs=128, raw=False):
        """Per-window tier probabilities sigmoid(RMD - tau); raw=True returns RMD - tau (no saturation)."""
        P = []
        for i in range(0, len(X), bs):
            x = torch.from_numpy(np.asarray(X[i:i + bs], np.float32)).to(self.dev)
            n, v = normalize_batch(x)
            hm = heatmaps(n, v)
            co = coords_input(n)
            p = []
            for m, kind, ref, tau in self.models:
                _, f = m(co if kind == "t3" else hm, return_features=True)
                z = rmd(f.float(), ref) - tau
                p.append(z if raw else torch.sigmoid(z))
            P.append(torch.stack(p, -1).cpu())
        return torch.cat(P).numpy()

    def alarm(self, P):
        lo1, hi1, lo2, hi2, tau3 = self.b
        p1, p2, p3 = P[:, 0], P[:, 1], P[:, 2]
        return np.where(p1 < lo1, False, np.where(p1 >= hi1, True,
                        np.where(p2 < lo2, False, np.where(p2 >= hi2, True, p3 >= tau3))))


def lab_windows():
    """Lab segments as 50-frame live-style windows. Segments shorter than 50 frames are widened
    symmetrically inside their take, so every segment can be scored."""
    import csv
    takes = {}
    out = []
    with open(os.path.join(LAB, "segments", "segments.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            take = r["take"]
            if take not in takes:
                takes[take] = np.load(os.path.join(LAB, "analysis", take + "_analysis.npz"))["kpts"][..., :2]
            K = takes[take]
            f0, f1 = int(round(float(r["start_s"]) * 30)), int(round(float(r["end_s"]) * 30))
            if f1 - f0 < 50:
                pad = 50 - (f1 - f0)
                f0 = max(0, f0 - pad // 2 - pad % 2)
                f1 = min(len(K), f0 + 50)
                f0 = f1 - 50
            ws = [live_prep(K[s:s + 50]) for s in range(f0, f1 - 49, STRIDE)]
            out.append((r["id"] + "_" + r["label"], r["category"], ws))
    return out, takes


def take_replay(takes):
    """Slides over each whole lab take like the live pipeline (stride 10) and labels each window by
    its overlap with the hand-labelled attack segments."""
    import csv
    att = {}
    with open(os.path.join(LAB, "segments", "segments.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["category"] == "attack":
                att.setdefault(r["take"], []).append((float(r["start_s"]) * 30, float(r["end_s"]) * 30, r["id"]))
    res = []
    for take, K in takes.items():
        for s in range(0, len(K) - 49, STRIDE):
            ov = [(min(s + 50, b) - max(s, a), i) for a, b, i in att.get(take, [])]
            best = max(ov, default=(0, None))
            lab = "attack" if best[0] >= 15 else ("normal" if best[0] <= 0 else "edge")
            res.append((take, s, lab, best[1], live_prep(K[s:s + 50])))
    return res


def live_clips(live_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(live_dir, "**", "skeleton_raw.npy"), recursive=True)):
        raw = np.load(f)
        if (np.abs(raw).sum(-1) > 0).mean() < 0.5:
            continue
        sess = os.path.relpath(f, live_dir).split(os.sep)[0]
        out.append((sess, os.path.basename(os.path.dirname(f)), live_prep(raw[:50])))
    return out


def evaluate(args):
    dev = torch.device(args.device)
    c = load_cache()
    ca = c["split"] == "calib"
    report = {}
    labs, takes = lab_windows()
    if args.test_take:  # takes used for calibration are not test data
        takes = {k: v for k, v in takes.items() if k.startswith(args.test_take)}
        labs = [x for x in labs if int(x[0][:3]) >= 4]  # segments 000-003 are take 001
    replay = take_replay(takes)
    live = live_clips(args.live_dir)
    for which in ("old", "new"):
        cas = Cascade(which, dev, version=args.version)
        r = {}
        # calibration split, per source folder, clip level
        P = cas.probs(c["X"][ca])
        A = cas.alarm(P)
        per = {}
        for g, a in zip(c["group"][ca], A):
            per[g] = per.get(g, False) or bool(a)
        groups = {}
        for g, a in per.items():
            key = "/".join(g.split("/")[:-1]) if g.startswith("validate") else g.split("/")[0]
            groups.setdefault(key, []).append(a)
        r["calib"] = {k: (len(v), float(np.mean(v))) for k, v in sorted(groups.items())}
        # lab segments
        segs = {}
        for name, cat, ws in labs:
            a = cas.alarm(cas.probs(np.stack(ws)))
            segs[name] = (cat, bool(a.any()), int(a.sum()), len(a))
        r["lab_segments"] = segs
        # whole-take replay
        Pw = cas.probs(np.stack([w for *_, w in replay]))
        Aw = cas.alarm(Pw)
        att_ids = {i for _, _, lab, i, _ in replay if lab == "attack"}
        hit = {i for (_, _, lab, i, _), a in zip(replay, Aw) if lab == "attack" and a}
        normal = np.array([lab == "normal" for _, _, lab, _, _ in replay])
        minutes = normal.sum() * STRIDE / 30 / 60
        # false-alarm events: alarmed normal windows closer than the 5 s alert hold count as one event
        events, last = 0, {}
        for (take, st, lab, _, _), a in zip(replay, Aw):
            if lab == "normal" and a:
                if st - last.get(take, -10 ** 9) > 150:
                    events += 1
                last[take] = st
        r["replay_events"] = events
        r["replay"] = dict(attacks_detected=len(hit), attacks=len(att_ids),
                           attack_window_alarm_rate=float(Aw[[lab == "attack" for _, _, lab, _, _ in replay]].mean()),
                           normal_windows=int(normal.sum()), normal_alarm_windows=int(Aw[normal].sum()),
                           normal_minutes=float(minutes))
        # live debug clips
        if live:
            Pl = cas.probs(np.stack([w for *_, w in live]))
            Al = cas.alarm(Pl)
            sess = {}
            for (s, clip, _), a, p in zip(live, Al, Pl):
                sess.setdefault(s, []).append((clip, bool(a), [round(float(x), 3) for x in p]))
            r["live"] = sess
        report[which] = r
    with open(os.path.join(OUT, f"evaluation_{args.version}.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    o, n = report["old"], report["new"]
    print("\n== Calibration split (clip has >=1 alarm window) ==")
    print(f"{'group':42s} {'n':>5s} {'old':>7s} {'new':>7s}")
    for k in n["calib"]:
        print(f"{k:42s} {n['calib'][k][0]:5d} {100 * o['calib'][k][1]:6.1f}% {100 * n['calib'][k][1]:6.1f}%")
    print("\n== Lab segments 2026-10-31 (TEST, never trained on) ==")
    for cat in ("attack", "normal"):
        ks = [k for k, v in n["lab_segments"].items() if v[0] == cat]
        ao = np.mean([o["lab_segments"][k][1] for k in ks])
        an = np.mean([n["lab_segments"][k][1] for k in ks])
        print(f"  {cat:6s} segments: {len(ks)} | with alarm: old {100 * ao:5.1f}%  new {100 * an:5.1f}%")
    print("  per attack segment (alarm windows / windows):")
    for k, v in n["lab_segments"].items():
        if v[0] == "attack":
            ov = o["lab_segments"][k]
            print(f"    {k:40s} old {ov[2]:2d}/{ov[3]:<2d}  new {v[2]:2d}/{v[3]:<2d}")
    print("  normal segments with an alarm (new):",
          ", ".join(k for k, v in n["lab_segments"].items() if v[0] == "normal" and v[1]) or "none")
    print("\n== Whole lab takes replayed like the live pipeline (TEST) ==")
    for w, r in (("old", o["replay"]), ("new", n["replay"])):
        print(f"  {w}: attacks detected {r['attacks_detected']}/{r['attacks']} | attack windows alarmed "
              f"{100 * r['attack_window_alarm_rate']:.1f}% | normal windows alarmed {r['normal_alarm_windows']}/"
              f"{r['normal_windows']} ({r['normal_alarm_windows'] / max(r['normal_minutes'], 1e-9):.1f} per minute "
              f"of normal activity) | false-alarm events {report[w]['replay_events']} "
              f"({report[w]['replay_events'] / max(r['normal_minutes'], 1e-9):.2f} per minute)")
    if "live" in n:
        print("\n== Live debug clips (unlabelled; alarms per session) ==")
        for s in n["live"]:
            print(f"  {s}: old {sum(a for _, a, _ in o['live'][s])}/{len(o['live'][s])}  "
                  f"new {sum(a for _, a, _ in n['live'][s])}/{len(n['live'][s])}")
    print(f"\nFull results: {os.path.join(OUT, f'evaluation_{args.version}.json')}")


# ------------------------------------------------------------------ deployment calibration
# The engine turns each tier score into p = sigmoid(RMD - tau) and uses fixed bounds: T1 discards below
# 0.20 and alarms from 0.99, T2 discards below 0.35 and alarms from 0.95, T3 alarms from 0.55.
# Scaling both precision matrices by 1/T scales RMD by 1/T, so a temperature T and a tau can place those
# fixed bounds on any two score values without touching the engine.
LOGIT = lambda p: float(np.log(p / (1 - p)))
# The simulation-trained ST-GCN transfers best to the deployment camera, so it gets the final say (tier 3).
# Tier 1 = the distilled PoseConv3D (t2), tier 2 = the plain one (t1). Calibrated as discard filters they threw
# away 82 % of the real lab attack windows (heatmap models trained on simulation do not transfer), so by
# default they are pass-through (p = 0.5: never discard, never alarm) until real deployment-camera data exists.
SIM_ORDER = ("t2", "t1", "t3")
BOUNDS = {"t2": (0.20, 0.99), "t1": (0.35, 0.95), "t3": (None, 0.55)}
# The sim3 blend: model name -> (window cache under results/train_sim, experiment weight tag), all seed 1.
BLEND = {"old": ("exp_A_old", "_A_old_s1"), "fixed": ("exp_H_all", "_H_all_s1"), "mix": ("exp_H_mix", "_H_mixB_s1")}
BLEND_NORMAL_EXCL = {"s111247": (30, 40), "s114942": (34, 42)}


def deployment_normals(args):
    """Normal activity seen by the deployment camera, kept apart from the test take:
    lab take 001 (walking only) + live debug clips minus the clips around known attack tests."""
    K = np.load(os.path.join(LAB, "analysis", "001_floor_markers_142701_analysis.npz"))["kpts"][..., :2]
    W = [live_prep(K[s:s + 50]) for s in range(0, len(K) - 49, STRIDE)]
    excl = {}
    for item in filter(None, args.exclude.split(",")):
        sess, rng_ = item.split(":")
        a, b = map(int, rng_.split("-"))
        excl.setdefault(sess, []).append((a, b))
    n_live = 0
    for f in sorted(glob.glob(os.path.join(args.live_dir, "**", "skeleton_raw.npy"), recursive=True)):
        clip = os.path.basename(os.path.dirname(f))
        num = int(clip.split("_")[1])
        if any(sess in f and a <= num <= b for sess, rs in excl.items() for a, b in rs):
            continue
        raw = np.load(f)
        if (np.abs(raw).sum(-1) > 0).mean() < 0.5:
            continue
        W.append(live_prep(raw[:50]))
        n_live += 1
    print(f"  deployment-camera normal windows: {len(W) - n_live} from lab take 001 + {n_live} live clips")
    return np.stack(W)


def deploy_calibrate(args):
    dev = torch.device(args.device)
    c = load_cache()
    tr, ca = c["split"] == "train", c["split"] == "calib"
    sim_att = ca & np.isin(c["source"], ["sim_attack"])
    groups = c["group"][sim_att]
    normals = deployment_normals(args)
    summary = {}
    passthrough = set(filter(None, args.passthrough.split(",")))
    for kind in SIM_ORDER:
        m = load_trained(kind, dev, args.tag)
        ref = rmd_reference(features(m, kind, c["X"][tr], dev), torch.from_numpy(c["cls"][tr]), args.eps)
        s_att = rmd(features(m, kind, c["X"][sim_att], dev), ref).numpy()
        clip_max = {}
        for g, v in zip(groups, s_att):
            clip_max[g] = max(clip_max.get(g, -1e18), v)
        s_norm = rmd(features(m, kind, normals, dev), ref).numpy()
        lo_p, hi_p = BOUNDS[kind]
        if kind in passthrough:
            # p = sigmoid(RMD / 1e9 - 0) = 0.5 for any clip: inside both tier bands, so tier 3 decides
            hi, lo, T, tau = None, None, 1e9, 0.0
        elif kind == "t3":
            # final verdict: alarm above a high percentile of deployment-camera normal activity
            hi = float(np.quantile(s_norm, args.t3_quantile))
            T = float(np.std(s_norm) + 1e-6)
            tau = hi / T - LOGIT(hi_p)
            lo = None
        else:
            # discard below the score that keeps 98 % of simulated attack clips; the alarm bound is put
            # out of reach (far above every attack and normal score), so these tiers never alarm alone
            vals = np.array(list(clip_max.values()))
            lo = float(np.quantile(vals, args.discard_quantile))
            hi = float(max(vals.max(), s_norm.max()) + 10 * np.std(np.r_[vals, s_norm]))
            T = (hi - lo) / (LOGIT(hi_p) - LOGIT(lo_p))
            tau = lo / T - LOGIT(lo_p)
        ref_out = {"mu_0": ref["mu_0"], "mu_c": ref["mu_c"], "inv_Sigma_0": ref["inv_Sigma_0"] / T,
                   "inv_Sigma_c": ref["inv_Sigma_c"] / T}
        z = lambda v: v / T - tau
        stats = dict(temperature=T, tau=tau, score_lo=lo, score_hi=hi, passthrough=kind in passthrough,
                     normal_windows_discarded=float((1 / (1 + np.exp(-z(s_norm))) < (lo_p or 0)).mean()) if lo_p else None,
                     normal_windows_over_alarm_bound=float((1 / (1 + np.exp(-z(s_norm))) >= hi_p).mean()),
                     sim_attack_clips_over_alarm_bound=float(np.mean([1 / (1 + np.exp(-z(v))) >= hi_p
                                                                      for v in clip_max.values()])))
        torch.save({"version": f"{kind}_sim_v2", **ref_out, "optimal_epsilon": args.eps, "optimal_threshold": tau,
                    "temperature": T, "classes": CLASSES[1:], "calibration": stats},
                   os.path.join(REPO_DIR, "weights", REF_FILES[kind].replace("_v1", "_v2")))
        torch.save(m.state_dict(), os.path.join(REPO_DIR, "weights", WEIGHT_FILES[kind].replace("_v1", "_v2")))
        summary[kind] = stats
        print(f"[DEPLOY] {kind}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                              for k, v in stats.items()), flush=True)
    with open(os.path.join(OUT, "deploy_calibration_v2.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


def blend_normals(live_dir):
    """Deployment-camera normals not used by --real-normal-live: second half of take 001 and two live sessions."""
    K = np.load(os.path.join(LAB, "analysis", "001_floor_markers_142701_analysis.npz"))["kpts"][..., :2]
    W = [live_prep(K[s:s + 50]) for s in range(len(K) // 2, len(K) - 49, STRIDE)]
    for sess, clip, w in live_clips(live_dir):
        a, b = BLEND_NORMAL_EXCL.get(sess, (0, 10 ** 9))  # other sessions are left out completely
        if not a <= int(clip.split("_")[1]) <= b:
            W.append(w)
    return np.stack(W).astype(np.float32)


def deploy_blend(args):
    """Writes weights/stgcn_sim3_*: each model's score becomes z = (RMD - mean) / std on the held-out normals, and the
    alarm bound sits at the highest blended z of those normals, so sigmoid(mean z - bound) >= 0.5 means alarm."""
    dev = torch.device(args.device)
    normals = blend_normals(args.live_dir)
    models, z = [], []
    for name, (cache, tag) in BLEND.items():
        c = np.load(os.path.join(REPO_DIR, "results", "train_sim", cache, "windows.npz"))
        tr = c["split"] == "train"
        m = build("t3").to(dev)
        m.load_state_dict(torch.load(os.path.join(REPO_DIR, "results", "train_sim", "experiment_weights",
                                                  f"stgcn_sim_v1{tag}.pth"), map_location=dev, weights_only=True))
        m.eval()
        ref = rmd_reference(features(m, "t3", c["X"][tr], dev), torch.from_numpy(c["cls"][tr]), args.eps)
        s = rmd(features(m, "t3", normals, dev), ref).numpy()
        T = float(s.std() + 1e-6)
        models.append((name, m, ref, float(s.mean()) / T, T))
        z.append((s - s.mean()) / T)
    bound = float(np.mean(z, axis=0).max())
    for name, m, ref, tau, T in models:
        torch.save({"version": f"stgcn_sim3_{name}", "mu_0": ref["mu_0"], "mu_c": ref["mu_c"],
                    "inv_Sigma_0": ref["inv_Sigma_0"] / T, "inv_Sigma_c": ref["inv_Sigma_c"] / T, "optimal_epsilon": args.eps,
                    "optimal_threshold": tau + bound, "temperature": T, "classes": CLASSES[1:], "blend_bound_z": bound},
                   os.path.join(REPO_DIR, "weights", f"reference_data_stgcn_sim3_{name}.pt"))
        torch.save(m.state_dict(), os.path.join(REPO_DIR, "weights", f"stgcn_sim3_{name}.pth"))
    print(f"[BLEND] {len(normals)} held-out normal windows, bound z {bound:.3f} -> weights/stgcn_sim3_*")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "train", "calibrate", "deploy", "evaluate", "blend"])
    ap.add_argument("--real-normal-live", default="", help="prepare: live sessions added as real normals, each with "
                                                           "its attack-test range, e.g. s1031:28-34,s110035:29-35")
    ap.add_argument("--model", choices=["t1", "t2", "t3"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix", default="real:0,sim:16,normal:12,real_normal:0",
                    help="windows per batch from each pool: real (real+perturbed attacks), sim (simulated attacks), "
                         "normal (simulated normal activity), real_normal (real non-violent clips)")
    ap.add_argument("--tag", default="", help="suffix for experiment weights, e.g. _simonly")
    ap.add_argument("--models", default="t1,t2,t3", help="models to calibrate")
    ap.add_argument("--eps", type=float, default=0.05, help="covariance shrinkage for the deployment reference")
    ap.add_argument("--live-dir", default=os.path.join(REPO_DIR, "results", "debug_stage2"),
                    help="folder with live debug sessions (clips/*/skeleton_raw.npy), used as normal activity")
    ap.add_argument("--exclude", default="", help="live clips to leave out, e.g. s111247:30-40,s114942:34-42")
    ap.add_argument("--discard-quantile", type=float, default=0.02,
                    help="deploy: tier 1/2 discard bound = this quantile of simulated attack clip scores")
    ap.add_argument("--version", default="v1", help="evaluate: new weights to compare with the live ones")
    ap.add_argument("--test-take", default="", help="evaluate: only this lab take (e.g. 002) counts as test")
    ap.add_argument("--passthrough", default="t1,t2", help="deploy: models that never discard or alarm")
    ap.add_argument("--t3-quantile", type=float, default=0.98,
                    help="deploy: tier 3 alarms above this quantile of deployment-camera normal windows")
    args = ap.parse_args()
    {"prepare": prepare, "train": train, "calibrate": calibrate, "deploy": deploy_calibrate,
     "evaluate": evaluate, "blend": deploy_blend}[args.stage](args)


if __name__ == "__main__":
    main()
