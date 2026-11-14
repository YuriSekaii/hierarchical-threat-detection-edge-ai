"""Stage 2 three-model recipe (ST-GCN, PoseConv3D, PoseConv3D distilled; as poc_relabel3.py) trained on the cut and
relabelled clips of all lab days, scored on the takes never trained on, next to the shipped models.

  python results/samples/poc/poc_segments.py <REPO> [--train nov5|clips|all] [--seeds 1,2,3] [--save]

--train nov5 : Nov 5 clips + Oct 31 take 001          (what the earlier session had, now with fixed labels)
--train clips: + Nov 14 training clips                 (the complete lab set)
--train all  : + data/Validate clips + 3,000 + 3,000 simulated windows of exp_FH_all (earlier best add-on)
Test (never trained on): Nov 14 test_* takes, Oct 31 take 002 (clip 043, a kick, is left out), Nov 14 confuser takes.
Calibration (never trained on, sets the alarm bound): the 8 held-out Nov 5 takes, the second repeat of each Nov 14
normal take. Whole takes are replayed like the live pipeline (50-frame windows, stride 10).
"""
import argparse, glob, json, os, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--train", default="clips", choices=["nov5", "clips", "all"])
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--save", action="store_true", help="save seed-1 models to results/train_sim/exp_segments_<train>")
args = ap.parse_args()
REPO = os.path.abspath(args.repo)
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import train_stage2_sim as T
from src.distillation_losses import RelationalKDLoss

dev = torch.device(os.environ.get("DEV", "cuda:0"))
STEPS, BS, W, STRIDE = 700, 48, 50, 10
NC = len(T.CLASSES)
CAL5 = ("003", "008", "010", "017", "020", "023", "028", "031")
CAL14 = ("003", "006", "008", "010", "012", "014", "016", "018", "020", "022", "024")
D31, D5, D14 = "2026-10-31", "2026-11-05", "2026-11-14"


def role(r):
    n = r["take"][:3]
    if r["day"] == D31:
        return "train" if n == "001" else ("drop" if r["id"] == "043" else "test31")
    if r["day"] == D5:
        return "cal" if n in CAL5 else "train"
    if "_test_" in r["take"]:
        return "test14"
    if "048" <= n <= "057":  # object-in-hand takes: all of their clips, strikes and pauses
        return "confuser"
    return "cal" if n in CAL14 else "train"


def cls_of(label):
    L = label.lower()
    if "thrust" in L or "lunge" in L: return 1
    if "stab" in L: return 2
    if "cut_down" in L or "overhand" in L: return 3
    if "slash" in L: return 4
    return 1


rows, takes = T.lab_segments()
for r in rows:
    r["role"] = role(r)


def ok(K, s):  # a person is in at least 40 of the 50 frames
    return 0 <= s and s + W <= len(K) and (np.abs(K[s:s + W]).sum((1, 2)) > 0).sum() >= 40


GUARD = 30  # frames: a window this close to a strike holds its wind-up or recovery, so it is neither normal nor scored
ATT = {}
for r in rows:
    if r["category"] == "attack" or (r["category"] == "confuser" and "strike" in r["label"]):
        ATT.setdefault((r["day"], r["take"]), []).append(r)


def near_strike(key, s):
    return any(s < r["f1"] + GUARD and s + W > r["f0"] - GUARD for r in ATT.get(key, []))


# ---------------- training windows ----------------
X, Y = [], []
for r in rows:
    if r["role"] != "train" or "timing unreliable" in r["note"] or (args.train == "nov5" and r["day"] == D14):
        continue
    K = takes[r["day"], r["take"]]
    att = r["category"] == "attack"
    # an attack clip is 51 frames around the strike: three windows that all hold the strike
    for s in (range(r["f0"] - 10, r["f0"] + 1, 5) if att else range(r["f0"], r["f1"] - W + 1, STRIDE)):
        if ok(K, s) and (att or not near_strike((r["day"], r["take"]), s)):
            X.append(T.live_prep(K[s:s + W])); Y.append(cls_of(r["label"]) if att else 0)
if args.train == "all":
    import xml.etree.ElementTree as ET
    def raw_xml(path):
        imgs = ET.parse(path).getroot().findall(".//image")
        K = np.zeros((len(imgs), 17, 2), np.float32)
        for i, im in enumerate(imgs):
            p = im.find("points")
            pts = p.get("points", "").split(";") if p is not None else []
            if len(pts) >= 17:
                K[i] = [[float(v) for v in q.split(",")] for q in pts[:17]]
        return K
    for p in sorted(glob.glob(os.path.join(REPO, "data", "Validate", "*", "**", "*.xml"), recursive=True)):
        parts = os.path.relpath(p, os.path.join(REPO, "data", "Validate")).split(os.sep)
        K = raw_xml(p)
        for s in range(0, len(K) - W + 1, STRIDE):
            X.append(T.live_prep(K[s:s + W])); Y.append(0 if parts[0] == "OOD" else cls_of(parts[1]))
    z = np.load(os.path.join(T.OUT, "exp_FH_all", "windows.npz"))
    r0 = np.random.default_rng(0)
    for name, cap in (("sim_attack", 3000), ("sim_normal", 3000), ("real", None), ("real_normal", None)):
        i = np.flatnonzero((z["source"] == name) & (z["split"] == "train"))
        if cap and len(i) > cap:
            i = np.sort(r0.choice(i, cap, replace=False))
        X += list(z["X"][i]); Y += z["cls"][i].tolist()
X, Y = np.stack(X).astype(np.float32), np.array(Y, np.int64)
print(f"[{args.train}] train windows per class {dict(zip(T.CLASSES, np.bincount(Y, minlength=NC).tolist()))}", flush=True)


# ---------------- replay of the takes not trained on ----------------
def replay(which):
    """Windows over whole takes: (take, start, kind, strike id, label). kind attack = overlaps an attack clip by >= 15
    frames (rule of T.take_replay), normal = touches none, edge otherwise (not scored)."""
    out, Ws = [], []
    by_take = {}
    for r in rows:
        if r["role"] == which:
            by_take.setdefault((r["day"], r["take"]), []).append(r)
    for key, rs in sorted(by_take.items()):
        K = takes[key]
        att = [r for r in rs if r["category"] == "attack" or (r["category"] == "confuser" and "strike" in r["label"])]
        drop = [r for r in rows if r["role"] == "drop" and (r["day"], r["take"]) == key]
        lo, hi = min(r["f0"] for r in rs), max(r["f1"] for r in rs)
        for s in range(max(lo, 0), min(hi, len(K)) - W + 1, STRIDE):
            if not ok(K, s) or any(s < r["f1"] and s + W > r["f0"] for r in drop):
                continue
            ov = max(((min(s + W, r["f1"]) - max(s, r["f0"]), r["day"] + r["take"] + r["id"], r["label"]) for r in att), default=(0, None, None))
            if ov[0] >= 15:
                kind, sid, lab = "attack", ov[1], ov[2]
            elif near_strike(key, s):
                continue
            else:
                mid = s + W // 2
                lab = next((r["label"] for r in rs if r["f0"] <= mid < r["f1"]), "unlabelled")
                kind, sid = "normal", None
            out.append((key[1], s, kind, sid, lab)); Ws.append(T.live_prep(K[s:s + W]))
    return out, np.stack(Ws).astype(np.float32)


SETS = {k: replay(k) for k in ("cal", "test14", "test31", "confuser")}
for k, (m, _) in SETS.items():
    print(f"{k}: {sum(x[2] == 'attack' for x in m)} attack windows ({len({x[3] for x in m if x[3]})} strikes), "
          f"{sum(x[2] == 'normal' for x in m)} normal windows ({sum(x[2] == 'normal' for x in m) * STRIDE / 1800:.1f} min)", flush=True)


# ---------------- models (fit, run, reference as poc_relabel3.py) ----------------
def fit(kind, seed, teacher=None):
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
    return m.eval()


@torch.no_grad()
def run(m, kind, Wn):
    L, Fs = [], []
    for i in range(0, len(Wn), 96):
        n, v = T.normalize_batch(torch.from_numpy(np.asarray(Wn[i:i + 96], np.float32)).to(dev))
        l, f = m(T.model_input(kind, n, v), return_features=True)
        L.append(l.float().reshape(-1).cpu()); Fs.append(f.float().cpu())
    return torch.cat(L), torch.cat(Fs)


def reference(f, y, eps=1e-2):  # background = not-attack windows only (POC findings, section 7)
    f0 = f[y == 0]; mu_0 = f0.mean(0, keepdim=True); d0 = f0 - mu_0
    eye = torch.eye(f.shape[1]); mus, inv = [], []
    for c in range(1, NC):
        fc = f[y == c]
        if len(fc) < 2:
            continue
        mus.append(fc.mean(0)); dc = fc - mus[-1]
        inv.append(torch.linalg.pinv(dc.T @ dc / (len(fc) - 1) + eps * eye))
    return dict(mu_0=mu_0, mu_c=torch.stack(mus), inv_Sigma_0=torch.linalg.pinv(d0.T @ d0 / (len(f0) - 1) + eps * eye), inv_Sigma_c=torch.stack(inv))


# ---------------- scoring ----------------
def events(meta, alarm):
    """False-alarm events: alarmed normal windows of one take closer than the 5 s alert hold count once."""
    n, last = 0, {}
    for (take, s, kind, _, _), a in zip(meta, alarm):
        if kind == "normal" and a:
            n += s - last.get(take, -10 ** 9) > 150
            last[take] = s
    return n


def strikes(meta, score):
    by = {}
    for (_, _, kind, sid, lab), q in zip(meta, score):
        if kind == "attack":
            by[sid] = (max(by.get(sid, (-1e18,))[0], q), lab)
    return by


def measure(meta, score, bounds):
    """score: one number per window, higher = attack. bounds: {name: threshold}."""
    kind = np.array([m[2] for m in meta])
    by = strikes(meta, score)
    smax = np.array([v[0] for v in by.values()])
    mins = (kind == "normal").sum() * STRIDE / 1800
    out = dict(strikes=len(by), normal_min=round(float(mins), 2),
               auc_window=float(roc_auc_score(kind == "attack", score)) if len(by) and (kind == "normal").any() else None)
    # pauses inside attack takes were cut automatically and may hide missed strikes: second view without them
    pure = [m for m in meta if m[2] == "attack" or m[4] != "between_strikes"]
    psc = np.array([q for m, q in zip(meta, score) if m[2] == "attack" or m[4] != "between_strikes"])
    pk = np.array([m[2] for m in pure])
    pmin = (pk == "normal").sum() * STRIDE / 1800
    out["pure_min"] = round(float(pmin), 2)
    out["auc_pure"] = float(roc_auc_score(pk == "attack", psc)) if len(by) and (pk == "normal").any() else float("nan")
    for name, thr in bounds.items():
        ev = events(meta, score >= thr)
        out[name] = dict(caught=int((smax >= thr).sum()), fa_events=ev, fa_per_min=round(ev / max(mins, 1e-9), 2),
                         fa_pure=events(pure, psc >= thr),
                         caught_by_label=dict(Counter(v[1] for v in by.values() if v[0] >= thr)),
                         fa_labels=dict(Counter(m[4] for m, q in zip(meta, score) if m[2] == "normal" and q >= thr)))
    for k in (0, 2):  # best threshold with at most k false-alarm events on this same set: an upper limit, tuned on test
        best = 0
        for thr in np.unique(np.r_[score[kind == "normal"] + 1e-6, smax]):
            if events(meta, score >= thr) <= k:
                best = max(best, int((smax >= thr).sum()))
        out[f"oracle_fa{k}"] = best
    return out


def bounds_from_cal(score_cal):
    meta = SETS["cal"][0]
    kind = np.array([m[2] for m in meta])
    smax = np.array([v[0] for v in strikes(meta, score_cal).values()])
    return {"normal_bound": float(score_cal[kind == "normal"].max() + 1e-6),  # above every calibration normal window
            "normal_q99": float(np.quantile(score_cal[kind == "normal"], 0.99)),
            "attack_bound": float(np.quantile(smax, 0.10))}  # keeps 90 % of the calibration strikes


def report(tag, sc, bounds, results):
    for k in ("test14", "test31", "confuser"):
        results.setdefault(tag, {}).setdefault(k, []).append(measure(SETS[k][0], sc[k], bounds))


results = {}
# ---- shipped models, each with its own shipped thresholds (bound 0) and with the calibration bounds ----
live = T.Cascade("old", dev)
P = {k: live.probs(Wn) for k, (_, Wn) in SETS.items()}
Z = {k: live.probs(Wn, raw=True) for k, (_, Wn) in SETS.items()}
for k in ("test14", "test31", "confuser"):  # the cascade's own decision, as shipped
    results.setdefault("live cascade, shipped thresholds", {}).setdefault(k, []).append(
        measure(SETS[k][0], live.alarm(P[k]).astype(np.float32), {"shipped": 0.5}))
sc = {k: Z[k].mean(1) for k in Z}
report("live models, mean score", sc, bounds_from_cal(sc["cal"]), results)

sim3 = []
for name in ("old", "fixed", "mix"):
    m = T.build("t3").to(dev)
    m.load_state_dict(torch.load(os.path.join(REPO, "weights", f"stgcn_sim3_{name}.pth"), map_location=dev, weights_only=True))
    ref = torch.load(os.path.join(REPO, "weights", f"reference_data_stgcn_sim3_{name}.pt"), map_location="cpu", weights_only=False)
    sim3.append((m.eval(), ref))
sc = {k: np.mean([T.rmd(run(m, "t3", Wn)[1], ref).numpy() - float(ref["optimal_threshold"]) for m, ref in sim3], 0)
      for k, (_, Wn) in SETS.items()}
report("sim3", sc, {"shipped": 0.0, **bounds_from_cal(sc["cal"])}, results)

# ---- retrained three-model recipe ----
for seed in [int(s) for s in args.seeds.split(",")]:
    t3 = fit("t3", seed); t1 = fit("t1", seed); t2 = fit("t2", seed, teacher=t3)
    out, rmd = {k: [] for k in SETS}, {k: [] for k in SETS}
    for kind, m in (("t3", t3), ("t1", t1), ("t2", t2)):
        ref = reference(run(m, kind, X)[1], torch.from_numpy(Y))
        o = {k: run(m, kind, Wn) for k, (_, Wn) in SETS.items()}
        nor = np.array([x[2] == "normal" for x in SETS["cal"][0]])
        for store, vals in ((out, {k: v[0].numpy() for k, v in o.items()}), (rmd, {k: T.rmd(v[1], ref).numpy() for k, v in o.items()})):
            mu, sd = vals["cal"][nor].mean(), vals["cal"][nor].std() + 1e-6  # each model in units of its calibration normals
            for k in SETS:
                store[k].append((vals[k] - mu) / sd)
        if args.save and seed == 1:
            d = os.path.join(T.OUT, f"exp_segments_{args.train}"); os.makedirs(d, exist_ok=True)
            torch.save(m.state_dict(), os.path.join(d, f"{kind}_s1.pth")); torch.save(ref, os.path.join(d, f"{kind}_s1_ref.pt"))
    for tag, store in (("output blend", out), ("RMD blend", rmd)):
        sc = {k: np.mean(store[k], 0) for k in SETS}
        report(f"{args.train}: {tag}", sc, bounds_from_cal(sc["cal"]), results)
        if args.save and seed == 1:  # per-window scores for a closer look
            np.savez(os.path.join(REPO, "results", "stage2_segments", f"scores_{args.train}_{tag.split()[0]}.npz"),
                     **sc, **{k + "_meta": np.array([[str(v) for v in m] for m in SETS[k][0]]) for k in SETS})
    print(f"seed {seed} done", flush=True)

# ---------------- print ----------------
def mean(ms, *path):
    v = []
    for m in ms:
        for p in path:
            m = m[p]
        v.append(m)
    return float(np.mean(v))


for tag, r in results.items():
    print(f"\n== {tag} ({len(r['test14'])} run(s)) ==")
    for k in ("test14", "test31", "confuser"):
        ms = r[k]
        line = f"  {k:8s} strikes {ms[0]['strikes']:3d} normal {ms[0]['normal_min']:5.1f} min | window AUC {mean(ms, 'auc_window'):.3f}, without pauses {mean(ms, 'auc_pure'):.3f} ({ms[0]['pure_min']} min)"
        for b in ("shipped", "normal_bound", "normal_q99", "attack_bound"):
            if b in ms[0]:
                line += f" | {b}: {mean(ms, b, 'caught'):.1f} caught, {mean(ms, b, 'fa_events'):.1f} FA ({mean(ms, b, 'fa_per_min'):.2f}/min; {mean(ms, b, 'fa_pure'):.1f} outside pauses)"
        print(line + f" | best at 0 FA {mean(ms, 'oracle_fa0'):.1f}, at <=2 FA {mean(ms, 'oracle_fa2'):.1f}")
    for b in ("shipped", "normal_bound", "attack_bound"):
        if b in r["test14"][0]:
            c, f = Counter(), Counter()
            for m in r["test14"]:
                c.update(m[b]["caught_by_label"]); f.update(m[b]["fa_labels"])
            n = len(r["test14"])
            print(f"  test14 at {b}: caught by label {({k: round(v / n, 1) for k, v in c.items()})}; normal windows alarmed by label {({k: round(v / n, 1) for k, v in f.items()})}")
os.makedirs(os.path.join(REPO, "results", "stage2_segments"), exist_ok=True)
with open(os.path.join(REPO, "results", "stage2_segments", f"{args.train}{'' if args.seeds == '1,2,3' else '_s' + args.seeds}.json"), "w") as fh:
    json.dump(results, fh, indent=1)
