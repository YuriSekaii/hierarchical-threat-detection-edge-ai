"""
Can one recorded action become another? Generates each real strike of one action from strikes of a different action,
given only where the real hand landed, and measures every other joint against the real strike.

The poses the coupling learns from never contain the target action or held-out takes.
  python scripts/evaluate_motion_edit.py --out results/samples/edit
"""
import argparse
import os
import sys

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [REPO_DIR, os.path.join(REPO_DIR, "scripts")]

import generate_from_edits as M  # noqa: E402
import generate_from_samples as S  # noqa: E402
from src.skeleton_sim import from_samples as G  # noqa: E402
from src.skeleton_sim import motion_edit as E  # noqa: E402

GROUPS = {"hand": [E.R_WR], "elbow": [E.R_EL], "shoulder": [E.R_SH], "free arm": [G.L_SH, G.L_EL, G.L_WR], "head": G.FACE,
          "legs": [G.L_KN, G.R_KN, G.L_AN, G.R_AN]}
# (source take, target take): same direction, a different action. Take 9 is left out: most of it is off the person's spot.
PAIRS = [(2, 6, "chest to low, front"), (4, 8, "chest to low, back"), (3, 7, "chest to low, left"),
         (6, 2, "low to chest, front"), (8, 4, "low to chest, back"), (7, 3, "low to chest, left")]
HALF = 20  # frames compared on each side of the landing
# body fit: the source in the body, unedited. body arm: only the striking arm and trunk may change; body arm only: only the arm
METHODS = ("no edit", "hand only", "geometry", "global", "local", "body fit", "body", "body arm", "body arm only", "floor")


def strikes(t):
    """[(start, land, end)] and the body."""
    t = M.prepare(t)
    return t["strikes"], t["body"]


def segment(body, land):
    a, b = land - HALF, land + HALF + 1
    return body[a:b] if a >= 0 and b <= len(body) else None


def errors(gen, real):
    """Mean distance per joint group, torso lengths."""
    d = np.linalg.norm(gen - real, axis=-1)
    return {g: d[:, j].mean() for g, j in GROUPS.items()}


def ik_two_bone(body, hand, upper, fore):
    """Plain geometry baseline: shoulder fixed, bones as long as in the source pose, elbow on the source's side."""
    shoulder, elbow = E.ARM[hand]
    out = body.copy()
    s, h = body[:, shoulder], body[:, hand]
    d = np.maximum(np.linalg.norm(h - s, axis=1), 1e-6)
    a = np.minimum(d, upper + fore)
    along = np.clip((upper ** 2 - fore ** 2 + a ** 2) / (2 * a), -upper, upper)
    up = np.sqrt(np.maximum(upper ** 2 - along ** 2, 0))
    u = (h - s) / d[:, None]
    n = np.stack([-u[:, 1], u[:, 0]], 1)
    side = np.sign(np.sum((body[:, elbow] - s) * n, axis=1))
    out[:, elbow] = s + u * along[:, None] + n * (up * np.where(side == 0, 1, side))[:, None]
    return out


def body_takes(takes, nums, device):
    """The takes fitted into the 3D body: from the saved fit of the training takes, others fitted now with the same
    bone sizes and camera. Fitting reads only the take's own 2D points."""
    import torch
    from src.skeleton_sim import body_edit as BE
    from src.skeleton_sim import body_model as B
    path = os.path.join(REPO_DIR, "results", "samples", "body_model")
    z = np.load(os.path.join(path, "fit_train_s0.05_t13_b0.1.npz"))
    body = B.Body(os.path.join(path, "Rajagopal2016.osim"))
    shared = dict(log_scale=torch.tensor(z["log_scale"], device=device), face=torch.tensor(z["face"], device=device),
                  tilt=torch.tensor(float(z["tilt"]), device=device), log_dist=torch.tensor(float(np.log(z["dist"])), device=device))
    out = {int(n): BE.BodyTake.from_fit(body, shared, z, i, device) for i, n in enumerate(z["nums"])}
    new = [n for n in nums if n not in out]
    if new:
        obs, w = [], []
        for n in new:
            t = M.prepare(takes[n])
            c = np.load(os.path.join(S.KP, t["name"] + ".npy"))[t["offset"]:t["offset"] + len(t["xy"]), :, 2].astype(np.float32)
            obs.append(t["body"].astype(np.float32))
            w.append(np.where(t["missing"][:, None], 0, c))
        f, _, _ = B.fit_takes(body, obs, w, [0.0] * len(new), shared=shared, device=device)
        with torch.no_grad():
            angles, yaw, lean = f.angles().cpu().numpy(), f.yaw.cpu().numpy(), f.lean.cpu().numpy()
        ends = np.cumsum([len(o) for o in obs])
        for i, (n, a, b) in enumerate(zip(new, np.r_[0, ends[:-1]], ends)):
            out[n] = BE.BodyTake(body, shared, angles[a:b], lean[a:b], yaw[a:b], float(f.logk[i]), device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/samples/edit")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--body", type=int, default=0, help="also the editor inside the 3D body (src/skeleton_sim/body_edit.py)")
    args = ap.parse_args()
    takes = {t["num"]: t for t in S.load_takes()}
    if args.body:
        import torch
        from src.skeleton_sim import body_edit as BE
        bodies = body_takes(takes, {s for s, _, _ in PAIRS}, args.device)
        fitted = [n for n in bodies if n not in S.HELD_OUT]
    lab = np.load(os.path.join(S.T.LAB, "analysis", "001_floor_markers_142701_analysis.npz"))["kpts"][..., :2]
    lab = S.T.interpolate_missing_joints(lab[:len(lab) // 2].astype(np.float32))
    results = {}
    for src, dst, name in PAIRS:
        ts, td = takes[src], takes[dst]
        pool = [E.to_body(t["xy"])[0][::2] for n, t in takes.items()
                if n not in S.HELD_OUT and (t["kind"] == ts["kind"] or "thrust" not in t["kind"])] + [E.to_body(lab)[0][::2]]
        poses = np.concatenate(pool)
        local, glob = E.PoseCoupling(poses, k=300, device=args.device), E.PoseCoupling(poses, k=len(poses), device=args.device)
        ev_s, body_s = strikes(ts)
        ev_d, body_d = strikes(td)
        limits = E.arm_limits(body_s, ts["hand"])
        rows = {m: [] for m in METHODS}
        if args.body:
            bt = bodies[src]
            bcoupling = BE.StateCoupling(np.concatenate([bodies[n].states[::2] for n in fitted
                                                          if takes[n]["kind"] == ts["kind"] or "thrust" not in takes[n]["kind"]]), device=args.device)
            with torch.no_grad():
                body_fit = bt.render(bt.states).cpu().numpy()
        for a, land, b in ev_s:
            for _, land_d, _ in ev_d:
                real = segment(body_d, land_d)
                if real is None or segment(body_s, land) is None:
                    continue
                target = body_d[land_d, td["hand"]]
                delta = E.strike_weight(body_s[:, ts["hand"]], a, land, b)[:, None] * (target - body_s[land, ts["hand"]])
                moved = body_s.copy()
                moved[:, ts["hand"]] += delta
                gens = {"no edit": body_s, "hand only": moved,
                        "geometry": ik_two_bone(moved, ts["hand"], *limits),
                        "global": E.reperform(body_s, ts["hand"], a, land, b, target, glob, limits),
                        "local": E.reperform(body_s, ts["hand"], a, land, b, target, local, limits)}
                if args.body:
                    side = "r" if ts["hand"] == E.R_WR else "l"
                    for m, free in (("body", None), ("body arm", BE.free_dims(bt, side)), ("body arm only", BE.free_dims(bt, side, trunk=False))):
                        new = BE.reperform(bt, ts["hand"], a, land, b, target, bcoupling, free=free)  # needs gradients: not under no_grad
                        with torch.no_grad():
                            gens[m] = bt.render(new).cpu().numpy()
                    gens["body fit"] = body_fit
                for m, g in gens.items():
                    rows[m].append(errors(segment(g, land), real))
        for i, (_, land_i, _) in enumerate(ev_d):  # two real strikes of the target action: the best anyone can do
            for j, (_, land_j, _) in enumerate(ev_d):
                x, y = segment(body_d, land_i), segment(body_d, land_j)
                if i != j and x is not None and y is not None:
                    rows["floor"].append(errors(x, y))
        results[name] = {m: {g: float(np.mean([r[g] for r in v])) for g in GROUPS} for m, v in rows.items() if v}
        print(f"\n{name}: take {src} -> {dst}, {len(rows['local'])} pairs, library {len(poses)} poses", flush=True)
        print(f"  {'':14s}" + "".join(f"{g:>10s}" for g in GROUPS))
        for m, v in results[name].items():
            print(f"  {m:14s}" + "".join(f"{v[g]:10.3f}" for g in GROUPS))
    print("\nMean over all pairs")
    print(f"  {'':14s}" + "".join(f"{g:>10s}" for g in GROUPS))
    for m in METHODS:
        if not any(m in r for r in results.values()):
            continue
        print(f"  {m:14s}" + "".join(f"{np.mean([r[m][g] for r in results.values() if m in r]):10.3f}" for g in GROUPS))
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "scores.npy"), results, allow_pickle=True)


if __name__ == "__main__":
    main()
