"""
Builds Stage 2 window caches with the motion editor (src/skeleton_sim/motion_edit.py).

Attacks: a recorded strike re-performed to a new landing point. Gestures: real pauses and walking, with the hand and
other joints sent through several points (to the face, head, chest, floor). Real walk-ins and desk visits are also normals.
  --attacks edit|raw|view  --gestures 0|1  --hide-low 0|1 (no recorded low thrust in the samples: the missing-action test)
  --attacks view: recorded strikes plus the same strikes from other directions; --turn-only 1 keeps to 180 degree turns

  python scripts/generate_from_edits.py --base results/train_sim/exp_H_mix --out results/train_sim/exp_E_full --attacks edit --gestures 1
  add --strike-gestures 0.5 for half of the gestures built on real strikes
Train with HTD_TRAIN_OUT=train_sim/exp_E_full and --mix real:3,sim:12,normal:6,real_normal:7,gen:8,gen_normal:6
Also writes results/samples/heldout2.npz: held-out and low-thrust windows, labelled with the corrected strike finder.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [REPO_DIR, os.path.join(REPO_DIR, "scripts")]

import generate_from_samples as S  # noqa: E402
import train_stage2_sim as T  # noqa: E402
from src.skeleton_sim import from_samples as G  # noqa: E402
from src.skeleton_sim import motion_edit as E  # noqa: E402

LOW = {6, 7, 8, 9, 28}
THRUST = 1
GESTURES = ("face", "head", "chest", "shoulder", "inspect", "point", "overhead", "floor")
DIRECTIONS = ("front", "right", "back", "left")
TILT = 0.2  # fitted by building recorded directions from two others, training takes only (results/samples/edit/camera_fit.txt)


def full_take(t):
    """Everything the person did in the take, including walking in and out and desk visits: joints, the first frame,
    and which frames nobody was detected in (their joints are interpolated, not seen)."""
    k = np.load(os.path.join(S.KP, t["name"] + ".npy"))
    here = np.where(k[..., 2].sum(-1) > 0)[0]
    k = k[here[0]:here[-1] + 1]
    return T.interpolate_missing_joints(k[..., :2].astype(np.float32)), int(here[0]), k[..., 2].sum(-1) == 0


def prepare(t):
    t["body"], t["root"], t["torso"] = E.to_body(t["xy"])
    t["trips"] = E.find_strikes(t["body"], t["root"] / t["torso"], t["hand"], "walking" in t["kind"])
    t["any_trips"] = E.find_strikes(t["body"], t["root"] / t["torso"], t["hand"], True)  # also those rejected as too long or off the spot
    # From the side, a thrust that does not end in front of the person is its wind-up. Overhand strikes end anywhere.
    facing = {"left": -1, "right": 1}.get(t["dir"]) if t["cls"] == THRUST else None
    t["strikes"] = [e for e in t["trips"] if not facing or t["body"][e[1], t["hand"], 0] * facing > 0.3]
    t["limits"] = E.arm_limits(t["body"], t["hand"])
    return t


def walk_in_windows(t):
    """Stride-10 windows of the walk to the mark and the walk back, including desk visits. Only outside the first and
    last hand trip of any kind: a trip the finder rejected may still be a strike, so it is never called normal. For the
    same reason nothing comes from a take with no strike found, or from after the last trip of a take cut short, whose
    later strikes were never searched. Windows with over 10 undetected frames are skipped, as in label_windows."""
    if not t["strikes"]:
        return []
    xy, first, missing = full_take(t)
    lo, hi = t["any_trips"][0][0] + t["offset"] - first, t["any_trips"][-1][2] + t["offset"] - first
    return [xy[s:s + 50] for s in range(0, len(xy) - 49, T.STRIDE)
            if (s + 50 < lo - 15 or (s > hi + 15 and not t["cut"])) and missing[s:s + 50].sum() <= 10]


def clip_around(t, ev, length, rng):
    """Slice of a take holding the whole strike, and where the wanted window sits in it."""
    a, land, b = ev
    s = land - int(length * rng.uniform(0.2, 0.8))
    if s < 0 or s + length >= len(t["body"]):
        return None
    lo, hi = min(s, a), min(max(s + length + 1, b + 1), len(t["body"]))
    return lo, hi, s - lo


def to_window(t, gen, lo, offset, length, rng, legs, p_legs=0.25, p_view=0.7):
    """Back to pixels on the take's own hip path, then the usual 50-frame window and variation."""
    seq = gen * t["torso"] + t["root"][lo:lo + len(gen), None]
    return G.vary(G.time_warp(seq.astype(np.float32), offset, length, rng), rng, legs=legs[rng.integers(len(legs))], p_legs=p_legs, p_view=p_view)


def side_of(a, b):
    """+1 if direction b is 90 degrees on from a in the sense front -> right -> back -> left, otherwise -1."""
    return 1 if (DIRECTIONS.index(b) - DIRECTIONS.index(a)) % 4 == 1 else -1


def view_partner(t, ev, lo, hi, by_kind, rng):
    """The same action recorded 90 degrees away, aligned to this strike. Height does not change with the view, so of the
    other take's strikes, one with the closest heights is the nearest thing to the same motion seen from the side."""
    near = [o for o in by_kind.get(t["kind"], []) if t["dir"] and o["dir"] in S.AROUND[t["dir"]] and o["strikes"]]
    if not near:
        return None
    o = near[rng.integers(len(near))]
    cand = [E.align_strike(o["body"], e, ev, np.arange(lo, hi)) for e in o["strikes"]]
    err = [np.abs(c[..., 1] - t["body"][lo:hi, :, 1]).mean() for c in cand]
    return cand[rng.choice(np.argsort(err)[:3])], side_of(t["dir"], o["dir"])


def viewed_attack(t, ev, rng, legs, by_kind, turn_only=False):
    """A recorded strike from another direction. Two recorded directions 90 degrees apart give x from any direction;
    with only one, a turn of 180 degrees still works: x is negated and joint names stay, which mirroring cannot do."""
    length = int(50 * rng.uniform(0.75, 1.35))
    c = clip_around(t, ev, length, rng)
    if c is None:
        return None
    lo, hi, offset = c
    body = t["body"][lo:hi]
    partner = None if turn_only else view_partner(t, ev, lo, hi, by_kind, rng)
    if partner is not None:
        gen = E.turned_view(body, partner[0], partner[1], rng.uniform(-np.pi, np.pi), TILT)
    else:
        gen = G.view_between(body, body, np.pi) if rng.random() < 0.5 else body
    return to_window(t, gen, lo, offset, length, rng, legs, p_legs=0.0, p_view=0.0)


def viewed_normal(w, rng, pauses=None):
    """The same steps for a normal window, so that looking turned carries no label. A pause mixes with a pause from a
    neighbouring direction; other windows can only turn, as their walking direction is unknown."""
    if pauses:
        other, side = pauses[rng.integers(len(pauses))]
        hip_a, hip_b = G.centre(w, G.L_HIP, G.R_HIP)[:, None], G.centre(other, G.L_HIP, G.R_HIP)[:, None]
        seq = (hip_a + E.turned_view(w - hip_a, other - hip_b, side, rng.uniform(-np.pi, np.pi), TILT)).astype(np.float32)
    else:
        seq = G.view_between(w, w, np.pi) if rng.random() < 0.5 else w
    length = int(50 * rng.uniform(0.75, 1.0))
    return G.vary(G.time_warp(seq, int(rng.integers(0, 50 - length + 1)), length, rng), rng, p_legs=0.0, p_view=0.0)


def edited_attack(t, ev, coupling, rng, legs):
    length = int(50 * rng.uniform(0.75, 1.35))
    c = clip_around(t, ev, length, rng)
    if c is None:
        return None
    lo, hi, offset = c
    a, land, b = ev[0] - lo, ev[1] - lo, min(ev[2] - lo, hi - lo - 1)
    body = t["body"][lo:hi]
    target = strike_target(t, body, land, rng)
    return to_window(t, E.reperform(body, t["hand"], a, land, b, target, coupling, t["limits"]), lo, offset, length, rng, legs)


def strike_target(t, body, land, rng):
    h = body[land, t["hand"]]
    if t["cls"] == THRUST:  # any height from hip to above the head, at about the same reach
        target = (h[0] * rng.uniform(0.8, 1.1), rng.uniform(-1.6, 0.1))
    else:
        target = h + rng.uniform(-0.5, 0.5, 2)
    return E.reachable(body[land], t["hand"], target, *t["limits"])


def random_turn(rng, p_turn):
    return rng.uniform(-np.pi, np.pi) if rng.random() < p_turn else 0.0


BODY_MODEL = os.path.join(REPO_DIR, "results", "samples", "body_model")


def load_clips(name, takes, body, shared, device):
    """Clips in the 3D body fitted to two views (results/samples/tools/body_clips.py): their depth was seen, so they can
    be shown from any direction. Each: take, BodyTake, unturned render, noise (recorded minus render), strike, first frame."""
    import torch
    from src.skeleton_sim import body_edit as BE
    z = np.load(os.path.join(BODY_MODEL, name))
    out = []
    for i, n in enumerate(z["take"]):
        bt = BE.BodyTake(body, shared, z["angles"][i], z["lean"][i], z["yaw"][i], z["logk"][i], device)
        with torch.no_grad():
            fit = bt.render(bt.states).cpu().numpy()
        noise = z["residual"][i]
        # turned, only the tracker's jitter (fast) still applies; the slow rest is where the body missed this view
        jitter = noise - gaussian_filter1d(noise, 2.0, axis=0)
        out.append(dict(t=takes[int(n)], bt=bt, fit=fit, noise=noise, jitter=jitter, ev=tuple(int(v) for v in z["ev"][i]), first=int(z["first"][i])))
    return out


def clip_noise(clip, turn):
    return clip["noise"] if turn == 0 else clip["jitter"]


def body_attacks(clip, coupling, rng, legs, n, free, p_turn, gesture=False):
    """n re-performances of one strike clip inside the 3D body (src/skeleton_sim/body_edit.py), each to its own landing
    point, solved together, each seen from a random direction. gesture: the hand goes to the person's own body instead
    (face, head, chest, ...), at the strike's own speed: a normal that moves as fast as an attack."""
    import torch
    from src.skeleton_sim import body_edit as BE
    t, bt, (a, land, b) = clip["t"], clip["bt"], clip["ev"]
    recorded = clip["fit"] + clip["noise"]
    if gesture:
        side = float(np.sign(recorded[a, t["hand"], 0] - recorded[a, 0, 0]) or 1.0)
        # floor gestures also move the shoulders (a shift the batch solve does not take); without it they are skipped
        kinds = [k for k in GESTURES if k != "floor"]
        targets = np.stack([gesture_target(kinds[rng.integers(len(kinds))], recorded[land], t["hand"], side, rng, t["limits"])[0]
                            for _ in range(n)]).astype(np.float32)
    else:
        targets = np.stack([strike_target(t, recorded, land, rng) for _ in range(n)]).astype(np.float32)
    delta = BE.solve_landing(bt, np.repeat(bt.states[land][None], n, 0), [t["hand"]], targets[:, None], coupling, free=free)
    w = E.strike_weight(clip["fit"][:, t["hand"]], a, land, b)
    out = []
    for d in delta:
        length = min(int(50 * rng.uniform(0.75, 1.35)), len(w) - 1)
        s = int(np.clip(land - int(length * rng.uniform(0.2, 0.8)), 0, len(w) - 1 - length))
        turn = random_turn(rng, p_turn)
        with torch.no_grad():
            img = bt.render(bt.states + w[:, None] * d[None], turn).cpu().numpy()
        out.append(to_window(t, img + clip_noise(clip, turn), clip["first"], s, length, rng, legs, p_legs=0.0, p_view=0.0))
    return out


def body_normal(clip, rng, p_turn):
    """A real pause in the body (two views), seen from a random direction as body_attacks shows strikes, so that being
    turned carries no label."""
    import torch
    t = clip["t"]
    turn = random_turn(rng, p_turn)
    with torch.no_grad():
        img = clip["bt"].render(clip["bt"].states, turn).cpu().numpy()
    s = clip["first"]
    seq = ((img + clip_noise(clip, turn)) * t["torso"] + t["root"][s:s + len(img), None]).astype(np.float32)
    length = int(50 * rng.uniform(0.75, 1.0))
    return G.vary(G.time_warp(seq, int(rng.integers(0, len(img) - length + 1)), length, rng), rng, p_legs=0.0, p_view=0.0)


def gesture_target(kind, pose, hand, side, rng, limits):
    """Where the hand goes in a gesture, and how other joints shift (a bend). Relative to the person's pose at that frame."""
    sh = G.centre(pose[None], G.L_SH, G.R_SH)[0]
    other = E.L_WR if hand == E.R_WR else E.R_WR
    shift = {}
    if kind == "face":
        target = pose[0] + (side * 0.1, 0.1)
    elif kind == "head":
        target = pose[0] + (side * rng.uniform(0, 0.25), rng.uniform(-0.3, -0.1))
    elif kind == "chest":
        target = sh + (rng.uniform(-0.2, 0.2), rng.uniform(0.15, 0.45))
    elif kind == "shoulder":
        target = pose[E.ARM[other][0]] + (0, 0.05)
    elif kind == "inspect":
        target = sh + (rng.uniform(-0.3, 0.3), rng.uniform(0.2, 0.7))
    elif kind == "point":
        target = sh + (side * rng.uniform(0.9, 1.4), rng.uniform(-0.1, 0.5))
    elif kind == "overhead":
        target = pose[0] + (side * rng.uniform(0.1, 0.4), rng.uniform(-0.8, -0.5))
    else:  # floor: the shoulders and head drop with the hand
        drop = np.array((side * rng.uniform(0.1, 0.5), rng.uniform(0.35, 0.7)))
        target = np.array((side * rng.uniform(0.1, 0.6), rng.uniform(0.3, 1.0)))
        shift = {j: drop for j in (G.L_SH, G.R_SH, 0)}
    if not shift:
        target = E.reachable(pose, hand, target, *limits)
    return target, shift


def strike_gesture(t, ev, coupling, rng, legs, limits, kind=None):
    """A real strike with the hand sent to the person's own body instead: the torso, head and feet keep the real motion
    of the strike, and only where the hand ends up changes."""
    length = int(50 * rng.uniform(0.75, 1.35))
    c = clip_around(t, ev, length, rng)
    if c is None:
        return None
    lo, hi, offset = c
    a, land, b = ev[0] - lo, ev[1] - lo, min(ev[2] - lo, hi - lo - 1)
    body = t["body"][lo:hi]
    kind = kind or GESTURES[rng.integers(len(GESTURES))]
    side = float(np.sign(body[a, t["hand"], 0] - body[a, 0, 0]) or 1.0)
    target, shift = gesture_target(kind, body[land], t["hand"], side, rng, limits)
    return to_window(t, E.reperform(body, t["hand"], a, land, b, target, coupling, limits, shift or None), lo, offset, length, rng, legs, p_legs=0.6)


def gesture_keys(kind, body, hand, side, t0, t1, t2, t3, rng, limits):
    """The points a gesture passes through, on each joint it controls."""
    other = E.L_WR if hand == E.R_WR else E.R_WR
    target, shift = gesture_target(kind, body[t1], hand, side, rng, limits)
    rest0, rest3 = body[t0, hand], body[t3, hand]
    bow = np.array((side * rng.uniform(0.05, 0.25), 0.0))  # hands travel around the body, not through it
    hand_keys = [(t0, rest0), ((t0 + t1) // 2, (rest0 + target) / 2 + bow), (t1, target)]
    for f in np.sort(rng.integers(t1 + 1, max(t2, t1 + 2), rng.integers(0, 4))):  # a fidget while the hand stays there
        hand_keys.append((int(f), target + rng.normal(0, 0.12, 2)))
    hand_keys += [(t2, target + rng.normal(0, 0.05, 2)), (t3, rest3)]
    # real gestures often go twice: scratch, drop, scratch again
    t4 = t3 + int(rng.integers(3, 12))
    t5, t6 = t4 + int(rng.integers(5, 20)), t4 + int(rng.integers(26, 40))
    t7 = t6 + int(rng.integers(5, 20))
    if rng.random() < 0.5 and t7 < len(body):
        hand_keys += [(t4, body[t4, hand]), (t5, target + rng.normal(0, 0.15, 2)), (t6, target + rng.normal(0, 0.15, 2)), (t7, body[t7, hand])]
    seen = {}
    for f, p in hand_keys:
        seen[f] = p
    keys = {hand: list(seen.items())}
    for j, v in shift.items():
        keys[j] = [(t0, body[t0, j]), (t1, body[t1, j] + v), (t2, body[t2, j] + v), (t3, body[t3, j])]
    if kind == "overhead" and rng.random() < 0.5:  # both arms up
        free = body[t1, 0] + (-side * rng.uniform(0.1, 0.4), rng.uniform(-0.8, -0.5))
        keys[other] = [(t0, body[t0, other]), (t1, free), (t2, free), (t3, body[t3, other])]
    return keys


def is_upright(window):
    """A gesture starts from a person standing or walking. A window already bent over a desk would bend twice."""
    w = np.asarray(window)
    hip, sh = G.centre(w, G.L_HIP, G.R_HIP), G.centre(w, G.L_SH, G.R_SH)
    d = np.median(sh - hip, axis=0)
    leg = np.median(np.linalg.norm((w[:, G.L_AN] + w[:, G.R_AN]) / 2 - hip, axis=1))
    return d[1] < 0 and abs(d[0]) < 0.45 * abs(d[1]) and leg < 2.2 * np.linalg.norm(d)


def gesture(window, coupling, rng, legs, limits, kind=None):
    """A real normal window with a gesture performed through it. The window is mirrored in time on both sides, so a
    gesture can start or end outside the 50 frames, as it would in a live clip."""
    body, root, torso = E.to_body(np.asarray(window))
    pad = 40
    long_body = np.concatenate([body[pad:0:-1], body, body[-2:-pad - 2:-1]])
    t1 = pad + int(rng.integers(-15, 45))
    # Real gestures of take 002 move about as fast as strikes, so the timing overlaps theirs (fitted once, on that take).
    t0, t2 = max(t1 - int(rng.integers(5, 20)), 0), t1 + int(rng.integers(2, 25))
    t3 = min(t2 + int(rng.integers(5, 20)), len(long_body) - 1)
    t2 = min(t2, t3 - 1)
    kind = kind or GESTURES[rng.integers(len(GESTURES))]
    hand = E.R_WR if rng.random() < 0.7 else E.L_WR
    side = -1.0 if hand == E.R_WR else 1.0  # image left is the person's right when facing the camera
    if rng.random() < 0.3:
        side = -side
    keys = gesture_keys(kind, long_body, hand, side, t0, t1, t2, t3, rng, limits)
    gen = E.follow_keys(long_body, keys, coupling, hand, limits)[pad:pad + 50]
    return G.vary((gen * torso + root[:, None]).astype(np.float32), rng, legs=legs[rng.integers(len(legs))], p_legs=0.6)


def relabelled(t):
    """The take with the corrected strikes as its events, for S.label_windows."""
    return dict(t, events=t["strikes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attacks", choices=["edit", "raw", "view", "body"], required=True)
    ap.add_argument("--body-free", choices=["all", "arm", "armonly"], default="arm",
                    help="with --attacks body: which body dims a strike may change (arm: the striking arm and trunk)")
    ap.add_argument("--p-turn", type=float, default=0.7, help="with --attacks body: share of windows seen from a random direction")
    ap.add_argument("--body-gestures", type=int, default=0,
                    help="with --attacks body: normals made from the same strikes with the hand sent to the person's own body")
    ap.add_argument("--turn-only", type=int, default=0, help="with --attacks view: only 180 degree turns, no mix of two takes")
    ap.add_argument("--view-normals", type=int, default=6000)
    ap.add_argument("--gestures", type=int, default=0)
    ap.add_argument("--hide-low", type=int, default=0)
    ap.add_argument("--strike-gestures", type=float, default=0.0, help="share of gestures built on real strikes, not on pauses")
    ap.add_argument("--attacks-per-class", type=int, default=3000)
    ap.add_argument("--gesture-windows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    takes = [prepare(t) for t in S.load_takes()]
    hidden = LOW if args.hide_low else set()
    train = [t for t in takes if t["num"] not in S.HELD_OUT and t["num"] not in hidden]
    print(f"{len(train)} training takes, {sum(len(t['strikes']) for t in train)} strikes")

    X, take, strike, cls = [], [], [], []
    for t in takes:
        if t["num"] in S.HELD_OUT | LOW:  # low thrust takes are scored in the missing-action test
            for s, i in S.label_windows(relabelled(t)):
                X.append(T.live_prep(t["xy"][s:s + 50]))
                take.append(t["num"])
                strike.append(i)
                cls.append(t["cls"])
    path = os.path.join(REPO_DIR, "results", "samples", "heldout2.npz")
    tmp = f"{path}.{os.getpid()}.npz"  # parallel builds write this too, and scorers read it
    np.savez(tmp, X=np.stack(X).astype(np.float32), take=np.array(take), strike=np.array(strike), cls=np.array(cls))
    for attempt in range(10):  # Windows refuses the replace while another build or a scorer has the file open
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 9 and os.path.exists(path):  # every build writes the same windows, so the existing file is fine
                os.remove(tmp)
                break
            if attempt == 9:
                raise
            time.sleep(1 + attempt)
    print(f"scored windows: {len(X)}, attack {sum(i >= 0 for i in strike)}")

    lab = np.load(os.path.join(T.LAB, "analysis", "001_floor_markers_142701_analysis.npz"))["kpts"][..., :2]
    lab = T.interpolate_missing_joints(lab[:len(lab) // 2].astype(np.float32))
    poses = [E.to_body(lab)[0][::2]]
    for t in train:  # only frames the person was seen in
        xy, _, missing = full_take(t)
        poses.append(E.to_body(xy)[0][~missing][::2])
    coupling = E.PoseCoupling(np.concatenate(poses), device=args.device)

    z = np.load(os.path.join(REPO_DIR, args.base, "windows.npz"))
    new_X, new_cls, new_src, new_grp = [], [], [], []

    def add(w, c, source, group):
        new_X.append(T.live_prep(w))
        new_cls.append(c)
        new_src.append(source)
        new_grp.append(group)

    # real normals: pauses, plus walking in and out and desk visits around the strikes
    real_normals = [t["xy"][s:s + 50] for t in train for s, i in S.label_windows(relabelled(t)) if i < 0]
    for t in train:
        real_normals += walk_in_windows(t)
    for n, w in enumerate(real_normals):
        add(w, 0, "gen_normal", f"edit/real_normal/{n}")
    legs = [np.asarray(w, np.float32) for w in real_normals]
    print(f"real normal windows: {len(real_normals)}", flush=True)
    by_kind = {}
    for t in train:
        by_kind.setdefault(t["kind"], []).append(t)

    if args.attacks == "body":
        import torch
        import evaluate_motion_edit as V
        from src.skeleton_sim import body_edit as BE
        names = {t["num"] for t in train}
        bodies = {n: b for n, b in V.body_takes({}, [], args.device).items() if n in names}  # the saved fit of training takes
        any_take = next(iter(bodies.values()))
        by_num = {t["num"]: t for t in train}
        clips = [c for c in load_clips("clips_two_view.npz", by_num, any_take.body, any_take.shared, args.device) if c["t"]["num"] in names]
        still = [c for c in load_clips("pauses_two_view.npz", by_num, any_take.body, any_take.shared, args.device) if c["t"]["num"] in names]
        # the coupling learns from the single-view fits of whole takes: turning is not used there, only nearness of poses
        bcoupling = BE.StateCoupling(np.concatenate([b.states[::2] for b in bodies.values()]), device=args.device)
        free = {h: None if args.body_free == "all" else BE.free_dims(any_take, side, trunk=args.body_free == "arm")
                for h, side in ((E.R_WR, "r"), (E.L_WR, "l"))}
        print(f"body: {len(clips)} two-view strikes ({sum(len(t['strikes']) for t in train)} in all), {len(still)} pauses, "
              f"{len(bcoupling.states)} states", flush=True)

    for c in (1, 2, 3, 4):
        pool = [t for t in train if t["cls"] == c and t["strikes"]]
        made = 0
        if args.attacks in ("raw", "view"):
            for t in pool:
                for s, i in S.label_windows(relabelled(t)):
                    if i >= 0:
                        add(t["xy"][s:s + 50], c, "gen_attack", f"edit/{t['num']}/{s}")
                        made += 1
            viewed = 0
            while args.attacks == "view" and viewed < args.attacks_per_class:
                t = pool[rng.integers(len(pool))]
                w = viewed_attack(t, t["strikes"][rng.integers(len(t["strikes"]))], rng, legs, by_kind, bool(args.turn_only))
                if w is not None:
                    add(w, c, "gen_attack", f"view/{t['num']}/{viewed}")
                    viewed += 1
        elif args.attacks == "body":
            mine = [k for k in clips if k["t"]["cls"] == c]
            per = -(-args.attacks_per_class // len(mine))  # every clip about equally often
            for k in rng.permutation(len(mine)):
                n = min(per, args.attacks_per_class - made)
                if n <= 0:
                    break
                for w in body_attacks(mine[k], bcoupling, rng, legs, n, free[mine[k]["t"]["hand"]], args.p_turn):
                    add(w, c, "gen_attack", f"body/{mine[k]['t']['num']}/{made}")
                    made += 1
            pool = sorted({k["t"]["num"] for k in mine})
        else:
            while made < args.attacks_per_class:
                t = pool[rng.integers(len(pool))]
                w = edited_attack(t, t["strikes"][rng.integers(len(t["strikes"]))], coupling, rng, legs)
                if w is not None:
                    add(w, c, "gen_attack", f"edit/{t['num']}/{made}")
                    made += 1
        print(f"  {T.CLASSES[c]}: {made} windows from {len(pool)} takes", flush=True)

    if args.attacks == "view":
        pauses = {t["num"]: [np.asarray(t["xy"][s:s + 50], np.float32) for s, i in S.label_windows(relabelled(t)) if i < 0] for t in train}
        # half from pauses, the only normals that can change to any angle: attacks nearly all do, so this must not be a label
        still = [(p, t) for t in train if t["dir"] for p in pauses[t["num"]]]
        for n in range(args.view_normals):
            if rng.random() < 0.5:
                w, t = still[rng.integers(len(still))]
                near = [(p, side_of(t["dir"], o["dir"])) for o in train if o["dir"] in S.AROUND[t["dir"]] for p in pauses[o["num"]]]
            else:
                w, near = legs[rng.integers(len(legs))], None
            add(viewed_normal(w, rng, near if near and not args.turn_only else None), 0, "gen_normal", f"view/normal/{n}")
        print(f"  viewed normals: {args.view_normals} windows, {len(still)} pauses, {len(legs)} all", flush=True)

    if args.attacks == "body":
        for n in range(args.view_normals):
            add(body_normal(still[rng.integers(len(still))], rng, args.p_turn), 0, "gen_normal", f"body/normal/{n}")
        print(f"  body normals: {args.view_normals} windows from {len(still)} pauses", flush=True)
        made, per = 0, -(-args.body_gestures // max(len(clips), 1))
        for k in rng.permutation(len(clips)):
            n = min(per, args.body_gestures - made)
            if n <= 0:
                break
            for w in body_attacks(clips[k], bcoupling, rng, legs, n, free[clips[k]["t"]["hand"]], args.p_turn, gesture=True):
                add(w, 0, "gen_normal", f"body/gesture/{clips[k]['t']['num']}/{made}")
                made += 1
        if made:
            print(f"  body gestures: {made} windows from {len(clips)} strikes", flush=True)

    if args.gestures:
        limits =tuple(np.median([E.arm_limits(t["body"], t["hand"]) for t in train], axis=0))
        upright = [w for w in legs if is_upright(w)]
        print(f"  gesture sources: {len(upright)} upright of {len(legs)}", flush=True)
        pool = [t for t in train if t["strikes"]]
        made = 0
        while made < args.gesture_windows:
            if rng.random() < args.strike_gestures:
                t = pool[rng.integers(len(pool))]
                w = strike_gesture(t, t["strikes"][rng.integers(len(t["strikes"]))], coupling, rng, legs, limits)
            else:
                w = gesture(upright[rng.integers(len(upright))], coupling, rng, legs, limits)
            if w is not None:
                add(w, 0, "gen_normal", f"edit/gesture/{made}")
                made += 1
        print(f"  gestures: {args.gesture_windows} windows", flush=True)

    out = {"X": np.concatenate([z["X"], np.stack(new_X).astype(np.float32)]),
           "cls": np.concatenate([z["cls"], np.array(new_cls, np.int64)]),
           "split": np.concatenate([z["split"], np.array(["train"] * len(new_X))]),
           "group": np.concatenate([z["group"], np.array(new_grp)]),
           "source": np.concatenate([z["source"], np.array(new_src)])}
    os.makedirs(os.path.join(REPO_DIR, args.out), exist_ok=True)
    np.savez(os.path.join(REPO_DIR, args.out, "windows.npz"), **out)
    print("->", os.path.join(args.out, "windows.npz"), out["X"].shape)


if __name__ == "__main__":
    main()
