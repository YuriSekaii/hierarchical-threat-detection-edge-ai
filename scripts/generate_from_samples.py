"""
Builds Stage 2 window caches from the recorded motion samples (src/skeleton_sim/from_samples.py).

  python scripts/extract_take_skeletons.py data/lab_recordings/2026-11-05 --out results/samples/kp
  python scripts/generate_from_samples.py --base results/train_sim/exp_H_mix --out results/train_sim/exp_S_raw --mode raw
  python scripts/generate_from_samples.py --base results/train_sim/exp_H_mix --out results/train_sim/exp_S_gen --mode gen

Adds windows with source gen_attack and gen_normal to the base cache, for train_stage2_sim.py --mix gen:N,gen_normal:N.
Held-out takes never enter a cache. They go to results/samples/heldout.npz with one strike id per window.

Recipes in reports/MODIFICATION_REPORT_STAGE2_SAMPLES.md: HTD_TRAIN_OUT=train_sim/exp_S_raw or exp_S_gen, then
train --model t3 --steps 6000 --seed 1..3 with one of these --mix values:
  raw and generated:     real:3,sim:4,normal:6,real_normal:7,gen:8
  generated + normals:   real:3,sim:4,normal:5,real_normal:4,gen:8,gen_normal:4
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [REPO_DIR, os.path.join(REPO_DIR, "scripts")]

import train_stage2_sim as T  # noqa: E402
from src.skeleton_sim import from_samples as G  # noqa: E402

DAY = os.path.join(REPO_DIR, "data", "lab_recordings", "2026-11-05")
KP = os.path.join(REPO_DIR, "results", "samples", "kp")
THRUST, STAB, CUT, SLASH = 1, 2, 3, 4
# Takes 010-013 are labelled overhand_stab but show cut-downs, and 014-017 the reverse (the user mixed them up;
# hand travel confirms it). 026 is a 2 s false start.
BY_LABEL = {"thrust_chest": THRUST, "thrust_low": THRUST, "walking_thrust": THRUST, "overhand_stab": STAB,
            "cut_down": CUT, "slash": SLASH}
SWAPPED = {10: CUT, 11: CUT, 12: CUT, 13: CUT, 14: STAB, 15: STAB, 16: STAB, 17: STAB}
SKIP = {1, 26}
# Fixed before any model was trained on these takes: one direction per attack type, plus two left-hand takes.
HELD_OUT = {3, 8, 10, 17, 20, 23, 28, 31}
AROUND = {"front": ("left", "right"), "left": ("front", "back"), "back": ("left", "right"), "right": ("front", "back")}


def load_takes():
    takes = []
    with open(os.path.join(DAY, "takes.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            num = int(os.path.basename(r["file"].replace("\\", "/"))[:3])
            if num in SKIP or r["status"] != "ok" or r["category"] != "train_attack":
                continue
            name = os.path.splitext(os.path.basename(r["file"].replace("\\", "/")))[0]
            k = np.load(os.path.join(KP, name + ".npy"))
            here = np.where(k[..., 2].sum(-1) > 0)[0]
            a, b = here[0] + 60, here[-1] - 60  # walking to and from the desk
            if int(r["dropped"]) > 0:  # the old recorder dropped frames after about 30 s
                b = min(b, a + 900)
            xy = T.interpolate_missing_joints(k[a:b, :, :2].astype(np.float32))
            hand = G.L_WR if r["hand"] == "left" else G.R_WR
            env = G.hand_speed(xy, hand)
            attack = next(v for lab, v in BY_LABEL.items() if r["label"].startswith(lab))
            takes.append(dict(num=num, name=name, offset=a, cls=SWAPPED.get(num, attack), kind=r["label"].rsplit("_", 1)[0],
                              dir=r["direction"], xy=xy, missing=k[a:b, :, 2].sum(-1) == 0, env=env,
                              events=G.strike_events(env), pause=G.pause_frames(env), hand=hand))
    return takes


def label_windows(t):
    """Stride-10 windows of a take: (start, strike index or -1 for a pause). Others overlap a strike a little and are skipped."""
    out = []
    for s in range(0, len(t["xy"]) - 49, T.STRIDE):
        if t["missing"][s:s + 50].sum() > 10:
            continue
        best = max(((min(s + 50, e) - max(s, a), i) for i, (a, _, e) in enumerate(t["events"])), default=(0, -1))
        if best[0] >= 15:
            out.append((s, best[1]))
        elif best[0] <= 0 and t["pause"][s:s + 50].mean() >= 0.8:
            out.append((s, -1))
    return out


def attack_window(t, ev, rng, takes_by, legs):
    a, peak, e = ev
    length = int(50 * rng.uniform(0.75, 1.35))
    start = peak - int(length * rng.uniform(0.2, 0.8))
    if start < 0 or start + length >= len(t["xy"]):
        return None
    seq = G.time_warp(t["xy"], start, length, rng)
    r = rng.random()
    if r < 0.5:  # another performance of the same attack, aligned at its peak
        near = takes_by[t["kind"]]
        q = rng.random() < 0.6 and t["dir"] and [o for o in near if o["dir"] in AROUND[t["dir"]] and o["events"]]
        o = near[rng.integers(len(near))] if not q else q[rng.integers(len(q))]
        if not o["events"]:
            return None
        _, op, _ = o["events"][rng.integers(len(o["events"]))]
        ostart = op - (peak - start)
        if ostart < 0 or ostart + length >= len(o["xy"]):
            return None
        other = G.time_warp(o["xy"], ostart, length, rng, wobble=0.1)
        seq = G.view_between(seq, other, rng.uniform(0.2, 1.2)) if q else G.blend(seq, other, rng.uniform(0.3, 0.7))
    return G.vary(seq, rng, legs=legs[rng.integers(len(legs))])


def normal_window(src, rng, legs):
    def take(x):
        length = int(50 * rng.uniform(0.75, 1.35 if len(x) > 70 else 1.0))
        start = rng.integers(0, max(len(x) - length, 1))
        return G.time_warp(x, start, length, rng)

    seq = take(src[rng.integers(len(src))])
    if rng.random() < 0.5:  # same treatment as attacks, so that blending carries no label
        seq = G.blend(seq, take(src[rng.integers(len(src))]), rng.uniform(0.3, 0.7))
    return G.vary(seq, rng, legs=legs[rng.integers(len(legs))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["raw", "gen"], required=True)
    ap.add_argument("--attacks-per-class", type=int, default=6000)
    ap.add_argument("--normals", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    takes = load_takes()
    train = [t for t in takes if t["num"] not in HELD_OUT]
    print(f"{len(takes)} takes, {len(train)} for training")
    for t in takes:
        w = label_windows(t)
        print(f"  {t['num']:3d} {T.CLASSES[t['cls']]:9s} {t['kind']:16s} {t['dir'] or 'left hand':9s} {len(t['xy']):5d} f  "
              f"{len(t['events']):2d} strikes  {sum(i >= 0 for _, i in w):3d} attack / {sum(i < 0 for _, i in w):3d} pause windows"
              + ("  HELD OUT" if t["num"] in HELD_OUT else ""))

    # held-out test set
    X, take, strike, cls = [], [], [], []
    for t in takes:
        if t["num"] in HELD_OUT:
            for s, i in label_windows(t):
                X.append(T.live_prep(t["xy"][s:s + 50]))
                take.append(t["num"])
                strike.append(i)
                cls.append(t["cls"])
    os.makedirs(os.path.join(REPO_DIR, "results", "samples"), exist_ok=True)
    np.savez(os.path.join(REPO_DIR, "results", "samples", "heldout.npz"), X=np.stack(X).astype(np.float32),
             take=np.array(take), strike=np.array(strike), cls=np.array(cls))
    print(f"held out: {len(X)} windows, {sum(i >= 0 for i in strike)} attack")

    z = np.load(os.path.join(REPO_DIR, args.base, "windows.npz"))
    base = {k: z[k] for k in z.files}
    new_X, new_cls, new_src, new_grp = [], [], [], []

    def add(w, c, source, group):
        new_X.append(T.live_prep(w))
        new_cls.append(c)
        new_src.append(source)
        new_grp.append(group)

    pauses = [t["xy"][s:s + 50] for t in train for s, i in label_windows(t) if i < 0]
    if args.mode == "raw":
        for t in train:
            for s, i in label_windows(t):
                if i >= 0:
                    add(t["xy"][s:s + 50], t["cls"], "gen_attack", f"gen/{t['num']}/{s}")
        for n, p in enumerate(pauses):
            add(p, 0, "gen_normal", f"gen/pause/{n}")
    else:
        lab = np.load(os.path.join(T.LAB, "analysis", "001_floor_markers_142701_analysis.npz"))["kpts"][..., :2]
        real_normal = base["X"][base["source"] == "real_normal"]
        sources = pauses + [lab[:len(lab) // 2].astype(np.float32)] + list(real_normal)
        legs = list(real_normal[rng.choice(len(real_normal), min(3000, len(real_normal)), replace=False)]) + pauses
        by_kind = {}
        for t in train:
            by_kind.setdefault(t["kind"], []).append(t)
        for c in (THRUST, STAB, CUT, SLASH):
            pool = [t for t in train if t["cls"] == c and t["events"]]
            made = 0
            while made < args.attacks_per_class:
                t = pool[rng.integers(len(pool))]
                ev = t["events"][rng.integers(len(t["events"]))]
                w = attack_window(t, ev, rng, by_kind, legs)
                if w is not None:
                    add(w, c, "gen_attack", f"gen/{t['num']}/{made}")
                    made += 1
            print(f"  {T.CLASSES[c]}: {made} windows from {len(pool)} takes", flush=True)
        for n in range(args.normals):
            add(normal_window(sources, rng, legs), 0, "gen_normal", f"gen/normal/{n}")
    print(f"new windows: {len(new_X)} ({new_src.count('gen_attack')} attack, {new_src.count('gen_normal')} normal)")

    out = {"X": np.concatenate([base["X"], np.stack(new_X).astype(np.float32)]),
           "cls": np.concatenate([base["cls"], np.array(new_cls, np.int64)]),
           "split": np.concatenate([base["split"], np.array(["train"] * len(new_X))]),
           "group": np.concatenate([base["group"], np.array(new_grp)]),
           "source": np.concatenate([base["source"], np.array(new_src)])}
    os.makedirs(os.path.join(REPO_DIR, args.out), exist_ok=True)
    np.savez(os.path.join(REPO_DIR, args.out, "windows.npz"), **out)
    print("->", os.path.join(args.out, "windows.npz"), out["X"].shape)


if __name__ == "__main__":
    main()
