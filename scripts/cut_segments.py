"""
Cuts a recorded lab day into labelled clips, in the layout of data/lab_recordings/2026-10-31/segments/.

Strikes are found from the knife-hand speed (src/skeleton_sim/from_samples.py); each one becomes a 1.7 s attack clip,
the rest of the take becomes normal clips. Takes were not performed exactly as the guide said, so labels and rejected
strikes come from a hand review in <day>/segments/review.csv (take,event,action,value,note):
  action=reject  event=<n>   drop strike n of that take (numbering is printed by --dry-run)
  action=label   event=<n|*> new label for strike n, or for the whole take
  action=drop    event=*     skip the take
  action=note    event=*     note for all segments of the take

  python scripts/cut_segments.py data/lab_recordings/2026-11-05 --kp results/samples/kp [--dry-run]
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.skeleton_sim.from_samples import hand_speed, strike_events  # noqa: E402

FPS = 30
PRE, POST = 30, 21          # attack window around the speed peak
EDGE = 45                   # strikes this close to walking in or out are dropped
HEADER = "id,file,skeleton,take,start_s,end_s,frames,category,label,knife_hand,view,knife_seen,note".split(",")
CROP = "4:3 centre, 1920x1440"


def runs(mask, min_len=30):
    """[(a, b)] of True runs after filling gaps shorter than min_len."""
    m = mask.copy()
    idx = np.flatnonzero(m)
    for x, y in zip(idx[:-1], idx[1:]):
        if 1 < y - x < min_len:
            m[x:y] = True
    d = np.diff(np.r_[0, m.astype(int), 0])
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]


def strikes(k, hand, rr):
    """Kept strike windows [(n, a, b)] and the auto-dropped event numbers."""
    env = hand_speed(k[..., :2], hand)
    lvl = np.percentile(env, 98)
    ev = [p for _, p, _ in strike_events(env) if env[p] > 0.5 * lvl]
    keep, auto = [], []
    for n, p in enumerate(ev):
        a, b = p - PRE, p + POST
        if any(ra + EDGE <= a and b <= rb - EDGE for ra, rb in rr):
            keep.append([n, a, b])
        else:
            auto.append(n)
    for w0, w1 in zip(keep[:-1], keep[1:]):  # no overlapping clips
        if w0[2] > w1[1]:
            w0[2] = w1[1] = (w0[2] + w1[1]) // 2
    return [tuple(w) for w in keep], auto


def plan_take(r, k, review):
    """[(start, end, category, label, note)] for one take."""
    lab, cat = r["label"], r["category"]
    rv = review.get(r["take"], {})
    if "drop" in rv.get("*", {}):
        return [], []
    note = rv.get("*", {}).get("note", "")
    take_label = rv.get("*", {}).get("label", lab)
    rr = runs(k[..., 2].sum(-1) > 0)
    if not rr:
        return [], []
    out = []
    strike_take = cat in ("train_attack", "test_attack") or lab.endswith("_strike")
    if strike_take:
        hand = 9 if r["knife_hand"] == "L" else 10
        ws, auto = strikes(k, hand, rr)
        ws = [w for w in ws if "reject" not in rv.get(str(w[0]), {})]
        acat = "confuser" if lab.endswith("_strike") else "attack"
        for n, a, b in ws:
            e = rv.get(str(n), {})
            out.append((a, b, acat, e.get("label", take_label), e.get("note", note) or f"strike {n}"))
        for ra, rb in rr:  # what is left of each run is normal
            cuts = [ra] + [x for n, a, b in ws if ra <= a < rb for x in (a, b)] + [rb]
            for a, b in zip(cuts[::2], cuts[1::2]):
                if b - a >= 30:
                    out.append((a, b, "normal", "between_strikes", note))
        return sorted(out), auto
    ncat = "confuser" if cat == "confuser" else "normal"
    for i, (ra, rb) in enumerate(rr):
        a, b = ra, rb
        if i == 0 and b - a >= 180:
            out.append((a, a + 60, "normal", "walk", note))
            a += 60
        if i == len(rr) - 1 and b - a >= 120:
            out.append((b - 60, b, "normal", "walk", note))
            b -= 60
        cuts = list(range(a, b, 300)) + [b]
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < 60:
            cuts.pop(-2)
        out += [(x, y, ncat, take_label, note) for x, y in zip(cuts[:-1], cuts[1:])]
    return sorted(out), []


def read_review(path):
    rv = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for x in csv.DictReader(f):
                v = x["value"] if x["action"] != "note" else (x["value"] or x["note"])
                rv.setdefault(x["take"], {}).setdefault(x["event"], {})[x["action"]] = v
    return rv


def write_clip(src, dst, a, b):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    w = None
    for _ in range(b - a):
        ok, f = cap.read()
        if not ok:
            break
        if w is None:
            w = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"MJPG"), fps, (f.shape[1], f.shape[0]))
        w.write(f)
    cap.release()
    if w is not None:
        w.release()
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day", nargs="?")
    ap.add_argument("--kp")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if os.path.basename(os.path.normpath(args.day)) == "2026-10-31":
        sys.exit("2026-10-31 segments are hand-made; not overwriting them")
    out_dir = os.path.join(args.day, "segments")
    os.makedirs(out_dir, exist_ok=True)
    review = read_review(os.path.join(out_dir, "review.csv"))
    rows, n = [], 0
    with open(os.path.join(args.day, "takes.csv"), newline="", encoding="utf-8") as f:
        takes = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    for r in takes:
        r["take"] = os.path.splitext(os.path.basename(r["file"]))[0]
        r["knife_hand"] = "L" if "lefthand" in r["label"] or r.get("hand") == "left" else "R"
        p = os.path.join(args.kp, r["take"] + ".npy")
        if not os.path.exists(p):
            print(f"{r['take']}: no skeleton, skipped")
            continue
        k = np.load(p)
        if not len(k):
            print(f"{r['take']}: empty take, skipped")
            continue
        segs, auto = plan_take(r, k, review)
        print(f"{r['take']}: {len(segs)} segments" + (f", strikes dropped at walk-in/out: {auto}" if auto else ""))
        view = next((v for v in ("front", "left", "back", "right") if r["label"].endswith(v)), "mixed")
        src = os.path.join(args.day, r["file"].replace("\\", "/"))
        for a, b, cat, lab, note in segs:
            if int(r.get("dropped") or 0) > 0 and b > 900:  # the Nov 5 recorder dropped frames after about 30 s
                note = "; ".join(x for x in (note, "after 30 s in a take with dropped frames: timing unreliable") if x)
            sid = f"{n:03d}"
            name = f"{sid}_{cat}_{lab}"
            rows.append([sid, name + ".avi", name + ".npz", r["take"], f"{a / FPS:.2f}", f"{b / FPS:.2f}", b - a,
                         cat, lab, r["knife_hand"], view, "", note])
            if not args.dry_run:
                fps = write_clip(src, os.path.join(out_dir, name + ".avi"), a, b)
                np.savez(os.path.join(out_dir, name + ".npz"), kpts=k[a:b].astype(np.float32), take=r["take"],
                         start_frame=a, end_frame=b, fps=fps, crop=CROP)
            n += 1
    with open(os.path.join(out_dir, "segments.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([HEADER] + rows)
    print(f"{n} segments -> {out_dir}")


def selftest():
    t = np.arange(600)
    k = np.zeros((600, 17, 3), np.float32)
    k[..., 2] = 1
    k[:, [5, 6], 1], k[:, [11, 12], 1] = 100, 200          # shoulders above hips, torso 100 px
    k[:, :, 0] = 300
    for c in (200, 400):                                    # two fast wrist bursts
        k[:, 10, 0] += 150 * np.exp(-((t - c) / 6.0) ** 2)
    k[:, 10, 0] += np.random.default_rng(0).normal(0, 0.5, 600)
    r = {"take": "x", "label": "thrust", "category": "train_attack", "knife_hand": "R"}
    segs, _ = plan_take(r, k, {})
    att = [s for s in segs if s[2] == "attack"]
    assert len(att) == 2 and att[0][1] <= att[1][0], att
    assert any(s[2] == "normal" and s[1] <= att[0][0] for s in segs) and any(s[2] == "normal" and s[0] >= att[1][1] for s in segs)
    assert all(abs((a + PRE) - c) <= 3 for (a, _b, *_), c in zip(att, (200, 400))), att
    print("selftest ok")


if __name__ == "__main__":
    main()
