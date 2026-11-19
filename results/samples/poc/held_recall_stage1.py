"""Frame recall when a knife detection is held for N frames, as the pipeline holds a weapon (weapon_hold=3.0 s, 90
frames at 30 fps). A test knife box counts as found if the detector found a knife at conf >= T, IoU >= 0.2, in this
labelled frame, or found one near it (centre within 3 box sides) in a labelled frame of the same take up to N frames
earlier. Labelled frames are 10 apart, so this is an estimate. Uses the lab cache; no GPU.

  DETS=kdgtNL,labgtL,teachergtL T=0.40 .venv/Scripts/python.exe results/samples/poc/held_recall_stage1.py
"""
import collections, os, sys
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_stage1_rules as S

DETS = os.environ.get("DETS", "kdgtNL,labgtL,teachergtL").split(",")
T = float(os.environ.get("T", "0.40"))
recs = S.load_cache("lab")
fr = lambda p: int(os.path.splitext(p)[0].rsplit("_f", 1)[1])
by = {(r["category"], fr(r["path"])): r for r in recs}


def hit_here(r, g, d):
    return any(b["conf"] >= T and S.calculate_iou(b["xyxy"], g) >= S.IOU_THRESH for b in r["det"][d])


def near(r, g, d, k=3):
    sx, sy = g[2] - g[0], g[3] - g[1]
    s = max(sx, sy * 9 / 16)  # one box side, in frame-width units
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    return any(b["conf"] >= T and abs((b["xyxy"][0] + b["xyxy"][2]) / 2 - cx) < k * s
               and abs((b["xyxy"][1] + b["xyxy"][3]) / 2 - cy) < k * s * 16 / 9 for b in r["det"][d])


print(f"| detector | conf | no hold | hold 10 frames | 30 | 90 (3 s) |\n|---|---|---|---|---|---|")
per_take = {}
for d in DETS:
    found = collections.Counter(); n = 0; tk = collections.defaultdict(collections.Counter)
    for r in recs:
        f = fr(r["path"])
        for g in r["gt"]:
            n += 1; tk[r["category"]]["n"] += 1
            for N in (0, 10, 30, 90):
                ok = hit_here(r, g, d) or any(near(by[(r["category"], f - df)], g, d)
                                              for df in range(10, N + 1, 10) if (r["category"], f - df) in by)
                found[N] += ok; tk[r["category"]][N] += ok
    print(f"| {d} | {T:.2f} | " + " | ".join(f"{found[N] / n * 100:.1f} %" for N in (0, 10, 30, 90)) + " |")
    per_take[d] = tk
d = DETS[0]
print(f"\n{d}, per take: no hold / 3 s hold\n| take | boxes | no hold | 3 s hold |\n|---|---|---|---|")
for take, c in sorted(per_take[d].items()):
    print(f"| {take} | {c['n']} | {c[0] / c['n'] * 100:.1f} % | {c[90] / c['n'] * 100:.1f} % |")
