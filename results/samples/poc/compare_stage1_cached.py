"""Compare detectors from the sweep_stage1_rules.py caches: precision / recall at fixed thresholds, with and without the hand
gate (wrist within 0.5 torso lengths), and the best recall at a precision target. No GPU.

  EXTRA_DETS="labgt=x;labgtL=x" .venv/Scripts/python.exe results/samples/poc/compare_stage1_cached.py shipped teacher student labgt labgtL
  (EXTRA_DETS only names the extra detectors already in the caches; the paths are not used here)
"""
import os, sys

DETS = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_stage1_rules as S

CONFS = (0.25, 0.35, 0.45)
TS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
rule = lambda t, gate: ({"kind": "hyst", "hi": 1.01, "lo": t, "gate": {"kind": "wrist@k", "k": 0.5}} if gate
                        else {"kind": "single", "t": t})


def score(recs, d, r):
    tp = fp = fn = 0
    for rec in recs:
        a, b, c = S.evaluate_rule(rec, d, r)
        tp, fp, fn = tp + a, fp + b, fn + c
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), fp


for sname in ("lab", "gtv2"):
    recs = S.load_cache(sname)
    subs = {"all": recs}
    if sname == "lab":
        subs["Oct 31 take 002"] = [r for r in recs if "2026-10-31" in r["path"]]
    for sub, rr in subs.items():
        print(f"\n## {sname} / {sub}: {len(rr)} images, {sum(len(r['gt']) for r in rr)} boxes")
        print("det | " + " | ".join(f"{t:.2f}{' gate' if g else ''} P/R fp" for t in CONFS for g in (False, True)))
        for d in DETS:
            cells = [score(rr, d, rule(t, g)) for t in CONFS for g in (False, True)]
            print(d, "|", " | ".join(f"{p * 100:.1f}/{r * 100:.1f} {fp}" for p, r, fp in cells))
    print(f"\n## {sname}: best recall at precision >= target: R (conf, P, false boxes)")
    for d in DETS:
        row = []
        for g in (False, True):
            pts = [score(recs, d, rule(t, g)) + (t,) for t in TS]
            for tgt in (0.85, 0.90, 0.92):
                ok = [p for p in pts if p[0] >= tgt]
                b = max(ok, key=lambda p: p[1]) if ok else None
                row.append(f"{'gate' if g else 'raw'}@{tgt:.2f} " + (f"{b[1] * 100:.1f} (c{b[3]:.2f}, P {b[0] * 100:.1f}, fp {b[2]})" if b else "-"))
        print(d, "|", " | ".join(row))
