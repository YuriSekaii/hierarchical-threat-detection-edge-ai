"""Stage 1 on the lab takes no detector here was trained on: share of frames (every 3rd, 4:3 crop, 384x640) with a
knife detection, and share of ~1 s windows with >= 3 of 10 detections. System specs are those of bench_stage1.py.

  .venv/Scripts/python.exe results/samples/poc/bench_stage1_holdout.py "det,weights/yolo_weapon_distilled.pt,0.45" ...

Knife takes (higher is better): Nov 14 test takes 025-036, Oct 31 take 002, Nov 14 takes 020 and 041.
Object takes (lower is better): Nov 14 takes 050 (second phone) and 056 (remote), strikes with no knife.
Rows are appended to results/stage1_kd/holdout.jsonl.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_stage1 as B

LAB = os.path.join(B.WORKTREE, "data", "lab_recordings")
knife = sorted(glob.glob(os.path.join(LAB, "2026-11-14", "test_*", "*.avi"))) + [
    os.path.join(LAB, "2026-10-31", "raw", "002_checkerboard_142846.avi")] + [
    p for n in ("020", "041") for p in glob.glob(os.path.join(LAB, "2026-11-14", "train_*", n + "_*.avi"))]
other = [p for n in ("050", "056") for p in glob.glob(os.path.join(LAB, "2026-11-14", "confuser", n + "_*.avi"))]
assert len(knife) == 15 and len(other) == 2, (len(knife), len(other))

systems = [B.build_system(s) for s in sys.argv[1:]]
res = {name: {} for name, _ in systems}
for path in knife + other:
    take = os.path.basename(os.path.dirname(os.path.dirname(path)))[5:] + " " + os.path.basename(path)[:-11]
    for name, flags in B.eval_take(path, systems).items():
        res[name][take] = (round(B.share(flags), 4), round(B.windows_share(flags), 4))
    print(take, {n: res[n][take] for n in res}, flush=True)

os.makedirs(os.path.join(B.WORKTREE, "results", "stage1_kd"), exist_ok=True)
with open(os.path.join(B.WORKTREE, "results", "stage1_kd", "holdout.jsonl"), "a") as fh:
    for name in res:
        fh.write(json.dumps({"system": name, "takes": res[name]}) + "\n")
nk = len(knife)
print("\n| System | Nov 14 test takes, frames (mean / worst) | windows (mean / worst) | Oct 31 take 002 frames | Second phone 050 | Remote 056 |")
print("|---|---|---|---|---|---|")
for name, r in res.items():
    v = list(r.values())
    t = v[:12]
    print(f"| {name} | {sum(x[0] for x in t) / 12:.0%} / {min(x[0] for x in t):.0%} | {sum(x[1] for x in t) / 12:.0%} / "
          f"{min(x[1] for x in t):.0%} | {v[12][0]:.0%} | {v[nk][0]:.0%} | {v[nk + 1][0]:.0%} |")
