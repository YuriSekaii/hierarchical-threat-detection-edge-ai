"""Is the Stage 1 distillation wired right? Two checks.

1. GPU: the teacher loaded the way train_stage1_repro.py loads it for KD (YOLO(pt).model, eval) gives, on a 640x640
   letterboxed image, the same top knife score in its one2one head as model.predict() on that same 640x640 image (predict() on the raw
   frame pads to 640x384 instead, which moves scores by up to 0.23).
2. No GPU: does each KD student copy its teacher? For every lab test knife, the best conf at IoU >= 0.2 of teacher and
   student (cache of sweep_stage1_rules.py): correlation, and how often the student finds what the teacher finds.

  TEACHER=weights/stage1_repro846_teachergtaug_p250.pt DEV=0 .venv/Scripts/python.exe results/samples/poc/check_stage1_kd.py
"""
import os, sys

import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_stage1_rules as S

TEACHER = os.environ.get("TEACHER", "weights/stage1_repro846_teachergtaug_p250.pt")
DEV = os.environ.get("DEV", "0")
recs = S.load_cache("lab")

# 1. teacher as KD sees it vs predict()
import cv2, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

yolo = YOLO(TEACHER)
teacher = YOLO(TEACHER).model.to(f"cuda:{DEV}").eval()
knife = [k for k, v in yolo.names.items() if v == "knife"][0]
diffs = []
for r in recs[::60]:
    img = cv2.imread(r["path"])
    lb = LetterBox((640, 640), auto=False)(image=img)
    x = torch.from_numpy(lb[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255)[None].to(f"cuda:{DEV}")
    with torch.no_grad(), torch.autocast("cuda"):
        preds = teacher(x)[1]
    kd = preds["one2one"]["scores"][0, knife].float().sigmoid().max().item()
    p = yolo.predict(lb, imgsz=640, conf=0.001, device=DEV, verbose=False)[0].boxes
    pr = max([c for c, k in zip(p.conf.tolist(), p.cls.tolist()) if int(k) == knife], default=0.0)
    diffs.append(abs(kd - pr))
print(f"1. teacher in KD form vs predict(), {len(diffs)} images: max |top knife score diff| = {max(diffs):.4f}, mean {np.mean(diffs):.4f}")
assert max(diffs) < 0.05, "the teacher KD sees is not the teacher predict() runs"


# 2. does the student copy the teacher?
def best(r, d, g):
    return max([b["conf"] for b in r["det"][d] if S.calculate_iou(b["xyxy"], g) >= S.IOU_THRESH], default=0.0)


print("\n2. lab test knives: teacher vs student best conf at IoU >= 0.2")
print("| teacher | student | KD? | corr | student finds (0.25) when teacher finds (0.25) | student finds when teacher does not |")
print("|---|---|---|---|---|---|")
for t, s, kd in (("teachergtL", "kdgtNL", "yes"), ("teachergtL", "labgtL", "no"),
                 ("teachergtaug", "kdgtaugFL", "yes"), ("teachergtaug", "labgtaugL", "no"), ("teachergtaugL", "kdgtaugFL", "other ckpt")):
    a = np.array([[best(r, t, g), best(r, s, g)] for r in recs for g in r["gt"]])
    th = a[:, 0] >= 0.25
    print(f"| {t} | {s} | {kd} | {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.3f} | {(a[th, 1] >= 0.25).mean() * 100:.1f} % "
          f"| {(a[~th, 1] >= 0.25).mean() * 100:.1f} % |")


# 3. where do the students' false boxes come from? A box at conf >= 0.25 that touches no test knife at IoU >= 0.2;
# the teacher's best conf on a box at IoU >= 0.3 with it (0 when the teacher has no box there at conf >= 0.01).
print("\n3. student false boxes at conf >= 0.25 (lab test) and what the teacher scores at the same place")
print("| teacher | student | KD? | false boxes | teacher < 0.01 | 0.01-0.10 | 0.10-0.25 | >= 0.25 (teacher false box too) |")
print("|---|---|---|---|---|---|---|---|")
for t, s, kd in (("teachergtL", "kdgtNL", "yes"), ("teachergtL", "labgtL", "no"),
                 ("teachergtaug", "kdgtaugFL", "yes"), ("teachergtaug", "labgtaugL", "no")):
    tc = []
    for r in recs:
        for b in r["det"][s]:
            if b["conf"] >= 0.25 and all(S.calculate_iou(b["xyxy"], g) < S.IOU_THRESH for g in r["gt"]):
                tc.append(max([c["conf"] for c in r["det"][t] if S.calculate_iou(c["xyxy"], b["xyxy"]) >= 0.3], default=0.0))
    tc = np.array(tc)
    bins = [(tc < 0.01).sum(), ((tc >= 0.01) & (tc < 0.1)).sum(), ((tc >= 0.1) & (tc < 0.25)).sum(), (tc >= 0.25).sum()]
    print(f"| {t} | {s} | {kd} | {len(tc)} | " + " | ".join(str(v) for v in bins) + " |")


# 4. duplicates: the scoring counts a second box on an already-matched knife as a false box. YOLO26 runs without NMS
# (one2one head), so two boxes on one knife both reach the score. Split the false boxes, then re-score after a plain
# NMS (IoU 0.5) on the cached boxes.
def nms(bs, thr=0.5):
    out = []
    for b in sorted(bs, key=lambda v: -v[4]):
        if all(S.calculate_iou(b[:4], o[:4]) < thr for o in out):
            out.append(b)
    return out


assert len(nms([(0, 0, .1, .1, .9), (0, 0, .1, .11, .5), (.5, .5, .6, .6, .3)])) == 2
TS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]


def score(rs, d, t, use_nms):
    tp = fp = fn = dup = 0
    for r in rs:
        bs = [tuple(b["xyxy"]) + (b["conf"],) for b in r["det"][d] if b["conf"] >= t]
        if use_nms:
            bs = nms(bs)
        a, b, c = S.evaluate_matches(bs, r["gt"])
        far = sum(all(S.calculate_iou(x[:4], g) < S.IOU_THRESH for g in r["gt"]) for x in bs)
        tp, fp, fn, dup = tp + a, fp + b, fn + c, dup + b - far
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), fp, dup


for sname in ("lab", "gtv2"):
    rs = S.load_cache(sname)
    print(f"\n4. {sname}: false boxes at 0.25 (of which duplicates on a found knife), and best recall at P >= 0.90: R (conf, P)")
    print("| detector | 0.25: P / R, false boxes (duplicates) | P >= 0.90, as scored now | P >= 0.90, with NMS 0.5 |")
    print("|---|---|---|---|")
    for d in ("labgtL", "kdgtNL", "labgtaugL", "kdgtaugFL", "kdgtaugF", "teachergtL", "teachergtaug"):
        p, rr, fp, dup = score(rs, d, 0.25, False)
        cells = []
        for use in (False, True):
            ok = [(score(rs, d, t, use), t) for t in TS]
            ok = [(s, t) for s, t in ok if s[0] >= 0.90]
            b = max(ok, key=lambda v: v[0][1]) if ok else None
            cells.append(f"{b[0][1] * 100:.1f} % ({b[1]:.2f}, P {b[0][0] * 100:.1f})" if b else "not reached")
        print(f"| {d} | {p * 100:.1f} / {rr * 100:.1f}, {fp} ({dup}) | {cells[0]} | {cells[1]} |")
