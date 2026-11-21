"""Check by eye the spots where Stage 1 detectors fire but the lab labels say "no knife", and add empty-hand frames.

Every candidate is one item in results/stage1_kd/cand_review/items.json, in the lab_items.json tile format, so
label_review_ui.py --dir results/stage1_kd/cand_review can show it. Labels as in label_lab_stage1.py:
"a" = knife in the red box and the box fits it, "k" = knife but the box does not fit, "n" = no knife, "u" = cannot tell.

  .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py collect test     # false boxes on the lab test set (caches, no GPU)
  DEV=0 .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py collect train   # boxes on lab training frames that miss the label
  DEV=0 MAXNEG=300 .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py collect neg   # new frames from training takes
  .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py sheets    # numbered 4x3 sheets for the voters
  .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py tally     # votes/<voter>.json (+ claude.json) -> votes.csv
  .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py apply     # test_label_fix.json + data/stage1_negatives/lab_gt_aug_hn
  .venv/Scripts/python.exe results/samples/poc/review_stage1_candidates.py selftest

Vote rule (tally): all voters "a" -> a; all "n" and the best detector conf < 0.40 -> n; all "n" at conf >= 0.40, or any
mix -> Claude looks (claude.json). Claude sure -> final; Claude unsure, or "k"/"u" -> the human (review_candidates.bat).
Never written: data/lab_recordings, data/lab_stage1_gt, data/stage1_negatives/lab_gt*, weights.
"""
import csv, glob, json, os, random, re, shutil, sys, tempfile, time

import cv2
import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import sweep_stage1_rules as S

OUT = os.path.join(REPO, "results", "stage1_kd", "cand_review")
ITEMS = os.path.join(OUT, "items.json")
LAB = os.path.join(REPO, "data", "lab_stage1_gt")
HN = os.path.join(REPO, "data", "lab_stage1_hn", "images")
REC = os.path.join(REPO, "data", "lab_recordings")
AUG = os.path.join(REPO, "data", "stage1_negatives", "lab_gt_aug")
NEW = os.path.join(REPO, "data", "stage1_negatives", "lab_gt_aug_hn")
TEST_MODELS = ("labgtL", "labgtaugL", "labgtaugS1L", "labgtaugltL", "kdgtNL", "kdgtaugFL", "teachergtL", "teachergtaug")
TRAIN_MODELS = (("teachergtaug", "stage1_repro846_teachergtaug_p250.pt"), ("v2", "stage1_repro846_v2_p100.pt"),
                ("shipped", "yolo_weapon_distilled.pt"))  # v2 and shipped never saw a lab frame
NEG_TAKES = ("2026-11-05/train_attack/*.avi", "2026-11-14/train_attack/*.avi", "2026-11-14/train_normal/*.avi")
FIRST_ID = {"test": 0, "train": 100000, "neg": 200000}
TILE, CELL = 330, 320
MODEL_NO = 0.40  # unanimous "no knife" where a detector is this sure goes to Claude anyway
iou = S.calculate_iou


def nms(bs, thr=0.5):
    out = []
    for b in sorted(bs, key=lambda v: -v[4]):
        if all(iou(b[:4], o[:4]) < thr for o in out):
            out.append(b)
    return out


def merge(per_model, gts, gt_iou=S.IOU_THRESH):
    """Boxes of all models (normalized xyxy + conf) that match no labelled knife, clustered across models at IoU >= 0.3."""
    pool = [(b, m) for m, bs in per_model.items() for b in nms(bs) if all(iou(b[:4], g) < gt_iou for g in gts)]
    clusters = []
    for b, m in sorted(pool, key=lambda v: -v[0][4]):
        c = next((c for c in clusters if iou(c["box"], b[:4]) >= 0.3), None)
        if c is None:
            clusters.append({"box": list(b[:4]), "conf": b[4], "src": {m: b[4]}})
        else:
            c["src"][m] = max(c["src"].get(m, 0.0), b[4])
    return clusters


def name_parts(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    take, f = stem.rsplit("_f", 1)
    return take, int(f)


def make_item(i, img, kind, split, w, h, c, **extra):
    x1, y1, x2, y2 = c["box"][0] * w, c["box"][1] * h, c["box"][2] * w, c["box"][3] * h
    side = int(min(max(400, 3 * max(x2 - x1, y2 - y1)), w, h))
    x0 = int(min(max((x1 + x2) / 2 - side / 2, 0), w - side))
    y0 = int(min(max((y1 + y2) / 2 - side / 2, 0), h - side))
    take, f = name_parts(img)
    return dict(id=i, img=os.path.relpath(img, REPO).replace("\\", "/"), split=split, take=take, frame=f, x0=x0, y0=y0,
                crop=side, kind=kind, props=[{"letter": "a", "box": [round(v, 1) for v in (x1, y1, x2, y2)],
                                              "score": round(c["conf"], 3), "src": {k: round(v, 3) for k, v in c["src"].items()}}],
                conf=round(c["conf"], 3), n_models=len(c["src"]), **extra)


def save_kind(kind, new):
    old = json.load(open(ITEMS)) if os.path.exists(ITEMS) else []
    os.makedirs(OUT, exist_ok=True)
    json.dump([it for it in old if it["kind"] != kind] + new, open(ITEMS, "w"), indent=0)
    print(f"{kind}: {len(new)} items, {len({it['img'] for it in new})} frames, conf >= {MODEL_NO}: {sum(it['conf'] >= MODEL_NO for it in new)}")


def labels_of(img_name):
    p = os.path.join(LAB, "labels", os.path.splitext(img_name)[0] + ".txt")
    return [list(S.xywhn2xyxy(*g[1:])) for g in S.load_ground_truth(p)] if os.path.exists(p) else None


def detector():
    from ultralytics import YOLO
    dev = int(os.environ.get("DEV", 0))

    def run(model, img, conf):
        r = model(img, imgsz=640, conf=conf, half=True, device=dev, verbose=False)[0]
        return [tuple(b) + (c,) for b, c, k in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
                if model.names[int(k)] == "knife"]
    return YOLO, run


def collect_test():
    out = []
    for r in S.load_cache("lab"):
        for c in merge({m: [tuple(b["xyxy"]) + (b["conf"],) for b in r["det"][m] if b["conf"] >= 0.25] for m in TEST_MODELS}, r["gt"]):
            out.append(make_item(FIRST_ID["test"] + len(out), r["path"], "test", "test", r["w"], r["h"], c))
    save_kind("test", out)


def collect_train():
    YOLO, run = detector()
    models = {n: YOLO(os.path.join(REPO, "weights", w)) for n, w in TRAIN_MODELS}
    frames = [l.strip() for l in open(os.path.join(LAB, "train.txt")) if l.strip()]
    out = []
    for k, p in enumerate(frames):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        for c in merge({n: run(m, img, 0.25) for n, m in models.items()}, labels_of(os.path.basename(p)) or []):
            out.append(make_item(FIRST_ID["train"] + len(out), p, "train", "train", w, h, c))
        if k % 500 == 0:
            print(f"  {k}/{len(frames)}")
    save_kind("train", out)


def collect_neg():
    """Midpoint frames between the sampled frames of the training takes; keep frames with a box away from the knife."""
    YOLO, run = detector()
    model = YOLO(os.path.join(REPO, "weights", TRAIN_MODELS[0][1]))
    known = {os.path.basename(l.strip()) for l in open(os.path.join(LAB, "train.txt")) if l.strip()}
    sampled = {}
    for p in glob.glob(os.path.join(LAB, "images", "*.jpg")):
        take, f = name_parts(p)
        sampled.setdefault(take, []).append(f)
    os.makedirs(HN, exist_ok=True)
    frames = []
    for pat in NEG_TAKES:
        for video in sorted(glob.glob(os.path.join(REC, pat))):
            date = os.path.basename(os.path.dirname(os.path.dirname(video)))
            take = f"{date}_{os.path.basename(video)[:-4]}"
            fs = sorted(sampled.get(take, []))
            if not any(f"{take}_f{f:05d}.jpg" in known for f in fs):
                continue  # not a training take
            want = {(a + b) // 2: (a, b) for a, b in zip(fs, fs[1:]) if b - a >= 2}
            cap, k = cv2.VideoCapture(video), -1
            while want and cap.grab():
                k += 1
                if k not in want:
                    continue
                ok, img = cap.retrieve()
                if not ok:
                    break
                neigh = [labels_of(f"{take}_f{f:05d}.jpg") if f"{take}_f{f:05d}.jpg" in known else None for f in want.pop(k)]
                boxes = run(model, img, 0.10)
                knives = [g for n in neigh if n for g in n]
                if len(neigh) == 2 and neigh[0] and neigh[1]:  # where the knife is now: between its two labelled positions
                    knives.append([(neigh[0][0][j] + neigh[1][0][j]) / 2 for j in range(4)])
                # usable as an empty-hand frame: no knife in either neighbour and a box >= 0.25, or one box on the knife
                # and another box >= 0.25 well away from it (a single box away from the old position is usually the moved knife)
                empty = all(n == [] for n in neigh)
                on = [b for b in boxes if any(iou(b[:4], g) >= 0.1 for g in knives)]
                away = [b for b in boxes if b[4] >= 0.25 and all(iou(b[:4], g) < 0.05 for g in knives + [o[:4] for o in on])]
                if (empty and any(b[4] >= 0.25 for b in boxes)) or (on and away):
                    name = os.path.join(HN, f"{take}_f{k:05d}.jpg")
                    frames.append((name, img.shape[1], img.shape[0], boxes, any(n is None or len(n) > 0 for n in neigh), img))
            print(f"  {take}: {len(frames)} candidate frames so far")
    random.Random(0).shuffle(frames)
    frames = frames[:int(os.environ.get("MAXNEG", 300))]
    out = []
    for name, w, h, boxes, neigh_knife, img in frames:
        cv2.imwrite(name, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ids = []
        for c in merge({"teachergtaug": boxes}, []):
            ids.append(FIRST_ID["neg"] + len(out))
            out.append(make_item(ids[-1], name, "neg", "train", w, h, c, neigh_knife=neigh_knife))
        for it in out[-len(ids):]:
            it["frame_items"] = ids
    keep = {os.path.basename(f[0]) for f in frames}
    for p in glob.glob(os.path.join(HN, "*.jpg")):
        if os.path.basename(p) not in keep:
            os.remove(p)
    save_kind("neg", out)


def sheets(only=None, folder="sheets"):
    items = json.load(open(ITEMS))
    if only is not None:  # `sheets claude`: just the items tally left for Claude, in their own folder
        items = [it for it in items if it["id"] in set(only)]
    order = {"test": 0, "train": 1, "neg": 2}
    items.sort(key=lambda it: (order[it["kind"]], it["id"]))
    d = os.path.join(OUT, folder)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    index, cache = {}, {}
    for n in range(0, len(items), 12):
        cells = []
        for it in items[n:n + 12]:
            p = os.path.join(REPO, it["img"])
            img = cache.get(p)
            if img is None:
                cache.clear()
                img = cache[p] = cv2.imread(p)
            s = CELL / it["crop"]
            cell = cv2.resize(img[it["y0"]:it["y0"] + it["crop"], it["x0"]:it["x0"] + it["crop"]], (CELL, CELL), interpolation=cv2.INTER_CUBIC)
            b = it["props"][0]["box"]
            cv2.rectangle(cell, (int((b[0] - it["x0"]) * s) - 4, int((b[1] - it["y0"]) * s) - 4),
                          (int((b[2] - it["x0"]) * s) + 4, int((b[3] - it["y0"]) * s) + 4), (0, 0, 255), 2)
            cv2.rectangle(cell, (0, 0), (92, 30), (0, 0, 0), -1)
            cv2.putText(cell, str(it["id"]), (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), (60, 60, 60), 1)
            cells.append(cell)
        cells += [np.zeros((CELL, CELL, 3), np.uint8)] * (12 - len(cells))
        name = f"sheet_{n // 12:04d}.jpg"
        cv2.imwrite(os.path.join(d, name), np.vstack([np.hstack(cells[r:r + 4]) for r in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        index[name] = [it["id"] for it in items[n:n + 12]]
    json.dump(index, open(os.path.join(d, "index.json"), "w"), indent=0)
    print(f"{len(index)} sheets, {len(items)} items -> {d}")


def decide(votes, conf, claude=None):
    """votes: one label per voter (None = missing). -> (final_label, final_source, flag)."""
    if claude and claude.get("override"):  # Claude saw the voters agree on a wrong answer (e.g. a phone called a knife)
        return (claude["label"], "claude", "override") if claude["label"] in ("a", "n") else ("", "needs_human", "override")
    flag = ""
    if votes and None not in votes and len(set(votes)) == 1 and votes[0] == "a":
        return "a", "vote", ""
    if votes and None not in votes and len(set(votes)) == 1 and votes[0] == "n":
        if conf < MODEL_NO:
            return "n", "vote", ""
        flag = "no_vs_model"
    else:
        flag = "disagree"
    if claude is None:
        return "", "needs_claude", flag
    if claude.get("sure") and claude["label"] in ("a", "n"):
        return claude["label"], "claude", flag
    return "", "needs_human", flag


assert decide(["a", "a", "a"], 0.9) == ("a", "vote", "")
assert decide(["n", "n", "n"], 0.3) == ("n", "vote", "")
assert decide(["n", "n", "n"], 0.5) == ("", "needs_claude", "no_vs_model")
assert decide(["a", "n", "a"], 0.3, {"label": "a", "sure": True}) == ("a", "claude", "disagree")
assert decide(["a", None, "a"], 0.3, {"label": "a", "sure": False}) == ("", "needs_human", "disagree")
assert decide(["k", "k", "k"], 0.3) == ("", "needs_claude", "disagree")
assert decide(["a", "a", "a"], 0.9, {"label": "n", "sure": True, "override": True}) == ("n", "claude", "override")


def tally(out=OUT):
    items = json.load(open(os.path.join(out, "items.json")))
    vd = os.path.join(out, "votes")
    voters = sorted(os.path.splitext(f)[0] for f in os.listdir(vd) if f.endswith(".json") and f != "claude.json")
    votes = {v: json.load(open(os.path.join(vd, v + ".json"))) for v in voters}
    claude = json.load(open(os.path.join(vd, "claude.json"))) if os.path.exists(os.path.join(vd, "claude.json")) else {}
    path = os.path.join(out, "votes.csv")
    old = {r["tile_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))} if os.path.exists(path) else {}
    fields = ["tile_id", "img", "split", "take", "frame", "kind"] + voters + [
        "n_votes", "agree", "flag_reason", "claude_label", "claude_conf", "human_label", "human_box", "final_label",
        "final_source", "last_update"]
    rows, count, need = [], {}, []
    for it in items:
        tid = str(it["id"])
        vs = [votes[v].get(tid) for v in voters]
        cl = claude.get(tid)
        fl, fs, flag = decide(vs, it["conf"], cl)
        row = dict(tile_id=tid, img=it["img"], split=it["split"], take=it["take"], frame=it["frame"], kind=it["kind"],
                   n_votes=sum(v is not None for v in vs), agree=int(not flag), flag_reason=flag,
                   claude_label=cl["label"] if cl else "", claude_conf=(1.0 if cl.get("sure") else 0.5) if cl else "",
                   final_label=fl, final_source=fs, last_update=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   **{v: votes[v].get(tid, "") for v in voters})
        o = old.get(tid, {})
        if o.get("human_label"):
            row.update(human_label=o["human_label"], human_box=o.get("human_box", ""), final_label=o["final_label"],
                       final_source=o["final_source"], last_update=o.get("last_update", ""))
        rows.append(row)
        count[(it["kind"], row["final_source"], row["final_label"])] = count.get((it["kind"], row["final_source"], row["final_label"]), 0) + 1
        if row["final_source"] == "needs_claude":
            need.append(it["id"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json.dump(need, open(os.path.join(out, "needs_claude.json"), "w"))
    for k in sorted(count):
        print(f"  {k[0]:5} {k[1]:12} {k[2] or '-':3} {count[k]}")
    print(f"voters {voters}; needs Claude: {len(need)}; needs human: {sum(r['final_source'] == 'needs_human' for r in rows)}")


def final_box(it, row):
    """Normalized xyxy knife box decided for an item, None = no knife, "drop" = undecided."""
    lab = row.get("human_label") or row.get("final_label")
    if row.get("human_box"):
        v = json.loads(row["human_box"])
        s = it["crop"] / TILE
        box = [it["x0"] + v[0] * s, it["y0"] + v[1] * s, it["x0"] + v[2] * s, it["y0"] + v[3] * s]
    elif lab == "a":
        box = it["props"][0]["box"]
    elif lab == "n":
        return None
    else:
        return "drop"
    img = cv2.imread(os.path.join(REPO, it["img"]))
    h, w = img.shape[:2]
    return [box[0] / w, box[1] / h, box[2] / w, box[3] / h]


def yolo_line(b):
    return "1 %.6f %.6f %.6f %.6f" % ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1])


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def apply():
    items = {it["id"]: it for it in json.load(open(ITEMS))}
    rows = {int(r["tile_id"]): r for r in csv.DictReader(open(os.path.join(OUT, "votes.csv"), encoding="utf-8"))}
    dec = {i: final_box(items[i], rows[i]) for i in items if i in rows}
    # test label fixes, read by sweep_stage1_rules.load_cache when LAB_FIX is set
    fix = {}
    for i, b in dec.items():
        if items[i]["kind"] == "test" and isinstance(b, list):
            fix.setdefault(os.path.basename(items[i]["img"]), []).append(b)
    json.dump(fix, open(os.path.join(OUT, "test_label_fix.json"), "w"), indent=0)
    print(f"test: {sum(len(v) for v in fix.values())} knives added in {len(fix)} frames, "
          f"{sum(1 for i, b in dec.items() if items[i]['kind'] == 'test' and b is None)} confirmed empty, "
          f"{sum(1 for i, b in dec.items() if items[i]['kind'] == 'test' and b == 'drop')} undecided")
    # training set: lab_gt_aug + added knives on training frames (and their blur/dark copy) + empty-hand frames
    train = [l.strip() for l in open(os.path.join(AUG, "train.txt")) if l.strip()]
    add = {}
    for i, b in dec.items():
        if items[i]["kind"] == "train" and isinstance(b, list):
            add.setdefault(os.path.splitext(os.path.basename(items[i]["img"]))[0], []).append(yolo_line(b))
    out = []
    for p in train:
        stem = os.path.splitext(os.path.basename(p))[0]
        base = stem[:-4] if stem.endswith("_aug") else stem
        if base not in add:
            out.append(p)
            continue
        lbl = os.path.splitext(p.replace("\\", "/").replace("/images/", "/labels/"))[0] + ".txt"
        lines = [l.strip() for l in open(lbl) if l.strip()] if os.path.exists(lbl) else []
        dst = os.path.join(NEW, "images", "fix", stem + ".jpg")
        link(p, dst)
        os.makedirs(os.path.join(NEW, "labels", "fix"), exist_ok=True)
        open(os.path.join(NEW, "labels", "fix", stem + ".txt"), "w").write("".join(l + "\n" for l in lines + add[base]))
        out.append(dst)
    used, why, kinds = 0, {}, {"with knife": 0, "empty hand": 0}
    frames = {}
    for i, it in items.items():
        if it["kind"] == "neg":
            frames.setdefault(it["img"], []).append(i)
    for img, ids in frames.items():
        bs = [dec.get(i, "drop") for i in ids]
        if "drop" in bs:
            why["undecided item"] = why.get("undecided item", 0) + 1
            continue
        # kept when it adds something: an empty-hand box the detector scored >= 0.25, or a held knife now located by eye
        if not any((b is None and items[i]["conf"] >= 0.25) or isinstance(b, list) for i, b in zip(ids, bs)):
            why["nothing decided to learn from"] = why.get("nothing decided to learn from", 0) + 1
            continue
        if items[ids[0]]["neigh_knife"] and not any(isinstance(b, list) for b in bs):
            why["knife near but not located"] = why.get("knife near but not located", 0) + 1
            continue
        stem = os.path.splitext(os.path.basename(img))[0]
        dst = os.path.join(NEW, "images", "neg", stem + ".jpg")
        link(os.path.join(REPO, img), dst)
        os.makedirs(os.path.join(NEW, "labels", "neg"), exist_ok=True)
        open(os.path.join(NEW, "labels", "neg", stem + ".txt"), "w").write("".join(yolo_line(b) + "\n" for b in bs if isinstance(b, list)))
        out.append(dst)
        used += 1
        kinds[("with knife" if any(isinstance(b, list) for b in bs) else "empty hand")] += 1
    open(os.path.join(NEW, "train.txt"), "w").write("".join(l + "\n" for l in out))
    shutil.copy2(os.path.join(AUG, "val.txt"), os.path.join(NEW, "val.txt"))
    y = open(os.path.join(AUG, "data.yaml")).read().replace("lab_gt_aug", "lab_gt_aug_hn")
    open(os.path.join(NEW, "data.yaml"), "w").write(y)
    print(f"train: {sum(len(v) for v in add.values())} knives added on {len(add)} frames; new frames used {used} {kinds}, dropped {why}; "
          f"train.txt {len(out)} lines -> {NEW}")


if __name__ == "__main__":
    mode = ARGS[:2]
    if mode == ["collect", "test"]:
        collect_test()
    elif mode == ["collect", "train"]:
        collect_train()
    elif mode == ["collect", "neg"]:
        collect_neg()
    elif mode == ["sheets"]:
        sheets()
    elif mode == ["sheets", "vote_a"]:  # unanimous "knife" votes, for a second look
        ok = [int(r["tile_id"]) for r in csv.DictReader(open(os.path.join(OUT, "votes.csv"), encoding="utf-8"))
              if r["final_source"] == "vote" and r["final_label"] == "a"]
        sheets(ok, "vote_a_sheets")
    elif mode == ["sheets", "claude"]:
        sheets(json.load(open(os.path.join(OUT, "needs_claude.json"))), "claude_sheets")
    elif mode == ["tally"]:
        tally()
    elif mode == ["apply"]:
        apply()
    elif mode == ["selftest"]:
        c = merge({"x": [(0, 0, .1, .1, .9), (0, 0, .1, .11, .5)], "y": [(0, 0, .1, .1, .6), (.5, .5, .6, .6, .4)]}, [[.5, .5, .6, .6]])
        assert len(c) == 1 and c[0]["src"] == {"x": .9, "y": .6}, c
        print("selftest ok")
    else:
        print(__doc__)
