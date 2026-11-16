"""Knife labels for the lab recordings (Oct 31, Nov 5, Nov 14), made from hand tiles with detector boxes as suggestions.

  DEV=0 python label_lab_stage1.py <REPO> propose
      -> data/lab_stage1_gt/images/<date>_<take>_fNNNNN.jpg (sampled frames), results/stage1_kd/lab_label/lab_items.json and
         sheets/sheet_NNNN.jpg: one 400-px tile around the knife hand (pose wrist; hand from takes.csv) per frame, 4x4 tiles
         per sheet, with up to three suggested boxes (a green, b cyan, c magenta; letter only). Suggestions: four detectors on the
         full frame at 0.02 and the teacher on the tile itself. TAKES=<substring> limits the takes.
  python label_lab_stage1.py <REPO> apply
      -> reads lab_label/decisions/*.json then lab_decisions_fix.json (overrides), {tile id: "a"|"b"|"c"|"n"|"u"|"k"|[x1,y1,x2,y2]}
         ("n" no knife in the tile, "u" cannot tell, "k" knife but no box fits, a list = box in 330-px tile pixels).
         Writes data/lab_stage1_gt/{labels,test.txt,train.txt,data.yaml} and lab_label/lab_labels.json. An image with no box
         and a "u", "k" or missing tile is left out.
  MAXS=<n> python label_lab_stage1.py <REPO> suspects
      -> lab_label/review/review_NN.jpg (6x6) of the tiles where the decision and the detector scores disagree.

Test split: Nov 14 test takes and Oct 31 take 002, every 10th frame. Train split: Nov 5 and Nov 14 attack takes every 20th
frame, Nov 14 normal takes every 45th. Old labels and recordings are never changed.
"""
import csv, glob, json, os, random, sys

import cv2
import numpy as np

REPO, MODE = os.path.abspath(sys.argv[1]), sys.argv[2]
LAB = os.path.join(REPO, "data", "lab_recordings")
DST = os.path.join(REPO, "data", "lab_stage1_gt")
OUT = os.path.join(REPO, "results", "stage1_kd", "lab_label")
CROP, TILE = 400, 330
COLORS = ((0, 255, 0), (255, 255, 0), (255, 0, 255))
DETS = (("shipped", "yolo_weapon_distilled.pt"), ("teacher", "stage1_repro846_teacher.pt"),
        ("v2", "stage1_repro846_v2_p100.pt"), ("labv2", "stage1_repro846_labv2_p50.pt"))
TAKES = [("test", "2026-11-14/test_attack/*.avi", 10), ("test", "2026-11-14/test_normal/*.avi", 10),
         ("test", "2026-10-31/raw/002_*.avi", 10), ("train", "2026-11-05/train_attack/*.avi", 20),
         ("train", "2026-11-14/train_attack/*.avi", 20), ("train", "2026-11-14/train_normal/*.avi", 45)]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def takes():
    out = []
    for split, pat, step in TAKES:
        for p in sorted(glob.glob(os.path.join(LAB, pat))):
            if os.environ.get("TAKES", "") not in p.replace("\\", "/"):
                continue
            date = os.path.basename(os.path.dirname(os.path.dirname(p)))
            hand = "right"
            tc = os.path.join(LAB, date, "takes.csv")
            if os.path.exists(tc):
                for r in csv.DictReader(open(tc)):
                    if os.path.basename(r["file"].replace("\\", "/")) == os.path.basename(p):
                        hand = r.get("hand") or "right"
            out.append((split, p, step, date, hand))
    return out


def draw(img, it, labels=None):
    s = TILE / it["crop"]  # boxes and letters are drawn after the resize so a large person crop keeps readable letters
    tile = cv2.resize(img[it["y0"]:it["y0"] + it["crop"], it["x0"]:it["x0"] + it["crop"]], (TILE, TILE))
    for p, col in zip(it["props"], COLORS):
        b = p["box"]
        p1 = (int((b[0] - it["x0"]) * s) - 4, int((b[1] - it["y0"]) * s) - 4)
        p2 = (int((b[2] - it["x0"]) * s) + 4, int((b[3] - it["y0"]) * s) + 4)
        cv2.rectangle(tile, p1, p2, col, 1)
        txt = labels(p) if labels else p["letter"]
        cv2.putText(tile, txt, (p1[0], max(14, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return tile


def merge(found):
    """Boxes of all sources, best score first; an overlapping box only adds its source's score to the kept one."""
    keep = []
    for n, c, b in sorted(found, key=lambda t: -t[1]):
        hit = [kp for kp in keep if iou(b, kp["box"]) >= 0.3]
        for kp in hit:
            kp["src"][n] = max(kp["src"].get(n, 0.0), round(c, 3))
        if not hit:
            keep.append(dict(box=[round(v, 1) for v in b], score=round(c, 3), src={n: round(c, 3)}))
    return keep


def inside(keep, x0, y0, crop):
    out = [dict(kp) for kp in keep if x0 <= (kp["box"][0] + kp["box"][2]) / 2 < x0 + crop and y0 <= (kp["box"][1] + kp["box"][3]) / 2 < y0 + crop][:3]
    for kp, letter in zip(out, "abc"):
        kp["letter"] = letter
    return out


def write_sheets(tiles, ids, cols, path_fmt, first=0):
    per = cols * cols
    for n in range(0, len(tiles), per):
        t = tiles[n:n + per]
        t = t + [np.zeros((TILE, TILE, 3), np.uint8)] * (-len(t) % cols)
        cv2.imwrite(path_fmt % (first + n // per), np.vstack([np.hstack(t[i:i + cols]) for i in range(0, len(t), cols)]))
    return -(-len(tiles) // per)


def propose():
    from ultralytics import YOLO
    dev = int(os.environ.get("DEV", 0))
    models = [(n, YOLO(os.path.join(REPO, "weights", w), task="detect")) for n, w in DETS]
    teacher = dict(models)["teacher"]
    pose = YOLO(os.path.join(REPO, "weights", "yolo26s-pose.pt"))
    os.makedirs(os.path.join(DST, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    items, tiles = [], []

    def boxes(m, img, imgsz, conf):
        r = m(img, conf=conf, imgsz=imgsz, half=True, device=dev, verbose=False)[0]
        return [(float(b.conf[0]), b.xyxy[0].tolist()) for b in r.boxes if m.names[int(b.cls[0])] not in ("phone", "hand_grip_knife")]

    for split, path, step, date, hand in takes():
        stem = os.path.basename(path)[:-4]
        cap, k = cv2.VideoCapture(path), -1
        while cap.grab():
            k += 1
            if k % step:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            name = f"{date}_{stem}_f{k:05d}.jpg"
            f = os.path.join(DST, "images", name)
            if not os.path.exists(f):
                cv2.imwrite(f, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = img.shape[:2]
            found = [(n, c, b) for n, m in models for c, b in boxes(m, img, (384, 640), 0.02)]
            r = pose(img, imgsz=(384, 640), conf=0.25, device=dev, verbose=False)[0]
            centre, hand_found = None, False
            if r.keypoints is not None and r.keypoints.conf is not None and len(r.boxes):
                big = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in r.boxes.xyxy.tolist()]))
                xy, kc = r.keypoints.xy[big].cpu().numpy(), r.keypoints.conf[big].cpu().numpy()
                e, wr = (7, 9) if hand == "left" else (8, 10)
                if kc[wr] >= 0.3:
                    d = (xy[wr] - xy[e]) * 0.4 if kc[e] >= 0.3 else 0
                    centre, hand_found = tuple(xy[wr] + d), True
                else:
                    pb = r.boxes.xyxy[big].tolist()
                    centre = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
            best = max(found, key=lambda t: t[1], default=None)
            if not hand_found and best and best[1] >= 0.10:
                centre = ((best[2][0] + best[2][2]) / 2, (best[2][1] + best[2][3]) / 2)
            centre = centre or (w / 2, h / 2)
            crop_at = lambda cx, cy: (int(min(max(cx - CROP / 2, 0), w - CROP)), int(min(max(cy - CROP / 2, 0), h - CROP)))
            x0, y0 = crop_at(*centre)
            found += [("tile", c, [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
                      for c, b in boxes(teacher, img[y0:y0 + CROP, x0:x0 + CROP], 640, 0.05)]
            keep = merge(found)
            centres = [(x0, y0)]
            for kp in keep:
                b = kp["box"]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if kp["score"] >= 0.25 and (abs(cx - centre[0]) > CROP / 3 or abs(cy - centre[1]) > CROP / 3):
                    c0 = crop_at(cx, cy)
                    if all(abs(c0[0] - a) > CROP / 3 or abs(c0[1] - b_) > CROP / 3 for a, b_ in centres):
                        centres.append(c0)
            frame_dets = {n: max([c for n2, c, _ in found if n2 == n], default=0.0) for n, _ in DETS}
            for tx, ty in centres:
                it = dict(id=len(items), img=os.path.relpath(f, REPO).replace("\\", "/"), take=f"{date}/{stem}", split=split,
                          frame=k, x0=tx, y0=ty, crop=CROP, hand_found=hand_found, props=inside(keep, tx, ty, CROP), frame_dets=frame_dets)
                tile = draw(img, it)
                cv2.putText(tile, str(it["id"]), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                items.append(it)
                tiles.append(tile)
        print(split, date, stem, "->", len(items), "tiles so far", flush=True)
    json.dump(items, open(os.path.join(OUT, "lab_items.json"), "w"))
    n = write_sheets(tiles, None, 4, os.path.join(OUT, "sheets", "sheet_%04d.jpg"))
    for s in ("test", "train"):
        print(s, len({i["img"] for i in items if i["split"] == s}), "frames,", sum(i["split"] == s for i in items), "tiles")
    print(n, "sheets ->", os.path.join(OUT, "sheets"))


def persons():
    """Frames whose knife-hand wrist was not found but that show a person get one more tile: a square around the person."""
    from ultralytics import YOLO
    dev = int(os.environ.get("DEV", 0))
    path = os.path.join(OUT, "lab_items.json")
    items = json.load(open(path))
    if any(it.get("kind") == "person" for it in items):
        sys.exit("person tiles already added")
    per = {}
    for it in items:
        per.setdefault(it["img"], []).append(it)
    todo = [f for f, its in per.items() if not any(t["hand_found"] for t in its)]
    models = [(n, YOLO(os.path.join(REPO, "weights", w), task="detect")) for n, w in DETS]
    pose = YOLO(os.path.join(REPO, "weights", "yolo26s-pose.pt"))
    first_sheet, tiles = -(-len(items) // 16), []
    first_id = len(items)
    while first_id % 16:  # start the person tiles on a fresh sheet
        first_id += 1
    for f in todo:
        img = cv2.imread(os.path.join(REPO, f))
        h, w = img.shape[:2]
        r = pose(img, imgsz=(384, 640), conf=0.25, device=dev, verbose=False)[0]
        if not len(r.boxes):
            continue
        pb = max(r.boxes.xyxy.tolist(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        side = int(min(max(pb[2] - pb[0], pb[3] - pb[1]) * 1.15, h, w))
        x0 = int(min(max((pb[0] + pb[2]) / 2 - side / 2, 0), w - side))
        y0 = int(min(max((pb[1] + pb[3]) / 2 - side / 2, 0), h - side))
        found = []
        for n, m in models:
            res = m(img, conf=0.02, imgsz=(384, 640), half=True, device=dev, verbose=False)[0]
            found += [(n, float(b.conf[0]), b.xyxy[0].tolist()) for b in res.boxes if m.names[int(b.cls[0])] not in ("phone", "hand_grip_knife")]
        src = per[f][0]
        it = dict(id=first_id + len(tiles), img=f, take=src["take"], split=src["split"], frame=src["frame"], x0=x0, y0=y0, crop=side,
                  hand_found=False, kind="person", props=inside(merge(found), x0, y0, side), frame_dets=src["frame_dets"])
        tile = draw(img, it)
        cv2.putText(tile, str(it["id"]), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        items.append(it)
        tiles.append(tile)
    json.dump(items, open(path, "w"))
    n = write_sheets(tiles, None, 4, os.path.join(OUT, "sheets", "sheet_%04d.jpg"), first_sheet)
    print(len(todo), "frames without the knife hand,", len(tiles), "person tiles, sheets", first_sheet, "to", first_sheet + n - 1)


def decisions():
    dec = {}
    for p in sorted(glob.glob(os.path.join(OUT, "decisions", "*.json"))) + [os.path.join(OUT, "lab_decisions_fix.json")]:
        if os.path.exists(p):
            dec.update({int(k): v for k, v in json.load(open(p)).items()})
    return dec


def tile_box(it, v):
    if isinstance(v, str):
        return next(p["box"] for p in it["props"] if p["letter"] == v)
    s = it["crop"] / TILE
    return [it["x0"] + v[0] * s, it["y0"] + v[1] * s, it["x0"] + v[2] * s, it["y0"] + v[3] * s]


def apply():
    items, dec = json.load(open(os.path.join(OUT, "lab_items.json"))), decisions()
    per = {}
    for it in items:
        per.setdefault(it["img"], []).append(it)
    out, unsure = {}, []
    stats = {s: dict(images=0, knife=0, no_knife=0, unsure=0, missing=0, k=0) for s in ("test", "train")}
    lists = {"test": [], "train": []}
    os.makedirs(os.path.join(DST, "labels"), exist_ok=True)
    for img, its in per.items():
        st, boxes, doubt = stats[its[0]["split"]], [], False
        st["images"] += 1
        for it in its:
            v = dec.get(it["id"])
            if v is None:
                st["missing"] += 1
                doubt = True
            elif v in ("u", "k"):
                st["k"] += v == "k"
                doubt = True
            elif v != "n" and not (isinstance(v, str) and v not in [p["letter"] for p in it["props"]]):
                b = tile_box(it, v)
                if all(iou(b, o) < 0.3 for o in boxes):
                    boxes.append(b)
            elif v != "n":
                doubt = True  # a letter with no such box
        if not boxes and doubt:
            st["unsure"] += 1
            unsure.append(img)
            continue
        st["knife" if boxes else "no_knife"] += 1
        out[img] = [[round(x, 1) for x in b] for b in boxes]
        h, w = cv2.imread(os.path.join(REPO, img)).shape[:2] if boxes else (1, 1)
        name = os.path.splitext(os.path.basename(img))[0]
        open(os.path.join(DST, "labels", name + ".txt"), "w").write("".join(
            f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}\n" for b in boxes))
        lists[its[0]["split"]].append(os.path.join(REPO, img).replace("\\", "/"))
    for s, l in lists.items():
        open(os.path.join(DST, f"{s}.txt"), "w").write("".join(x + "\n" for x in sorted(l)))
    d = DST.replace("\\", "/")
    open(os.path.join(DST, "data.yaml"), "w").write(f"path: {d}\ntrain: train.txt\nval: test.txt\nnames:\n  0: knife\n")
    json.dump(dict(boxes=out, unsure=sorted(unsure)), open(os.path.join(OUT, "lab_labels.json"), "w"), indent=0)
    for s, st in stats.items():
        print(s, st)


def suspects():
    items, dec = json.load(open(os.path.join(OUT, "lab_items.json"))), decisions()
    sus = {}
    for it in items:
        v, src = dec.get(it["id"]), lambda p, n: p["src"].get(n, 0.0)
        if v is None:
            sus[it["id"]] = "missing"
        elif v == "k":
            sus[it["id"]] = "k"
        elif v == "n" and any(src(p, "shipped") >= 0.45 or src(p, "teacher") >= 0.25 for p in it["props"]):
            sus[it["id"]] = "n_but_detector"
        elif isinstance(v, str) and v in "abc":
            p = next((p for p in it["props"] if p["letter"] == v), None)
            if p is None or (src(p, "shipped") < 0.05 and src(p, "teacher") < 0.05 and src(p, "tile") < 0.3):
                sus[it["id"]] = "box_low_score"
    ids = sorted(sus)
    if os.environ.get("MAXS") and len(ids) > int(os.environ["MAXS"]):
        ids = sorted(random.Random(0).sample(ids, int(os.environ["MAXS"])))
    byid = {it["id"]: it for it in items}
    tiles = []
    for i in ids:
        it = byid[i]
        t = draw(cv2.imread(os.path.join(REPO, it["img"])), it,
                 lambda p: f"{p['letter']} t{p['src'].get('teacher', 0):.2f} s{p['src'].get('shipped', 0):.2f}"[:16].replace("0.", "."))
        cv2.putText(t, f"{i} {dec.get(i)}", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        tiles.append(t)
    os.makedirs(os.path.join(OUT, "review"), exist_ok=True)
    n = write_sheets(tiles, ids, 6, os.path.join(OUT, "review", "review_%02d.jpg"))
    json.dump({f"review_{k:02d}": ids[k * 36:(k + 1) * 36] for k in range(n)}, open(os.path.join(OUT, "review", "review_items.json"), "w"))
    reasons = {}
    for i in sus.values():
        reasons[i] = reasons.get(i, 0) + 1
    print(len(sus), "suspect tiles", reasons, "->", n, "review sheets of", len(ids))


if __name__ == "__main__":
    {"propose": propose, "persons": persons, "apply": apply, "suspects": suspects}[MODE]()
