"""Relabel the images of a Stage 1 set that have no knife box, by eye, with detector boxes as suggestions.

  DEV=0 python relabel_stage1.py <REPO> propose test|train|val <weights> [<weights> ...]
      -> results/stage1_kd/relabel/<set>_items.json and <set>_sheet_NN.jpg: one tile per hand (pose wrists) of every
         box-less image, with up to two detector boxes (a = green, b = cyan) drawn in it. HAND=R: right hand only
  DEV=0 python relabel_stage1.py <REPO> hand test|train|val
      -> in the hand-checked labels the knife is in the right hand (230 of 234 boxes, by the pose model's nearest wrist).
         A decision on a left-hand tile is set to "n-left" (not the knife) in <set>_decisions.json.
  python relabel_stage1.py <REPO> apply test|train|val
      -> reads <set>_decisions.json {tile id: "a" | "b" | "u" | [x1, y1, x2, y2 in tile pixels]}; a tile not listed = no knife.
         "u" = cannot tell: the image is left out of the new set. Writes <set>_labels_v2.json (new boxes only) and the set:
         test -> data/Ground_Truth_Dataset_v2, train/val -> data/stage1_negatives/orig_v2 (images are links, old labels are copied)
  python relabel_stage1.py <REPO> lab all
      -> data/stage1_negatives/lab_v2: the lab1114 set with its copy of the original images replaced by orig_v2

Old label files are never changed.
"""
import glob, json, os, shutil, sys

import cv2
import numpy as np

REPO, MODE, SET = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
OUT = os.path.join(REPO, "results", "stage1_kd", "relabel")
SPLIT = "D:/Backup/Weapon_Detection_Split"
GT = os.path.join(REPO, "data", "Ground_Truth_Dataset").replace("\\", "/")
CROP, TILE, COLS, ROWS = 300, 330, 6, 6  # native pixels around the hand, tile size on the sheet


def images():
    """(image path, label path) of the set."""
    if SET == "test":
        ims = [f for c in ("With_Coat", "Without_Coat") for f in sorted(glob.glob(f"{GT}/{c}/images/*"))]
    else:
        ims = sorted(glob.glob(f"{SPLIT}/images/{SET}/*"))
    ims = [f.replace("\\", "/") for f in ims]
    return [(f, f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt") for f in ims]


def has_box(lab):
    return os.path.exists(lab) and bool(open(lab).read().strip())


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def crop_at(img, cx, cy):
    h, w = img.shape[:2]
    x0, y0 = int(min(max(cx - CROP / 2, 0), w - CROP)), int(min(max(cy - CROP / 2, 0), h - CROP))
    return x0, y0


def propose(weights):
    from ultralytics import YOLO
    dev = int(os.environ.get("DEV", 0))
    todo = [f for f, lab in images() if not has_box(lab)]
    dets = {f: [] for f in todo}
    for w in weights:
        m = YOLO(w, task="detect")
        for f in todo:
            r = m(cv2.imread(f), conf=0.02, imgsz=(384, 640), half=True, device=dev, verbose=False)[0]
            dets[f] += [(float(b.conf[0]), b.xyxy[0].tolist()) for b in r.boxes if m.names[int(b.cls[0])] != "phone"]
    pose = YOLO(os.path.join(REPO, "weights", "yolo26s-pose.pt"))
    items, tiles = [], []
    for f in todo:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        keep = []
        for c, b in sorted(dets[f], reverse=True):  # merge the detectors: best score first, drop overlaps
            if all(iou(b, k[1]) < 0.3 for k in keep):
                keep.append((c, b))
        r = pose(img, imgsz=(384, 640), conf=0.25, device=dev, verbose=False)[0]
        hands = []
        if r.keypoints is not None and r.keypoints.conf is not None:
            for xy, kc in zip(r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()):
                for e, wr in ((7, 9), (8, 10))[1 if os.environ.get("HAND") == "R" else 0:]:  # HAND=R: tiles of the right hand only
                    if kc[wr] >= 0.3:
                        d = (xy[wr] - xy[e]) * 0.4 if kc[e] >= 0.3 else 0  # the hand is past the wrist
                        hands.append(tuple(xy[wr] + d))
        centres = [h_ for h_ in hands]
        for c, b in keep:  # a strong box away from every hand gets its own tile
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if c >= 0.25 and not os.environ.get("HAND") and all(abs(cx - x) > CROP / 3 or abs(cy - y) > CROP / 3 for x, y in centres):
                centres.append((cx, cy))
        if not centres:
            centres = [(w / 2, h / 2)]
        near = lambda x, y, c0: any(c >= c0 and abs((b[0] + b[2]) / 2 - x) < CROP / 2 and abs((b[1] + b[3]) / 2 - y) < CROP / 2 for c, b in keep)
        if any(near(x, y, 0.25) for x, y in centres):  # one knife per video: with a strong box at one hand, a hand with no box is not shown
            centres = [(x, y) for x, y in centres if near(x, y, 0.02)]
        for cx, cy in centres:
            x0, y0 = crop_at(img, cx, cy)
            inside = [(c, b) for c, b in keep if x0 <= (b[0] + b[2]) / 2 < x0 + CROP and y0 <= (b[1] + b[3]) / 2 < y0 + CROP][:2]
            tile = img[y0:y0 + CROP, x0:x0 + CROP].copy()
            for (c, b), col, name in zip(inside, ((0, 255, 0), (255, 255, 0)), "ab"):
                p1, p2 = (int(b[0] - x0) - 5, int(b[1] - y0) - 5), (int(b[2] - x0) + 5, int(b[3] - y0) + 5)
                cv2.rectangle(tile, p1, p2, col, 1)
                cv2.putText(tile, f"{name}{c:.2f}", (p1[0], max(12, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            tile = cv2.resize(tile, (TILE, TILE))
            cv2.putText(tile, str(len(items)), (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tiles.append(tile)
            other = max([c for c, b in keep if (c, b) not in inside], default=0.0)  # best detector score outside this tile
            items.append(dict(id=len(items), img=f, x0=x0, y0=y0, other=other, props=[dict(score=c, box=b) for c, b in inside]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(items, open(os.path.join(OUT, f"{SET}_items.json"), "w"))
    per = COLS * ROWS
    for n in range(0, len(tiles), per):
        t = tiles[n:n + per] + [np.zeros((TILE, TILE, 3), np.uint8)] * (-len(tiles[n:n + per]) % COLS)
        cv2.imwrite(os.path.join(OUT, f"{SET}_sheet_{n // per:02d}.jpg"), np.vstack([np.hstack(t[i:i + COLS]) for i in range(0, len(t), COLS)]))
    print(len(todo), "box-less images,", len(items), "tiles,", -(-len(tiles) // per), "sheets ->", OUT)


def tile_box(it, v):
    if isinstance(v, str):
        return it["props"]["ab".index(v)]["box"]
    return [it["x0"] + v[0] * CROP / TILE, it["y0"] + v[1] * CROP / TILE, it["x0"] + v[2] * CROP / TILE, it["y0"] + v[3] * CROP / TILE]


def hand():
    from ultralytics import YOLO
    pose = YOLO(os.path.join(REPO, "weights", "yolo26s-pose.pt"))
    items = {i["id"]: i for i in json.load(open(os.path.join(OUT, f"{SET}_items.json")))}
    path = os.path.join(OUT, f"{SET}_decisions.json")
    dec, left = json.load(open(path)), []
    for k, v in dec.items():
        it = items[int(k)]
        if isinstance(v, str) and v.startswith("n"):
            continue
        b = [it["x0"], it["y0"], it["x0"] + CROP, it["y0"] + CROP] if v == "u" else tile_box(it, v)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        r = pose(cv2.imread(it["img"]), imgsz=(384, 640), conf=0.25, device=int(os.environ.get("DEV", 0)), verbose=False)[0]
        near = [(np.hypot(xy[wr][0] - cx, xy[wr][1] - cy), name) for xy, kc in zip(r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy())
                for name, wr in (("L", 9), ("R", 10)) if kc[wr] >= 0.3] if r.keypoints is not None and r.keypoints.conf is not None else []
        if near and min(near)[0] < 250 and min(near)[1] == "L":
            left.append((k, v if isinstance(v, str) else "box"))
            dec[k] = "n-left"
    json.dump(dec, open(path, "w"), indent=0)
    print(len(left), "left-hand tiles set to n-left:", left)


def apply():
    items = {i["id"]: i for i in json.load(open(os.path.join(OUT, f"{SET}_items.json")))}
    dec = {int(k): v for k, v in json.load(open(os.path.join(OUT, f"{SET}_decisions.json"))).items() if not str(v).startswith("n")}
    new, unsure = {}, set()
    for k, v in dec.items():
        it = items[k]
        if v == "u":
            unsure.add(it["img"])
            continue
        b = tile_box(it, v)
        if all(iou(b, o) < 0.3 for o in new.get(it["img"], [])):  # the same knife can be in two tiles
            new.setdefault(it["img"], []).append(b)
    # a tile with no decision is "no knife" only for what the tile shows: a detector box elsewhere in the image makes it "cannot tell"
    unsure |= {it["img"] for k, it in items.items() if k not in dec and it.get("other", 0.0) >= 0.10}
    unsure -= set(new)
    root = GT if SET == "test" else SPLIT
    json.dump(dict(new={f[len(root) + 1:]: b for f, b in new.items()}, unsure=sorted(f[len(root) + 1:] for f in unsure)),
              open(os.path.join(OUT, f"{SET}_labels_v2.json"), "w"), indent=0)
    import _winapi
    for f, lab in images() + ([(f.replace("\\", "/"), "") for f in glob.glob(f"{GT}/OOD/images/*")] if SET == "test" else []):
        rel = os.path.dirname(f)[len(root) + 1:]  # With_Coat/images or images/train
        dst_root = os.path.join(REPO, "data", "Ground_Truth_Dataset_v2" if SET == "test" else "stage1_negatives/orig_v2")
        dst_im, dst_lab = os.path.join(dst_root, rel), os.path.join(dst_root, rel.replace("images", "labels"))
        if not os.path.exists(dst_im):
            os.makedirs(os.path.dirname(dst_im), exist_ok=True)
            if SET == "test":  # the test set leaves out the "cannot tell" images, so it gets copies, not a link to the folder
                os.makedirs(dst_im)
            else:
                _winapi.CreateJunction(os.path.dirname(f), dst_im)
        os.makedirs(dst_lab, exist_ok=True)
        name = os.path.splitext(os.path.basename(f))[0]
        if SET == "test":
            if f in unsure:
                continue
            shutil.copy2(f, dst_im)
        if f in new:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            open(os.path.join(dst_lab, name + ".txt"), "w").write("".join(
                f"0 {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}\n" for b in new[f]))
        elif lab and os.path.exists(lab):
            shutil.copy2(lab, dst_lab)
    print(SET, ":", len(new), "images get a box,", sum(len(b) for b in new.values()), "boxes,", len(unsure), "images cannot be told")
    if SET != "test":  # training list without the "cannot tell" images
        dst_root = os.path.join(REPO, "data", "stage1_negatives", "orig_v2").replace("\\", "/")
        open(f"{dst_root}/{SET}.txt", "w").write("".join(f"{dst_root}/images/{SET}/{os.path.basename(f)}\n" for f, _ in images() if f not in unsure))
        open(f"{dst_root}/data.yaml", "w").write(f"path: {dst_root}\ntrain: train.txt\nval: val.txt\nnames:\n  0: knife\n")


def lab_set():
    import _winapi
    neg = os.path.join(REPO, "data", "stage1_negatives").replace("\\", "/")
    for sp in ("train", "val"):
        orig = {os.path.basename(f) for f in glob.glob(f"{SPLIT}/images/{sp}/*")}
        lab = [l.strip() for l in open(f"{neg}/lab1114/{sp}.txt") if l.strip() and os.path.basename(l.strip()) not in orig]
        os.makedirs(f"{neg}/lab_v2/labels/{sp}", exist_ok=True)
        if not os.path.exists(f"{neg}/lab_v2/images/{sp}"):
            os.makedirs(f"{neg}/lab_v2/images", exist_ok=True)
            _winapi.CreateJunction(f"{SPLIT}/images/{sp}", f"{neg}/lab_v2/images/{sp}")
        v2 = [os.path.basename(l.strip()) for l in open(f"{neg}/orig_v2/{sp}.txt") if l.strip()]
        for name in v2:  # class 0 (knife) of the original set is class 1 in the lab set
            src = f"{neg}/orig_v2/labels/{sp}/{os.path.splitext(name)[0]}.txt"
            rows = [l.split() for l in open(src)] if os.path.exists(src) else []
            open(f"{neg}/lab_v2/labels/{sp}/{os.path.splitext(name)[0]}.txt", "w").write("".join("1 " + " ".join(r[1:]) + "\n" for r in rows if r))
        open(f"{neg}/lab_v2/{sp}.txt", "w").write("".join(l + "\n" for l in lab + [f"{neg}/lab_v2/images/{sp}/{n}" for n in v2]))
        print(sp, len(lab), "lab images +", len(v2), "relabelled original images")
    yaml = [f"path: {neg}/lab_v2", "train: train.txt", "val: val.txt", "names:", "  0: hand_grip_knife", "  1: knife", "  2: phone"]
    open(f"{neg}/lab_v2/data.yaml", "w").write("".join(l + "\n" for l in yaml))


if __name__ == "__main__":
    propose(sys.argv[4:]) if MODE == "propose" else hand() if MODE == "hand" else lab_set() if MODE == "lab" else apply()
