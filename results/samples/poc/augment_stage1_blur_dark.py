"""Offline motion-blur / darkening augmentation for the Stage 1 lab_gt set.

  .venv/Scripts/python.exe results/samples/poc/augment_stage1_blur_dark.py

Reads data/stage1_negatives/lab_gt, writes one augmented copy of every lab knife frame to
data/stage1_negatives/lab_gt_aug (train.txt = lab_gt train lines + the copies) and a contact sheet
results/stage1_kd/aug_samples.jpg. Fixed seed 0, cv2 + numpy only. lab_gt, data/lab_stage1_gt and
data/lab_recordings are never written.
"""
import os
import random

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "data", "stage1_negatives", "lab_gt")
DST = os.path.join(REPO, "data", "stage1_negatives", "lab_gt_aug")
LAB_IMAGES = os.path.join(REPO, "data", "lab_stage1_gt", "images")
SHEET = os.path.join(REPO, "results", "stage1_kd", "aug_samples.jpg")
SEED = 0
BLUR_P, DARK_P = 0.7, 0.5
REGION_SCALE = 2.5
SHEET_N, COLS, TILE = 24, 8, 256


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def label_for(image):
    """Ultralytics images -> labels rule: .../images/<split>/x.jpg -> .../labels/<split>/x.txt."""
    p = image.replace("\\", "/")
    if "/images/" not in p:
        raise ValueError("no /images/ in " + p)
    return p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def motion_kernel(angle, length):
    length = max(5, int(round(length)))
    if length % 2 == 0:
        length += 1
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2.0
    a = np.deg2rad(angle)
    dx, dy = float(np.cos(a)), float(np.sin(a))
    p1 = (int(round(c - dx * c)), int(round(c - dy * c)))
    p2 = (int(round(c + dx * c)), int(round(c + dy * c)))
    cv2.line(k, p1, p2, 1.0, 1, lineType=cv2.LINE_AA)
    s = float(k.sum())
    if s <= 0:
        k[length // 2, length // 2] = 1.0
        s = 1.0
    return k / s


def kernel_length(box, rng):
    longer = max(box[2] - box[0], box[3] - box[1])
    return max(5, int(round(rng.uniform(0.3, 1.2) * longer)))


def expand_box(box, length, angle, w, h):
    a = np.deg2rad(angle)
    ex = length / 4.0 * abs(float(np.cos(a)))
    ey = length / 4.0 * abs(float(np.sin(a)))
    return [max(0.0, box[0] - ex), max(0.0, box[1] - ey),
            min(float(w), box[2] + ex), min(float(h), box[3] + ey)]


def region(box, w, h, scale=REGION_SCALE):
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    rw, rh = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
    return (int(max(0, np.floor(cx - rw / 2.0))), int(max(0, np.floor(cy - rh / 2.0))),
            int(min(w, np.ceil(cx + rw / 2.0))), int(min(h, np.ceil(cy + rh / 2.0))))


def feathered_blend(img, box, fn, scale=REGION_SCALE, soft=0.08):
    """Blend fn(roi) into img only around box, through a Gaussian-blurred rectangle mask."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = region(box, w, h, scale)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    sigma = max(2.0, soft * min(x2 - x1, y2 - y1))
    pad = int(np.ceil(3.0 * sigma))
    ax1, ay1 = max(0, x1 - pad), max(0, y1 - pad)
    ax2, ay2 = min(w, x2 + pad), min(h, y2 + pad)
    mask = np.zeros((ay2 - ay1, ax2 - ax1), np.float32)
    mask[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1] = 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigma) / 255.0
    roi = img[ay1:ay2, ax1:ax2].astype(np.float32)
    out = roi * (1.0 - mask[..., None]) + fn(roi.copy()) * mask[..., None]
    img[ay1:ay2, ax1:ax2] = np.clip(out, 0, 255).astype(np.uint8)


def blur_fn(kernel):
    return lambda roi: cv2.filter2D(roi, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def dark_fn(brightness, contrast):
    def fn(roi):
        mean = roi.reshape(-1, roi.shape[2]).mean(axis=0, keepdims=True)
        return mean + contrast * (roi * brightness - mean)
    return fn


def read_lines(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def self_check():
    k = motion_kernel(0, 21)
    assert abs(float(k.sum()) - 1.0) < 1e-6, k.sum()
    box = [20.0, 20.0, 40.0, 40.0]
    out = expand_box(box, 21, 0, 100, 100)
    assert out[1] == box[1] and out[3] == box[3] and out[0] < box[0] and out[2] > box[2], out
    print("self_check: kernel sum %.6f, angle 0 widens x only" % k.sum())


def main():
    self_check()
    assert os.path.isdir(SRC), SRC
    orig = read_lines(os.path.join(SRC, "train.txt"))
    val = read_lines(os.path.join(SRC, "val.txt"))
    lab_root = norm(os.path.realpath(LAB_IMAGES))

    selected = []
    for p in orig:
        if not norm(os.path.realpath(p)).startswith(lab_root):
            continue
        lf = label_for(p)
        if not os.path.exists(lf):
            continue
        rows = read_lines(lf)
        if any(r.split()[0] == "1" for r in rows):
            selected.append(p)

    for d in ("images/aug", "labels/aug"):
        os.makedirs(os.path.join(DST, *d.split("/")), exist_ok=True)

    rng = random.Random(SEED)
    records, added = [], []
    n_blur = n_dark = n_both = 0
    for p in selected:
        img = cv2.imread(p)
        if img is None:
            print("skip unreadable", p)
            continue
        h, w = img.shape[:2]
        lf = label_for(p)
        rows = read_lines(lf)
        knife = [(i, r) for i, r in enumerate(rows) if r.split()[0] == "1"]
        if not knife:
            continue

        do_blur = rng.random() < BLUR_P
        do_dark = rng.random() < DARK_P
        if not do_blur and not do_dark:
            do_blur = True
        n_blur += do_blur
        n_dark += do_dark
        n_both += do_blur and do_dark

        new_rows = list(rows)
        for i, r in knife:
            v = r.split()
            cx, cy, bw, bh = (float(x) for x in v[1:5])
            box = [(cx - bw / 2.0) * w, (cy - bh / 2.0) * h,
                   (cx + bw / 2.0) * w, (cy + bh / 2.0) * h]
            new_box = box
            if do_blur:
                angle = rng.uniform(0.0, 180.0)
                length = kernel_length(box, rng)
                new_box = expand_box(box, length, angle, w, h)
                feathered_blend(img, new_box, blur_fn(motion_kernel(angle, length)), 1.5)  # the smear and the hand, not the whole arm
            if do_dark:
                brightness = rng.uniform(0.35, 0.7)
                contrast = rng.uniform(0.5, 0.8)
                feathered_blend(img, box, dark_fn(brightness, contrast), soft=0.35)  # a hard-edged dark patch would be a shortcut to 'knife'
            if do_blur:
                nx1, ny1, nx2, ny2 = new_box
                new_rows[i] = "%s %.6f %.6f %.6f %.6f" % (
                    v[0], (nx1 + nx2) / 2.0 / w, (ny1 + ny2) / 2.0 / h,
                    (nx2 - nx1) / w, (ny2 - ny1) / h)

        stem = os.path.splitext(os.path.basename(p))[0] + "_aug"
        out_img = os.path.join(DST, "images", "aug", stem + ".jpg")
        out_lbl = os.path.join(DST, "labels", "aug", stem + ".txt")
        if not cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError("cannot write " + out_img)
        open(out_lbl, "w", encoding="utf-8").write("".join(r + "\n" for r in new_rows))

        first = next(r for i, r in enumerate(new_rows) if r.split()[0] == "1")
        v = first.split()
        cx, cy, bw, bh = (float(x) for x in v[1:5])
        new_box = [(cx - bw / 2.0) * w, (cy - bh / 2.0) * h,
                   (cx + bw / 2.0) * w, (cy + bh / 2.0) * h]
        records.append((out_img, new_box))
        added.append(out_img)

    train = orig + added
    assert not any("_test_" in p or "2026-10-31" in p for p in train), "test take in train.txt"
    trainset = {norm(os.path.realpath(p)) for p in train}
    assert not any(norm(os.path.realpath(p)) in trainset for p in val), "val image in train.txt"
    assert len(train) == len(orig) + len(added)

    open(os.path.join(DST, "train.txt"), "w", encoding="utf-8").write("".join(p + "\n" for p in train))
    open(os.path.join(DST, "val.txt"), "w", encoding="utf-8").write("".join(p + "\n" for p in val))
    src_yaml = open(os.path.join(SRC, "data.yaml"), encoding="utf-8").read().splitlines()
    dst_yaml = ["path: " + DST.replace(os.sep, "/") if l.strip().startswith("path:") else l for l in src_yaml]
    open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8").write("\n".join(dst_yaml) + "\n")

    sheet_rng = random.Random(SEED)
    picks = sheet_rng.sample(records, min(SHEET_N, len(records)))
    tiles = []
    for path, box in picks:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        bw, bh = box[2] - box[0], box[3] - box[1]
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        cw, ch = max(1.0, bw * 4.0), max(1.0, bh * 4.0)
        x1, y1 = max(0, int(cx - cw / 2.0)), max(0, int(cy - ch / 2.0))
        x2, y2 = min(w, int(cx + cw / 2.0)), min(h, int(cy + ch / 2.0))
        crop = cv2.resize(img[y1:y2, x1:x2], (TILE, TILE))
        sx, sy = TILE / float(max(1, x2 - x1)), TILE / float(max(1, y2 - y1))
        p1 = (int(round((box[0] - x1) * sx)), int(round((box[1] - y1) * sy)))
        p2 = (int(round((box[2] - x1) * sx)), int(round((box[3] - y1) * sy)))
        cv2.rectangle(crop, p1, p2, (0, 255, 0), 2)
        tiles.append(crop)
    while len(tiles) % COLS:
        tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    if rows:
        cv2.imwrite(SHEET, np.vstack(rows))

    print("original train lines:", len(orig))
    print("selected lab knife frames:", len(selected))
    print("augmented copies:", len(added))
    print("blurred:", n_blur, "darkened:", n_dark, "both:", n_both)
    print("train.txt lines:", len(train))
    print("sheet ->", SHEET)


if __name__ == "__main__":
    main()
