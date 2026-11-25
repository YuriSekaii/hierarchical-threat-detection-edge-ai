"""Hybrid out-of-fold + feature-space + goal-alignment audit for Stage 1 training labels (lab_gt).

Modes:
  split        Group lab_gt/train.txt by take/clip into balanced Fold A and Fold B (no take leakage).
  train <a|b>  Train 60-epoch YOLO26s on Fold A or Fold B using Ultralytics 8.4.6 MuSGD recipe.
  score        Out-of-fold inference, E2ELoss, detect-head TracIn alignment, YOLO P3/P4 k-NN, pose, v2_p100, cropcls2.
  triage       Rank Safe vs Suspect frames, check multi-box duplicates & temporal flips, build /swarm contact sheets.
  selftest     Quick invariant checks on grouping, box/cover math, and fold disjointness.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "results", "stage1_kd", "train_audit")
LAB_GT = os.path.join(REPO, "data", "stage1_negatives", "lab_gt")
LAB_HN = os.path.join(REPO, "data", "stage1_negatives", "lab_gt_aug_hn")
VOTES_CSV = os.path.join(REPO, "results", "stage1_kd", "lab_label", "votes.csv")


def norm_path(p):
    return p.strip().replace("\\", "/")


def label_path_for(img_path):
    p = norm_path(img_path)
    return os.path.splitext(p.replace("/images/", "/labels/"))[0] + ".txt"


def read_labels(img_path):
    """Return list of (cls_id, xc, yc, w, h) normalized floats."""
    lp = label_path_for(img_path)
    if not os.path.exists(lp):
        return []
    rows = []
    for line in open(lp, encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) >= 5:
            rows.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def take_group(img_path):
    """Group frames by recording take / clip so no take leaks across folds."""
    p = norm_path(img_path)
    b = os.path.basename(p)
    if "/lab_v2/" in p:
        stem = re.sub(r"_frame_\d+.*$", "", os.path.splitext(b)[0])
        return "orig_" + stem
    stem = re.sub(r"^(n14_|\d{4}-\d\d-\d\d_)", "", os.path.splitext(b)[0])
    if "_f" in stem:
        return "lab_" + stem.rsplit("_f", 1)[0]
    return "misc_" + os.path.splitext(b)[0]


def source_kind(img_path):
    p = norm_path(img_path)
    if "/lab_v2/" in p:
        return "orig_v2"
    if "/ft_phonecls/" in p:
        return "phone"
    if "/lab_gt/" in p:
        return "lab_gt"
    return "other"


def xywhn2xyxy(xc, yc, w, h):
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def box_cover(a, b):
    """Fraction of smaller box covered by intersection."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(min(aa, bb), 1e-9)


def human_train_basenames():
    if not os.path.exists(VOTES_CSV):
        return set()
    with open(VOTES_CSV, encoding="utf-8") as f:
        return {
            os.path.basename(r["img"])
            for r in csv.DictReader(f)
            if r.get("split") == "train" and r.get("final_source") == "human"
        }


def split():
    os.makedirs(OUT, exist_ok=True)
    lines = [norm_path(l) for l in open(os.path.join(LAB_GT, "train.txt"), encoding="utf-8") if l.strip()]
    assert not any("_test_" in p or "2026-10-31" in p or "Ground_Truth_Dataset" in p for p in lines), "Test leak in train.txt!"
    humans = human_train_basenames()

    by_group = defaultdict(list)
    for p in lines:
        by_group[take_group(p)].append(p)

    # Greedy stratified assignment by group size (largest first) keeping sources + positives + humans balanced
    groups = sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    folds = {"a": [], "b": []}
    stats = {"a": Counter(), "b": Counter()}

    for gname, paths in groups:
        src = Counter(source_kind(p) for p in paths).most_common(1)[0][0]
        n_pos = sum(any(r[0] == 1 for r in read_labels(p)) for p in paths)
        n_hum = sum(os.path.basename(p) in humans for p in paths)
        # Pick fold with fewer total images (tie-break by same-source count and human count)
        score_a = (stats["a"][src], stats["a"]["total"], stats["a"]["pos"], stats["a"]["hum"])
        score_b = (stats["b"][src], stats["b"]["total"], stats["b"]["pos"], stats["b"]["hum"])
        pick = "a" if score_a <= score_b else "b"
        folds[pick].extend(paths)
        stats[pick][src] += len(paths)
        stats[pick]["total"] += len(paths)
        stats[pick]["pos"] += n_pos
        stats[pick]["hum"] += n_hum

    ga = {take_group(p) for p in folds["a"]}
    gb = {take_group(p) for p in folds["b"]}
    assert ga.isdisjoint(gb), f"Group overlap: {ga & gb}"
    assert set(folds["a"]).isdisjoint(set(folds["b"]))
    assert len(folds["a"]) + len(folds["b"]) == len(lines)

    val_txt = os.path.join(LAB_GT, "val.txt").replace("\\", "/")
    meta = {}
    for k in ("a", "b"):
        fdir = os.path.join(OUT, f"fold_{k}")
        os.makedirs(fdir, exist_ok=True)
        with open(os.path.join(fdir, "train.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(folds[k])) + "\n")
        with open(os.path.join(fdir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(
                f"path: {LAB_GT.replace(chr(92), '/')}\n"
                f"train: {os.path.join(fdir, 'train.txt').replace(chr(92), '/')}\n"
                f"val: {val_txt}\n"
                f"names:\n  0: hand_grip_knife\n  1: knife\n  2: phone\n"
            )
        meta[k] = dict(stats[k])
        print(f"fold_{k}:", dict(stats[k]))
    with open(os.path.join(OUT, "fold_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_fold(fold):
    assert fold in ("a", "b")
    sys.path.insert(0, os.path.join(REPO, ".ul846"))
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer

    assert ultralytics.__version__ == "8.4.6", ultralytics.__version__
    dev = int(os.environ.get("DEV", 0 if fold == "a" else 1))
    epochs = int(os.environ.get("EPOCHS", 60))
    workers = int(os.environ.get("WORKERS", 2))
    data_yaml = os.path.join(OUT, f"fold_{fold}", "data.yaml")
    assert os.path.exists(data_yaml), "Run split first"

    build = BaseTrainer.build_optimizer
    BaseTrainer.build_optimizer = lambda self, iterations=1e5, **kw: build(self, iterations=min(iterations, 9999), **kw)
    model = YOLO(r"D:/Backup/yolo26s.pt")
    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=epochs,
        patience=epochs,
        batch=32,
        optimizer="MuSGD",
        lr0=0.002,
        momentum=0.9,
        warmup_bias_lr=0.0,
        degrees=15.0,
        mosaic=1.0,
        mixup=0.1,
        plots=False,
        exist_ok=True,
        workers=workers,
        device=dev,
        project=OUT,
        name=f"run_fold_{fold}",
    )


def letterbox_640(img_bgr):
    """Letterbox BGR uint8 image to 640x640, return (tensor_1x3x640x640_rgb_01, scale, pad_x, pad_y, w0, h0)."""
    import torch

    h0, w0 = img_bgr.shape[:2]
    r = min(640.0 / h0, 640.0 / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (640 - nw) / 2.0, (640 - nh) / 2.0
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0) / 255.0
    return t, r, left, top, w0, h0


def norm_boxes_to_letterbox(labels, r, left, top, w0, h0):
    """Convert original normalized xywh labels to letterboxed 640x640 normalized xywh."""
    out = []
    for cls_id, xc, yc, bw, bh in labels:
        x_lb = (xc * w0 * r + left) / 640.0
        y_lb = (yc * h0 * r + top) / 640.0
        w_lb = (bw * w0 * r) / 640.0
        h_lb = (bh * h0 * r) / 640.0
        out.append((cls_id, x_lb, y_lb, w_lb, h_lb))
    return out


def pool_neck_vec(feats, box_xyxy_norm_lb):
    """Pool 1x1 from P3 (layer 16) + P4 (layer 19) inside letterboxed normalized xyxy box -> 384-d L2 vector."""
    import torch
    import torch.nn.functional as F

    x1, y1, x2, y2 = box_xyxy_norm_lb
    vecs = []
    for key in (16, 19):
        fm = feats[key]  # [1, C, H, W]
        _, c, fh, fw = fm.shape
        gx1 = max(0, min(fw - 1, int(np.floor(x1 * fw))))
        gy1 = max(0, min(fh - 1, int(np.floor(y1 * fh))))
        gx2 = max(gx1 + 1, min(fw, int(np.ceil(x2 * fw))))
        gy2 = max(gy1 + 1, min(fh, int(np.ceil(y2 * fh))))
        sub = fm[0, :, gy1:gy2, gx1:gx2].mean(dim=(1, 2))
        vecs.append(sub)
    v = torch.cat(vecs, dim=0).float()
    return F.normalize(v, dim=0).cpu().numpy()


def extract_wrists(pose_model, img_bgr, dev):
    """Return list of (wx_norm, wy_norm, torso_norm) for wrists with conf >= 0.3."""
    h, w = img_bgr.shape[:2]
    res = pose_model.predict(img_bgr, imgsz=(384, 640), conf=0.25, device=dev, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        return []
    kpts = res.keypoints.xy.cpu().numpy()
    confs = res.keypoints.conf.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()
    out = []
    for kp, kc, pb in zip(kpts, confs, boxes):
        # Torso scale from shoulder-hip midpoint distance, fallback to 0.4 * person height
        sho = [kp[i] for i in (5, 6) if kc[i] >= 0.3]
        hip = [kp[i] for i in (11, 12) if kc[i] >= 0.3]
        if sho and hip:
            sm, hm = np.mean(sho, axis=0), np.mean(hip, axis=0)
            torso = float(np.linalg.norm(sm - hm)) / max(h, w)
        else:
            torso = 0.4 * float(pb[3] - pb[1]) / max(h, w)
        torso = max(torso, 0.06)
        for wi in (9, 10):
            if kc[wi] >= 0.3:
                out.append((float(kp[wi, 0]) / w, float(kp[wi, 1]) / h, torso))
    return out


def cropcls_prob(cls_model, img_bgr, xyxy_norm, dev=0, enlarge=0.5):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy_norm[0] * w, xyxy_norm[1] * h, xyxy_norm[2] * w, xyxy_norm[3] * h
    bw, bh = max(x2 - x1, 12.0), max(y2 - y1, 12.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = bw * (0.5 + enlarge), bh * (0.5 + enlarge)
    ix1, iy1 = max(0, int(cx - hw)), max(0, int(cy - hh))
    ix2, iy2 = min(w, int(cx + hw)), min(h, int(cy + hh))
    crop = img_bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return 0.0
    r = cls_model.predict(crop, imgsz=224, device=dev, verbose=False)[0]
    names = cls_model.names
    k_idx = next(k for k, v in names.items() if v == "knife")
    return float(r.probs.data[k_idx].item())


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score():
    import torch
    sys.path.insert(0, os.path.join(REPO, ".ul846"))
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    dev = int(os.environ.get("DEV", 0))
    humans = human_train_basenames()

    # Load external (non-lab_gt) helpers using .ul846 or default ultralytics
    pose_model = YOLO(os.path.join(REPO, "weights", "yolo26s-pose.pt"))
    v2_model = YOLO(os.path.join(REPO, "weights", "stage1_repro846_v2_p100.pt"))
    cls_model = YOLO(os.path.join(REPO, "weights", "stage1_cropcls2.pt"))

    records = []
    for eval_fold, model_fold in (("a", "b"), ("b", "a")):
        wpath = os.path.join(OUT, f"run_fold_{model_fold}", "weights", "last.pt")
        assert os.path.exists(wpath), f"Missing {wpath}; run train {model_fold} first"
        eval_paths = [norm_path(l) for l in open(os.path.join(OUT, f"fold_{eval_fold}", "train.txt"), encoding="utf-8") if l.strip()]

        # Setup separate predict wrapper (predict() fuses/mutates its model) and raw model for loss/grads/features
        yolo_oof = YOLO(wpath)
        m = YOLO(wpath).model.cuda(dev).eval()
        m.args = get_cfg()
        for p in m.parameters():
            p.requires_grad_(False)
        head_params = list(m.model[-1].parameters())
        cls_params = list(m.model[-1].cv3.parameters()) + list(m.model[-1].one2one_cv3.parameters())
        for p in head_params:
            p.requires_grad_(True)
        crit = m.init_criterion()

        feats = {}
        hooks = [
            m.model[16].register_forward_hook(lambda mod, inp, out: feats.__setitem__(16, out.detach())),
            m.model[19].register_forward_hook(lambda mod, inp, out: feats.__setitem__(19, out.detach())),
        ]

        # Pass 1a: compute clean anchor gradient sums on unseen human-checked frames in this fold
        def img_batch(lbls_lb):
            if lbls_lb:
                return {
                    "batch_idx": torch.zeros(len(lbls_lb), device=f"cuda:{dev}", dtype=torch.float32),
                    "cls": torch.tensor([[x[0]] for x in lbls_lb], device=f"cuda:{dev}", dtype=torch.float32),
                    "bboxes": torch.tensor([x[1:] for x in lbls_lb], device=f"cuda:{dev}", dtype=torch.float32),
                }
            return {
                "batch_idx": torch.zeros(0, device=f"cuda:{dev}", dtype=torch.float32),
                "cls": torch.zeros((0, 1), device=f"cuda:{dev}", dtype=torch.float32),
                "bboxes": torch.zeros((0, 4), device=f"cuda:{dev}", dtype=torch.float32),
            }

        def grad_vec(img_bgr, lbls):
            t, r, left, top, w0, h0 = letterbox_640(img_bgr)
            t = t.cuda(dev)
            lb_lbls = norm_boxes_to_letterbox(lbls, r, left, top, w0, h0)
            m.zero_grad()
            preds = m(t)
            loss, items = crit(preds, img_batch(lb_lbls))
            loss.sum().backward()
            gf = torch.cat([
                (q.grad.detach() if q.grad is not None else torch.zeros_like(q)).reshape(-1)
                for q in head_params
            ]).float().cpu().numpy()
            gc = torch.cat([
                (q.grad.detach() if q.grad is not None else torch.zeros_like(q)).reshape(-1)
                for q in cls_params
            ]).float().cpu().numpy()
            return gf, gc, items.detach().cpu().tolist(), (r, left, top, w0, h0, lb_lbls)

        sum_all = sum_pos = sum_neg = None
        sum_cls_all = sum_cls_pos = sum_cls_neg = None
        n_pos_hum = n_neg_hum = 0
        for p in eval_paths:
            if os.path.basename(p) not in humans:
                continue
            img_bgr = cv2.imread(p)
            lbls = read_labels(p)
            has_k = any(x[0] == 1 for x in lbls)
            gf, gc, _, _ = grad_vec(img_bgr, lbls)
            sum_all = gf if sum_all is None else (sum_all + gf)
            sum_cls_all = gc if sum_cls_all is None else (sum_cls_all + gc)
            if has_k:
                sum_pos = gf if sum_pos is None else (sum_pos + gf)
                sum_cls_pos = gc if sum_cls_pos is None else (sum_cls_pos + gc)
                n_pos_hum += 1
            else:
                sum_neg = gf if sum_neg is None else (sum_neg + gf)
                sum_cls_neg = gc if sum_cls_neg is None else (sum_cls_neg + gc)
                n_neg_hum += 1
        assert n_pos_hum > 10 and n_neg_hum > 10, (n_pos_hum, n_neg_hum)

        # Pass 1b: run OOF forward/backward + pose + v2 + cropcls + collect features & streaming gradient cosine
        fold_recs = []
        roi_vecs = []       # (rec_idx, kind='gt'|'wrist', is_knife_label, take_grp, is_human, vec)

        print(f"[score] fold_{eval_fold} ({len(eval_paths)} imgs, {n_pos_hum}+{n_neg_hum} human anchor) evaluated by fold_{model_fold}", flush=True)
        for idx, p in enumerate(eval_paths):
            img_bgr = cv2.imread(p)
            assert img_bgr is not None, p
            lbls = read_labels(p)
            gf, gc, items_list, (r, left, top, w0, h0, lb_lbls) = grad_vec(img_bgr, lbls)
            is_hum = os.path.basename(p) in humans
            gt_knife_boxes = [xywhn2xyxy(*x[1:]) for x in lbls if x[0] == 1]
            has_k = len(gt_knife_boxes) > 0

            a_all = (sum_all - gf) if is_hum else sum_all
            a_pos = (sum_pos - gf) if (is_hum and has_k) else sum_pos
            a_neg = (sum_neg - gf) if (is_hum and not has_k) else sum_neg
            ca_all = (sum_cls_all - gc) if is_hum else sum_cls_all
            ca_pos = (sum_cls_pos - gc) if (is_hum and has_k) else sum_cls_pos
            ca_neg = (sum_cls_neg - gc) if (is_hum and not has_k) else sum_cls_neg

            # Decode OOF + external v2 predictions
            with torch.no_grad():
                oof_res = yolo_oof.predict(img_bgr, imgsz=640, conf=0.05, device=dev, verbose=False)[0]
                oof_boxes = [
                    {"xyxy": [float(v) for v in b], "conf": float(c), "cls": int(k)}
                    for b, c, k in zip(oof_res.boxes.xyxyn.tolist(), oof_res.boxes.conf.tolist(), oof_res.boxes.cls.tolist())
                ]
                v2_res = v2_model.predict(img_bgr, imgsz=640, conf=0.05, device=dev, verbose=False)[0]
                v2_boxes = [
                    {"xyxy": [float(v) for v in b], "conf": float(c), "cls": int(k)}
                    for b, c, k in zip(v2_res.boxes.xyxyn.tolist(), v2_res.boxes.conf.tolist(), v2_res.boxes.cls.tolist())
                ]

            wrists = extract_wrists(pose_model, img_bgr, dev)
            grp = take_group(p)
            gt_all_boxes = [(x[0], xywhn2xyxy(*x[1:])) for x in lbls]

            # Pool P3/P4 features & cropcls2 on GT knife boxes
            gt_cls_probs = []
            for kb_lb, kb_orig in zip([xywhn2xyxy(*x[1:]) for x in lb_lbls if x[0] == 1], gt_knife_boxes):
                vec = pool_neck_vec(feats, kb_lb)
                roi_vecs.append((len(fold_recs), "gt", 1, grp, is_hum, vec))
                gt_cls_probs.append(cropcls_prob(cls_model, img_bgr, kb_orig, dev=dev, enlarge=0.5))

            # Pool P3/P4 features & cropcls2 on wrists for negative frames
            wrist_cls_probs = []
            if not gt_knife_boxes:
                for wx, wy, torso in wrists:
                    s = 0.15 * max(torso / 0.15, 1.0)
                    wb_orig = [max(0.0, wx - s), max(0.0, wy - s), min(1.0, wx + s), min(1.0, wy + s)]
                    wx1 = (wb_orig[0] * w0 * r + left) / 640.0
                    wy1 = (wb_orig[1] * h0 * r + top) / 640.0
                    wx2 = (wb_orig[2] * w0 * r + left) / 640.0
                    wy2 = (wb_orig[3] * h0 * r + top) / 640.0
                    vec = pool_neck_vec(feats, [wx1, wy1, wx2, wy2])
                    roi_vecs.append((len(fold_recs), "wrist", 0, grp, is_hum, vec))
                    wrist_cls_probs.append(cropcls_prob(cls_model, img_bgr, wb_orig, dev=dev, enlarge=0.25))

            rec = dict(
                path=p,
                fold=eval_fold,
                take=grp,
                source=source_kind(p),
                is_human=is_hum,
                has_knife=has_k,
                gt_all=gt_all_boxes,
                gt_knives=gt_knife_boxes,
                oof_boxes=oof_boxes,
                v2_boxes=v2_boxes,
                wrists=wrists,
                loss=[round(float(v), 5) for v in items_list],
                gt_cls_probs=[round(v, 4) for v in gt_cls_probs],
                wrist_cls_probs=[round(v, 4) for v in wrist_cls_probs],
                grad_cos_all=round(cosine_sim(gf, a_all), 5),
                grad_cos_pos=round(cosine_sim(gf, a_pos), 5),
                grad_cos_neg=round(cosine_sim(gf, a_neg), 5),
                grad_cls_cos_all=round(cosine_sim(gc, ca_all), 5),
                grad_cls_cos_pos=round(cosine_sim(gc, ca_pos), 5),
                grad_cls_cos_neg=round(cosine_sim(gc, ca_neg), 5),
            )
            fold_recs.append(rec)
            if (idx + 1) % 400 == 0 or idx + 1 == len(eval_paths):
                print(f"  fold_{eval_fold}: {idx + 1}/{len(eval_paths)}", flush=True)

        for h in hooks:
            h.remove()

        # Pass 3: Cross-take k-NN in unseen YOLO P3/P4 neck feature space
        if roi_vecs:
            V = np.stack([x[5] for x in roi_vecs], axis=0)  # [N_roi, 384] already L2-normed
            labels_arr = np.array([x[2] for x in roi_vecs])
            takes_arr = np.array([x[3] for x in roi_vecs])
            hum_arr = np.array([x[4] for x in roi_vecs])
            sim = V @ V.T  # [N_roi, N_roi]

            clean_knife_mask = (labels_arr == 1) & hum_arr
            knife_indices = np.where(labels_arr == 1)[0]
            for r_i, (rec_idx, kind, lbl, grp, _, _) in enumerate(roi_vecs):
                same_take = takes_arr == grp
                row_sim = sim[r_i].copy()
                row_sim[same_take] = -2.0
                # Top-15 cross-take neighbors across all ROIs (GT knives + negative wrists)
                top15 = np.argpartition(-row_sim, 15)[:15]
                knife_frac = float(labels_arr[top15].mean())
                # Mean sim to top-5 cross-take human-checked knives (or all knives)
                ksim = row_sim[clean_knife_mask if clean_knife_mask.sum() >= 5 else knife_indices]
                top5_k = np.sort(ksim)[-5:]
                knife_sim = float(top5_k.mean())
                rec = fold_recs[rec_idx]
                if kind == "gt":
                    rec.setdefault("gt_knn_knife_frac", []).append(round(knife_frac, 4))
                    rec.setdefault("gt_knn_knife_sim", []).append(round(knife_sim, 4))
                else:
                    rec.setdefault("wrist_knn_knife_frac", []).append(round(knife_frac, 4))
                    rec.setdefault("wrist_knn_knife_sim", []).append(round(knife_sim, 4))

        records.extend(fold_recs)

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "audit_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} scored records -> {out_path}")


def frame_num(path):
    m = re.search(r"_f(\d+)\.", os.path.basename(path))
    return int(m.group(1)) if m else None


def triage():
    scores_path = os.path.join(OUT, "audit_scores.json")
    assert os.path.exists(scores_path), "Run score first"
    recs = json.load(open(scores_path, encoding="utf-8"))

    # Group lab takes temporally for flip detection
    by_take = defaultdict(list)
    for idx, r in enumerate(recs):
        fn = frame_num(r["path"])
        if fn is not None and r["take"].startswith("lab_"):
            by_take[r["take"]].append((fn, idx))
    temporal_flip = set()
    for take, seq in by_take.items():
        seq.sort()
        for k in range(1, len(seq) - 1):
            f_prev, i_prev = seq[k - 1]
            f_cur, i_cur = seq[k]
            f_next, i_next = seq[k + 1]
            if f_next - f_prev > 120:
                continue
            p_prev = recs[i_prev]["has_knife"]
            p_cur = recs[i_cur]["has_knife"]
            p_next = recs[i_next]["has_knife"]
            if p_prev == p_next and p_cur != p_prev:
                temporal_flip.add(i_cur)

    # Check multi-box frames in lab_gt AND lab_gt_aug_hn additions
    multi_dups = []
    for idx, r in enumerate(recs):
        if len(r["gt_knives"]) > 1:
            bs = r["gt_knives"]
            max_cov = max(box_cover(bs[i], bs[j]) for i in range(len(bs)) for j in range(i + 1, len(bs)))
            max_iou = max(box_iou(bs[i], bs[j]) for i in range(len(bs)) for j in range(i + 1, len(bs)))
            multi_dups.append(dict(idx=idx, img=os.path.basename(r["path"]), n=len(bs), max_iou=round(max_iou, 3), max_cov=round(max_cov, 3)))

    hn_dups = []
    hn_train = os.path.join(LAB_HN, "train.txt")
    if os.path.exists(hn_train):
        for line in open(hn_train, encoding="utf-8"):
            p = norm_path(line)
            if "/fix/" in p or "/neg/" in p:
                lbls = [xywhn2xyxy(*x[1:]) for x in read_labels(p) if x[0] == 1]
                if len(lbls) > 1:
                    max_cov = max(box_cover(lbls[i], lbls[j]) for i in range(len(lbls)) for j in range(i + 1, len(lbls)))
                    max_iou = max(box_iou(lbls[i], lbls[j]) for i in range(len(lbls)) for j in range(i + 1, len(lbls)))
                    if max_cov >= 0.3 or max_iou >= 0.1:
                        hn_dups.append(dict(img=os.path.basename(p), n=len(lbls), max_iou=round(max_iou, 3), max_cov=round(max_cov, 3)))

    # Thresholds calibrated against distributions (separate lab+orig negatives from phone-class frames)
    pos_recs = [r for r in recs if r["has_knife"]]
    neg_lab = [r for r in recs if not r["has_knife"] and r["source"] in ("lab_gt", "orig_v2")]
    pos_losses = np.array([sum(r["loss"]) for r in pos_recs])
    neg_losses = np.array([r["loss"][1] for r in neg_lab])  # cls loss on non-phone negs
    pos_loss_hi = float(np.quantile(pos_losses, 0.90))
    neg_loss_hi = float(np.quantile(neg_losses, 0.90))

    suspects = []
    safe_count = 0
    by_source = defaultdict(Counter)
    flag_counts = Counter()

    for idx, r in enumerate(recs):
        flags = []
        oof_knives = [b for b in r["oof_boxes"] if b["cls"] == 1]
        v2_knives = [b for b in r["v2_boxes"] if b["cls"] == 0]  # orig_v2 is single-class 0: knife
        wrists = r["wrists"]

        if r["has_knife"]:
            best_oof_confs, best_v2_confs, wrist_dists = [], [], []
            for kb in r["gt_knives"]:
                c_oof = max([b["conf"] for b in oof_knives if box_iou(b["xyxy"], kb) >= 0.2 or box_cover(b["xyxy"], kb) >= 0.5], default=0.0)
                c_v2 = max([b["conf"] for b in v2_knives if box_iou(b["xyxy"], kb) >= 0.2 or box_cover(b["xyxy"], kb) >= 0.5], default=0.0)
                best_oof_confs.append(c_oof)
                best_v2_confs.append(c_v2)
                cx, cy = (kb[0] + kb[2]) / 2, (kb[1] + kb[3]) / 2
                if wrists:
                    d = min(np.hypot(cx - wx, cy - wy) / torso for wx, wy, torso in wrists)
                    wrist_dists.append(d)

            min_oof = min(best_oof_confs)
            min_v2 = min(best_v2_confs)
            min_cls = min(r["gt_cls_probs"]) if r["gt_cls_probs"] else 0.0
            min_knn_sim = min(r.get("gt_knn_knife_sim") or [1.0])
            max_wdist = max(wrist_dists) if wrist_dists else 0.0

            if min_oof < 0.15 and min_v2 < 0.20:
                flags.append("pos_unseen_oof_v2")
            if sum(r["loss"]) >= pos_loss_hi:
                flags.append("pos_high_oof_loss")
            if r["grad_cos_all"] < -0.02 or r["grad_cls_cos_pos"] < -0.05:
                flags.append("pos_neg_goal_grad")
            if min_knn_sim < 0.90:
                flags.append("pos_feat_outlier")
            if min_cls < 0.20 and min_oof < 0.35:
                flags.append("pos_cropcls_low")
            if max_wdist > 0.65:
                flags.append("pos_far_from_wrist")
            if len(r["gt_knives"]) > 1:
                flags.append("pos_multibox")
            if idx in temporal_flip and min_oof < 0.45:
                flags.append("pos_temporal_spike")

            worst_i = int(np.argmin(best_oof_confs))
            focus_box = r["gt_knives"][worst_i]
            is_suspect = (
                "pos_multibox" in flags
                or "pos_far_from_wrist" in flags
                or len(flags) >= 2
            )
            best_conf = min_oof
        elif r["source"] in ("phone", "other"):
            # Phone-only takes (class 2): check whether the phone box itself has high OOF loss & missed cls==2
            oof_phones = [b for b in r["oof_boxes"] if b["cls"] == 2]
            phone_boxes = [b for c, b in r["gt_all"] if c == 2]
            min_ph_oof = min(
                (max([b["conf"] for b in oof_phones if box_iou(b["xyxy"], pb) >= 0.2], default=0.0) for pb in phone_boxes),
                default=1.0,
            )
            if phone_boxes and min_ph_oof < 0.15 and sum(r["loss"]) >= pos_loss_hi:
                flags.append("phone_unseen_hi_loss")
            focus_box = phone_boxes[0] if phone_boxes else [0.25, 0.25, 0.75, 0.75]
            is_suspect = len(flags) >= 1
            best_conf = min_ph_oof
        else:
            max_oof = max([b["conf"] for b in oof_knives], default=0.0)
            max_v2 = max([b["conf"] for b in v2_knives], default=0.0)
            max_cls = max(r["wrist_cls_probs"]) if r["wrist_cls_probs"] else 0.0

            if max_oof >= 0.10 or max_v2 >= 0.20:
                flags.append("neg_det_hit")
            if r["loss"][1] >= neg_loss_hi:
                flags.append("neg_high_oof_loss")
            if r["grad_cls_cos_neg"] < 0.50:
                flags.append("neg_neg_goal_grad")
            if max_cls >= 0.85:
                flags.append("neg_cropcls_hi")
            if idx in temporal_flip:
                flags.append("neg_temporal_sandwich")

            cand_boxes = sorted(oof_knives + v2_knives, key=lambda b: -b["conf"])
            if cand_boxes:
                focus_box = cand_boxes[0]["xyxy"]
            elif wrists:
                wi = int(np.argmax(r["wrist_cls_probs"])) if r["wrist_cls_probs"] else 0
                wx, wy, torso = wrists[wi]
                s = 0.20
                focus_box = [max(0.0, wx - s), max(0.0, wy - s), min(1.0, wx + s), min(1.0, wy + s)]
            else:
                focus_box = [0.25, 0.25, 0.75, 0.75]

            is_suspect = (
                (max_oof >= 0.25 and max_v2 >= 0.15)
                or len(flags) >= 2
                or ("neg_temporal_sandwich" in flags and (max_oof >= 0.05 or max_cls >= 0.50))
            )
            best_conf = max_oof

        src = r["source"]
        kind = "pos" if r["has_knife"] else "neg"
        if is_suspect:
            by_source[src][f"suspect_{kind}"] += 1
            for fl in flags:
                flag_counts[fl] += 1
            suspects.append(dict(
                id=len(suspects),
                rec_idx=idx,
                img=os.path.relpath(r["path"], REPO).replace("\\", "/"),
                take=r["take"],
                source=src,
                kind=kind,
                is_human=r["is_human"],
                flags=flags,
                n_flags=len(flags),
                oof_conf=round(best_conf, 3),
                loss=r["loss"],
                grad_cos_all=r["grad_cos_all"],
                grad_cls_cos_pos=r["grad_cls_cos_pos"],
                gt_knives=r["gt_knives"],
                oof_boxes=[b for b in oof_knives if b["conf"] >= 0.05],
                v2_boxes=[b for b in v2_knives if b["conf"] >= 0.10],
                wrists=wrists,
                focus_box=focus_box,
            ))
        else:
            safe_count += 1
            by_source[src][f"safe_{kind}"] += 1

    # Sort suspects: highest number of signals first, non-human before human
    suspects.sort(key=lambda s: (s["is_human"], -s["n_flags"], -s["loss"][1]))
    for i, s in enumerate(suspects):
        s["id"] = i

    # Render 4x3 contact sheets (320px cells) for /swarm
    sdir = os.path.join(OUT, "sheets")
    os.makedirs(sdir, exist_ok=True)
    for old in glob.glob(os.path.join(sdir, "sheet_*.jpg")):
        os.remove(old)

    CELL = 320
    tiles = []
    for s in suspects:
        img = cv2.imread(os.path.join(REPO, s["img"]))
        h, w = img.shape[:2]
        fb = s["focus_box"]
        fx1, fy1, fx2, fy2 = fb[0] * w, fb[1] * h, fb[2] * w, fb[3] * h
        side = int(min(max(400, 3.0 * max(fx2 - fx1, fy2 - fy1)), w, h))
        x0 = int(min(max((fx1 + fx2) / 2 - side / 2, 0), w - side))
        y0 = int(min(max((fy1 + fy2) / 2 - side / 2, 0), h - side))
        crop = img[y0:y0 + side, x0:x0 + side].copy()
        sc = CELL / side
        tile = cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_AREA)

        # Draw GT boxes (green), OOF preds (yellow), v2 preds (cyan)
        for gb in s["gt_knives"]:
            p1 = (int((gb[0] * w - x0) * sc), int((gb[1] * h - y0) * sc))
            p2 = (int((gb[2] * w - x0) * sc), int((gb[3] * h - y0) * sc))
            cv2.rectangle(tile, p1, p2, (0, 230, 0), 2)
        for ob in s["oof_boxes"][:3]:
            b = ob["xyxy"]
            p1 = (int((b[0] * w - x0) * sc), int((b[1] * h - y0) * sc))
            p2 = (int((b[2] * w - x0) * sc), int((b[3] * h - y0) * sc))
            cv2.rectangle(tile, p1, p2, (0, 220, 255), 1)
        for vb in s["v2_boxes"][:2]:
            b = vb["xyxy"]
            p1 = (int((b[0] * w - x0) * sc), int((b[1] * h - y0) * sc))
            p2 = (int((b[2] * w - x0) * sc), int((b[3] * h - y0) * sc))
            cv2.rectangle(tile, p1, p2, (255, 220, 0), 1)
        for wx, wy, _ in s["wrists"]:
            pt = (int((wx * w - x0) * sc), int((wy * h - y0) * sc))
            if 0 <= pt[0] < CELL and 0 <= pt[1] < CELL:
                cv2.circle(tile, pt, 4, (255, 80, 255), -1)

        tag = f"#{s['id']} {s['kind'].upper()}{' H' if s['is_human'] else ''} c={s['oof_conf']:.2f}"
        fl_str = ",".join(f.replace("pos_", "").replace("neg_", "") for f in s["flags"][:3])
        cv2.rectangle(tile, (0, 0), (CELL, 36), (0, 0, 0), -1)
        cv2.putText(tile, tag, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.putText(tile, fl_str[:42], (4, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        s["crop_meta"] = dict(x0=x0, y0=y0, side=side)
        tiles.append(tile)

    n_sheets = 0
    for k in range(0, len(tiles), 12):
        batch_t = tiles[k:k + 12]
        while len(batch_t) < 12:
            batch_t.append(np.zeros((CELL, CELL, 3), dtype=np.uint8))
        rows_img = [np.hstack(batch_t[r * 4:(r + 1) * 4]) for r in range(3)]
        sheet = np.vstack(rows_img)
        cv2.imwrite(os.path.join(sdir, f"sheet_{n_sheets:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_sheets += 1

    summary = dict(
        total=len(recs),
        safe=safe_count,
        suspect=len(suspects),
        suspect_pos=sum(1 for s in suspects if s["kind"] == "pos"),
        suspect_neg=sum(1 for s in suspects if s["kind"] == "neg"),
        by_source={k: dict(v) for k, v in by_source.items()},
        flag_counts=dict(flag_counts),
        multi_dups=multi_dups,
        hn_dups=hn_dups,
        n_sheets=n_sheets,
        pos_loss_hi=round(pos_loss_hi, 3),
        neg_loss_hi=round(neg_loss_hi, 3),
    )
    with open(os.path.join(OUT, "triage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "suspect_items.json"), "w", encoding="utf-8") as f:
        json.dump(suspects, f, indent=1)

    print(f"Total: {len(recs)} | Safe: {safe_count} ({safe_count/len(recs)*100:.1f}%) | Suspect: {len(suspects)} (pos={summary['suspect_pos']}, neg={summary['suspect_neg']}) -> {n_sheets} sheets")
    print("By source:", summary["by_source"])
    print("Flags:", summary["flag_counts"])
    print("lab_gt multi-box:", len(multi_dups), "hn_dups:", len(hn_dups))


def tally_eye():
    items = json.load(open(os.path.join(OUT, "suspect_items.json"), encoding="utf-8"))
    votes = {}
    for p in sorted(glob.glob(os.path.join(OUT, "eye_votes", "shard_*.json"))):
        votes.update(json.load(open(p, encoding="utf-8")))
    assert len(votes) == len(items), f"Votes {len(votes)} != items {len(items)}"

    by_verdict, by_src, by_kind, by_flag = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    corrupt_rows = []
    for it in items:
        v_obj = votes[str(it["id"])]
        v = v_obj["verdict"]
        note = v_obj.get("note", "")
        by_verdict[v] += 1
        by_src[it["source"]][v] += 1
        by_kind[it["kind"]][v] += 1
        for fl in it["flags"]:
            by_flag[fl][v] += 1
        if v != "ok":
            corrupt_rows.append(dict(
                id=it["id"], verdict=v, kind=it["kind"], source=it["source"],
                is_human=it["is_human"], img=it["img"], flags=it["flags"], note=note,
            ))

    print("Eye verdicts (n=%d):" % len(items), dict(by_verdict))
    print("By kind:", {k: dict(v) for k, v in by_kind.items()})
    print("By source:", {k: dict(v) for k, v in by_src.items()})
    print("Signal precision (corrupt = wrong_pos + bad_box + missed_knife):")
    for fl, c in sorted(by_flag.items()):
        bad = c["wrong_pos"] + c["bad_box"] + c["missed_knife"]
        tot = sum(c.values())
        print(f"  {fl:24s}: total={tot:3d} corrupt={bad:3d} ({bad/tot*100:4.1f}%) unclear={c['unclear']:3d} ok={c['ok']:3d}")

    out = dict(
        total_suspect=len(items),
        counts=dict(by_verdict),
        by_kind={k: dict(v) for k, v in by_kind.items()},
        by_source={k: dict(v) for k, v in by_src.items()},
        by_flag={k: dict(v) for k, v in by_flag.items()},
        non_ok=corrupt_rows,
    )
    with open(os.path.join(OUT, "eye_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("Saved ->", os.path.join(OUT, "eye_audit.json"))


def build_review():
    """Export non-ok lab_gt + orig_v2 audit items to results/stage1_kd/train_audit/human_review for label_review_ui.py."""
    eye = json.load(open(os.path.join(OUT, "eye_audit.json"), encoding="utf-8"))
    by_id = {s["id"]: s for s in json.load(open(os.path.join(OUT, "suspect_items.json"), encoding="utf-8"))}
    rdir = os.path.join(OUT, "human_review")
    os.makedirs(rdir, exist_ok=True)

    # Order: confirmed corruptions first (wrong_pos, missed_knife, bad_box), then unclear; exclude phone/other takes
    order = {"wrong_pos": 0, "missed_knife": 1, "bad_box": 2, "unclear": 3}
    chosen = [x for x in eye["non_ok"] if x["source"] in ("lab_gt", "orig_v2") and x["verdict"] in order]
    chosen.sort(key=lambda x: (order[x["verdict"]], x["id"]))

    items, rows = [], []
    fields = [
        "tile_id", "img", "split", "take", "frame", "kind",
        "swarm", "audit_flags", "n_votes", "agree", "flag_reason",
        "claude_label", "claude_conf", "human_label", "human_box",
        "final_label", "final_source", "last_update",
    ]
    for new_id, x in enumerate(chosen):
        s = by_id[x["id"]]
        img_path = os.path.join(REPO, s["img"])
        img = cv2.imread(img_path)
        assert img is not None, img_path
        h, w = img.shape[:2]
        cm = s["crop_meta"]

        # Build up to 3 box suggestions: a = GT knife (or best OOF if NEG), b = best non-matching OOF/v2 box
        props = []
        if s["gt_knives"]:
            gb = s["gt_knives"][0]
            props.append({"letter": "a", "box": [round(gb[0] * w, 1), round(gb[1] * h, 1), round(gb[2] * w, 1), round(gb[3] * h, 1)], "src": {"gt": 1.0}})
            for cand in sorted(s["oof_boxes"] + s["v2_boxes"], key=lambda b: -b["conf"]):
                if box_iou(cand["xyxy"], gb) < 0.6 and len(props) < 3:
                    let = "b" if len(props) == 1 else "c"
                    cb = cand["xyxy"]
                    props.append({"letter": let, "box": [round(cb[0] * w, 1), round(cb[1] * h, 1), round(cb[2] * w, 1), round(cb[3] * h, 1)], "src": {"det": round(cand["conf"], 3)}})
                    break
            if len(s["gt_knives"]) > 1 and len(props) < 3:
                gb2 = s["gt_knives"][1]
                let = "b" if len(props) == 1 else "c"
                props.append({"letter": let, "box": [round(gb2[0] * w, 1), round(gb2[1] * h, 1), round(gb2[2] * w, 1), round(gb2[3] * h, 1)], "src": {"gt2": 1.0}})
        else:
            cands = sorted(s["oof_boxes"] + s["v2_boxes"], key=lambda b: -b["conf"])
            for cand in cands:
                if all(box_iou(cand["xyxy"], [p["box"][0] / w, p["box"][1] / h, p["box"][2] / w, p["box"][3] / h]) < 0.5 for p in props):
                    let = "abc"[len(props)]
                    cb = cand["xyxy"]
                    props.append({"letter": let, "box": [round(cb[0] * w, 1), round(cb[1] * h, 1), round(cb[2] * w, 1), round(cb[3] * h, 1)], "src": {"det": round(cand["conf"], 3)}})
                    if len(props) >= 2:
                        break

        fn = frame_num(s["img"]) or 0
        items.append({
            "id": new_id, "orig_suspect_id": x["id"], "img": s["img"], "split": "train",
            "take": s["take"], "frame": fn, "x0": cm["x0"], "y0": cm["y0"], "crop": cm["side"],
            "kind": s["kind"], "props": props, "source": s["source"],
        })
        rows.append({
            "tile_id": str(new_id), "img": s["img"], "split": "train", "take": s["take"],
            "frame": str(fn), "kind": s["kind"], "swarm": x["verdict"],
            "audit_flags": ",".join(f.replace("pos_", "").replace("neg_", "") for f in s["flags"][:2]),
            "n_votes": "1", "agree": "0", "flag_reason": x["note"][:80],
            "claude_label": x["verdict"], "claude_conf": "0.5",
            "human_label": "", "human_box": "", "final_label": "", "final_source": "needs_human", "last_update": "",
        })

    json.dump(items, open(os.path.join(rdir, "items.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(rdir, "votes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Prepared {len(items)} items for human review -> {rdir}", Counter(x["verdict"] for x in chosen))


def build_clean():
    """Build data/stage1_negatives/lab_gt_clean from lab_gt + human_review/votes.csv + verified 006-009 phone fix."""
    import shutil
    rdir = os.path.join(OUT, "human_review")
    rows = list(csv.DictReader(open(os.path.join(rdir, "votes.csv"), encoding="utf-8")))
    items = {it["id"]: it for it in json.load(open(os.path.join(rdir, "items.json"), encoding="utf-8"))}
    rev_by_img = {}
    for r in rows:
        tid = int(r["tile_id"])
        rev_by_img[os.path.basename(r["img"])] = (r["human_label"], r["human_box"], items[tid])

    dst = os.path.join(REPO, "data", "stage1_negatives", "lab_gt_clean")
    img_dir = os.path.join(dst, "images", "train")
    lbl_dir = os.path.join(dst, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    lines = [norm_path(l) for l in open(os.path.join(LAB_GT, "train.txt"), encoding="utf-8") if l.strip()]
    out_paths = []
    counts = Counter()

    for p in lines:
        bn = os.path.basename(p)
        lbls = read_labels(p)
        img = cv2.imread(p)
        assert img is not None, p
        h, w = img.shape[:2]

        # 1. Apply human review decision if present
        if bn in rev_by_img:
            hl, hb_str, it = rev_by_img[bn]
            if hl == "u":
                counts["dropped_unclear"] += 1
                continue
            non_knife = [x for x in lbls if x[0] != 1]
            if hl == "n":
                lbls = non_knife
                counts["human_set_neg"] += 1
            else:
                if hl == "box" and hb_str:
                    v = json.loads(hb_str)
                    s = it["crop"] / 330.0
                    box_px = [it["x0"] + v[0] * s, it["y0"] + v[1] * s, it["x0"] + v[2] * s, it["y0"] + v[3] * s]
                else:
                    box_px = next(pr["box"] for pr in it["props"] if pr["letter"] == hl)
                xc = max(0.0, min(1.0, (box_px[0] + box_px[2]) / 2.0 / w))
                yc = max(0.0, min(1.0, (box_px[1] + box_px[3]) / 2.0 / h))
                bw = max(0.001, min(1.0, (box_px[2] - box_px[0]) / w))
                bh = max(0.001, min(1.0, (box_px[3] - box_px[1]) / h))
                lbls = non_knife + [(1, xc, yc, bw, bh)]
                counts["human_fixed_box"] += 1

        # 2. Verified phone-take override (2026-11-05 takes 006, 007, 008, 009; NOT 028 which has a real knife)
        if any(tk in bn for tk in ("2026-11-05_006_", "2026-11-05_007_", "2026-11-05_008_")):
            new_lbls = []
            for cls_id, xc, yc, bw, bh in lbls:
                if cls_id == 1:
                    new_lbls.append((2, xc, yc, bw, bh))
                    counts["phone_take_to_cls2"] += 1
                else:
                    new_lbls.append((cls_id, xc, yc, bw, bh))
            lbls = new_lbls
        elif "2026-11-05_009_" in bn:
            new_lbls = []
            for cls_id, xc, yc, bw, bh in lbls:
                if cls_id == 1 and not (xc < 0.25 and yc < 0.25):  # keep desk knife at ~(0.13, 0.14); convert hand phone
                    new_lbls.append((2, xc, yc, bw, bh))
                    counts["phone_take_to_cls2"] += 1
                else:
                    new_lbls.append((cls_id, xc, yc, bw, bh))
            lbls = new_lbls

        # 3. Link/copy image (handle distinct prefixes cleanly if basename collides across ft_phonecls vs lab_gt)
        dst_name = ("ph_" + bn) if "/ft_phonecls/" in p else bn
        dst_img = os.path.join(img_dir, dst_name)
        dst_lbl = os.path.join(lbl_dir, os.path.splitext(dst_name)[0] + ".txt")
        if not os.path.exists(dst_img):
            try:
                os.link(p, dst_img)
            except OSError:
                shutil.copy2(p, dst_img)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for cls_id, xc, yc, bw, bh in lbls:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        out_paths.append(dst_img.replace("\\", "/"))
        counts["kept_pos" if any(x[0] == 1 for x in lbls) else "kept_neg"] += 1

    # Invariant checks: no test leaks, no knife labels on 006/007/008, no multibox knife duplicates
    assert not any("_test_" in x or "2026-10-31" in x or "Ground_Truth_Dataset" in x for x in out_paths)
    for x in out_paths:
        knives = [xywhn2xyxy(*r[1:]) for r in read_labels(x) if r[0] == 1]
        if any(tk in os.path.basename(x) for tk in ("2026-11-05_006_", "2026-11-05_007_", "2026-11-05_008_")):
            assert len(knives) == 0, x
        if len(knives) > 1:
            for i in range(len(knives)):
                for j in range(i + 1, len(knives)):
                    assert box_iou(knives[i], knives[j]) < 0.2 and box_cover(knives[i], knives[j]) < 0.5, (x, knives)

    val_src = os.path.join(LAB_GT, "val.txt")
    shutil.copy2(val_src, os.path.join(dst, "val.txt"))
    with open(os.path.join(dst, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_paths) + "\n")
    with open(os.path.join(dst, "data.yaml"), "w", encoding="utf-8") as f:
        d = dst.replace("\\", "/")
        f.write(f"path: {d}\ntrain: train.txt\nval: val.txt\nnames:\n  0: hand_grip_knife\n  1: knife\n  2: phone\n")
    print(f"Built lab_gt_clean: {len(out_paths)} train imgs -> {dst}", dict(counts))


def selftest():
    assert take_group("D:/x/ft_phonecls/images/train/006_thrust_low_front_164412_f00006.jpg") == "lab_006_thrust_low_front_164412"
    assert take_group("D:/x/lab_gt/images/train/2026-11-05_006_thrust_low_front_164412_f00100.jpg") == "lab_006_thrust_low_front_164412"
    assert take_group("D:/x/lab_v2/images/train/2025-09-27 11-50-49_frame_0010.jpg") == "orig_2025-09-27 11-50-49"
    assert abs(box_cover([0.1, 0.1, 0.3, 0.3], [0.0, 0.0, 0.5, 0.5]) - 1.0) < 1e-6
    assert abs(box_iou([0.0, 0.0, 0.2, 0.2], [0.0, 0.0, 0.2, 0.2]) - 1.0) < 1e-6
    print("selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd == ["selftest"]:
        selftest()
    elif cmd == ["split"]:
        split()
    elif len(cmd) == 2 and cmd[0] == "train" and cmd[1] in ("a", "b"):
        train_fold(cmd[1])
    elif cmd == ["score"]:
        score()
    elif cmd == ["triage"]:
        triage()
    elif cmd == ["tally_eye"]:
        tally_eye()
    elif cmd == ["review"]:
        build_review()
    elif cmd == ["build_clean"]:
        build_clean()
    else:
        print(__doc__)
