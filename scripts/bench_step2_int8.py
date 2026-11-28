"""
scripts/bench_step2_int8.py

Step 2a Benchmark: Mixed-Precision INT8 PTQ vs FP16 Baseline.
- Confidence threshold sweep on data/Ground_Truth_Dataset (715 images, IoU=0.20, nms_iou=0.50)
- Lab holdout + phone/remote confuser parity (crop_4_3, every 3rd frame)
- Pure GPU engine latency (CUDA events, 50 warmup + 500 iters) + end-to-end 1920x1440 latency
- Stage 2a pose latency & ByteTrack/keypoint parity (optional --with-pose)
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from scripts.record_takes import crop_4_3
from scripts.validate_stage1_ground_truth import (
    GT_DATASET_DIR,
    calculate_iou,
    calculate_metrics,
    get_category_from_path,
    load_ground_truth,
    xywhn2xyxy,
)
from src.stage1_detector import STAGE1_NMS_IOU, Stage1WeaponDetector
from src.stage2a_pose_tracker import Stage2aPoseTracker, compute_iou

CONFS = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35]


def _extract_raw_knife_boxes(res):
    """Extract all knife/weapon boxes above min conf=0.10 before thresholding/NMS."""
    if res.boxes is None or len(res.boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0, 4)), torch.empty((0,))
    names = res.names if isinstance(res.names, dict) else dict(enumerate(res.names or ()))
    keep_ids = {int(k) for k, v in names.items() if str(v).lower() in {"knife", "weapon"}}
    cls_t = res.boxes.cls
    mask = torch.zeros_like(cls_t, dtype=torch.bool)
    for cid in keep_ids:
        mask |= (cls_t.long() == cid)
    if not mask.any():
        return torch.empty((0, 4)), torch.empty((0, 4)), torch.empty((0,))
    return res.boxes.xyxyn[mask].float().cpu(), res.boxes.xyxy[mask].float().cpu(), res.boxes.conf[mask].float().cpu()


def sweep_gt(engine_path: str, image_files, confs=CONFS):
    det = Stage1WeaponDetector(weights_path=engine_path, conf_threshold=min(confs))
    raw_preds = []
    for img_path in image_files:
        cat = get_category_from_path(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_txt = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", f"{stem}.txt")
        gt_boxes = [xywhn2xyxy(*g[1:]) for g in load_ground_truth(gt_txt)]
        r = det.model.predict(source=img_path, conf=min(confs), device="cuda:0", imgsz=(384, 640), verbose=False)[0]
        xyxyn, xyxy_px, scores = _extract_raw_knife_boxes(r)
        raw_preds.append((cat, gt_boxes, xyxyn, xyxy_px, scores))

    sweep = {}
    for t in confs:
        s = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
        for cat, gt_boxes, xyxyn, xyxy_px, scores in raw_preds:
            m = scores >= t
            if m.any():
                sc = scores[m]
                bx_n = xyxyn[m]
                bx_p = xyxy_px[m]
                if bx_n.shape[0] > 1:
                    keep = tv_nms(bx_p, sc, STAGE1_NMS_IOU)
                    bx_n = bx_n[keep]
                pred_norm = bx_n.tolist()
            else:
                pred_norm = []

            matched_gt = set()
            for pb in pred_norm:
                best_iou, best_g = 0.0, -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = calculate_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_g = iou, gi
                if best_iou >= 0.20:
                    s[cat]["TP"] += 1
                    s["Overall"]["TP"] += 1
                    matched_gt.add(best_g)
                else:
                    s[cat]["FP"] += 1
                    s["Overall"]["FP"] += 1
            fn = len(gt_boxes) - len(matched_gt)
            s[cat]["FN"] += fn
            s["Overall"]["FN"] += fn
            if len(gt_boxes) == 0 and len(pred_norm) == 0:
                s[cat]["TN"] += 1
                s["Overall"]["TN"] += 1

        ov = s["Overall"]
        p, r, f1, spec, acc = calculate_metrics(ov["TP"], ov["FP"], ov["FN"], ov["TN"])
        wc = s["With_Coat"]
        woc = s["Without_Coat"]
        _, wc_r, _, _, _ = calculate_metrics(wc["TP"], wc["FP"], wc["FN"], wc["TN"])
        _, woc_r, _, _, _ = calculate_metrics(woc["TP"], woc["FP"], woc["FN"], woc["TN"])
        sweep[str(t)] = {
            "precision": round(p * 100, 4),
            "recall": round(r * 100, 4),
            "f1": round(f1 * 100, 4),
            "ood_fp": s["OOD"]["FP"],
            "with_coat_r": round(wc_r * 100, 4),
            "without_coat_r": round(woc_r * 100, 4),
            **ov,
        }
    return sweep, det


def eval_lab_holdout(engines_with_confs):
    lab = os.path.join(REPO_DIR, "data", "lab_recordings")
    knife_paths = (
        sorted(glob.glob(os.path.join(lab, "2026-11-14", "test_*", "*.avi")))
        + [os.path.join(lab, "2026-10-31", "raw", "002_checkerboard_142846.avi")]
    )
    other_paths = [
        p for n in ("050", "056")
        for p in glob.glob(os.path.join(lab, "2026-11-14", "confuser", f"{n}_*.avi"))
    ]

    def win_share(flags, win=10, min_hits=3):
        if len(flags) < win:
            return 0.0
        pos = sum(1 for i in range(len(flags) - win + 1) if sum(flags[i : i + win]) >= min_hits)
        return pos / (len(flags) - win + 1)

    loaded = {}
    for label, path, _ in engines_with_confs:
        if path not in loaded:
            loaded[path] = Stage1WeaponDetector(weights_path=path, conf_threshold=0.10)

    raw_flags = {label: {} for label, _, _ in engines_with_confs}
    for vpath in knife_paths + other_paths:
        tag = os.path.basename(os.path.dirname(os.path.dirname(vpath))) + "/" + os.path.basename(vpath)
        cap = cv2.VideoCapture(vpath)
        per_label = {label: [] for label, _, _ in engines_with_confs}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % 3 == 0:
                img = crop_4_3(frame)
                for label, path, conf in engines_with_confs:
                    res = loaded[path].predict(img, custom_conf=conf)
                    per_label[label].append(res["has_threat"])
            idx += 1
        cap.release()
        for label, _, _ in engines_with_confs:
            fl = per_label[label]
            raw_flags[label][tag] = {
                "frame_share": round(sum(fl) / max(len(fl), 1), 4),
                "win_share": round(win_share(fl), 4),
            }

    summary = {}
    for label, _, _ in engines_with_confs:
        d = raw_flags[label]
        nov14 = [v for k, v in d.items() if "2026-11-14" in k and "confuser" not in k]
        t002 = [v for k, v in d.items() if "2026-10-31" in k][0]
        p050 = [v for k, v in d.items() if "050_" in k][0]
        r056 = [v for k, v in d.items() if "056_" in k][0]
        summary[label] = {
            "nov14_knife_frame_mean": round(float(np.mean([x["frame_share"] for x in nov14])), 4),
            "nov14_knife_frame_min": round(float(np.min([x["frame_share"] for x in nov14])), 4),
            "nov14_knife_win_mean": round(float(np.mean([x["win_share"] for x in nov14])), 4),
            "nov14_knife_win_min": round(float(np.min([x["win_share"] for x in nov14])), 4),
            "take002_frame_share": t002["frame_share"],
            "phone050_fa_share": p050["frame_share"],
            "remote056_fa_share": r056["frame_share"],
        }
    return summary


def bench_gpu_latency(engine_path: str, task: str = "detect"):
    if task == "detect":
        wrap = Stage1WeaponDetector(weights_path=engine_path)
        ymodel = wrap.model
    else:
        wrap = Stage2aPoseTracker(weights_path=engine_path)
        ymodel = wrap.pose_model

    dummy_img = np.random.randint(0, 255, (1440, 1920, 3), dtype=np.uint8)
    _ = ymodel.predict(dummy_img, imgsz=(384, 640), verbose=False, device="cuda:0")
    backend = ymodel.predictor.model
    dtype = torch.float16 if getattr(backend, "fp16", False) else torch.float32
    inp = torch.rand(1, 3, 384, 640, device="cuda:0", dtype=dtype)

    for _ in range(50):
        _ = backend(inp)
    torch.cuda.synchronize()

    ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(500)]
    ev_end = [torch.cuda.Event(enable_timing=True) for _ in range(500)]
    for i in range(500):
        ev_start[i].record()
        _ = backend(inp)
        ev_end[i].record()
    torch.cuda.synchronize()
    pure_ms = np.array([s.elapsed_time(e) for s, e in zip(ev_start, ev_end)], dtype=np.float64)

    # End-to-end 1920x1440
    for _ in range(30):
        if task == "detect":
            wrap.predict(dummy_img)
        else:
            wrap.extract_actor_pose(dummy_img)
    torch.cuda.synchronize()

    e2e_ms = []
    for _ in range(200):
        t0 = time.perf_counter()
        if task == "detect":
            wrap.predict(dummy_img)
        else:
            wrap.extract_actor_pose(dummy_img)
        torch.cuda.synchronize()
        e2e_ms.append((time.perf_counter() - t0) * 1000.0)
    e2e_arr = np.array(e2e_ms, dtype=np.float64)

    return {
        "input_dtype": str(dtype),
        "pure_gpu_mean_ms": round(float(pure_ms.mean()), 4),
        "pure_gpu_median_ms": round(float(np.median(pure_ms)), 4),
        "pure_gpu_p95_ms": round(float(np.percentile(pure_ms, 95)), 4),
        "pure_gpu_fps": round(1000.0 / float(pure_ms.mean()), 1),
        "e2e_1920x1440_mean_ms": round(float(e2e_arr.mean()), 3),
        "e2e_1920x1440_fps": round(1000.0 / float(e2e_arr.mean()), 1),
    }


def bench_pose_parity(fp16_engine: str, int8_engine: str):
    vid = os.path.join(REPO_DIR, "data", "lab_recordings", "2026-10-31", "raw", "002_checkerboard_142846.avi")
    cap = cv2.VideoCapture(vid)
    frames = []
    while len(frames) < 150:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(crop_4_3(f))
    cap.release()

    out = {}
    for label, ep in (("fp16", fp16_engine), ("int8", int8_engine)):
        tr = Stage2aPoseTracker(weights_path=ep)
        tr.pose_model.predict(np.zeros((1440, 1920, 3), dtype=np.uint8), imgsz=(384, 640), verbose=False, device="cuda:0")
        tr.reset_tracking()
        boxes, kpts, tids = [], [], []
        for f in frames:
            kp, bx, tid = tr.extract_actor_pose(f)
            boxes.append(bx)
            kpts.append(kp)
            tids.append(tid)
        out[label] = {"boxes": boxes, "kpts": kpts, "tids": tids}

    ious, bx_drifts, kp_drifts = [], [], []
    for i in range(len(frames)):
        bf, bi = out["fp16"]["boxes"][i], out["int8"]["boxes"][i]
        if bf is not None and bi is not None:
            ious.append(compute_iou(bf, bi))
            bx_drifts.append(max(abs(a - b) for a, b in zip(bf, bi)))
        kf, ki = out["fp16"]["kpts"][i], out["int8"]["kpts"][i]
        valid = (np.any(kf != 0, axis=-1)) & (np.any(ki != 0, axis=-1))
        if np.any(valid):
            kp_drifts.extend(np.linalg.norm(kf[valid] - ki[valid], axis=-1).tolist())

    def id_sw(tids):
        nn = [t for t in tids if t is not None]
        return len(set(nn)), sum(1 for a, b in zip(nn, nn[1:]) if a != b)

    uf, sf = id_sw(out["fp16"]["tids"])
    ui, si = id_sw(out["int8"]["tids"])
    return {
        "n_frames": len(frames),
        "fp16_detected": sum(1 for b in out["fp16"]["boxes"] if b is not None),
        "int8_detected": sum(1 for b in out["int8"]["boxes"] if b is not None),
        "fp16_ids_switches": [uf, sf],
        "int8_ids_switches": [ui, si],
        "bbox_iou_median": round(float(np.median(ious)), 5) if ious else 0.0,
        "bbox_px_drift_median": round(float(np.median(bx_drifts)), 3) if bx_drifts else 0.0,
        "kpt_l2_px_median": round(float(np.median(kp_drifts)), 3) if kp_drifts else 0.0,
        "kpt_l2_px_max": round(float(np.max(kp_drifts)), 3) if kp_drifts else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8-engine", type=str, default=os.path.join(REPO_DIR, "weights", "stage1_repro846_kdgtclean_p250_int8ptq.engine"))
    parser.add_argument("--extra-engine", type=str, default=None)
    parser.add_argument("--pose-int8-engine", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=os.path.join(REPO_DIR, "results", "amp_edge_speed", "step2a_int8_ptq.json"))
    args = parser.parse_args()

    fp16_engine = os.path.join(REPO_DIR, "weights", "stage1_repro846_kdgtclean_p250.engine")
    gt_images = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))

    engines = [("fp16_baseline", fp16_engine), ("int8_ptq", args.int8_engine)]
    if args.extra_engine and os.path.exists(args.extra_engine):
        engines.append(("int8_ptq_lockp3", args.extra_engine))

    gt_sweeps = {}
    latencies = {}
    for label, ep in engines:
        print(f"[GT SWEEP] {label}: {os.path.basename(ep)}...", flush=True)
        sw, _ = sweep_gt(ep, gt_images)
        gt_sweeps[label] = sw
        print(f"  @0.20: {sw['0.2']}", flush=True)
        latencies[label] = bench_gpu_latency(ep, task="detect")
        print(f"  Latency: {latencies[label]}", flush=True)

    # Pick best conf on int8_ptq matching baseline recall >= 89.58% (or max F1)
    base_r = gt_sweeps["fp16_baseline"]["0.2"]["recall"]
    cands = [(float(c), d) for c, d in gt_sweeps["int8_ptq"].items() if d["recall"] >= base_r - 0.5]
    if cands:
        best_c, _ = max(cands, key=lambda x: (x[1]["f1"], -x[1]["ood_fp"]))
    else:
        best_c_str, _ = max(gt_sweeps["int8_ptq"].items(), key=lambda x: x[1]["f1"])
        best_c = float(best_c_str)

    holdout_specs = [
        ("fp16_baseline@0.20", fp16_engine, 0.20),
        ("int8_ptq@0.20", args.int8_engine, 0.20),
    ]
    if abs(best_c - 0.20) > 1e-6:
        holdout_specs.append((f"int8_ptq@{best_c:.2f}", args.int8_engine, best_c))
    print(f"[LAB HOLDOUT] Running specs: {[(x[0], x[2]) for x in holdout_specs]}", flush=True)
    lab_summary = eval_lab_holdout(holdout_specs)
    print("Lab Summary:", lab_summary, flush=True)

    pose_res = None
    if args.pose_int8_engine and os.path.exists(args.pose_int8_engine):
        fp16_pose = os.path.join(REPO_DIR, "weights", "yolo26s-pose.engine")
        print("[POSE] Benchmarking Stage 2a FP16 vs INT8 PTQ...", flush=True)
        pose_res = {
            "fp16_latency": bench_gpu_latency(fp16_pose, task="pose"),
            "int8_latency": bench_gpu_latency(args.pose_int8_engine, task="pose"),
            "parity": bench_pose_parity(fp16_pose, args.pose_int8_engine),
        }
        print("Pose Results:", pose_res, flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gt_sweeps": gt_sweeps,
                "best_int8_conf": best_c,
                "lab_summary": lab_summary,
                "latencies": latencies,
                "pose": pose_res,
            },
            f,
            indent=2,
        )
    print(f"Saved {args.out_json}")


if __name__ == "__main__":
    main()
