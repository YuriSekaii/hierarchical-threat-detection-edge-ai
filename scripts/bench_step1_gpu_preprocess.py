"""
scripts/bench_step1_gpu_preprocess.py

Step 1 Parity & Latency Benchmark:
Compares CPU Ultralytics LetterBox vs portable GPU letterbox (src/gpu_preprocess.py)
on Stage 1 (stage1_repro846_kdgtclean_p250.engine) and Stage 2a (yolo26s-pose.engine).
"""

import glob
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

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
from src.gpu_preprocess import letterbox_bgr_to_tensor
from src.stage1_detector import STAGE1_DEFAULT_CONF, STAGE1_NMS_IOU, Stage1WeaponDetector
from src.stage2a_pose_tracker import Stage2aPoseTracker, compute_iou


def _norm_boxes(boxes, h, w):
    return [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in boxes]


def bench_stage1_gt(det_cpu, det_gpu, image_files):
    stats = {}
    per_img_res = {}
    for mode, det in (("cpu", det_cpu), ("gpu", det_gpu)):
        s = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_images": 0})
        preds_by_img = {}
        for img_path in image_files:
            cat = get_category_from_path(img_path)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gt_txt = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", f"{stem}.txt")
            gt_boxes = load_ground_truth(gt_txt)
            img = cv2.imread(img_path)
            oh, ow = img.shape[:2]
            res = det.predict(img, custom_conf=STAGE1_DEFAULT_CONF, imgsz=(384, 640))
            pred_px = res["boxes"]
            pred_norm = _norm_boxes(pred_px, oh, ow)
            preds_by_img[img_path] = {"boxes_px": pred_px, "scores": res["scores"], "hw": (oh, ow)}

            matched_gt = set()
            for p_box in pred_norm:
                best_iou, best_g = 0.0, -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = calculate_iou(p_box, xywhn2xyxy(*gb[1:]))
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
            s[cat]["total_images"] += 1
            s["Overall"]["total_images"] += 1

        out_cat = {}
        for cat in ["OOD", "With_Coat", "Without_Coat", "Overall"]:
            d = s[cat]
            p, r, f1, spec, acc = calculate_metrics(d["TP"], d["FP"], d["FN"], d["TN"])
            out_cat[cat] = {
                **d,
                "precision": round(p * 100, 4),
                "recall": round(r * 100, 4),
                "f1": round(f1 * 100, 4),
                "specificity": round(spec * 100, 4),
                "accuracy": round(acc * 100, 4),
            }
        stats[mode] = out_cat
        per_img_res[mode] = preds_by_img

    # Box drift & borderline flips
    drifts_orig, drifts_lb = [], []
    flips = []
    for p in image_files:
        c = per_img_res["cpu"][p]
        g = per_img_res["gpu"][p]
        oh, ow = c["hw"]
        ratio = min(384.0 / oh, 640.0 / ow)
        cb, gs = c["boxes_px"], g["boxes_px"]
        if len(cb) != len(gs):
            flips.append({
                "img": os.path.relpath(p, GT_DATASET_DIR),
                "cpu_n": len(cb),
                "gpu_n": len(gs),
                "cpu_scores": [round(x, 4) for x in c["scores"]],
                "gpu_scores": [round(x, 4) for x in g["scores"]],
            })
        used = set()
        for b1 in cb:
            best_iou, best_j = 0.0, -1
            for j, b2 in enumerate(gs):
                if j in used:
                    continue
                iou = compute_iou(b1, b2)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
                b2 = gs[best_j]
                d_orig = max(abs(a - b) for a, b in zip(b1, b2))
                drifts_orig.append(d_orig)
                drifts_lb.append(d_orig * ratio)

    drift_summary = {
        "matched_boxes": len(drifts_orig),
        "orig_px_median": round(float(np.median(drifts_orig)), 4) if drifts_orig else 0.0,
        "orig_px_max": round(float(np.max(drifts_orig)), 4) if drifts_orig else 0.0,
        "lb384_px_median": round(float(np.median(drifts_lb)), 4) if drifts_lb else 0.0,
        "lb384_px_max": round(float(np.max(drifts_lb)), 4) if drifts_lb else 0.0,
        "count_mismatches": len(flips),
        "flips_sample": flips[:15],
    }
    return stats, drift_summary


def bench_lab_holdout(det_cpu, det_gpu):
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

    res = {"cpu": {}, "gpu": {}}
    for path in knife_paths + other_paths:
        tag = os.path.basename(os.path.dirname(os.path.dirname(path))) + "/" + os.path.basename(path)
        cap = cv2.VideoCapture(path)
        flags_c, flags_g = [], []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % 3 == 0:
                img = crop_4_3(frame)
                flags_c.append(det_cpu.predict(img)["has_threat"])
                flags_g.append(det_gpu.predict(img)["has_threat"])
            idx += 1
        cap.release()
        for m, fl in (("cpu", flags_c), ("gpu", flags_g)):
            res[m][tag] = {
                "n_frames": len(fl),
                "frame_share": round(sum(fl) / max(len(fl), 1), 4),
                "win_share": round(win_share(fl), 4),
            }

    summary = {}
    for m in ("cpu", "gpu"):
        nov14 = [v for k, v in res[m].items() if "2026-11-14" in k and "confuser" not in k]
        t002 = [v for k, v in res[m].items() if "2026-10-31" in k][0]
        p050 = [v for k, v in res[m].items() if "050_" in k][0]
        r056 = [v for k, v in res[m].items() if "056_" in k][0]
        summary[m] = {
            "nov14_knife_frame_mean": round(float(np.mean([x["frame_share"] for x in nov14])), 4),
            "nov14_knife_frame_min": round(float(np.min([x["frame_share"] for x in nov14])), 4),
            "nov14_knife_win_mean": round(float(np.mean([x["win_share"] for x in nov14])), 4),
            "nov14_knife_win_min": round(float(np.min([x["win_share"] for x in nov14])), 4),
            "take002_frame_share": t002["frame_share"],
            "phone050_fa_share": p050["frame_share"],
            "remote056_fa_share": r056["frame_share"],
        }
    return summary, res


def bench_stage2a_parity():
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
    for mode, gp in (("cpu", False), ("gpu", True)):
        tr = Stage2aPoseTracker(gpu_preprocess=gp)
        # warmup
        tr.pose_model.predict(np.zeros((1440, 1920, 3), dtype=np.uint8), imgsz=(384, 640), verbose=False, device="cuda:0")
        tr.reset_tracking()
        boxes, kpts, tids = [], [], []
        for f in frames:
            kp, bx, tid = tr.extract_actor_pose(f)
            boxes.append(bx)
            kpts.append(kp)
            tids.append(tid)
        out[mode] = {"boxes": boxes, "kpts": kpts, "tids": tids}

    ious, bx_drifts, kp_drifts = [], [], []
    for i in range(len(frames)):
        bc, bg = out["cpu"]["boxes"][i], out["gpu"]["boxes"][i]
        if bc is not None and bg is not None:
            ious.append(compute_iou(bc, bg))
            bx_drifts.append(max(abs(a - b) for a, b in zip(bc, bg)))
        kc, kg = out["cpu"]["kpts"][i], out["gpu"]["kpts"][i]
        valid = (np.any(kc != 0, axis=-1)) & (np.any(kg != 0, axis=-1))
        if np.any(valid):
            d = np.linalg.norm(kc[valid] - kg[valid], axis=-1)
            kp_drifts.extend(d.tolist())

    def id_switches(tids):
        non_none = [t for t in tids if t is not None]
        if not non_none:
            return 0, 0
        sw = sum(1 for a, b in zip(non_none, non_none[1:]) if a != b)
        return len(set(non_none)), sw

    uc, sc = id_switches(out["cpu"]["tids"])
    ug, sg = id_switches(out["gpu"]["tids"])
    return {
        "n_frames": len(frames),
        "cpu_detected": sum(1 for b in out["cpu"]["boxes"] if b is not None),
        "gpu_detected": sum(1 for b in out["gpu"]["boxes"] if b is not None),
        "cpu_unique_ids": uc, "cpu_id_switches": sc,
        "gpu_unique_ids": ug, "gpu_id_switches": sg,
        "bbox_iou_median": round(float(np.median(ious)), 5) if ious else 0.0,
        "bbox_iou_min": round(float(np.min(ious)), 5) if ious else 0.0,
        "bbox_px_drift_median": round(float(np.median(bx_drifts)), 3) if bx_drifts else 0.0,
        "bbox_px_drift_max": round(float(np.max(bx_drifts)), 3) if bx_drifts else 0.0,
        "kpt_l2_px_median": round(float(np.median(kp_drifts)), 3) if kp_drifts else 0.0,
        "kpt_l2_px_max": round(float(np.max(kp_drifts)), 3) if kp_drifts else 0.0,
    }


def _stats_ms(arr):
    a = np.asarray(arr, dtype=np.float64)
    mean_ms = float(a.mean())
    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(float(np.median(a)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "fps": round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
    }


def bench_latency(det_cpu, det_gpu, pose_cpu, pose_gpu):
    res = {}
    lb = LetterBox((384, 640), auto=False, stride=32)

    def cpu_prep(img):
        x = lb(image=img)
        x = x[..., ::-1].transpose((2, 0, 1))
        x = np.ascontiguousarray(x)
        t = torch.from_numpy(x).to("cuda:0").float() / 255.0
        return t.unsqueeze(0)

    shapes = [
        ("1920x1440_slice", lambda: np.random.randint(0, 255, (1440, 2560, 3), dtype=np.uint8)[:, 320:2240]),
        ("1280x720", lambda: np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)),
        ("3840x2160", lambda: np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)),
    ]

    for label, mk in shapes:
        img = mk()
        # Preprocess only
        for _ in range(30):
            _ = cpu_prep(img)
            _ = letterbox_bgr_to_tensor(img, (384, 640), "cuda:0")
        torch.cuda.synchronize()

        tc, tg = [], []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = cpu_prep(img)
            torch.cuda.synchronize()
            tc.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(200):
            t0 = time.perf_counter()
            _ = letterbox_bgr_to_tensor(img, (384, 640), "cuda:0")
            torch.cuda.synchronize()
            tg.append((time.perf_counter() - t0) * 1000.0)

        # Stage 1 end-to-end
        for _ in range(30):
            det_cpu.predict(img)
            det_gpu.predict(img)
        torch.cuda.synchronize()

        s1c, s1g = [], []
        for _ in range(200):
            t0 = time.perf_counter()
            det_cpu.predict(img)
            torch.cuda.synchronize()
            s1c.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(200):
            t0 = time.perf_counter()
            det_gpu.predict(img)
            torch.cuda.synchronize()
            s1g.append((time.perf_counter() - t0) * 1000.0)

        # Stage 2a end-to-end
        for _ in range(30):
            pose_cpu.extract_actor_pose(img)
            pose_gpu.extract_actor_pose(img)
        torch.cuda.synchronize()

        s2c, s2g = [], []
        for _ in range(200):
            t0 = time.perf_counter()
            pose_cpu.extract_actor_pose(img)
            torch.cuda.synchronize()
            s2c.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(200):
            t0 = time.perf_counter()
            pose_gpu.extract_actor_pose(img)
            torch.cuda.synchronize()
            s2g.append((time.perf_counter() - t0) * 1000.0)

        res[label] = {
            "prep_cpu": _stats_ms(tc),
            "prep_gpu": _stats_ms(tg),
            "stage1_cpu": _stats_ms(s1c),
            "stage1_gpu": _stats_ms(s1g),
            "stage2a_cpu": _stats_ms(s2c),
            "stage2a_gpu": _stats_ms(s2g),
        }

    # Micro-overhead isolation on 1920x1440
    inp_t = torch.rand(1, 3, 384, 640, device="cuda:0", dtype=torch.float32)
    # ensure predictor is initialized
    det_gpu.predict(np.zeros((1440, 1920, 3), dtype=np.uint8))
    backend_model = det_gpu.model.predictor.model
    for _ in range(30):
        _ = backend_model(inp_t)
        _ = bool(inp_t.max() > 1.0 + 1e-6)
        _ = ops.convert_torch2numpy_batch(inp_t)[..., ::-1]
    torch.cuda.synchronize()

    t_fwd, t_max, t_d2h = [], [], []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = backend_model(inp_t)
        torch.cuda.synchronize()
        t_fwd.append((time.perf_counter() - t0) * 1000.0)
    for _ in range(200):
        t0 = time.perf_counter()
        _ = bool(inp_t.max() > 1.0 + 1e-6)
        torch.cuda.synchronize()
        t_max.append((time.perf_counter() - t0) * 1000.0)
    for _ in range(200):
        t0 = time.perf_counter()
        _ = ops.convert_torch2numpy_batch(inp_t)[..., ::-1]
        torch.cuda.synchronize()
        t_d2h.append((time.perf_counter() - t0) * 1000.0)

    res["micro_overhead"] = {
        "autobackend_forward": _stats_ms(t_fwd),
        "loadtensor_max_sync": _stats_ms(t_max),
        "convert_torch2numpy_batch_d2h": _stats_ms(t_d2h),
    }
    return res


def main():
    os.makedirs(os.path.join(REPO_DIR, "results", "amp_edge_speed"), exist_ok=True)
    det_cpu = Stage1WeaponDetector(gpu_preprocess=False)
    det_gpu = Stage1WeaponDetector(gpu_preprocess=True)
    pose_cpu = Stage2aPoseTracker(gpu_preprocess=False)
    pose_gpu = Stage2aPoseTracker(gpu_preprocess=True)

    gt_images = sorted(glob.glob(os.path.join(GT_DATASET_DIR, "**", "*.jpg"), recursive=True))
    print(f"[1/4] Stage 1 GT parity ({len(gt_images)} images)...", flush=True)
    gt_stats, gt_drift = bench_stage1_gt(det_cpu, det_gpu, gt_images)
    print("GT CPU Overall:", gt_stats["cpu"]["Overall"], flush=True)
    print("GT GPU Overall:", gt_stats["gpu"]["Overall"], flush=True)
    print("GT Drift:", gt_drift, flush=True)

    print("[2/4] Stage 1 Lab Holdout + Confuser parity...", flush=True)
    lab_summary, lab_detail = bench_lab_holdout(det_cpu, det_gpu)
    print("Lab Summary:", lab_summary, flush=True)

    print("[3/4] Stage 2a ByteTrack + Pose parity...", flush=True)
    s2a_parity = bench_stage2a_parity()
    print("Stage 2a Parity:", s2a_parity, flush=True)

    print("[4/4] Latency & FPS benchmarks...", flush=True)
    lat = bench_latency(det_cpu, det_gpu, pose_cpu, pose_gpu)
    for k, v in lat.items():
        print(f"  {k}: {v}", flush=True)

    out_path = os.path.join(REPO_DIR, "results", "amp_edge_speed", "step1_gpu_preprocess.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gt_stats": gt_stats,
                "gt_drift": gt_drift,
                "lab_summary": lab_summary,
                "lab_detail": lab_detail,
                "stage2a_parity": s2a_parity,
                "latency": lat,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
