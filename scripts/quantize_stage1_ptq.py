"""
scripts/quantize_stage1_ptq.py

Step 2a: Mixed-Precision INT8 PTQ Gate via NVIDIA ModelOpt (Entropy/Histogram Calibration).
Builds an explicit Q/DQ ONNX from Stage 1 (stage1_repro846_kdgtclean_p250.pt) or
Stage 2a (yolo26s-pose.pt), patches residual-add quantizers into Bottleneck/SPPF shortcuts
for TensorRT Conv+Add fusion, locks precision-sensitive layers to FP16 via
modelopt.onnx.autocast, compiles a STRONGLY_TYPED TensorRT engine, and profiles layers.

Never overwrites shipped weights/ or calibrates on holdout test sets.
"""

import argparse
import copy
import glob
import json
import os
import random
import sys
import types
from collections import Counter

import cv2
import numpy as np
import onnx
import tensorrt as trt
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

import modelopt.onnx.autocast as moa
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.calib import HistogramCalibrator
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import enable_stats_collection
from modelopt.torch.quantization.nn import TensorQuantizer

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.engine_builder import build_engine_from_onnx

TRAIN_LIST = os.path.join(REPO_DIR, "data", "stage1_negatives", "lab_gt_clean", "train.txt")


def load_stratified_calibration_tensors(n_frames: int = 600, imgsz=(384, 640), seed: int = 42):
    """
    Deterministically sample >= 500 frames from Stage 1 training set ONLY
    (stratified across knife / phone / empty), preprocessed with exact inference LetterBox.
    """
    with open(TRAIN_LIST, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    for p in paths:
        norm = p.replace("\\", "/").lower()
        assert "ground_truth" not in norm, f"Forbidden test set path in train list: {p}"
        assert "002_checkerboard" not in norm and "2026-10-31" not in norm, f"Forbidden Take 002 in train list: {p}"

    buckets = {"knife": [], "phone": [], "empty": []}
    for p in paths:
        lbl = p.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        lbl = os.path.splitext(lbl)[0] + ".txt"
        cls_ids = set()
        if os.path.exists(lbl):
            with open(lbl, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if parts:
                        cls_ids.add(int(parts[0]))
        if 1 in cls_ids:
            buckets["knife"].append(p)
        elif 2 in cls_ids:
            buckets["phone"].append(p)
        else:
            buckets["empty"].append(p)

    rng = random.Random(seed)
    for k in buckets:
        rng.shuffle(buckets[k])

    targets = {"knife": n_frames // 2, "phone": n_frames // 4, "empty": n_frames - (n_frames // 2) - (n_frames // 4)}
    selected = []
    leftover = []
    for k, want in targets.items():
        take = buckets[k][:want]
        selected.extend(take)
        leftover.extend(buckets[k][want:])
    if len(selected) < n_frames:
        rng.shuffle(leftover)
        selected.extend(leftover[: n_frames - len(selected)])
    selected = selected[:n_frames]

    lb = LetterBox(imgsz, auto=False, stride=32)
    tensors = []
    for p in selected:
        img = cv2.imread(p)
        if img is None:
            continue
        x = lb(image=img)
        x = x[..., ::-1].transpose((2, 0, 1))
        x = np.ascontiguousarray(x)
        tensors.append(torch.from_numpy(x).float() / 255.0)
    print(f"[CALIB] Loaded {len(tensors)} stratified training frames "
          f"(pool: knife={len(buckets['knife'])}, phone={len(buckets['phone'])}, empty={len(buckets['empty'])})")
    return torch.stack(tensors, dim=0)


def patch_residual_add_quantizers(nn_model: torch.nn.Module):
    """
    Attach 8-bit histogram-calibrated TensorQuantizers to residual shortcut inputs
    in Bottleneck(add=True) and SPPF(add=True) so TensorRT fuses Conv+Add in INT8.
    """
    qattr = QuantizerAttributeConfig(num_bits=8, axis=None, calibrator="histogram")
    patched = []
    for name, mod in nn_model.named_modules():
        cname = type(mod).__name__
        if cname == "Bottleneck" and getattr(mod, "add", False):
            mod.shortcut_quantizer = TensorQuantizer(qattr)

            def _bn_fwd(self, x):
                return self.shortcut_quantizer(x) + self.cv2(self.cv1(x))

            mod.forward = types.MethodType(_bn_fwd, mod)
            patched.append(name)
        elif cname == "SPPF" and getattr(mod, "add", False):
            mod.shortcut_quantizer = TensorQuantizer(qattr)

            def _sppf_fwd(self, x):
                y = [self.cv1(x)]
                y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
                y = self.cv2(torch.cat(y, 1))
                return y + self.shortcut_quantizer(x)

            mod.forward = types.MethodType(_sppf_fwd, mod)
            patched.append(name)
    print(f"[PATCH] Patched {len(patched)} residual shortcut quantizers: {patched}")
    return patched


def build_entropy_int8_config():
    """INT8 config using HistogramCalibrator (entropy) for activations, per-channel weights."""
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    cfg["quant_cfg"]["*input_quantizer"] = {"num_bits": 8, "axis": None, "calibrator": "histogram"}
    cfg["quant_cfg"]["*output_quantizer"] = {"enable": False}
    cfg["algorithm"] = None  # skip max_calibrate; calibrate_entropy() runs HistogramCalibrator(method='entropy')
    return cfg


def calibrate_entropy(nn_model: torch.nn.Module, calib_tensor: torch.Tensor, batch_size: int = 8, device: str = "cuda:0"):
    """Collect activation histograms and compute amax via KL-entropy (NOT Min-Max)."""
    enable_stats_collection(nn_model)
    with torch.no_grad():
        for i in range(0, calib_tensor.shape[0], batch_size):
            batch = calib_tensor[i : i + batch_size].to(device, non_blocking=True)
            _ = nn_model(batch)

    n_hist, n_max = 0, 0
    for _, mod in nn_model.named_modules():
        if isinstance(mod, TensorQuantizer) and not mod._disabled:
            if mod._calibrator is not None and not mod._dynamic:
                if isinstance(mod._calibrator, HistogramCalibrator):
                    if mod._calibrator.compute_amax("entropy") is not None:
                        mod.load_calib_amax("entropy")
                    n_hist += 1
                else:
                    if mod._calibrator.compute_amax() is not None:
                        mod.load_calib_amax()
                    n_max += 1
            mod.enable_quant()
            mod.disable_calib()
    nn_model.to(device)
    print(f"[CALIB] Entropy calibration complete: {n_hist} HistogramCalibrator(entropy) + {n_max} MaxCalibrator(per-ch weights)", flush=True)


def lock_sensitive_layers_fp16(nn_model: torch.nn.Module, lock_p3: bool = False):
    """Disable INT8 quantizers on precision-critical layers so autocast keeps them in FP16."""
    patterns = [
        "*model.0.*",
        "*model.10.*",
        "*model.22.m.0.1.*",
        "*model.23.*",
    ]
    if lock_p3:
        patterns.append("*model.16.*")
    for pat in patterns:
        mtq.disable_quantizer(nn_model, pat)
    n_enabled = sum(1 for _, m in nn_model.named_modules() if isinstance(m, TensorQuantizer) and not m._disabled)
    n_disabled = sum(1 for _, m in nn_model.named_modules() if isinstance(m, TensorQuantizer) and m._disabled)
    print(f"[MIXED] Locked patterns {patterns} -> enabled={n_enabled}, disabled(FP16)={n_disabled}")


def inspect_onnx_qdq(onnx_path: str):
    m = onnx.load(onnx_path, load_external_data=False)
    counts = Counter(n.op_type for n in m.graph.node)
    q_cnt = counts.get("QuantizeLinear", 0)
    dq_cnt = counts.get("DequantizeLinear", 0)
    for forbidden in ("QLinearConv", "ConvInteger", "QLinearMatMul", "MatMulInteger"):
        assert counts.get(forbidden, 0) == 0, f"Forbidden pre-quantized op {forbidden} in {onnx_path}"
    assert q_cnt > 0 and dq_cnt > 0, f"Missing Q/DQ nodes in {onnx_path}: Q={q_cnt}, DQ={dq_cnt}"
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim], o.type.tensor_type.elem_type) for o in m.graph.output]
    print(f"[ONNX] {os.path.basename(onnx_path)}: Q={q_cnt}, DQ={dq_cnt}, outputs={outs}")


def inspect_engine_layers(engine_path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder="little")
            json.loads(f.read(meta_len).decode("utf-8"))
        except Exception:
            f.seek(0)
        engine = runtime.deserialize_cuda_engine(f.read())
    inspector = engine.create_engine_inspector()
    prec_counts, type_counts = Counter(), Counter()
    reformat_cnt, fused_q_conv = 0, 0
    for i in range(engine.num_layers):
        raw = json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
        if isinstance(raw, dict):
            prec_counts[raw.get("Precision", "UNKNOWN")] += 1
            lt = raw.get("LayerType", "UNKNOWN")
            type_counts[lt] += 1
            nm = raw.get("Name", "")
        else:
            nm = str(raw)
            prec_counts["LAYER_NAMES_ONLY"] += 1
            lt = "Reformat" if ("Reformatting" in nm or "reformat" in nm.lower()) else "Fused"
            type_counts[lt] += 1
        if "Reformat" in lt or "reformat" in nm.lower():
            reformat_cnt += 1
        if "weight_quantizer" in nm and "Conv" in nm:
            fused_q_conv += 1
    summary = {
        "total_layers": engine.num_layers,
        "precisions": dict(prec_counts),
        "reformat_layers": reformat_cnt,
        "fused_q_conv_layers": fused_q_conv,
        "layer_types": dict(type_counts),
    }
    print(f"[TRT PROFILE] {os.path.basename(engine_path)}: {summary}")
    return summary


def run_ptq(
    pt_path: str,
    out_stem: str,
    task: str = "detect",
    n_calib: int = 600,
    lock_p3: bool = False,
    load_calib: str = None,
    device: str = "cuda:0",
):
    weights_dir = os.path.join(REPO_DIR, "weights")
    fp32_onnx = os.path.join(weights_dir, f"{out_stem}_fp32.onnx")
    fp16_onnx = os.path.join(weights_dir, f"{out_stem}_fp16.onnx")
    engine_path = os.path.join(weights_dir, f"{out_stem}.engine")

    yolo = YOLO(pt_path, task=task)
    nn_model = yolo.model.to(device).eval()
    nn_model.fuse()
    nn_model.end2end = True
    head = nn_model.model[-1]
    head.export = True
    head.format = "onnx"
    head.end2end = True
    head.max_det = 300
    head.shape = None

    quant_cfg = build_entropy_int8_config()
    mtq.quantize(nn_model, quant_cfg, forward_loop=None)
    patch_residual_add_quantizers(nn_model)
    nn_model.to(device)
    if load_calib:
        sd = torch.load(load_calib, map_location=device, weights_only=False)
        for name, mod in nn_model.named_modules():
            if isinstance(mod, TensorQuantizer):
                k = f"{name}._amax"
                if k in sd:
                    mod.amax = sd[k].to(device)
                    mod.enable_quant()
                    mod.disable_calib()
        nn_model.load_state_dict(sd, strict=False)
        nn_model.to(device)
        lock_sensitive_layers_fp16(nn_model, lock_p3=lock_p3)
        print(f"[CALIB] Loaded calibrated state from {os.path.basename(load_calib)}", flush=True)
    else:
        calib_tensor = load_stratified_calibration_tensors(n_frames=n_calib)
        lock_sensitive_layers_fp16(nn_model, lock_p3=lock_p3)
        calibrate_entropy(nn_model, calib_tensor, batch_size=8, device=device)

        # Save calibrated student state for instant reuse (e.g., Step 2b QAD or --lock-p3)
        calib_ckpt = os.path.join(weights_dir, f"{out_stem}_calib.pt")
        torch.save(nn_model.state_dict(), calib_ckpt)
        print(f"[CKPT] Saved calibrated state_dict to {os.path.basename(calib_ckpt)}", flush=True)

    # Export explicit Q/DQ FP32 ONNX on CPU (avoids Windows MSVC modelopt_cuda_ext JIT fallback)
    nn_model.cpu().eval()
    dummy = torch.zeros(1, 3, 384, 640, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            nn_model,
            dummy,
            fp32_onnx,
            opset_version=17,
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
        )
    inspect_onnx_qdq(fp32_onnx)

    # Autocast non-quantized subgraphs to FP16
    print(f"[AUTOCAST] Converting {os.path.basename(fp32_onnx)} -> {os.path.basename(fp16_onnx)} (fp16, data_max=2048)...")
    mp = moa.convert_to_mixed_precision(
        onnx_path=fp32_onnx,
        low_precision_type="fp16",
        data_max=2048,
        keep_io_types=False,
        providers=["cpu"],
    )
    onnx.save(mp, fp16_onnx)
    inspect_onnx_qdq(fp16_onnx)

    metadata = {
        "description": f"Mixed-Precision INT8 PTQ ({out_stem})",
        "author": "Ultralytics",
        "version": "8.4.9",
        "stride": 32,
        "task": task,
        "batch": 1,
        "imgsz": [384, 640],
        "names": yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names)),
        "end2end": True,
    }
    if task == "pose":
        metadata["kpt_shape"] = [17, 3]

    build_engine_from_onnx(
        onnx_path=fp16_onnx,
        engine_path=engine_path,
        metadata=metadata,
        workspace_gb=1.0,
    )
    profile = inspect_engine_layers(engine_path)
    return {
        "fp32_onnx": fp32_onnx,
        "fp16_onnx": fp16_onnx,
        "engine_path": engine_path,
        "trt_profile": profile,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", action="store_true", help="Quantize Stage 2a yolo26s-pose.pt instead of Stage 1")
    parser.add_argument("--lock-p3", action="store_true", help="Also lock model.16 (P3 neck exit) to FP16")
    parser.add_argument("--n-calib", type=int, default=600)
    parser.add_argument("--load-calib", type=str, default=None, help="Reuse saved calibrated state_dict")
    args = parser.parse_args()

    if args.pose:
        pt = os.path.join(REPO_DIR, "weights", "yolo26s-pose.pt")
        stem = "yolo26s-pose_int8ptq"
        res = run_ptq(pt, stem, task="pose", n_calib=min(args.n_calib, 500), lock_p3=args.lock_p3, load_calib=args.load_calib)
    else:
        pt = os.path.join(REPO_DIR, "weights", "stage1_repro846_kdgtclean_p250.pt")
        stem = "stage1_repro846_kdgtclean_p250_int8ptq" + ("_lockp3" if args.lock_p3 else "")
        res = run_ptq(pt, stem, task="detect", n_calib=args.n_calib, lock_p3=args.lock_p3, load_calib=args.load_calib)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
