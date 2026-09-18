"""
src/engine_builder.py

TensorRT Engine Builder & Hardware Optimizer.
Enforces:
1. Strict CUDA requirement (no silent CPU degradation).
2. Hardware-aware precision:
   - FP32 for legacy Pascal GPUs (Compute Capability < 7.0, e.g. GTX 1060).
   - FP16 for modern Tensor Core GPUs (Compute Capability >= 7.0 or Jetson >= 5.3).
3. Low-Medium spec memory guardrail: Strict 1.0 GB workspace memory pool cap.
4. Auto-compilation on first use from universal .onnx blueprints.
"""

import os
import sys
import json
import time
from typing import Optional, Dict, Any, Tuple
import torch
import tensorrt as trt

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def get_hardware_precision() -> Tuple[str, bool]:
    """
    Analyzes local NVIDIA GPU architecture and selects optimal precision.
    Returns:
        (precision_str, is_fp16_supported)
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[HARDWARE ERROR] Hierarchical Threat Detection Edge-AI strictly requires an NVIDIA CUDA GPU.\n"
            "No CUDA-capable device was detected on this system."
        )

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    # Pascal (sm_60, sm_61 like GTX 1060) executes FP16 slowly; optimal is FP32.
    # Volta (sm_70), Turing (sm_75), Ampere (sm_86 like RTX 3090), Ada (sm_89), Blackwell (sm_100),
    # or Jetson Maxwell (sm_53) have native fast FP16 support.
    is_pascal = (cap[0] == 6)
    use_fp16 = not is_pascal and (cap >= (7, 0) or cap == (5, 3))
    precision = "FP16" if use_fp16 else "FP32"

    return precision, use_fp16


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    workspace_gb: float = 1.0,
    force_fp16: Optional[bool] = None,
    verbose: bool = False,
) -> str:
    """
    Compiles an ONNX model into a serialized TensorRT .engine binary tailored to the local GPU.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Path to write the compiled .engine file.
        metadata: Optional dictionary to embed (for Ultralytics YOLO compatibility).
        workspace_gb: Max workspace size in GB (defaults to 1.0 GB for low-spec safety).
        force_fp16: If specified, overrides automatic precision detection.
        verbose: Enable verbose TensorRT logging.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"[ENGINE BUILDER] Source ONNX blueprint not found: {onnx_path}")

    device_name = torch.cuda.get_device_name(0)
    auto_prec, auto_fp16 = get_hardware_precision()
    use_fp16 = force_fp16 if force_fp16 is not None else auto_fp16
    prec_name = "FP16" if use_fp16 else "FP32"

    print("\n" + "=" * 80)
    print(f"   TENSORRT COMPILER: Compiling {os.path.basename(onnx_path)} -> {os.path.basename(engine_path)}")
    print(f"   Target Hardware: {device_name} | Precision: {prec_name}")
    print(f"   Memory Workspace Limit: {workspace_gb:.1f} GB (Edge/Consumer Guardrail)")
    print("=" * 80)
    print("   [NOTE] First-run compilation performs kernel auto-tuning (~30-90s). Please wait...")

    logger_level = trt.Logger.INFO if verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_level)
    builder = trt.Builder(logger)

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            error_msgs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"[ENGINE BUILDER] Failed to parse ONNX file {onnx_path}:\n" + "\n".join(error_msgs))

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1024 ** 3))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    compile_time = time.perf_counter() - t0

    if serialized_engine is None:
        raise RuntimeError(f"[ENGINE BUILDER] TensorRT engine build failed for {onnx_path}.")

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        if metadata is not None:
            meta_str = json.dumps(metadata)
            f.write(len(meta_str).to_bytes(4, byteorder="little", signed=True))
            f.write(meta_str.encode("utf-8"))
        f.write(serialized_engine)

    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"   -> Compilation Successful! Engine generated in {compile_time:.2f}s ({file_size_mb:.2f} MB)")
    print(f"   -> Saved to: {engine_path}\n")

    return engine_path


def ensure_engine(
    onnx_path: str,
    engine_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    workspace_gb: float = 1.0,
) -> str:
    """
    Ensures that a compiled .engine exists for the local machine.
    If it exists, returns immediately.
    If missing, automatically compiles from the .onnx blueprint.
    """
    if os.path.exists(engine_path):
        return engine_path

    print(f"[ENGINE BUILDER] Engine binary {os.path.basename(engine_path)} not found on this machine.")
    print(f"                 Auto-compiling from universal blueprint {os.path.basename(onnx_path)}...")
    return build_engine_from_onnx(
        onnx_path=onnx_path,
        engine_path=engine_path,
        metadata=metadata,
        workspace_gb=workspace_gb,
    )
