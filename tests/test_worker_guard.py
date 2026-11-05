"""python tests/test_worker_guard.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.engine_builder as eb  # noqa: E402
from src.debug_stage2_tracer import DebugHierarchicalPipeline  # noqa: E402
from src.inference_production_pipeline import ProductionHierarchicalPipeline  # noqa: E402


def boom():
    raise AssertionError("input size not equal to max model size")


# A worker exception stops the run and is kept for the error exit
p = ProductionHierarchicalPipeline.__new__(ProductionHierarchicalPipeline)
p.is_running, p.fatal_error = True, None
p._guarded(boom)()
assert p.is_running is False and "boom failed" in p.fatal_error and "max model size" in p.fatal_error

# Debug mode tolerates 4 exceptions in a row and re-raises the 5th; a success resets the count
d = DebugHierarchicalPipeline.__new__(DebugHierarchicalPipeline)
d._dbg_fail_streak = {}
d._dbg = type("Log", (), {"exception": lambda self, where, exc: None})()
err = RuntimeError("x")
for _ in range(4):
    d._dbg_worker_failed("w", err)
d._dbg_fail_streak["w"] = 0
for _ in range(4):
    d._dbg_worker_failed("w", err)
try:
    d._dbg_worker_failed("w", err)
    raise SystemExit("5th exception in a row did not re-raise")
except RuntimeError:
    pass

# The pipeline runs at the size both engines take; engines of different sizes stop it, not a silent run
engine_sizes = {"weapon": (384, 640), "pose": (384, 640), "other": (480, 640), "dynamic": None}
eb.engine_input_size = engine_sizes.get
assert eb.shared_input_size("weapon", "pose") == (384, 640)
assert eb.shared_input_size("dynamic") is None
try:
    eb.shared_input_size("weapon", "other")
    raise SystemExit("engines of different sizes were accepted")
except RuntimeError as e:
    assert "different input sizes" in str(e)
print("ok")
