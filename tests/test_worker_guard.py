"""python tests/test_worker_guard.py"""
import os
import sys
import tempfile

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

# An engine of the right size is used as it is; a wrong one falls back to an existing <name>_<h>x<w>.engine
accepted = {"a.engine", "a_480x640.engine"}
eb.engine_accepts = lambda path, imgsz: os.path.basename(path) in accepted
assert eb.engine_for_imgsz("a.engine", "a.pt", {}, (480, 640)) == "a.engine"
with tempfile.TemporaryDirectory() as tmp:
    base = os.path.join(tmp, "a.engine")
    sized = os.path.join(tmp, "a_480x640.engine")
    open(sized, "wb").close()
    accepted = {"a_480x640.engine"}
    assert eb.engine_for_imgsz(base, "a.pt", {}, (480, 640)) == sized
    # ... and a sized engine that is still wrong is an error, not a silent run
    accepted = set()
    try:
        eb.engine_for_imgsz(base, "a.pt", {}, (480, 640))
        raise SystemExit("wrong-size engine was accepted")
    except RuntimeError as e:
        assert "does not take 480x640" in str(e)
print("ok")
