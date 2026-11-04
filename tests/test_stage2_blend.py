"""python tests/test_stage2_blend.py"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.stage2b_staircase import blend_probs  # noqa: E402

assert math.isclose(blend_probs([0.5, 0.5, 0.5]), 0.5)
assert math.isclose(blend_probs([0.9, 0.1, 0.5]), 0.5)  # opposite scores cancel
assert blend_probs([0.9, 0.9, 0.2]) > 0.5 > blend_probs([0.1, 0.1, 0.8])
assert 0.0 < blend_probs([0.0, 0.0, 0.0]) < blend_probs([1.0, 1.0, 1.0]) < 1.0  # saturated tiers stay finite
print("ok")
