"""The sample-driven transforms keep what they promise."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.skeleton_sim import from_samples as G


def skeleton(n=60):
    rng = np.random.default_rng(0)
    base = rng.uniform(100, 500, (17, 2))
    return (base + np.arange(n)[:, None, None] * rng.uniform(-1, 1, (17, 2))).astype(np.float32)


def test_retarget_with_unit_scales_changes_nothing():
    seq = skeleton()
    one = dict(torso=1, shoulder=1, hip=1, head=1, upper_arm=1, forearm=1, thigh=1, shin=1)
    assert np.allclose(G.retarget(seq, one), seq, atol=1e-3)


def test_retarget_scales_the_named_bone_only():
    seq = skeleton()
    one = dict(torso=1, shoulder=1, hip=1, head=1, upper_arm=1, forearm=2, thigh=1, shin=1)
    out = G.retarget(seq, one)
    assert np.allclose(out[:, G.R_WR] - out[:, G.R_EL], 2 * (seq[:, G.R_WR] - seq[:, G.R_EL]), atol=1e-3)
    assert np.allclose(out[:, G.R_EL], seq[:, G.R_EL], atol=1e-3)


def test_view_between_returns_the_first_view_at_zero():
    a, b = skeleton(), skeleton()[::-1]
    assert np.allclose(G.view_between(a, b, 0.0), a, atol=1e-3)


def test_time_warp_without_wobble_returns_the_same_frames():
    seq = skeleton(50)
    out = G.time_warp(seq, 0, 50, np.random.default_rng(0), wobble=0.0)
    assert out.shape == (50, 17, 2) and np.allclose(out, seq, atol=1e-3)


def test_strike_events_finds_each_burst():
    seq = np.zeros((300, 17, 2), np.float32)
    seq[:, [G.L_SH, G.R_SH]] = [100, 100]
    seq[:, [G.L_HIP, G.R_HIP]] = [100, 300]
    x = np.zeros(300)
    for start in (50, 150, 250):
        x[start:start + 15] = np.linspace(0, 300, 15)
        x[start + 15:start + 30] = np.linspace(300, 0, 15)
    seq[:, G.R_WR, 0] = 100 + x
    seq[:, G.R_WR, 1] = 200
    events = G.strike_events(G.hand_speed(seq, G.R_WR))
    assert len(events) == 3
    for (a, peak, e), start in zip(events, (50, 150, 250)):
        assert a <= peak < e and abs(peak - (start + 15)) < 25


def test_graft_legs_keeps_the_upper_body():
    a, b = skeleton(), skeleton()[::-1]
    out = G.graft_legs(a, b)
    assert np.array_equal(out[:, G.R_WR], a[:, G.R_WR]) and not np.allclose(out[:, G.R_AN], a[:, G.R_AN])
