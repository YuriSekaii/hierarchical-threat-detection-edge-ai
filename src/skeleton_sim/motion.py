"""
Keyframe motion for the skeleton simulator.

A Timeline holds keys per control (see body.CONTROLS). Between keys the value follows a cubic
Hermite curve: a "stop" key has zero velocity (start/end of a strike, a hold), a "pass" key
keeps the motion flowing (the middle of a slash arc). Feet lift off the floor automatically
when they move between two keys. Slow random sway is added on top so no two takes are identical.
"""

import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.skeleton_sim.body import CONTROLS, BodyShape, basis, solve_pose

FPS = 30.0


class Timeline:
    def __init__(self, shape: BodyShape, rng, start_xy=(0.0, 0.0), start_yaw=0.0):
        self.shape = shape
        self.rng = rng
        self.keys = {k: [] for k in CONTROLS}
        self.t = 0.0  # cursor: the end of the last scheduled move
        self.tags = []  # (t0, t1, label) of each scheduled move, for metadata
        # start pose: feet under the hips, slightly apart and staggered
        B = basis(start_yaw)
        right, fwd = B[:2, 0], B[:2, 1]
        p = np.asarray(start_xy, float)
        w = shape.m(shape.hip_half) * rng.uniform(1.0, 1.9)
        stag = rng.uniform(-0.12, 0.12)
        self.state = copy.deepcopy(CONTROLS)
        self.state.update(pelvis=p.copy(), yaw=start_yaw,
                          foot_R=p + w * right + stag * fwd, foot_L=p - w * right - stag * fwd)
        for k, v in self.state.items():
            self.keys[k].append((0.0, np.array(v, float), "stop"))

    # ---------------------------------------------------------------- keys
    def set(self, t, kind="stop", **controls):
        """Adds keys at absolute time t. Values are absolute (world for pelvis/feet/yaw)."""
        for k, v in controls.items():
            v = np.array(v, float)
            lst = self.keys[k]
            # replace a key at the same time, otherwise insert in order
            lst[:] = [kv for kv in lst if abs(kv[0] - t) > 1e-6]
            lst.append((t, v, kind))
            lst.sort(key=lambda kv: kv[0])
            if t >= max(kv[0] for kv in lst) - 1e-9:
                self.state[k] = v

    def hold(self, t, *names):
        """Pins the current value of controls at time t, so a later key does not start moving earlier."""
        self.set(t, **{n: self.state[n] for n in names})

    def move(self, dur, start=None, kind="stop", advance=True, label=None, **controls):
        """Moves the given controls from their current value to the new one over `dur` seconds,
        starting at `start` (default: the cursor). A control keeps still until its move starts."""
        t0 = self.t if start is None else start
        for k in controls:
            if self.keys[k][-1][0] < t0 - 1e-6:
                self.keys[k].append((t0, np.array(self.state[k], float), "stop"))
        self.set(t0 + dur, kind=kind, **controls)
        if label:
            self.tags.append((t0, t0 + dur, label))
        if advance:
            self.t = max(self.t, t0 + dur)
        return t0 + dur

    def truncate(self, t):
        """Drops every key after t (the pose there stays as it is at t) and moves the cursor to t."""
        for k, lst in self.keys.items():
            if lst[-1][0] > t:
                v = np.array(_interp(lst, t), float)
                lst[:] = [kv for kv in lst if kv[0] < t - 1e-6] + [(t, v, "stop")]
                self.state[k] = v
        self.tags = [(a, min(b, t), n) for a, b, n in self.tags if a < t]
        self.t = t

    def wait(self, dur):
        self.t += dur
        return self.t

    def value_at(self, name, t):
        return _interp(self.keys[name], t)

    # ---------------------------------------------------------------- body-relative helpers
    def local_to_world(self, xy_local, yaw=None, origin=None):
        yaw = self.state["yaw"] if yaw is None else yaw
        origin = self.state["pelvis"] if origin is None else origin
        B = basis(yaw)
        return np.asarray(origin, float) + B[:2, 0] * xy_local[0] + B[:2, 1] * xy_local[1]

    def heading(self):
        return basis(self.state["yaw"])[:2, 1]

    # ---------------------------------------------------------------- sampling
    def sample(self, duration=None, fps=FPS, sway=1.0):
        """Returns (T, 17, 3) joints and the per-frame controls."""
        dur = duration if duration is not None else max(kv[0] for lst in self.keys.values() for kv in lst)
        n = int(round(dur * fps)) + 1
        ts = np.arange(n) / fps
        ctrl = {k: np.stack([np.atleast_1d(_interp(self.keys[k], t)) for t in ts]) for k in self.keys}
        # feet: add a lift arc while a foot travels between two keys
        for side in ("R", "L"):
            ctrl["lift_" + side][:, 0] += _foot_lift(self.keys["foot_" + side], ts, self.rng)
        if sway > 0:
            _add_sway(ctrl, self.rng, sway, n)
        joints = np.zeros((n, 17, 3))
        for i in range(n):
            c = {k: (v[i] if v.shape[1] > 1 else float(v[i, 0])) for k, v in ctrl.items()}
            joints[i] = solve_pose(self.shape, c)
        return joints, ctrl


def _interp(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    i = np.searchsorted([k[0] for k in keys], t) - 1
    t0, v0, k0 = keys[i]
    t1, v1, k1 = keys[i + 1]
    h = t1 - t0
    u = (t - t0) / h

    def tangent(j):
        tj, vj, kj = keys[j]
        if kj == "stop" or j == 0 or j == len(keys) - 1:
            return np.zeros_like(vj)
        tp, vp, _ = keys[j - 1]
        tn, vn, _ = keys[j + 1]
        return (vn - vp) / (tn - tp)

    m0, m1 = tangent(i) * h, tangent(i + 1) * h
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * m0
            + (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * m1)


def _foot_lift(keys, ts, rng):
    lift = np.zeros(len(ts))
    for (t0, v0, _), (t1, v1, _) in zip(keys[:-1], keys[1:]):
        d = np.linalg.norm(v1 - v0)
        if d < 0.04 or t1 - t0 < 1e-3:
            continue
        h = min(0.13, 0.25 * d) * rng.uniform(0.7, 1.2)
        m = (ts > t0) & (ts < t1)
        u = (ts[m] - t0) / (t1 - t0)
        lift[m] += h * np.sin(np.pi * u) ** 1.5
    return lift


def _smooth_noise(rng, n, sigma_frames, scale, dims=1):
    x = rng.normal(0, 1, (n + 60, dims))
    x = gaussian_filter1d(x, sigma_frames, axis=0)[30:30 + n]
    x /= max(x.std(), 1e-9)
    return x * scale


def _add_sway(ctrl, rng, k, n):
    """Small slow drift: weight shift, breathing, wandering hands and head."""
    ctrl["pelvis"] += _smooth_noise(rng, n, 12, 0.008 * k, 2)
    ctrl["pitch"] += _smooth_noise(rng, n, 10, np.radians(1.2) * k)
    ctrl["roll"] += _smooth_noise(rng, n, 10, np.radians(1.0) * k)
    ctrl["twist"] += _smooth_noise(rng, n, 10, np.radians(1.5) * k)
    ctrl["head_pitch"] += _smooth_noise(rng, n, 8, np.radians(4) * k)
    ctrl["head_yaw"] += _smooth_noise(rng, n, 8, np.radians(6) * k)
    ctrl["crouch"] += _smooth_noise(rng, n, 15, 0.006 * k)
    for s in ("R", "L"):
        ctrl["wr_" + s] += _smooth_noise(rng, n, 6, 0.018 * k, 3)
        ctrl["swv_" + s] += _smooth_noise(rng, n, 10, np.radians(6) * k)
