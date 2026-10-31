"""
Perturbs real 2D skeleton clips (training XML or lab segments) into new, slightly different takes:
the knife arm a little lower/higher or more bent, the body leaning, a wider or narrower stance,
other body proportions, a different speed, a slightly different view, or the mirror image.

All edits work on pixel coordinates (T, 17, 2) and keep the clip length near the original.
Bone lengths of edited limbs are kept by solving the elbow/knee again with 2D two-bone IK.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.skeleton_sim.camera import JITTER

FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
ARM = {"L": (5, 7, 9), "R": (6, 8, 10)}
LEG = {"L": (11, 13, 15), "R": (12, 14, 16)}


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def _torso(k):
    return np.linalg.norm((k[:, 5] + k[:, 6]) / 2 - (k[:, 11] + k[:, 12]) / 2, axis=-1)


def knife_arm(k):
    """The arm whose wrist moves most relative to its shoulder."""
    rng_ = {s: np.ptp(np.linalg.norm(k[:, w] - k[:, sh], axis=-1)) for s, (sh, _, w) in ARM.items()}
    return max(rng_, key=rng_.get)


def _ik2d(root, mid, end_new, a, b):
    """Elbow/knee position for a new end point, bending to the same side as before."""
    v = end_new - root
    d = np.clip(np.linalg.norm(v, axis=-1), np.abs(a - b) + 1e-3, (a + b) * 0.999)
    u = v / np.maximum(np.linalg.norm(v, axis=-1), 1e-6)[:, None]
    end = root + u * d[:, None]
    cos_a = np.clip((a * a + d * d - b * b) / (2 * a * d), -1, 1)
    m = mid - root
    side = np.sign(u[:, 0] * m[:, 1] - u[:, 1] * m[:, 0])
    side[side == 0] = 1
    ang = np.arccos(cos_a) * side
    return root + a[:, None] * _rot(u, ang), end


def arm_edit(k, side, lower_deg=None, bend=None):
    """Swings the knife arm's path about the shoulder (lower_deg > 0 = lower) and/or shortens its reach.
    The edit grows with how far the arm is extended, so the resting arm stays where it was."""
    sh, el, wr = ARM[side]
    k = k.copy()
    a = np.linalg.norm(k[:, el] - k[:, sh], axis=-1)
    b = np.linalg.norm(k[:, wr] - k[:, el], axis=-1)
    a, b = np.full_like(a, np.median(a)), np.full_like(b, np.median(b))
    v = k[:, wr] - k[:, sh]
    ext = np.linalg.norm(v, axis=-1)
    w = np.clip((ext - ext.min()) / max(np.ptp(ext), 1e-6), 0, 1) ** 1.5
    w = gaussian_filter1d(w, 2)
    peak = int(np.argmax(ext))
    if lower_deg:
        # rotating toward image-down: sign depends on which way the arm points at its peak
        sgn = 1.0 if v[peak, 0] >= 0 else -1.0
        v = _rot(v, np.radians(lower_deg) * sgn * w)
    if bend:
        v = v * (1 - (1 - bend) * w)[:, None]
    k[:, el], k[:, wr] = _ik2d(k[:, sh], k[:, el], k[:, sh] + v, a, b)
    return k


def lean(k, deg):
    """Leans the upper body (shoulders, head, arms) about the hip centre."""
    k = k.copy()
    hip = (k[:, 11] + k[:, 12]) / 2
    up = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    k[:, up] = hip[:, None] + _rot(k[:, up] - hip[:, None], np.radians(deg))
    return k


def stance(k, scale):
    """Widens/narrows the stance: feet (and half of the knees) move away from the hip centre line."""
    k = k.copy()
    hip_x = ((k[:, 11, 0] + k[:, 12, 0]) / 2)[:, None]
    for j, f in ((15, 1.0), (16, 1.0), (13, 0.5), (14, 0.5)):
        k[:, j, 0] = hip_x[:, 0] + (k[:, j, 0] - hip_x[:, 0]) * (1 + (scale - 1) * f)
    for side in ("L", "R"):  # keep shin/thigh lengths
        hp, kn, an = LEG[side]
        a = np.full(len(k), np.median(np.linalg.norm(k[:, kn] - k[:, hp], axis=-1)))
        b = np.full(len(k), np.median(np.linalg.norm(k[:, an] - k[:, kn], axis=-1)))
        k[:, kn], _ = _ik2d(k[:, hp], k[:, kn], k[:, an], a, b)
    return k


def proportions(k, rng, amount=0.07):
    """Rescales limbs and shoulder/hip width (each child joint moves with its parent bone)."""
    k = k.copy()
    chains = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)]
    for root, mid, end in chains:
        s1, s2 = rng.normal(1, amount, 2)
        e = k[:, end] - k[:, mid]
        k[:, mid] = k[:, root] + (k[:, mid] - k[:, root]) * s1
        k[:, end] = k[:, mid] + e * s2
    for (l, r), s in (((5, 6), rng.normal(1, amount)), ((11, 12), rng.normal(1, amount))):
        c = (k[:, l] + k[:, r]) / 2
        d_l, d_r = (k[:, l] - c) * (s - 1), (k[:, r] - c) * (s - 1)
        arm_l, arm_r = ((7, 9), (8, 10)) if l == 5 else ((13, 15), (14, 16))
        k[:, l] += d_l
        k[:, r] += d_r
        k[:, list(arm_l)] += d_l[:, None]
        k[:, list(arm_r)] += d_r[:, None]
    return k


def view(k, squeeze=1.0, vsquash=1.0, shear=0.0, roll_deg=0.0):
    """Approximate view change: horizontal squeeze (turning the camera around the person),
    vertical squash (higher camera), shear and camera roll, about the body centre."""
    c = k.reshape(-1, 2).mean(0)
    p = k - c
    p = np.stack([p[..., 0] * squeeze + shear * p[..., 1], p[..., 1] * vsquash], -1)
    return _rot(p, np.radians(roll_deg)) + c


def mirror(k):
    c = k[..., 0].mean()
    m = k[:, FLIP].copy()
    m[..., 0] = 2 * c - m[..., 0]
    return m


def time_warp(k, rng, speed=None):
    """Plays the clip at a different, slightly uneven speed."""
    T = len(k)
    speed = speed or float(np.clip(rng.lognormal(0, 0.2), 0.65, 1.5))
    rate = speed * np.exp(gaussian_filter1d(rng.normal(0, 0.25, T * 2), 6))
    t = np.concatenate([[0], np.cumsum(rate)])
    t = t[t <= T - 1]
    if len(t) < 50:  # too short: slow down the whole clip a little
        t = np.linspace(0, T - 1, max(50, len(t)))
    out = np.empty((len(t), 17, 2))
    for j in range(17):
        for c in range(2):
            out[:, j, c] = np.interp(t, np.arange(T), k[:, j, c])
    return out, speed


def jitter(k, rng, level=1.0):
    tor = gaussian_filter1d(np.maximum(_torso(k), 5), 2)[:, None, None]
    return k + rng.normal(0, 1, k.shape) * JITTER[None, :, None] * tor * level


def perturb(k, rng, strength=1.0):
    """One random perturbed copy of a real clip. Returns (kpts, description)."""
    k = np.asarray(k, np.float64)[..., :2]
    desc = []
    side = knife_arm(k)
    if rng.random() < 0.7:
        low = rng.uniform(-15, 45) * strength  # mostly lower: the training thrusts are all at shoulder height
        bend = rng.uniform(0.7, 1.0) if rng.random() < 0.5 else None
        k = arm_edit(k, side, lower_deg=low, bend=bend)
        desc.append(f"arm{low:+.0f}" + (f"_bend{bend:.2f}" if bend else ""))
    if rng.random() < 0.5:
        d = rng.normal(0, 6) * strength
        k = lean(k, d)
        desc.append(f"lean{d:+.0f}")
    if rng.random() < 0.4:
        s = rng.uniform(0.75, 1.35)
        k = stance(k, s)
        desc.append(f"stance{s:.2f}")
    if rng.random() < 0.5:
        k = proportions(k, rng, 0.06 * strength)
        desc.append("body")
    if rng.random() < 0.6:
        sq, vs, sh, ro = rng.uniform(0.7, 1.05), rng.uniform(0.85, 1.05), rng.normal(0, 0.08), rng.normal(0, 4)
        k = view(k, sq, vs, sh, ro)
        desc.append(f"view{sq:.2f}")
    if rng.random() < 0.6:
        k, sp = time_warp(k, rng)
        desc.append(f"speed{sp:.2f}")
    if rng.random() < 0.5:
        k = mirror(k)
        desc.append("mirror")
    k = jitter(k, rng, 0.5)
    return k.astype(np.float32), "+".join(desc) or "jitter"
