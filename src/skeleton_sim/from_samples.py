"""
Generates skeleton windows from recorded samples: real motion supplies timing and joint paths, and the transforms
here supply variety (speed, timing, body proportions, view, legs). Coordinates are changed directly; nothing is rendered.

Attacks and normals go through the same transforms, so "looks generated" carries no label.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
FACE = [0, 1, 2, 3, 4]


def centre(seq, a, b):
    return (seq[:, a] + seq[:, b]) / 2


def hand_speed(seq, hand):
    """Smoothed speed of the knife hand relative to the shoulders, in torso lengths per second."""
    sh, hip = centre(seq, L_SH, R_SH), centre(seq, L_HIP, R_HIP)
    torso = max(np.median(np.linalg.norm(sh - hip, axis=1)), 1.0)
    rel = gaussian_filter1d((seq[:, hand] - sh) / torso, 1.5, axis=0)
    return gaussian_filter1d(np.r_[0.0, np.linalg.norm(np.diff(rel, axis=0), axis=1)] * 30, 12)


def strike_events(env):
    """[(start, peak, end)] of each burst of hand speed. Heights are relative to the take: a thrust at the camera
    moves little in 2D."""
    level = np.percentile(env, 98)
    peaks, _ = find_peaks(env, prominence=0.25 * level, distance=40)
    if not len(peaks):
        return []
    _, _, left, right = peak_widths(env, peaks, rel_height=0.6)
    return [(int(a), int(p), int(np.ceil(b))) for a, p, b in zip(left, peaks, right)]


def pause_frames(env):
    """Frames where the hand is nearly still."""
    return env < 0.2 * np.percentile(env, 98)


def time_warp(seq, start, length, rng, n=50, wobble=0.25):
    """n frames from seq[start : start + length], at a speed that varies smoothly by about +-wobble."""
    rate = np.exp(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 4), rng.uniform(-wobble, wobble, 4)))
    t = np.cumsum(rate) - rate[0]
    t = start + t / max(t[-1], 1e-6) * (length - 1)
    i = np.clip(t.astype(int), 0, len(seq) - 2)
    return seq[i] + (seq[i + 1] - seq[i]) * (t - i)[:, None, None]


def retarget(seq, s):
    """Same joint directions on a body with other proportions. s: torso, shoulder, hip, head, upper_arm, forearm, thigh, shin."""
    hip = centre(seq, L_HIP, R_HIP)
    sh0 = centre(seq, L_SH, R_SH)
    sh = hip + (sh0 - hip) * s["torso"]
    out = seq.copy()
    for j in FACE:
        out[:, j] = sh + (seq[:, j] - sh0) * s["head"]
    for shoulder, elbow, wrist in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        out[:, shoulder] = sh + (seq[:, shoulder] - sh0) * s["shoulder"]
        out[:, elbow] = out[:, shoulder] + (seq[:, elbow] - seq[:, shoulder]) * s["upper_arm"]
        out[:, wrist] = out[:, elbow] + (seq[:, wrist] - seq[:, elbow]) * s["forearm"]
    for hp, knee, ankle in ((L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)):
        out[:, hp] = hip + (seq[:, hp] - hip) * s["hip"]
        out[:, knee] = out[:, hp] + (seq[:, knee] - seq[:, hp]) * s["thigh"]
        out[:, ankle] = out[:, knee] + (seq[:, ankle] - seq[:, knee]) * s["shin"]
    return out


def random_body(rng, spread=0.15):
    return {k: rng.uniform(1 - spread, 1 + spread) for k in
            ("torso", "shoulder", "hip", "head", "upper_arm", "forearm", "thigh", "shin")}


def view_between(a, b, theta):
    """Approximates a view between two recorded azimuths 90 degrees apart. In an orthographic view x is a linear
    mix of the x of two other views, x = cos(theta) x_a + sin(theta) x_b, so no 3D lifting is needed.
    shortcut: a and b are different performances aligned in time, so this is only close; 3D lifting would be exact."""
    ha, hb = centre(a, L_HIP, R_HIP)[:, None], centre(b, L_HIP, R_HIP)[:, None]
    ra, rb = a - ha, b - hb
    c, s = np.cos(theta), np.sin(theta)
    w = c * c / max(c * c + s * s, 1e-6)
    return ha + np.stack([c * ra[..., 0] + s * rb[..., 0], w * ra[..., 1] + (1 - w) * rb[..., 1]], -1)


def blend(a, b, w):
    """A motion between two performances. The hip paths are blended too."""
    return w * a + (1 - w) * b


def graft_legs(seq, legs):
    """seq's upper body on the legs of another window, scaled by the torso lengths."""
    def torso(x):
        return max(np.median(np.linalg.norm(centre(x, L_SH, R_SH) - centre(x, L_HIP, R_HIP), axis=1)), 1.0)

    k = torso(seq) / torso(legs)
    hip, leg_hip = centre(seq, L_HIP, R_HIP)[:, None], centre(legs, L_HIP, R_HIP)[:, None]
    out = seq.copy()
    out[:, [L_KN, R_KN, L_AN, R_AN]] = hip + (legs[:, [L_KN, R_KN, L_AN, R_AN]] - leg_hip) * k
    return out


def warp_view(seq, rng):
    """Another camera height or distance, as a 2D linear map around the hip: squash and shear."""
    hip = centre(seq, L_HIP, R_HIP)[:, None]
    sx, sy, sh = rng.uniform(0.75, 1.15), rng.uniform(0.8, 1.15), rng.uniform(-0.15, 0.15)
    r = seq - hip
    return hip + np.stack([sx * (r[..., 0] + sh * r[..., 1]), sy * r[..., 1]], -1)


def vary(seq, rng, legs=None, p_body=0.8, p_legs=0.25, p_view=0.7):
    """The shared last step for attacks and normals alike."""
    if rng.random() < p_body:
        seq = retarget(seq, random_body(rng))
    if legs is not None and rng.random() < p_legs:
        seq = graft_legs(seq, legs)
    if rng.random() < p_view:
        seq = warp_view(seq, rng)
    return seq.astype(np.float32)
