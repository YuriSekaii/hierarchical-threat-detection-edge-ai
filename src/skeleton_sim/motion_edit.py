"""
Re-performs a recorded action so that the hand lands on a chosen point.

The hand path is bent smoothly towards the target (motion warping). The other joints follow a coupling learned from
recorded poses: among the recorded poses closest to the current one, how did each joint move when the hand moved?
The edit walks towards the target in small steps and re-learns this coupling at each step, so it stays near poses a
person really took. No body rules are written by hand except a limit on how far each arm bone can stretch.

Poses are (17, 2) arrays relative to the hip centre, in torso lengths.
"""

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from .from_samples import L_EL, L_HIP, L_SH, L_WR, R_EL, R_HIP, R_SH, R_WR, centre

ARM = {L_WR: (L_SH, L_EL), R_WR: (R_SH, R_EL)}


def to_body(seq):
    """(T, 17, 2) in pixels -> body coordinates, hip path, torso length."""
    root = centre(seq, L_HIP, R_HIP)
    torso = max(float(np.median(np.linalg.norm(centre(seq, L_SH, R_SH) - root, axis=1))), 1.0)
    return (seq - root[:, None]) / torso, root, torso


def guard_position(rel):
    """The hand's rest: of the positions held still for long, the lowest. An arm held out at landing can be still for
    longer than the rest pose, so the longest-held position alone is not enough. y grows downwards."""
    speed = np.r_[0.0, np.linalg.norm(np.diff(rel, axis=0), axis=1)]
    still = rel[speed <= np.percentile(speed, 40)][::5]
    near = (np.linalg.norm(still[:, None] - still[None], axis=2) < 0.15).sum(0)
    held = still[near >= 0.3 * near.max()]
    return held[int(np.argmax(held[:, 1]))]


def find_strikes(body, root, hand, walking=False):
    """[(start, land, end)]: each trip of the hand away from its rest and back. It lands where it first reaches the far
    end of the trip, so a wind-up on the way stays inside the strike. Needs no direction, so it works from any view.
    Trips made while off the person's usual spot, or too long to be one strike, are dropped unless the take is walking.
    root: hip path in torso lengths."""
    rel = gaussian_filter1d(body[:, hand] - centre(body, L_SH, R_SH), 1.5, axis=0)
    away = np.linalg.norm(rel - guard_position(rel), axis=1)
    out_trip = away > 0.25 * np.percentile(away, 98)
    edges = np.flatnonzero(np.diff(np.r_[0, out_trip.astype(np.int8), 0]))
    spot = np.median(root, axis=0)
    out = []
    for a, b in zip(edges[::2], edges[1::2] - 1):
        if b - a < 10 or (not walking and (b - a > 150 or np.linalg.norm(root[a:b + 1] - spot, axis=1).max() > 0.8)):
            continue
        land = a + int(np.argmax(away[a:b + 1] >= 0.95 * away[a:b + 1].max()))
        out.append((int(a), land, int(b)))
    return out


def align_strike(other, other_ev, ev, frames):
    """other (T, 17, 2) at the same phase as the given frames of a strike ev: start, landing and end line up, and frames
    outside the strike run at normal speed. Lets two takes of one action act as two views of one motion."""
    a, land, b = ev
    oa, oland, ob = other_ev
    t = np.clip(np.interp(frames, [a - 1000, a, land, b, b + 1000], [oa - 1000, oa, oland, ob, ob + 1000]), 0, len(other) - 1)
    i = np.minimum(t.astype(int), len(other) - 2)
    return other[i] + (other[i + 1] - other[i]) * (t - i)[:, None, None]


def turned_view(a, b, side, theta, tilt=0.0):
    """a and b: one motion from directions 90 degrees apart, b at side (+1 or -1) from a, in body coordinates. Returns it
    from theta radians further that way. Depth in a's view is side * b's x. tilt: how far depth shows up in image height,
    because the camera looks slightly down. theta 0 returns a's x, pi/2 returns b's x."""
    xa, xb, ya, yb = a[..., 0], b[..., 0], a[..., 1], b[..., 1]
    height = (ya - tilt * side * xb + yb + tilt * side * xa) / 2
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c * xa + s * xb, height + tilt * side * (c * xb - s * xa)], -1)


def strike_weight(path, start, land, end):
    """How much of the hand displacement applies at each frame: how far the hand has come along its own trip. A wind-up
    away from the target adds nothing, and a hold at the far end stays at 1. path: (T, 2) hand positions."""
    w = np.zeros(len(path))
    for frames, base in ((np.arange(start, land + 1), path[start]), (np.arange(land, end + 1), path[end])):
        v = path[land] - base
        w[frames] = np.clip((path[frames] - base) @ v / max(float(v @ v), 1e-6), 0, 1)
    return w


class PoseCoupling:
    """Learns from recorded poses how the body follows the hand, near any given pose."""

    def __init__(self, poses, k=300, ridge=1e-3, device="cpu"):
        self.poses = torch.as_tensor(np.asarray(poses, np.float32).reshape(len(poses), -1), device=device)
        self.k, self.ridge, self.device = min(k, len(poses)), ridge, device

    def step(self, p, joints, delta):
        """Moves the given joints of each pose in p (F, 34) by delta (F, 2 * len(joints)), and the other joints with them."""
        c = [i for j in joints for i in (2 * j, 2 * j + 1)]
        near = torch.cdist(p, self.poses).topk(self.k, largest=False).indices if self.k < len(self.poses) else None
        y = self.poses[near] if near is not None else self.poses[None].expand(len(p), -1, -1)
        y = y - y.mean(1, keepdim=True)
        x = y[..., c]
        cxx = x.transpose(1, 2) @ x / self.k + self.ridge * torch.eye(len(c), device=self.device)
        cyx = y.transpose(1, 2) @ x / self.k
        out = p + (cyx @ torch.linalg.solve(cxx, delta[..., None]))[..., 0]
        out[:, c] = p[:, c] + delta  # these joints go exactly where asked
        return out

    def move(self, body, joints, delta, steps=5):
        """body (T, 17, 2), joints: one joint or a list, delta (T, 2) or (T, len(joints), 2): displacement per frame."""
        joints = [int(j) for j in np.atleast_1d(joints)]
        delta = np.asarray(delta).reshape(len(body), -1)
        moving = np.linalg.norm(delta, axis=1) > 1e-4
        out = body.copy()
        if not moving.any():
            return out
        p = torch.as_tensor(body[moving].reshape(moving.sum(), -1), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(delta[moving] / steps, dtype=torch.float32, device=self.device)
        for _ in range(steps):
            p = self.step(p, joints, d)
        out[moving] = p.cpu().numpy().reshape(-1, 17, 2)
        return out


def limit_arm(body, hand, upper, fore):
    """Keeps the hand where it is and moves the elbow so that no arm bone is longer than the person's own."""
    shoulder, elbow = ARM[hand]
    out = body.copy()
    for _ in range(3):
        v = out[:, elbow] - out[:, shoulder]
        out[:, elbow] = out[:, shoulder] + v * np.minimum(1, upper / np.maximum(np.linalg.norm(v, axis=1), 1e-6))[:, None]
        v = out[:, elbow] - out[:, hand]
        out[:, elbow] = out[:, hand] + v * np.minimum(1, fore / np.maximum(np.linalg.norm(v, axis=1), 1e-6))[:, None]
    return out


def arm_limits(body, hand):
    """The longest each arm bone appears in a take; shorter than this is foreshortening."""
    shoulder, elbow = ARM[hand]
    upper = np.linalg.norm(body[:, elbow] - body[:, shoulder], axis=1)
    fore = np.linalg.norm(body[:, hand] - body[:, elbow], axis=1)
    return float(np.percentile(upper, 99)), float(np.percentile(fore, 99))


def follow_keys(body, keys, coupling, hand=None, limits=None):
    """Re-performs body (T, 17, 2) through any number of points on any joints. keys: {joint: [(frame, (x, y)), ...]}.
    Each joint's displacement is interpolated between its keys, so the recorded motion stays underneath, and the
    joints without keys follow the learned coupling."""
    joints = list(keys)
    t = np.arange(len(body))
    delta = np.zeros((len(body), len(joints), 2))
    for i, j in enumerate(joints):
        frames = np.array([f for f, _ in keys[j]])
        off = np.array([p for _, p in keys[j]], dtype=np.float64) - body[frames, j]
        order = np.argsort(frames)
        frames, off = frames[order], off[order]
        delta[:, i] = PchipInterpolator(frames, off, axis=0)(np.clip(t, frames[0], frames[-1])) if len(frames) > 1 else off
    out = coupling.move(body, joints, delta)
    others = [j for j in range(17) if j not in joints]
    out[:, others] = gaussian_filter1d(out, 1.0, axis=0)[:, others]  # coupling changes per frame; this removes the flicker
    return limit_arm(out, hand, *limits) if limits else out


def reachable(pose, hand, target, upper, fore):
    """The target, pulled towards the shoulder if the arm cannot reach it. pose: the body at landing."""
    shoulder = pose[ARM[hand][0]]
    v = np.asarray(target) - shoulder
    return shoulder + v * min(1.0, (upper + fore) / max(float(np.linalg.norm(v)), 1e-6))


def reperform(body, hand, start, land, end, target, coupling, limits, shift=None):
    """body (T, 17, 2) with one strike at frames start..land..end; the hand now lands on target (x, y).
    shift: {joint: (dx, dy)} other joints that also move by landing, such as the shoulders in a bend."""
    joints = [hand] + list(shift or {})
    moves = [np.asarray(target) - body[land, hand]] + [np.asarray(v) for v in (shift or {}).values()]
    w = strike_weight(body[:, hand], start, land, end)
    out = coupling.move(body, joints, w[:, None, None] * np.stack(moves)[None])
    others = [j for j in range(17) if j not in joints]
    out[:, others] = gaussian_filter1d(out, 1.0, axis=0)[:, others]  # coupling changes per frame; this removes the flicker
    return limit_arm(out, hand, *limits)
