"""
Re-performs a recorded action inside the 3D body (body_model.py), so every generated pose is one the body can take.

A take fitted into the body is a sequence of body states: joint angles, pelvis lean and turn. Angles are kept in an
unbounded form (a logit between the model's joint limits), so any change stays inside the limits. To land the hand on a
new point, the state at the landing is changed as little as the person's own recorded states allow: among the recorded
states closest to it, which ones vary together and by how much (their local spread). The change is found in a few small
steps, re-learning the spread at each one, and applied over the strike as far as the hand has come along its trip, as in
motion_edit.py. The result can be shown from any direction by turning the body.
"""

import numpy as np
import torch

from . import body_model as B
from .motion_edit import strike_weight

EPS = 1e-4


def unbounded(angles, lo, hi):
    """Joint angles -> their unbounded form (inverse of body_model._bounded), close to the limits without losing them."""
    f = np.clip((angles - lo) / (hi - lo), EPS, 1 - EPS)
    return np.log(f / (1 - f))


class BodyTake:
    """One take in the body. states (T, D): unbounded angles, unbounded lean (2), turn from the take's facing (1)."""

    def __init__(self, body, shared, angles, lean, yaw, logk, device):
        self.body, self.shared, self.device = body, shared, device
        self.facing = float(np.median(yaw))
        rel = (yaw - self.facing + np.pi) % (2 * np.pi) - np.pi
        self.states = np.concatenate([unbounded(angles, body.lo, body.hi), lean, rel[:, None]], 1).astype(np.float32)
        self.logk = float(logk)
        self.n = len(body.coords)
        self.lo, self.hi = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (body.lo, body.hi))

    @classmethod
    def from_fit(cls, body, shared, z, i, device):
        """Take i of a saved fit (results/samples/body_model/fit_train_*.npz)."""
        a, b = int(z["starts"][i]), int(z["ends"][i])
        return cls(body, shared, z["angles"][a:b], z["lean"][a:b], z["yaw"][a:b], z["logk"][i], device)

    def render(self, states, yaw_turn=0.0, balance=False):
        """(N, D) states -> image (N, 17, 2) in torso lengths, as the take's camera sees them, turned by yaw_turn radians."""
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        q = B._bounded(s[:, :self.n], self.lo, self.hi)
        yaw = s[:, self.n + 2] + self.facing + yaw_turn
        logk = torch.full((len(s),), self.logk, device=self.device)
        return B.render_states(self.body, self.shared, q, s[:, self.n:self.n + 2], yaw, logk, balance=balance)


class StateCoupling:
    """The local spread of recorded body states: near a given state, how the person's states vary."""

    def __init__(self, states, k=300, ridge=0.05, device="cpu"):
        s = np.asarray(states, np.float32)
        self.scale = torch.as_tensor(s.std(0) + 1e-3, device=device)
        self.states = torch.as_tensor(s, device=device) / self.scale
        self.k, self.ridge, self.device = min(k, len(s)), ridge, device

    def precision(self, state):
        """Inverse local covariance at states (B, D) -> (B, D, D), or at one state (D,) -> (D, D), in units of the
        recorded spread."""
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device) / self.scale
        near = self.states[torch.cdist(torch.atleast_2d(x), self.states).topk(self.k, largest=False).indices]  # (B, k, D)
        y = near - near.mean(1, keepdim=True)
        cov = y.transpose(1, 2) @ y / self.k + self.ridge * torch.eye(y.shape[2], device=self.device)
        p = torch.linalg.inv(cov)
        return p if x.dim() == 2 else p[0]


def free_dims(take, side, trunk=True):
    """Mask (D,) of the state dims a strike with the given arm ("r" or "l") may change: that arm, and the trunk if asked.
    Legs, the other arm, pelvis lean and facing stay as recorded."""
    names = [f"arm_flex_{side}", f"arm_add_{side}", f"arm_rot_{side}", f"elbow_flex_{side}"]
    names += ["lumbar_extension", "lumbar_bending", "lumbar_rotation"] if trunk else []
    m = np.zeros(take.states.shape[1], np.float32)
    m[[take.body.coords.index(n) for n in names]] = 1
    return m


def solve_landing(take, state, joints, targets, coupling, steps=5, iters=80, reach=3000.0, balance=10.0, free=None):
    """The smallest likely change of a state (D,) that puts the given joints at targets (len(joints), 2) in the take's
    image. Walks there in steps, each with the spread re-learned at the current state. free: mask (D,) of the dims that
    may change (default all). Several landings of one take solve at once: states (B, D), targets (B, len(joints), 2);
    they do not interact (Adam is per element). Returns the change, shaped like state."""
    dev = take.device
    one = np.ndim(state) == 1
    scale = coupling.scale
    mask = torch.ones_like(scale) if free is None else torch.as_tensor(free, dtype=torch.float32, device=dev)
    scale = scale * mask  # fixed dims: no change, and none counted by the prior (it is conditional on them)
    base = torch.as_tensor(np.atleast_2d(state), dtype=torch.float32, device=dev)
    with torch.no_grad():
        start = take.render(base)[:, joints]
    goal = torch.as_tensor(np.asarray(targets, np.float32), device=dev).reshape(start.shape)
    done = torch.zeros_like(base)
    for k in range(1, steps + 1):
        with torch.no_grad():
            p = coupling.precision(base + done * scale)
        aim = start + (goal - start) * k / steps
        d = done.clone().requires_grad_(True)
        opt = torch.optim.Adam([d], lr=0.05)
        for _ in range(iters):
            opt.zero_grad()
            img, off = take.render(base + d * scale, balance=True)
            step = (d - done) * mask
            prior = 0.5 * torch.einsum("bi,bij,bj->", step, p, step)
            loss = reach * ((img[:, joints] - aim) ** 2).sum() + prior + balance * (off ** 2).sum()
            loss.backward()
            opt.step()
        done = d.detach()
    out = (done * scale).cpu().numpy()
    return out[0] if one else out


def reperform(take, hand, start, land, end, target, coupling, rows=None, shift=None, **kw):
    """The take's states (or rows of it) with one strike at start..land..end (frames within rows) re-performed so the
    hand lands on target (x, y), in the take's image. shift: {joint: (dx, dy)} other joints that also move by landing.
    Returns the new states (N, D); render them from any direction with take.render(states, yaw_turn)."""
    states = take.states if rows is None else take.states[rows]
    with torch.no_grad():
        img = take.render(states).cpu().numpy()
    path = img[:, hand]
    joints = [hand] + list(shift or {})
    targets = [np.asarray(target)] + [img[land, j] + np.asarray(v) for j, v in (shift or {}).items()]
    delta = solve_landing(take, states[land], joints, targets, coupling, **kw)
    w = strike_weight(path, start, land, end)
    return states + w[:, None] * delta[None]
