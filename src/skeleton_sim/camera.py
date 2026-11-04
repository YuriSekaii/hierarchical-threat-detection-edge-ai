"""
Virtual camera and YOLO-pose sensor model for the skeleton simulator.

The camera is a pinhole looking at the person from any azimuth, height and distance, and
renders into the same 1920x1440 frame the pipeline analyses (the 4:3 centre crop of the
2560x1440 camera). The sensor model adds what YOLO26s-pose does to clean joints, with sizes
measured on the 2026-10-31 lab takes (residual jitter about 1 % of torso length, about 1.5 %
on the wrists, larger and drifting on joints hidden behind the body).
"""

from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import gaussian_filter1d

W, H = 1920, 1440

# robust jitter per joint (torso units), measured on the lab takes
JITTER = np.array([0.010, 0.010, 0.011, 0.009, 0.011, 0.010, 0.010, 0.012, 0.014,
                   0.014, 0.016, 0.010, 0.010, 0.012, 0.012, 0.009, 0.009])
# Real skeletons have little fast jitter but much slow wobble in bone lengths, most of all on hips and legs
FAST_JITTER_SCALE = 0.4
DRIFT = np.array([0.012, 0.012, 0.012, 0.012, 0.012, 0.025, 0.025, 0.025, 0.025,
                  0.025, 0.025, 0.035, 0.035, 0.035, 0.035, 0.030, 0.030])


@dataclass
class Camera:
    azimuth: float      # deg, camera position around the person; 0 = in front of them, 90 = their left
    height: float       # m above the floor
    distance: float     # m, horizontal, to the person's start position
    hfov: float         # deg, horizontal field of view of the 1920-wide frame
    aim_dx: float       # pan offset of the aim point (frame fractions), so people are not always centred
    aim_dy: float
    roll: float         # deg

    @staticmethod
    def random(rng, azimuth=None, height=None, distance=None):
        # half of the shots are close and looking down, like the real cameras (torso about 350 px)
        close = rng.random() < 0.5
        return Camera(azimuth=rng.uniform(0, 360) if azimuth is None else azimuth,
                      height=(rng.uniform(1.5, 3.2) if close else rng.uniform(0.9, 3.6)) if height is None else height,
                      distance=(rng.uniform(1.2, 2.5) if close else rng.uniform(2.5, 9.0)) if distance is None else distance,
                      hfov=rng.uniform(55, 95), aim_dx=rng.uniform(-0.25, 0.25), aim_dy=rng.uniform(-0.2, 0.2),
                      roll=rng.normal(0, 1.5))

    def setup(self, facing_yaw, center_xy, center_z=0.95):
        """Places the camera relative to a person facing `facing_yaw` at `center_xy`."""
        ang = facing_yaw + np.radians(self.azimuth)
        target = np.r_[center_xy, center_z]
        pos = np.r_[center_xy + self.distance * np.array([np.cos(ang), np.sin(ang)]), self.height]
        fwd = target - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        Rw = np.stack([right, down, fwd])  # world -> camera (x right, y down, z forward)
        f = (W / 2) / np.tan(np.radians(self.hfov) / 2)
        r = np.radians(self.roll)
        roll = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        self._R = roll @ Rw
        self._t = pos
        self._f = f
        self._c = np.array([W / 2 - self.aim_dx * W, H / 2 - self.aim_dy * H])
        return self

    def to_cam(self, X):
        return (X - self._t) @ self._R.T

    def project(self, X):
        """(..., 3) world -> (..., 2) pixels and (...) depth."""
        P = self.to_cam(X)
        z = np.maximum(P[..., 2], 1e-3)
        return self._f * P[..., :2] / z[..., None] + self._c, P[..., 2]

    def meta(self):
        return asdict(self)


def occlusion(joints3d, cam: Camera):
    """(T, 17) bool: joint is hidden behind the torso as seen from the camera."""
    P = cam.to_cam(joints3d)
    uv, z = cam.project(joints3d)
    torso_uv = uv[:, [5, 6, 12, 11]]
    torso_z = z[:, [5, 6, 11, 12]].mean(1)
    occ = np.zeros(z.shape, bool)
    for j in range(17):
        if j in (5, 6, 11, 12):
            continue
        behind = z[:, j] > torso_z + 0.06
        inside = np.array([_in_quad(uv[t, j], torso_uv[t], grow=1.15) for t in range(len(z))])
        occ[:, j] = behind & inside
    # the far side of the face is hidden when the head is turned away
    for far_eye, far_ear, near in ((1, 3, 2), (2, 4, 1)):
        hidden = z[:, far_ear] > z[:, 0] + 0.03
        occ[:, far_eye] |= hidden & (z[:, far_eye] > z[:, near] + 0.02)
        occ[:, far_ear] |= hidden
    face_back = (z[:, 0] > (z[:, 3] + z[:, 4]) / 2 + 0.02)  # nose behind the ears: back of the head
    occ[:, 0:3] |= face_back[:, None]
    return occ


def _in_quad(p, quad, grow=1.0):
    c = quad.mean(0)
    q = c + (quad - c) * grow
    s = None
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) < 1e-9:
            continue
        if s is None:
            s = np.sign(cr)
        elif np.sign(cr) != s:
            return False
    return True


def sensor(uv, occ, rng, level=1.0):
    """Adds YOLO-pose-like errors to clean 2D joints (T, 17, 2)."""
    T = len(uv)
    torso = np.linalg.norm((uv[:, 5] + uv[:, 6]) / 2 - (uv[:, 11] + uv[:, 12]) / 2, axis=-1)
    torso = gaussian_filter1d(np.maximum(torso, 5.0), 2)[:, None, None]
    out = uv.copy()
    # frame-to-frame jitter, slightly correlated in time
    j = rng.normal(0, 1, (T, 17, 2))
    j = 0.6 * j + 0.4 * gaussian_filter1d(j, 1.0, axis=0) / 0.55
    out += j * JITTER[None, :, None] * torso * level * rng.uniform(0.8, 1.3) * FAST_JITTER_SCALE
    # slow wobble of every joint, independent of whether it is hidden
    w = gaussian_filter1d(rng.normal(0, 1, (T + 40, 17, 2)), 5, axis=0)[20:20 + T]
    out += w / max(w.std(), 1e-9) * DRIFT[None, :, None] * torso * level * rng.uniform(0.6, 1.4)
    # hidden joints: the model guesses, so they wander slowly and jump more
    if occ.any():
        drift = gaussian_filter1d(rng.normal(0, 1, (T + 20, 17, 2)), 4, axis=0)[10:10 + T]
        drift /= max(drift.std(), 1e-9)
        guess = (drift * 0.05 + rng.normal(0, 0.006, (T, 17, 2))) * torso * level
        out += np.where(occ[..., None], guess, 0)
    # rare glitches: one joint jumps for a frame or two
    for _ in range(rng.poisson(0.4 * level * T / 60)):
        t, k = rng.integers(T), rng.integers(17)
        out[t:t + rng.integers(1, 3), k] += rng.normal(0, 0.15, 2) * torso[min(t, T - 1), 0, 0]
    return out


def in_frame(uv, margin=0.0):
    x, y = uv[..., 0], uv[..., 1]
    return (x >= -margin) & (x < W + margin) & (y >= -margin) & (y < H + margin)
