"""
3D body model for the skeleton simulator.

A person is a COCO-17 skeleton in metres, posed from a small set of high-level controls
(pelvis position and yaw, trunk lean/twist, head turn, wrist targets, foot positions).
Arms and legs are solved with two-bone IK, so bone lengths never change.

World frame: X/Y on the floor, Z up. A body facing yaw `psi` looks along (cos psi, sin psi, 0).
Body-local frame: x = the person's right, y = forward, z = up.
"""

from dataclasses import dataclass
import numpy as np

# COCO-17 joint order
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
         (5, 6), (11, 12), (5, 11), (6, 12), (0, 1), (0, 2), (1, 3), (2, 4)]

# Every control the pose solver reads, with its rest value. Wrist targets are in units of
# arm length, in the trunk frame, relative to that arm's shoulder.
CONTROLS = {
    "pelvis": np.zeros(2),          # world xy of the pelvis centre
    "yaw": 0.0,                     # pelvis facing (rad)
    "crouch": 0.0,                  # pelvis drop (m)
    "pitch": 0.0,                   # trunk lean forward (rad)
    "roll": 0.0,                    # trunk lean to the right (rad)
    "twist": 0.0,                   # shoulders turned left of the pelvis (rad)
    "head_pitch": 0.0,              # head down (rad)
    "head_yaw": 0.0,                # head turned left (rad)
    "wr_R": np.array([0.08, 0.10, -0.95]),
    "wr_L": np.array([-0.08, 0.05, -0.97]),
    "swv_R": 0.0,                   # elbow swivel about the shoulder-wrist axis (rad)
    "swv_L": 0.0,
    "foot_R": np.array([0.10, 0.0]),  # world xy of each ankle (set from the start pose)
    "foot_L": np.array([-0.10, 0.0]),
    "lift_R": 0.0,                  # ankle lift above the floor (m)
    "lift_L": 0.0,
}


@dataclass
class BodyShape:
    height: float = 1.72
    torso: float = 0.29      # mid-hip to mid-shoulder, fraction of height
    sh_half: float = 0.105   # half shoulder width
    hip_half: float = 0.06
    uarm: float = 0.172
    farm: float = 0.150
    thigh: float = 0.245
    shank: float = 0.245
    ankle: float = 0.04      # ankle height above the floor
    neck: float = 0.11       # mid-shoulder to nose, up
    nose_fwd: float = 0.055

    @staticmethod
    def random(rng):
        """A random adult; proportions vary about +-6 % around the anthropometric means
        (they match the bone ratios measured from YOLO-pose on the lab takes)."""
        j = lambda: rng.normal(1.0, 0.035)
        return BodyShape(height=rng.uniform(1.52, 1.92), torso=0.29 * j(), sh_half=0.105 * j(),
                         hip_half=0.06 * j(), uarm=0.172 * j(), farm=0.150 * j(), thigh=0.245 * j(),
                         shank=0.245 * j(), neck=0.11 * j(), nose_fwd=0.055 * j())

    def m(self, frac):
        return frac * self.height

    @property
    def arm(self):
        return self.m(self.uarm + self.farm)

    @property
    def leg(self):
        return self.m(self.thigh + self.shank)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def basis(yaw):
    """Columns: right, forward, up of a body facing `yaw`."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[s, c, 0], [-c, s, 0], [0, 0, 1.0]])


def _two_bone(root, target, a, b, pole):
    """Places the middle joint of a two-bone chain. The chain bends toward `pole`."""
    v = target - root
    d = np.linalg.norm(v)
    d_c = np.clip(d, abs(a - b) + 1e-3, (a + b) * 0.999)
    u = v / max(d, 1e-9)
    end = root + u * d_c
    p = pole - np.dot(pole, u) * u
    n = np.linalg.norm(p)
    if n < 1e-6:  # pole parallel to the chain: pick any perpendicular
        p = np.cross(u, [0, 0, 1.0])
        n = np.linalg.norm(p)
        if n < 1e-6:
            p, n = np.cross(u, [1.0, 0, 0]), 1.0
    p /= n
    cos_a = np.clip((a * a + d_c * d_c - b * b) / (2 * a * d_c), -1, 1)
    mid = root + a * (cos_a * u + np.sqrt(1 - cos_a * cos_a) * p)
    return mid, end


def _rot_about(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * np.dot(axis, v) * (1 - np.cos(ang))


def solve_pose(shape: BodyShape, c: dict) -> np.ndarray:
    """Controls -> (17, 3) joint positions in metres."""
    m = shape.m
    J = np.zeros((17, 3))
    B = basis(c["yaw"])
    right, fwd, up = B[:, 0], B[:, 1], B[:, 2]

    # pelvis height: requested crouch, lowered further if a foot would be out of reach
    rest_z = m(shape.ankle) + 0.985 * shape.leg
    hip_z = rest_z - c["crouch"]
    pel_xy = np.asarray(c["pelvis"], float)
    for side, key in ((+1, "foot_R"), (-1, "foot_L")):
        hip_xy = pel_xy + side * m(shape.hip_half) * right[:2]
        foot = np.r_[c[key], m(shape.ankle) + c["lift_" + key[-1]]]
        dh = np.linalg.norm(foot[:2] - hip_xy)
        L = 0.995 * shape.leg
        if dh < L:
            hip_z = min(hip_z, foot[2] + np.sqrt(L * L - dh * dh))
    pel = np.r_[pel_xy, hip_z]
    J[R_HIP] = pel + m(shape.hip_half) * right
    J[L_HIP] = pel - m(shape.hip_half) * right

    # trunk
    Rt = B @ _rz(c["twist"]) @ _rx(-c["pitch"]) @ _ry(c["roll"])
    t_right, t_fwd, t_up = Rt[:, 0], Rt[:, 1], Rt[:, 2]
    neck = pel + m(shape.torso) * t_up
    arm = shape.arm
    for side, sh_i, el_i, wr_i, key in ((+1, R_SH, R_EL, R_WR, "R"), (-1, L_SH, L_EL, L_WR, "L")):
        tgt = np.asarray(c["wr_" + key], float) * arm
        # the shoulder girdle follows the hand a little: shrug when raised, protract when reaching
        elev = np.clip(tgt[2] / arm + 0.2, 0, 1.2) * m(0.03)
        prot = np.clip(tgt[1] / arm, -0.5, 1) * m(0.03)
        sh = neck + side * m(shape.sh_half) * t_right + elev * t_up + prot * t_fwd
        J[sh_i] = sh
        wr = sh + Rt @ tgt
        h = np.clip(tgt[2] / arm + 0.3, 0, 1)  # elbow points down for low hands, out for high hands
        pole_local = (1 - h) * np.array([side * 0.35, -0.25, -1.0]) + h * np.array([side * 1.0, 0.25, -0.15])
        pole = Rt @ pole_local
        pole = _rot_about(pole, wr - sh, c["swv_" + key] * side)
        J[el_i], J[wr_i] = _two_bone(sh, wr, m(shape.uarm), m(shape.farm), pole)

    # head
    Rh = Rt @ _rz(c["head_yaw"]) @ _rx(-c["head_pitch"])
    mid_sh = (J[L_SH] + J[R_SH]) / 2
    head = lambda x, y, z: mid_sh + Rh @ np.array([m(x), m(y), m(z)])
    J[NOSE] = head(0, shape.nose_fwd, shape.neck)
    J[R_EYE] = head(0.018, shape.nose_fwd - 0.012, shape.neck + 0.018)
    J[L_EYE] = head(-0.018, shape.nose_fwd - 0.012, shape.neck + 0.018)
    J[R_EAR] = head(0.042, -0.01, shape.neck + 0.008)
    J[L_EAR] = head(-0.042, -0.01, shape.neck + 0.008)

    # legs: knees point forward and slightly out
    for side, hip_i, kn_i, an_i, key in ((+1, R_HIP, R_KN, R_AN, "R"), (-1, L_HIP, L_KN, L_AN, "L")):
        foot = np.r_[c["foot_" + key], m(shape.ankle) + c["lift_" + key]]
        pole = fwd + 0.15 * side * right
        J[kn_i], J[an_i] = _two_bone(J[hip_i], foot, m(shape.thigh), m(shape.shank), pole)
    return J
