"""
One simulated take: a random person does one attack (or normal activity), filmed by a random
camera, seen through the YOLO-pose sensor model.

Attack clips are cut like the training set: a short still/approach part, the attack, a short
recovery. Each 50-frame window the dataset loader takes from them therefore contains the attack.
"""

import numpy as np

from src.skeleton_sim import actions as A
from src.skeleton_sim.body import BodyShape
from src.skeleton_sim.camera import Camera, H, in_frame, occlusion, sensor
from src.skeleton_sim.motion import FPS, Timeline

CLIP_MIN = 52  # frames; a 50-frame window plus a little slack


def _prelude(tl, a):
    kind = str(a.rng.choice(["still", "walk_in", "step", "ready"], p=[0.4, 0.3, 0.15, 0.15]))
    if kind == "still":
        tl.wait(a.u(0.3, 1.2))
    elif kind == "walk_in":
        A.walk(tl, a, int(a.rng.integers(1, 5)), turn_deg=a.u(-30, 30))
        if a.chance(0.5):
            tl.wait(a.u(0.1, 0.5))
    elif kind == "step":
        tl.wait(a.u(0.2, 0.6))
        A.step(tl, a.k if a.chance(0.5) else a.f, a.u(0.2, 0.45), dur=a.d(0.4))
    else:
        tl.move(a.d(0.5), **{A.K(a): A.ready_knife(a), A.F(a): A.free_pose(a, "guard")})
        tl.wait(a.u(0.2, 0.8))
    return kind


def _postlude(tl, a):
    kind = str(a.rng.choice(["still", "settle", "step_back"], p=[0.35, 0.45, 0.2]))
    if kind == "settle":
        A.settle_feet(tl, a)
    elif kind == "step_back":
        A.step(tl, a.k if a.chance(0.5) else a.f, -a.u(0.2, 0.5), dur=a.d(0.45), pelvis_frac=0.5, crouch=0.0)
        tl.t += a.d(0.45)
    tl.wait(a.u(0.3, 1.0))
    return kind


def build_motion(rng, action, hand=None, speed=None):
    """Returns (joints3d (T,17,3), timeline, actor, info) for one attack or normal activity."""
    shape = BodyShape.random(rng)
    if speed is None and action not in A.ATTACKS:
        # everyday movement is brisk: real normal activity in the lab takes moves the wrist at about half
        # the speed of a strike, not a fifth of it
        speed = float(np.clip(rng.lognormal(np.log(1.2), 0.25), 0.7, 2.0))
    a = A.Actor(rng, hand=hand, speed=speed)
    tl = Timeline(shape, rng)
    tl.set(0.0, **{A.K(a): A.rest_knife(a), A.F(a): A.free_pose(a, "hang")})
    info = {"action": action, "hand": a.k, "speed": round(a.sp, 3), "body_height": round(shape.height, 3),
            "free_hand": a.free_style}
    if action in A.ATTACKS:
        info["prelude"] = _prelude(tl, a)
        t_att = tl.t
        variant, t_strike = A.ATTACKS[action][0](tl, a)
        t_end = tl.t
        info["postlude"] = _postlude(tl, a)
        joints, ctrl = tl.sample()
        # cut: up to 1 s before the attack starts, up to 1 s after it ends
        f_att, f_end = int(t_att * FPS), int(t_end * FPS)
        f0 = max(0, f_att - int(rng.uniform(0.25, 1.0) * FPS))
        f1 = min(len(joints), f_end + int(rng.uniform(0.25, 1.0) * FPS))
        while f1 - f0 < CLIP_MIN and (f0 > 0 or f1 < len(joints)):
            f0, f1 = max(0, f0 - 1), min(len(joints), f1 + 1)
        info["variant"] = variant
        info["attack_frames"] = f"{f_att - f0}-{f_end - f0}"
        facing = float(ctrl["yaw"][min(int(t_strike * FPS), len(joints) - 1), 0])
    else:
        names = []
        while not names or (tl.t < CLIP_MIN / FPS + 0.3 or rng.random() < 0.4) and len(names) < 4:
            names.append(A.NORMALS[action if not names else str(rng.choice(list(A.NORMALS)))](tl, a))
        joints, ctrl = tl.sample()
        f0, f1 = 0, len(joints)
        info["variant"] = "+".join(names)
        facing = float(ctrl["yaw"][0, 0])
    joints = joints[f0:f1]
    info["frames"] = len(joints)
    return joints, facing, info


def film(joints, facing, rng, cam=None, tries=40, noise=1.0):
    """Projects 3D joints through a random camera that keeps the whole person in the frame."""
    center = joints[:, [11, 12], :2].mean((0, 1))
    for i in range(tries):
        c = cam if cam is not None else Camera.random(rng)
        if i > tries // 2:  # struggling to fit: back off and centre the shot
            c.distance *= 1.3
            c.aim_dx *= 0.3
            c.aim_dy *= 0.3
        c.setup(facing, center)
        uv, z = c.project(joints)
        if z.min() < 0.8 or not in_frame(uv, -10).all():
            continue
        height_px = np.percentile(uv[..., 1].max(1) - uv[..., 1].min(1), 50)
        if height_px < 0.18 * H:  # YOLO-pose gets unreliable on tiny people; the pipeline needs a usable skeleton
            continue
        occ = occlusion(joints, c)
        noisy = sensor(uv, occ, rng, level=noise)
        return noisy.astype(np.float32), uv.astype(np.float32), occ, c
    return None
