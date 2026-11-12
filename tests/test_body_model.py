import os

import numpy as np
import torch

from src.skeleton_sim import body_model as B

MODEL = os.path.join(os.path.dirname(__file__), "..", "results", "samples", "body_model", "Rajagopal2016.osim")


def _body():
    if not os.path.exists(MODEL):
        print("skip: no model file at", MODEL)
        return None
    return B.Body(MODEL)


def _pose(b, **deg):
    q = torch.zeros(1, len(b.coords))
    for k, v in deg.items():
        q[0, b.coords.index(k)] = np.radians(v)
    return b.points(q, torch.eye(3)[None], torch.ones(6), torch.tensor(B.FACE0, dtype=torch.float32))[0].numpy()


def test_joints_bend_the_anatomical_way():
    b = _body()
    if b is None:
        return
    assert (_pose(b, elbow_flex_r=90)[10] - _pose(b, elbow_flex_r=90)[8])[0] > 0.2  # the hand comes forward
    assert (_pose(b, knee_angle_r=90)[16] - _pose(b, knee_angle_r=90)[14])[0] < -0.3  # the foot goes back
    assert (_pose(b, hip_flexion_l=60)[13] - _pose(b, hip_flexion_l=60)[11])[0] > 0.3  # the knee comes forward
    assert (_pose(b, arm_add_r=-60)[8] - _pose(b, arm_add_r=-60)[6])[2] > 0.2  # the right arm goes out to the right
    assert b.lo[b.coords.index("elbow_flex_r")] == 0 and b.lo[b.coords.index("knee_angle_l")] == 0  # neither bends backwards


def test_standing_is_balanced_and_sitting_on_air_is_not():
    b = _body()
    if b is None:
        return
    def off(**deg):
        q = torch.zeros(1, len(b.coords))
        for k, v in deg.items():
            q[0, b.coords.index(k)] = np.radians(v)
        root = torch.eye(3)[None]
        return float(b.balance(b.forward(q, root, torch.ones(6)), torch.ones(6), root)[0])
    assert off() == 0
    assert off(hip_flexion_r=40, hip_flexion_l=40) > 0.3  # both straight legs forward, as early fits of back views had
    assert off(lumbar_extension=-60) < 0.05  # bending forward over the feet is fine


def test_camera_shows_the_right_side_on_the_left_and_nearer_points_lower():
    p = torch.zeros(1, 17, 3)
    p[0, 6, 2] = 0.2  # right shoulder, to the person's right
    p[0, 9, 0] = 0.5  # left wrist, forward towards the camera
    img = B.project(p, torch.tensor(0.3))[0]
    assert img[6, 0] < 0 and img[9, 1] > 0
    feet = torch.zeros(1, 17, 3)
    feet[0, 15:, 1] = -0.9
    feet[0, 15:, 2] = 0.1
    near, far = B.project(feet, torch.tensor(0.3), torch.tensor(3.0))[0], B.project(feet, torch.tensor(0.3))[0]
    assert abs(near[15, 0]) < abs(far[15, 0])  # below a high camera, the feet are farther away and look smaller


def test_fit_recovers_a_motion_seen_from_an_angle():
    b = _body()
    if b is None:
        return
    rng = np.random.default_rng(0)
    n = 40
    q = np.zeros((n, len(b.coords)))
    t = np.linspace(0, 1, n)
    q[:, b.coords.index("arm_flex_r")] = np.radians(70) * np.sin(np.pi * t)
    q[:, b.coords.index("elbow_flex_r")] = np.radians(100) * (1 - np.sin(np.pi * t))
    q[:, b.coords.index("knee_angle_l")] = np.radians(20)
    shared = B.new_shared("cpu")
    root = B.root_rotation(torch.full((n,), np.radians(30.0)), torch.zeros(n), torch.zeros(n))
    with torch.no_grad():
        p = b.points(torch.tensor(q, dtype=torch.float32), root, torch.ones(6), shared["face"])
        obs = (B.project(p, shared["tilt"], torch.exp(shared["log_dist"])) * 1.9).numpy()
    obs += rng.normal(0, 0.005, obs.shape)
    f, err, _ = B.fit_takes(b, [obs], [np.ones((n, 17))], [0.0], steps=(150, 600), device="cpu", shared=shared)
    with torch.no_grad():
        got = f.render(b).numpy()
        angles = f.angles().numpy()
    assert np.abs(got - obs).mean() < 0.02
    assert np.all(angles >= b.lo - 1e-6) and np.all(angles <= b.hi + 1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
