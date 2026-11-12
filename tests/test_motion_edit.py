import numpy as np

from src.skeleton_sim import motion_edit as E


def _poses(n=2000, seed=0):
    """Hand x moves freely, and the elbow follows at half the distance."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.01, (n, 17, 2))
    p[:, E.R_WR] += rng.uniform(-1, 1, (n, 2))
    p[:, E.R_EL] += p[:, E.R_WR] * 0.5
    return p


def test_hand_lands_on_target_and_elbow_follows_the_learned_coupling():
    coupling = E.PoseCoupling(_poses(), k=500)
    body = np.zeros((1, 17, 2))
    out = coupling.move(body, E.R_WR, np.array([[0.4, -0.2]]))
    assert np.allclose(out[0, E.R_WR], [0.4, -0.2], atol=1e-5)
    assert np.allclose(out[0, E.R_EL], [0.2, -0.1], atol=0.03)


def test_no_displacement_changes_nothing():
    body = _poses(5)
    assert np.array_equal(E.PoseCoupling(_poses(), k=100).move(body, E.R_WR, np.zeros((5, 2))), body)


def test_weight_follows_the_hand_trip_and_ignores_a_wind_up_and_holds_at_the_far_end():
    x = np.zeros(30)
    x[5:8] = [-0.2, -0.3, -0.1]  # wind-up away from the target
    x[8:13] = [0.2, 0.4, 0.6, 0.8, 1.0]
    x[13:17] = 1.0  # hold
    x[17:21] = [0.75, 0.5, 0.25, 0.0]
    w = E.strike_weight(np.stack([x, 0 * x], 1), 5, 12, 20)
    assert w[:5].max() == 0 and w[21:].max() == 0
    assert w[5:7].max() == 0 and w[12] == 1 and np.all(w[12:17] == 1) and np.isclose(w[18], 0.5)


def test_limit_arm_keeps_hand_and_stops_bones_stretching():
    body = np.zeros((1, 17, 2))
    body[0, E.R_EL] = [1.0, 0]
    body[0, E.R_WR] = [1.4, 0]
    out = E.limit_arm(body, E.R_WR, 0.6, 0.6)
    assert np.array_equal(out[0, E.R_WR], body[0, E.R_WR])
    assert np.linalg.norm(out[0, E.R_WR] - out[0, E.R_EL]) <= 0.6 + 1e-6


def test_reperform_lands_on_target_and_leaves_frames_outside_the_strike():
    body = _poses(40, seed=1)
    limits = (9.0, 9.0)
    out = E.reperform(body, E.R_WR, 10, 18, 28, (0.3, -0.9), E.PoseCoupling(_poses(), k=500), limits)
    assert np.allclose(out[18, E.R_WR], [0.3, -0.9], atol=1e-5)
    assert np.array_equal(out[:5, E.R_WR], body[:5, E.R_WR])


def test_reperform_moves_extra_joints_and_targets_are_kept_within_reach():
    body = _poses(40, seed=2)
    out = E.reperform(body, E.R_WR, 10, 18, 28, (0.2, 0.5), E.PoseCoupling(_poses(), k=500), (9.0, 9.0),
                      shift={E.L_SH: (0.1, 0.4), E.R_SH: (0.1, 0.4)})
    assert np.allclose(out[18, E.L_SH], body[18, E.L_SH] + [0.1, 0.4], atol=0.05)  # the smoothing moves it slightly
    pose = np.zeros((17, 2))
    assert np.isclose(np.linalg.norm(E.reachable(pose, E.R_WR, (3.0, 4.0), 0.5, 0.5)), 1.0)


def test_follow_keys_passes_through_every_point_on_every_joint():
    body = _poses(40, seed=3)
    hand_keys = [(0, body[0, E.R_WR]), (10, (0.5, -0.8)), (20, (0.2, 0.4)), (39, body[39, E.R_WR])]
    out = E.follow_keys(body, {E.R_WR: hand_keys, E.L_SH: [(20, (0.1, 0.5))]}, E.PoseCoupling(_poses(), k=500))
    for f, p in hand_keys:
        assert np.allclose(out[f, E.R_WR], p, atol=1e-4)
    assert np.allclose(out[20, E.L_SH], (0.1, 0.5), atol=1e-4)


def test_align_strike_lines_up_start_landing_and_end():
    other = np.zeros((100, 17, 2))
    other[:, E.R_WR, 0] = np.arange(100)
    out = E.align_strike(other, (40, 50, 70), (10, 30, 40), np.array([0, 10, 20, 30, 35, 40, 45]))
    assert np.allclose(out[:, E.R_WR, 0], [30, 40, 45, 50, 60, 70, 75])
    assert E.align_strike(other, (40, 50, 70), (10, 30, 40), np.array([500]))[0, E.R_WR, 0] == 99  # never leaves the take


def test_turned_view_is_a_rotation_seen_by_a_tilted_camera():
    # one point in 3D: lateral X, height Y, forward Z, seen by a camera where depth adds tilt * depth to image height
    X, Y, Z, tilt = 0.3, -0.5, 0.8, 0.2
    def view(phi):
        return np.array([X * np.cos(phi) + Z * np.sin(phi), Y + tilt * (-X * np.sin(phi) + Z * np.cos(phi))])[None, None]
    for side in (1, -1):
        for theta in (0.0, 0.7, np.pi / 2, np.pi, -1.1):
            assert np.allclose(E.turned_view(view(0.0), view(side * np.pi / 2), side, theta, tilt), view(side * theta))


def test_find_strikes_rest_is_the_low_position_even_if_the_arm_is_held_out_longer():
    n = 400
    body = np.zeros((n, 17, 2))
    body[:, E.L_SH] = body[:, E.R_SH] = [0, -1]
    out = np.zeros(n)
    for s in range(20, 380, 90):  # 60 frames out (mostly still), 30 frames at rest
        out[s:s + 60] = np.clip(np.minimum(np.arange(60), 60 - np.arange(60)) / 8, 0, 1)
    body[:, E.R_WR] = np.stack([out * 1.2, -0.1 - out * 0.7], 1)
    ev = E.find_strikes(body, np.zeros((n, 2)), E.R_WR)
    assert len(ev) == 4
    assert all(body[land, E.R_WR, 0] > 1.1 and a < land < b for a, land, b in ev)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
