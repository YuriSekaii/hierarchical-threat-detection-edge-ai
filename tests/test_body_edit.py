import os

import numpy as np
import torch

from src.skeleton_sim import body_edit as BE
from src.skeleton_sim import body_model as B

MODEL = os.path.join(os.path.dirname(__file__), "..", "results", "samples", "body_model", "Rajagopal2016.osim")


def test_strike_lands_on_new_point_inside_the_body():
    if not os.path.exists(MODEL):
        print("skip: no model file at", MODEL)
        return
    b = B.Body(MODEL)
    rng = np.random.default_rng(0)
    n = 60
    t = np.linspace(0, 1, n)
    reach = np.clip(np.sin(np.pi * (t - 0.25) / 0.5), 0, None) * (np.abs(t - 0.5) < 0.25)
    q = np.full((n, len(b.coords)), 0.05)
    q[:, b.coords.index("arm_flex_r")] = np.radians(80) * reach
    q[:, b.coords.index("elbow_flex_r")] = np.radians(90) * (1 - reach)
    q = np.clip(q, b.lo + 1e-3, b.hi - 1e-3)
    shared = B.new_shared("cpu")
    take = BE.BodyTake(b, shared, q, np.zeros((n, 2), np.float32), np.radians(np.full(n, 30.0)), np.log(1.9), "cpu")
    # recorded states: this strike and others like it with the arm higher or lower, shoulder and elbow moving together
    pool = []
    for k in range(40):
        s = take.states.copy()
        s[:, b.coords.index("arm_flex_r")] += rng.normal(0, 0.6)
        s[:, b.coords.index("arm_add_r")] += rng.normal(0, 0.4)
        pool.append(s + rng.normal(0, 0.02, s.shape))
    coupling = BE.StateCoupling(np.concatenate(pool), k=400)
    with torch.no_grad():
        img = take.render(take.states).numpy()
    start, land, end = 15, 30, 45
    target = img[land, 10] + np.array([0.0, -0.4])  # the hand lands higher
    new = BE.reperform(take, 10, start, land, end, target, coupling, steps=3, iters=60)
    with torch.no_grad():
        got = take.render(new).numpy()
        angles = B._bounded(torch.as_tensor(new[:, :len(b.coords)]), torch.as_tensor(b.lo, dtype=torch.float32), torch.as_tensor(b.hi, dtype=torch.float32)).numpy()
    assert np.linalg.norm(got[land, 10] - target) < 0.05, np.linalg.norm(got[land, 10] - target)
    assert np.all(angles >= b.lo - 1e-6) and np.all(angles <= b.hi + 1e-6)
    assert np.allclose(new[:start], take.states[:start]) and np.allclose(new[end + 1:], take.states[end + 1:])
    assert np.abs(got[land, 9] - img[land, 9]).max() < 0.1  # the free hand stays about where it was

    # several landings at once give what each gives alone; a mask keeps the other dims as they were
    free = BE.free_dims(take, "r")
    targets = np.stack([target, img[25, 10] + np.array([0.2, 0.1])])
    both = BE.solve_landing(take, take.states[[land, 25]], [10], targets[:, None], coupling, steps=2, iters=30, free=free)
    alone = BE.solve_landing(take, take.states[25], [10], targets[1:], coupling, steps=2, iters=30, free=free)
    assert np.allclose(both[1], alone, atol=5e-3)  # float noise: Adam scales tiny gradients up, np.abs(both[1] - alone).max()
    assert np.all(both[:, free == 0] == 0)


if __name__ == "__main__":
    test_strike_lands_on_new_point_inside_the_body()
    print("ok test_strike_lands_on_new_point_inside_the_body")
