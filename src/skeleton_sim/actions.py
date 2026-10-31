"""
Action library for the skeleton simulator.

Every action writes keyframes into a Timeline starting at its cursor. Attacks are built from
the phases a real knife attack has: get ready -> wind up -> strike (fast) -> follow through or
hold -> recover. Each phase draws its own random parameters (height, direction, reach, speed,
lean, stance, step), so every call gives a different take of the same action.

Wrist targets are directions in the trunk frame scaled by arm extension; see `aim()`.
"""

import numpy as np

from src.skeleton_sim.motion import Timeline

R = np.radians


class Actor:
    """Per-clip constants of the person: knife hand, speed and style."""

    def __init__(self, rng, hand=None, speed=None):
        self.rng = rng
        self.k = hand or rng.choice(["R", "L"], p=[0.75, 0.25])  # knife hand
        self.f = "L" if self.k == "R" else "R"                     # free hand
        self.s = 1.0 if self.k == "R" else -1.0                    # knife side, +1 = right
        # speed 1.0 = phase durations as written; real strikes (training set and lab takes) measure about
        # 0.7 of that, so the default is centred there
        self.sp = speed or float(np.clip(rng.lognormal(np.log(0.72), 0.2), 0.45, 1.2))
        self.free_style = rng.choice(["hang", "guard", "grab", "counter"], p=[0.35, 0.3, 0.15, 0.2])

    def u(self, a, b):
        return float(self.rng.uniform(a, b))

    def d(self, base, jitter=0.25):
        """A phase duration: base seconds at normal speed, jittered and scaled by the actor speed."""
        return base * float(self.rng.uniform(1 - jitter, 1 + jitter)) / self.sp

    def chance(self, p):
        return self.rng.random() < p


def aim(side, az_deg, el_deg, ext):
    """Wrist target (arm-length units, trunk frame, from the shoulder).
    az: + toward the arm's own side, - across the body; 0 = straight ahead.
    el: + up, - down, 90 = overhead. ext: 1 = fully straight arm."""
    az, el = R(az_deg), R(el_deg)
    return ext * np.array([side * np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


def rest_knife(a):
    return aim(a.s, a.u(5, 20), a.u(-85, -70), a.u(0.88, 0.96))


def ready_knife(a):
    return aim(a.s, a.u(-5, 25), a.u(-60, -35), a.u(0.6, 0.8))


def windup_el(a):
    """Wind-up elevation for overhand strikes. Real takes mostly raise the hand to about head height
    (wrist peak ~0.3-0.4 torso above the shoulder in the training and lab clips); some go high."""
    return a.u(30, 65) if a.chance(0.7) else a.u(65, 105)


def free_pose(a, style=None):
    fs = -a.s
    style = style or a.free_style
    if style == "guard":
        return aim(fs, a.u(-30, 0), a.u(-35, -10), a.u(0.55, 0.75))
    if style == "grab":
        return aim(fs, a.u(-20, 5), a.u(-30, -5), a.u(0.85, 0.98))
    if style == "counter":
        return aim(fs, a.u(10, 30), a.u(-80, -60), a.u(0.85, 0.97)) + np.array([0, -0.25, 0])
    return aim(fs, a.u(3, 12), a.u(-88, -78), a.u(0.93, 0.98))


def K(a):
    return "wr_" + a.k


def F(a):
    return "wr_" + a.f


def SW(a):
    return "swv_" + a.k


# ---------------------------------------------------------------- footwork
def step(tl: Timeline, foot, fwd, side=0.0, dur=0.3, start=None, pelvis_frac=0.5, crouch=None):
    """Moves one foot by (fwd, side) metres in the body frame; the pelvis follows part of the way."""
    t0 = tl.t if start is None else start
    B_xy = tl.local_to_world(np.array([side, fwd]), origin=np.zeros(2))
    new_foot = tl.state["foot_" + foot] + B_xy
    pel = tl.state["pelvis"] + B_xy * pelvis_frac
    kw = {"foot_" + foot: new_foot}
    tl.move(dur, start=t0, advance=False, **kw)
    extra = {} if crouch is None else {"crouch": crouch}
    tl.move(dur * 1.1, start=t0, advance=False, pelvis=pel, **extra)
    return t0 + dur * 1.1


def walk(tl: Timeline, a: Actor, n_steps, turn_deg=0.0, step_len=None, step_time=None, arms=True):
    """Walks forward n steps, turning `turn_deg` in total. Arms swing against the legs."""
    L = step_len or a.u(0.3, 0.6)
    T = (step_time or a.u(0.48, 0.65)) / np.sqrt(a.sp / 0.72)
    w = tl.shape.m(tl.shape.hip_half) * a.u(0.6, 1.0)
    # lead with the foot that is further back
    fwd = tl.heading()
    order = "R" if np.dot(tl.state["foot_R"] - tl.state["foot_L"], fwd) < 0 else "L"
    for i in range(n_steps):
        t0 = tl.t
        yaw = tl.state["yaw"] + R(turn_deg) / n_steps
        pel = tl.local_to_world(np.array([0.0, L]), yaw=yaw)
        side = 1.0 if order == "R" else -1.0
        foot = tl.local_to_world(np.array([side * w, L / 2]), yaw=yaw, origin=pel)
        last = i == n_steps - 1
        tl.move(0.8 * T, start=t0 + 0.1 * T, advance=False, **{"foot_" + order: foot})
        tl.move(T, start=t0, kind="stop" if last else "pass", advance=False, pelvis=pel, yaw=yaw)
        if arms:  # the arm opposite the stepping foot swings forward
            sw = a.u(0.15, 0.3)
            arm_fwd, arm_back = ("wr_L", "wr_R") if order == "R" else ("wr_R", "wr_L")
            for key, sgn in ((arm_fwd, 1), (arm_back, -1)):
                amp = sw * (0.5 if key == K(a) else 1.0)
                side_k = 1.0 if key == "wr_R" else -1.0
                tgt = np.array([side_k * 0.08, 0.05 + sgn * amp, -0.95])
                if key == K(a):
                    tgt = 0.5 * tgt + 0.5 * rest_knife(a)
                tl.move(T, start=t0, kind="pass" if not last else "stop", advance=False, **{key: tgt})
        tl.t = t0 + T
        order = "L" if order == "R" else "R"
    tl.tags.append((tl.t - n_steps * T, tl.t, f"walk{n_steps}"))


def settle_feet(tl: Timeline, a: Actor, dur=0.35):
    """Brings the trailing foot beside the other one (end of a walk or lunge)."""
    fwd = tl.heading()
    back = "R" if np.dot(tl.state["foot_R"] - tl.state["foot_L"], fwd) < 0 else "L"
    other = "L" if back == "R" else "R"
    w = tl.shape.m(tl.shape.hip_half) * a.u(1.0, 1.8)
    side = 1.0 if back == "R" else -1.0
    pel_from_other = tl.local_to_world(np.array([0, 0]), origin=tl.state["foot_" + other])
    tgt = tl.local_to_world(np.array([2 * side * w, a.u(-0.15, 0.1)]), origin=pel_from_other)
    tl.move(dur, advance=False, **{"foot_" + back: tgt})
    mid = (tgt + tl.state["foot_" + other]) / 2
    tl.move(dur, pelvis=mid, crouch=0.0)


# ---------------------------------------------------------------- attack phases
def _lunge(tl, a, dur, start, depth=None):
    """Steps toward the target with the front foot while the strike happens."""
    lead = a.k if a.chance(0.5) else a.f
    L = depth if depth is not None else a.u(0.2, 0.6)
    step(tl, lead, L, side=a.u(-0.05, 0.08) * (1 if lead == "R" else -1), dur=dur, start=start,
         pelvis_frac=a.u(0.4, 0.6), crouch=a.u(0.03, 0.08) + 0.15 * L)
    return lead


def thrust(tl, a, style="any", reps=None, lunge=None):
    """Straight stab forward. style: 'train' (straight arm at shoulder height, as in the dataset),
    'low' (hip/belly level), 'bent' (short jab with a bent arm), 'any'."""
    style = a.rng.choice(["train", "low", "bent", "mid"]) if style == "any" else style
    reps = reps or int(a.rng.choice([1, 1, 2, 2, 3, 4]))
    lunge = a.chance(0.45) if lunge is None else lunge
    el = {"train": a.u(-12, 12), "low": a.u(-60, -35), "bent": a.u(-50, -10), "mid": a.u(-35, -10)}[style]
    ext = {"train": a.u(0.9, 0.99), "low": a.u(0.75, 0.97), "bent": a.u(0.55, 0.78), "mid": a.u(0.8, 0.98)}[style]
    tl.move(a.d(0.35), **{F(a): free_pose(a)})
    t_start = tl.t
    for r in range(reps):
        el_r, ext_r = el + a.u(-8, 8), np.clip(ext + a.u(-0.06, 0.04), 0.5, 0.99)
        # wind up: pull the hand back to the hip, turn the knife shoulder back
        chamber = aim(a.s, a.u(15, 45), a.u(-75, -45), a.u(0.55, 0.8)) + np.array([0, a.u(-0.35, -0.1), 0])
        tl.move(a.d(0.3 if r == 0 else 0.2), **{K(a): chamber, "twist": -a.s * R(a.u(5, 30)),
                                               "pitch": R(a.u(-5, 5))})
        dur = a.d(0.17, 0.35)
        if lunge and (r == 0 or a.chance(0.3)):
            _lunge(tl, a, dur * 1.3, tl.t - 0.05)
        tl.move(dur, **{K(a): aim(a.s, a.u(-25, 8), el_r, ext_r), "twist": a.s * R(a.u(10, 40)),
                        "pitch": R(a.u(5, 22)), SW(a): R(a.u(-15, 15))}, label="thrust_" + style)
        tl.wait(a.d(0.12, 0.6))
        if r < reps - 1:
            continue
        tl.move(a.d(0.4), **{K(a): ready_knife(a), "twist": a.s * R(a.u(-5, 10)), "pitch": R(a.u(0, 8))})
    return f"thrust_{style}x{reps}" + ("_lunge" if lunge else ""), t_start


def overhand_stab(tl, a, reps=None, deep=None):
    """Ice-pick stab: knife raised above the head, driven down and forward."""
    reps = reps or int(a.rng.choice([1, 1, 2, 3, 4, 5]))
    deep = a.chance(0.25) if deep is None else deep
    tl.move(a.d(0.3), **{F(a): free_pose(a)})
    t_start = tl.t
    for r in range(reps):
        up = aim(a.s, a.u(0, 30), windup_el(a), a.u(0.55, 0.9))
        tl.move(a.d(0.35 if r == 0 else 0.22), **{K(a): up, SW(a): R(a.u(10, 40)), "pitch": R(a.u(-8, 3)),
                                                 "twist": -a.s * R(a.u(0, 20)), "crouch": a.u(0, 0.04)})
        if r == 0 and a.chance(0.35):
            tl.wait(a.d(0.25, 0.6))  # hold the raised knife
        el_end = a.u(-75, -50) if deep else a.u(-55, -15)
        if a.chance(0.4):
            _lunge(tl, a, a.d(0.25), tl.t - 0.03)
        tl.move(a.d(0.18, 0.3), **{K(a): aim(a.s, a.u(-25, 5), el_end, a.u(0.7, 0.95)), SW(a): R(a.u(-10, 10)),
                                   "pitch": R(a.u(12, 30) + (20 if deep else 0)), "twist": a.s * R(a.u(5, 25)),
                                   "crouch": a.u(0.03, 0.12) + (0.15 if deep else 0)}, label="overhand_stab")
        tl.wait(a.d(0.08, 0.6))
    tl.move(a.d(0.45), **{K(a): ready_knife(a), "pitch": R(a.u(0, 8)), "twist": 0.0, "crouch": a.u(0, 0.04),
                          SW(a): 0.0})
    return f"overhand_stab_x{reps}" + ("_deep" if deep else ""), t_start


def cut_down(tl, a, reps=None):
    """Chop: wind up above / behind the head and cut straight down through the front."""
    reps = reps or int(a.rng.choice([1, 1, 2, 3]))
    tl.move(a.d(0.3), **{F(a): free_pose(a)})
    t_start = tl.t
    for r in range(reps):
        tl.move(a.d(0.38 if r == 0 else 0.25), **{K(a): aim(a.s, a.u(0, 25), windup_el(a) + 10, a.u(0.55, 0.9)),
                                                 SW(a): R(a.u(10, 35)), "pitch": R(a.u(-10, 0)),
                                                 "twist": -a.s * R(a.u(0, 20))})
        if a.chance(0.35):
            _lunge(tl, a, a.d(0.3), tl.t - 0.02)
        d = a.d(0.3, 0.3)
        tl.move(d * 0.45, kind="pass", **{K(a): aim(a.s, a.u(-10, 10), a.u(20, 40), a.u(0.9, 0.99))})
        tl.move(d * 0.55, **{K(a): aim(a.s, a.u(-20, 10), a.u(-80, -55), a.u(0.8, 0.97)), SW(a): 0.0,
                             "pitch": R(a.u(15, 38)), "twist": a.s * R(a.u(0, 20)), "crouch": a.u(0.05, 0.2)},
                label="cut_down")
        tl.wait(a.d(0.1, 0.6))
    tl.move(a.d(0.45), **{K(a): ready_knife(a), "pitch": R(a.u(0, 8)), "twist": 0.0, "crouch": a.u(0, 0.04)})
    return f"cut_down_x{reps}", t_start


def _sweep(tl, a, fore, height, dur, big=False, follow=True):
    """One slash arc from the current wind-up. fore: forehand (own side -> across the body),
    else backhand. height: 'high'/'mid'/'low'/'diag_down'/'diag_up'."""
    el0, el1 = {"high": (a.u(0, 25),) * 2, "mid": (a.u(-30, 0),) * 2, "low": (a.u(-60, -35),) * 2,
                "diag_down": (a.u(35, 75), a.u(-70, -35)),
                "diag_up": (a.u(-70, -40), a.u(10, 45))}[height]
    out_az, in_az = a.u(70, 110) + (35 if big else 0), -a.u(55, 85)
    az0, az1 = (out_az, in_az) if fore else (in_az, out_az)
    e1 = a.u(0.55, 0.75) if fore else a.u(0.85, 0.98)
    tw = R(a.u(20, 40) + (15 if big else 0))
    sgn = a.s if fore else -a.s  # direction the shoulders turn during the cut
    # the knife passes in front with the arm straight, the body turns into the cut
    tl.move(dur * 0.5, kind="pass", **{K(a): aim(a.s, (az0 + az1) / 2 + a.u(-10, 10), (el0 + el1) / 2, a.u(0.9, 0.99)),
                                        "twist": sgn * tw * 0.25, "pitch": R(a.u(5, 18))})
    tl.move(dur * 0.5, **{K(a): aim(a.s, az1, el1, e1), "twist": sgn * tw,
                          "yaw": tl.state["yaw"] + sgn * R(a.u(0, 6) + (10 if big else 0))})
    if follow:
        tl.wait(a.d(0.06, 0.6))


def _sweep_start(tl, a, fore, height, big=False, dur=None):
    """Wind-up position of a slash."""
    el0 = {"high": a.u(0, 25), "mid": a.u(-30, 0), "low": a.u(-60, -35), "diag_down": a.u(35, 75),
           "diag_up": a.u(-70, -40)}[height]
    if fore:
        tgt = aim(a.s, a.u(70, 110) + (35 if big else 0), el0, a.u(0.8, 0.95))
        tw = -a.s * R(a.u(20, 40) + (15 if big else 0))
    else:
        tgt = aim(a.s, -a.u(55, 85), el0, a.u(0.55, 0.75))
        tw = a.s * R(a.u(20, 40))
    tl.move(dur or a.d(0.35), **{K(a): tgt, "twist": tw, "pitch": R(a.u(-5, 8)), SW(a): R(a.u(-10, 25))})


def slash(tl, a, n=1, height=None, fore=None, big=False):
    """Horizontal or diagonal slash; n > 1 alternates forehand and backhand (multi-slash)."""
    height = height or str(a.rng.choice(["high", "mid", "mid", "low", "diag_down", "diag_up"]))
    fore = a.chance(0.65) if fore is None else fore
    tl.move(a.d(0.3), **{F(a): free_pose(a)})
    t_start = tl.t
    _sweep_start(tl, a, fore, height, big)
    if big or a.chance(0.3):
        _lunge(tl, a, a.d(0.3), tl.t - 0.02, depth=a.u(0.2, 0.6))
    for i in range(n):
        h = height if i == 0 or a.chance(0.6) else str(a.rng.choice(["high", "mid", "low", "diag_down", "diag_up"]))
        _sweep(tl, a, fore, h, a.d(0.26 if not big else 0.36, 0.3), big=big, follow=(i == n - 1))
        tl.tags.append((tl.t - 0.3, tl.t, "slash_" + ("fore" if fore else "back") + "_" + h))
        fore = not fore
    tl.wait(a.d(0.15, 0.6))
    tl.move(a.d(0.45), **{K(a): ready_knife(a), "twist": 0.0, "pitch": R(a.u(0, 8)), SW(a): 0.0,
                          "yaw": tl.state["yaw"] + R(a.u(-10, 10))})
    kind = "swing" if big else ("multi_slash" if n > 1 else "slash")
    return f"{kind}_x{n}_{height}", t_start


def frenzy(tl, a, reps=None):
    """Grabs the victim with the free hand and stabs fast and short, many times."""
    reps = reps or int(a.rng.integers(3, 8))
    tl.move(a.d(0.3), **{F(a): free_pose(a, "grab"), "pitch": R(a.u(5, 15))})
    if a.chance(0.5):
        _lunge(tl, a, a.d(0.3), tl.t - 0.2, depth=a.u(0.15, 0.4))
    t_start = tl.t
    el = a.u(-55, -20)
    for r in range(reps):
        tl.move(a.d(0.15, 0.3), **{K(a): aim(a.s, a.u(20, 45), el - 15, a.u(0.45, 0.6)) + np.array([0, -0.15, 0]),
                                   "twist": -a.s * R(a.u(0, 15))})
        tl.move(a.d(0.12, 0.3), **{K(a): aim(a.s, a.u(-25, 0), el + a.u(-10, 10), a.u(0.65, 0.9)),
                                   "twist": a.s * R(a.u(5, 25))}, label="frenzy_stab")
    tl.move(a.d(0.4), **{K(a): ready_knife(a), F(a): free_pose(a, "guard"), "twist": 0.0})
    return f"frenzy_x{reps}", t_start


def upper_stab(tl, a, reps=None):
    """Underhand rising stab from the hip toward the belly / ribs."""
    reps = reps or int(a.rng.choice([1, 1, 2, 3]))
    tl.move(a.d(0.3), **{F(a): free_pose(a)})
    t_start = tl.t
    for r in range(reps):
        tl.move(a.d(0.28), **{K(a): aim(a.s, a.u(10, 35), a.u(-90, -70), a.u(0.75, 0.9)),
                              "twist": -a.s * R(a.u(5, 20)), "crouch": a.u(0.02, 0.1)})
        if r == 0 and a.chance(0.4):
            _lunge(tl, a, a.d(0.3), tl.t - 0.05)
        tl.move(a.d(0.2), **{K(a): aim(a.s, a.u(-20, 5), a.u(-30, 10), a.u(0.8, 0.97)),
                             "twist": a.s * R(a.u(10, 30)), "pitch": R(a.u(5, 18))}, label="upper_stab")
        tl.wait(a.d(0.1, 0.6))
    tl.move(a.d(0.4), **{K(a): ready_knife(a), "twist": 0.0, "pitch": 0.0, "crouch": 0.0})
    return f"upper_stab_x{reps}", t_start


def ground_strike(tl, a, reps=None):
    """Strikes at someone on the floor: deep bend / crouch, stab or chop downward."""
    reps = reps or int(a.rng.choice([1, 2, 3, 4]))
    t_start = tl.t
    tl.move(a.d(0.5), pitch=R(a.u(30, 55)), crouch=a.u(0.15, 0.45), **{F(a): free_pose(a, "grab")})
    for r in range(reps):
        tl.move(a.d(0.25), **{K(a): aim(a.s, a.u(0, 25), a.u(30, 80), a.u(0.75, 0.95))})
        tl.move(a.d(0.18, 0.3), **{K(a): aim(a.s, a.u(-20, 10), a.u(-90, -65), a.u(0.8, 0.97)),
                                   "pitch": R(a.u(40, 65))}, label="ground_strike")
        tl.wait(a.d(0.08))
    tl.move(a.d(0.6), pitch=R(a.u(0, 10)), crouch=0.0, **{K(a): ready_knife(a), F(a): free_pose(a)})
    return f"ground_strike_x{reps}", t_start


def turn_in_place(tl, a, ang, dur, start=None):
    """Turns the body by `ang` (rad, + = left), stepping both feet around the pelvis."""
    t0 = tl.t if start is None else start
    tl.move(dur, start=t0, advance=False, yaw=tl.state["yaw"] + ang)
    c, s = np.cos(ang), np.sin(ang)
    for i, f in enumerate(("R", "L") if a.chance(0.5) else ("L", "R")):
        rel = tl.state["foot_" + f] - tl.state["pelvis"]
        tl.move(dur * 0.55, start=t0 + i * dur * 0.45, advance=False,
                **{"foot_" + f: tl.state["pelvis"] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])})
    tl.t = max(tl.t, t0 + dur)


def turn_then(tl, a, inner):
    """Faces away (or sideways) from the target, turns toward it while raising the knife, attacks."""
    ang = R(a.u(70, 180)) * (1 if a.chance(0.5) else -1)
    t0 = tl.t
    turn_in_place(tl, a, ang, a.d(0.5))
    tl.t = t0 + (tl.t - t0) * a.u(0.4, 0.8)  # the attack starts before the turn is finished
    name, _ = inner(tl, a)
    return "turn_" + name, t0


def combo(tl, a):
    """Two to four different attacks chained without a pause."""
    pool = [lambda t, x: thrust(t, x, reps=1), lambda t, x: overhand_stab(t, x, reps=1),
            lambda t, x: cut_down(t, x, reps=1), lambda t, x: slash(t, x, n=1),
            lambda t, x: slash(t, x, n=2), lambda t, x: upper_stab(t, x, reps=1)]
    n = int(a.rng.integers(2, 5))
    names, t_start = [], None
    for i in a.rng.choice(len(pool), n, replace=False):
        nm, ts = pool[i](tl, a)
        names.append(nm.split("_x")[0])
        t_start = ts if t_start is None else t_start
        tl.truncate(tl.t - a.d(0.3, 0.3))  # cut most of the recovery, go straight into the next attack
    tl.t += a.d(0.2)
    return "combo_" + "+".join(names), t_start


def walking_thrust(tl, a):
    """Thrust while walking in (the live demo style): hip-level, often bent arm, no stop."""
    n = int(a.rng.integers(2, 5))
    t0 = tl.t
    walk(tl, a, n, step_len=a.u(0.4, 0.65), arms=False)
    t_mid = t0 + (tl.t - t0) * a.u(0.45, 0.85)
    d_wind = a.d(0.3)
    tl.move(d_wind, start=t_mid - d_wind, advance=False,
            **{K(a): aim(a.s, a.u(20, 40), a.u(-80, -55), a.u(0.6, 0.8)) + np.array([0, -0.2, 0]),
               "twist": -a.s * R(a.u(5, 20))})
    end = tl.move(a.d(0.18, 0.35), start=t_mid, advance=False,
                  **{K(a): aim(a.s, a.u(-20, 5), a.u(-55, -20), a.u(0.6, 0.95)), "twist": a.s * R(a.u(10, 30)),
                     "pitch": R(a.u(5, 15))}, label="walking_thrust")
    tl.t = max(tl.t, end)
    tl.move(a.d(0.4), **{K(a): ready_knife(a), "twist": 0.0, "pitch": 0.0})
    return f"walking_thrust_{n}steps", t_mid


# attack name -> (builder, dataset folder)
ATTACKS = {
    "thrust_train": (lambda t, a: thrust(t, a, "train"), "Sim_Thrust"),
    "thrust_low": (lambda t, a: thrust(t, a, "low"), "Sim_Thrust"),
    "thrust_bent": (lambda t, a: thrust(t, a, "bent"), "Sim_Thrust"),
    "thrust_mid": (lambda t, a: thrust(t, a, "mid"), "Sim_Thrust"),
    "walking_thrust": (walking_thrust, "Sim_Thrust"),
    "upper_stab": (upper_stab, "Sim_Stab"),
    "overhand_stab": (overhand_stab, "Sim_Stab"),
    "frenzy": (frenzy, "Sim_Stab"),
    "cut_down": (cut_down, "Sim_Cut-Down"),
    "ground_strike": (ground_strike, "Sim_Cut-Down"),
    "slash": (lambda t, a: slash(t, a, n=1), "Sim_Slash"),
    "multi_slash": (lambda t, a: slash(t, a, n=int(a.rng.integers(2, 6))), "Sim_Multi-Slash"),
    "swing": (lambda t, a: slash(t, a, n=int(a.rng.choice([1, 1, 2])), big=True), "Sim_Swing"),
    "combo": (combo, "Sim_Combo"),
    "turn_attack": (lambda t, a: turn_then(t, a, lambda t2, a2: ATTACKS[str(a2.rng.choice(
        ["overhand_stab", "cut_down", "slash", "thrust_mid"]))][0](t2, a2)), "Sim_Combo"),
}


# ---------------------------------------------------------------- normal (non-violent) activity
def n_idle(tl, a):
    tl.move(a.d(1.0), **{K(a): rest_knife(a), F(a): free_pose(a, "hang")})
    tl.wait(a.d(1.5, 0.5))
    if a.chance(0.5):
        tl.move(a.d(0.8), head_yaw=R(a.u(-50, 50)))
        tl.wait(a.d(0.8))
    return "idle"


def n_walk(tl, a):
    walk(tl, a, int(a.rng.integers(3, 8)), turn_deg=a.u(-120, 120) if a.chance(0.5) else 0.0)
    return "walk"


def n_turn(tl, a):
    turn_in_place(tl, a, R(a.u(90, 200)) * (1 if a.chance(0.5) else -1), a.d(1.0))
    tl.wait(a.d(0.5))
    return "turn"


def n_point(tl, a):
    tl.move(a.d(0.9), **{K(a): aim(a.s, a.u(-20, 15), a.u(-25, 5), a.u(0.85, 0.98))})
    tl.wait(a.d(1.0))
    tl.move(a.d(0.9), **{K(a): rest_knife(a)})
    return "point_with_knife"


def n_inspect(tl, a):
    tl.move(a.d(0.9), head_pitch=R(a.u(15, 35)),
            **{K(a): aim(a.s, a.u(-35, -10), a.u(-30, -5), a.u(0.45, 0.6)), F(a): aim(-a.s, a.u(-40, -15), a.u(-35, -15), a.u(0.45, 0.6))})
    tl.wait(a.d(1.2))
    tl.move(a.d(0.8), head_pitch=0.0, **{K(a): rest_knife(a), F(a): free_pose(a, "hang")})
    return "inspect_knife"


def n_hand_to_head(tl, a):
    hand = K(a) if a.chance(0.5) else F(a)
    side = a.s if hand == K(a) else -a.s
    tl.move(a.d(0.9), **{hand: aim(side, a.u(-40, -15), a.u(50, 75), a.u(0.35, 0.5)), "swv_" + hand[-1]: R(a.u(20, 50))})
    tl.wait(a.d(1.0))
    tl.move(a.d(0.8), **{hand: free_pose(a, "hang") if hand == F(a) else rest_knife(a), "swv_" + hand[-1]: 0.0})
    return "hand_to_head"


def n_stretch(tl, a):
    tl.move(a.d(1.0), **{K(a): aim(a.s, a.u(80, 95), a.u(-5, 20), a.u(0.95, 0.99)),
                         F(a): aim(-a.s, a.u(80, 95), a.u(-5, 20), a.u(0.95, 0.99))})
    tl.wait(a.d(0.8))
    tl.move(a.d(1.0), **{K(a): rest_knife(a), F(a): free_pose(a, "hang")})
    return "arms_out"


def n_pick_up(tl, a):
    tl.move(a.d(1.0), pitch=R(a.u(45, 75)), crouch=a.u(0.1, 0.4),
            **{F(a): aim(-a.s, a.u(-10, 10), a.u(-85, -70), 0.97)})
    tl.wait(a.d(0.6))
    tl.move(a.d(1.0), pitch=0.0, crouch=0.0, **{F(a): free_pose(a, "hang")})
    return "pick_up"


def n_gesture(tl, a):
    for _ in range(int(a.rng.integers(2, 5))):
        tl.move(a.d(0.5), **{F(a): aim(-a.s, a.u(-30, 30), a.u(-50, -10), a.u(0.5, 0.8))})
    tl.move(a.d(0.6), **{F(a): free_pose(a, "hang")})
    return "talk_gesture"


NORMALS = {"idle": n_idle, "walk": n_walk, "turn": n_turn, "point": n_point, "inspect": n_inspect,
           "hand_to_head": n_hand_to_head, "stretch": n_stretch, "pick_up": n_pick_up, "gesture": n_gesture}
