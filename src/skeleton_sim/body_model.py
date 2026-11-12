"""
A 3D human body that carries recorded 2D skeletons. The body is the Rajagopal et al. (2016) full-body model from OpenSim,
read from its .osim file: bones, joint axes and joint ranges come from that model (anatomy, not motion data). Only the
joints behind the 17 COCO points are used. The face points are not in the model, so they are fitted as fixed points on
the torso, and the bone sizes are fitted to the person.

A recorded take is fitted by turning joints within their ranges until the body, seen by the camera, matches the 2D
points. The fitted motion can then be seen from any direction.

Model frame: X forward, Y up, Z to the person's right, metres. Image: x to the right, y down, as in the recordings.
"""

import xml.etree.ElementTree as ET

import numpy as np
import torch

USED = ("hip_r", "walker_knee_r", "ankle_r", "hip_l", "walker_knee_l", "ankle_l", "back",
        "acromial_r", "elbow_r", "radioulnar_r", "radius_hand_r", "acromial_l", "elbow_l", "radioulnar_l", "radius_hand_l")
STILL = {"ankle_angle_r", "ankle_angle_l", "wrist_flex_r", "wrist_dev_r", "wrist_flex_l", "wrist_dev_l"}  # move none of the 17 points
POINTS = {5: "humerus_l", 6: "humerus_r", 7: "ulna_l", 8: "ulna_r", 9: "hand_l", 10: "hand_r", 11: "femur_l", 12: "femur_r",
          13: "tibia_l", 14: "tibia_r", 15: "talus_l", 16: "talus_r"}  # COCO joint -> the body whose origin it is
SEGMENTS = ("pelvis", "torso", "femur", "tibia", "humerus", "forearm")
SEGMENT_OF = {"pelvis": 0, "torso": 1, "femur_r": 2, "femur_l": 2, "tibia_r": 3, "tibia_l": 3, "talus_r": 3, "talus_l": 3,
              "humerus_r": 4, "humerus_l": 4, "ulna_r": 5, "ulna_l": 5, "radius_r": 5, "radius_l": 5, "hand_r": 5, "hand_l": 5}
FACE0 = np.array([[0.10, 0.62, 0.0], [0.08, 0.66, -0.035], [0.08, 0.66, 0.035], [0.0, 0.64, -0.075], [0.0, 0.64, 0.075]])  # nose, L eye, R eye, L ear, R ear, in the torso frame
X_AXIS, Y_AXIS, Z_AXIS = np.eye(3)


def _vec(el, tag):
    return np.array(el.find(tag).text.split(), dtype=np.float64)


def _euler_xyz(r):
    """OpenSim frame orientation: body-fixed rotations about X, then Y, then Z."""
    out = np.eye(3)
    for axis, a in zip(np.eye(3), r):
        out = out @ _rod_np(axis, a)
    return out


def _rod_np(axis, a):
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def rotation(axis, angle):
    """(N,) angles about one fixed axis -> (N, 3, 3)."""
    axis = torch.as_tensor(axis / np.linalg.norm(axis), dtype=angle.dtype, device=angle.device)
    k = torch.zeros(3, 3, dtype=angle.dtype, device=angle.device)
    k[0, 1], k[0, 2], k[1, 0], k[1, 2], k[2, 0], k[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return torch.eye(3, dtype=angle.dtype, device=angle.device) + s * k + (1 - c) * (k @ k)


def _function(el):
    """An OpenSim function of one coordinate, as (kind, numbers)."""
    f = next(e for e in el if e.tag not in ("coordinates", "axis"))
    if f.tag == "LinearFunction":
        return "linear", _vec(f, "coefficients")
    if f.tag == "Constant":
        return "constant", np.array([float(f.find("value").text)])
    if f.tag == "SimmSpline":
        return "spline", np.stack([_vec(f, "x"), _vec(f, "y")])
    raise ValueError(f"unsupported function {f.tag}")


def _evaluate(kind, numbers, q):
    if kind == "linear":
        return numbers[0] * q + numbers[1]
    xs = torch.as_tensor(numbers[0], dtype=q.dtype, device=q.device)
    ys = torch.as_tensor(numbers[1], dtype=q.dtype, device=q.device)
    i = torch.clamp(torch.searchsorted(xs, q.detach().contiguous()) - 1, 0, len(xs) - 2)
    t = (q - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])  # linear between the spline's knots, which are 10 degrees apart


class Body:
    """The body model. coords: the joint angles that move the 17 points, with the model's ranges (radians)."""

    def __init__(self, path):
        model = ET.parse(path).getroot()
        joints = {j.get("name"): j for j in model.find(".//JointSet/objects")}
        bodies = {b.get("name"): b for b in model.iter("Body")}
        self.mass = {n: (float(bodies[n].find("mass").text), _vec(bodies[n], "mass_center")) for n in SEGMENT_OF}
        for side in "rl":  # the feet beyond the ankle are not in the chain: their mass sits at the ankle
            m, c = self.mass["talus_" + side]
            self.mass["talus_" + side] = (m + sum(float(bodies[f"{p}_{side}"].find("mass").text) for p in ("calcn", "toes")), c)
        self.joints, names = [], []
        for name in USED:
            j = joints[name]
            frames = {f.get("name"): (f.find("socket_parent").text.split("/")[-1], _vec(f, "translation"), _euler_xyz(_vec(f, "orientation")))
                      for f in j.find("frames")}
            parent, t_pf, r_pf = frames[j.find("socket_parent_frame").text]
            child, t_bm, r_bm = frames[j.find("socket_child_frame").text]
            coords = [c.get("name") for c in j.iter("Coordinate")]
            for c in j.iter("Coordinate"):
                if c.get("name") not in STILL:
                    names.append((c.get("name"), *map(float, c.find("range").text.split())))
            if j.tag == "PinJoint":
                axes = [("rot", Z_AXIS, coords[0], ("linear", np.array([1.0, 0.0])))]
            elif j.tag == "UniversalJoint":
                axes = [("rot", X_AXIS, coords[0], ("linear", np.array([1.0, 0.0]))), ("rot", Y_AXIS, coords[1], ("linear", np.array([1.0, 0.0])))]
            else:
                axes = []
                for ta in j.find("SpatialTransform"):
                    coord, fn = ta.find("coordinates").text, _function(ta)
                    if coord is None and not fn[1].any():
                        continue  # a constant zero
                    axes.append(("rot" if ta.get("name").startswith("rotation") else "trans", _vec(ta, "axis"), coord, fn))
            self.joints.append(dict(parent=parent, child=child, t_pf=t_pf, r_pf=r_pf, t_bm=t_bm, r_bm=r_bm, axes=axes))
        self.coords = [n for n, _, _ in names]
        self.lo = np.array([lo for _, lo, _ in names])
        self.hi = np.array([hi for _, _, hi in names])

    def forward(self, q, root, scale):
        """q (N, len(coords)), root (N, 3, 3) pelvis orientation, scale (6,) per segment -> {body: (R (N,3,3), p (N,3))}."""
        n, dt, dev = len(q), q.dtype, q.device
        index = {c: i for i, c in enumerate(self.coords)}
        world = {"pelvis": (root, torch.zeros(n, 3, dtype=dt, device=dev))}
        for j in self.joints:
            rp, pp = world[j["parent"]]
            sp, sc = scale[SEGMENT_OF[j["parent"]]], scale[SEGMENT_OF[j["child"]]]
            r = torch.eye(3, dtype=dt, device=dev).expand(n, 3, 3)
            t = torch.zeros(n, 3, dtype=dt, device=dev)
            for kind, axis, coord, (fk, fnum) in j["axes"]:
                v = q[:, index[coord]] if coord in index else torch.zeros(n, dtype=dt, device=dev)
                v = _evaluate(fk, fnum, v) if fk != "constant" else torch.full((n,), float(fnum[0]), dtype=dt, device=dev)
                if kind == "rot":
                    r = r @ rotation(axis, v)  # body-fixed: each axis turns with the ones before it
                else:
                    t = t + v[:, None] * torch.as_tensor(axis, dtype=dt, device=dev) * sp
            r_pf, r_bm = (torch.as_tensor(m, dtype=dt, device=dev) for m in (j["r_pf"], j["r_bm"]))
            t_pf, t_bm = torch.as_tensor(j["t_pf"], dtype=dt, device=dev) * sp, torch.as_tensor(j["t_bm"], dtype=dt, device=dev) * sc
            r_fb = r @ r_bm.T
            t_fb = t - (r_fb @ t_bm)
            world[j["child"]] = (rp @ r_pf @ r_fb, pp + (rp @ (t_pf + (r_pf @ t_fb[..., None])[..., 0])[..., None])[..., 0])
        return world

    def points(self, q, root, scale, face, world=None):
        """The 17 COCO points in 3D, (N, 17, 3). face (5, 3): face points in the torso frame, before scaling."""
        w = world or self.forward(q, root, scale)
        out = [None] * 17
        for i, b in POINTS.items():
            out[i] = w[b][1]
        rt, pt = w["torso"]
        for i in range(5):
            out[i] = pt + (rt @ (face[i] * scale[1]))
        return torch.stack(out, 1)

    def balance(self, world, scale, root, reach=0.06, slack=0.10):
        """How far (m) the body's centre of mass is, seen from above, outside the area the feet support: the line
        between the ankles, moved forward to the middle of the feet, widened by slack. Masses are the model's."""
        total, com = 0.0, 0.0
        for n, (m, c) in self.mass.items():
            r, p = world[n]
            com = com + m * (p + r @ (torch.as_tensor(c, dtype=p.dtype, device=p.device) * scale[SEGMENT_OF[n]]))
            total += m
        com = (com / total)[:, [0, 2]]
        fwd = root[:, [0, 2], 0]
        fwd = fwd / torch.linalg.norm(fwd, dim=1, keepdim=True).clamp_min(1e-6)
        a, b = world["talus_l"][1][:, [0, 2]] + reach * fwd, world["talus_r"][1][:, [0, 2]] + reach * fwd
        ab = b - a
        t = (((com - a) * ab).sum(1) / (ab * ab).sum(1).clamp_min(1e-6)).clamp(0, 1)
        return torch.relu(torch.linalg.norm(com - a - t[:, None] * ab, dim=1) - slack)


def root_rotation(yaw, lean, side):
    """Pelvis orientation: turned by yaw about the vertical, then leaning forward (about Z) and sideways (about X)."""
    return rotation(Y_AXIS, yaw) @ rotation(Z_AXIS, lean) @ rotation(X_AXIS, side)


def project(p, tilt, dist=None):
    """3D points (N, 17, 3) -> image (N, 17, 2) relative to the hip centre, for a camera in front (+X) looking down at
    the pelvis by tilt radians from dist metres away (None: far away, no perspective). The person's right (+Z) shows on
    the image left, nearer points show lower, as from above, and with perspective farther points (the feet, from a high
    camera) show smaller. Scaled so that sizes at the pelvis match the far camera."""
    if tilt.dim():  # one camera per frame
        tilt = tilt[:, None]
        dist = dist[:, None] if dist is not None else None
    x = -p[..., 2]
    y = p[..., 0] * torch.sin(tilt) - p[..., 1] * torch.cos(tilt)
    if dist is not None:
        z = dist - (p[..., 0] * torch.cos(tilt) + p[..., 1] * torch.sin(tilt))  # depth along the camera's axis
        x, y = x * dist / z, y * dist / z
    img = torch.stack([x, y], -1)
    return img - (img[:, 11:12] + img[:, 12:13]) / 2


def _bounded(u, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(u)


def _unbounded(v, lo, hi):
    f = np.clip((v - lo) / (hi - lo), 0.02, 0.98)
    return np.log(f / (1 - f))


class Fit:
    """Joint angles for a set of sequences, one start direction each. Angles stay in the model's ranges by construction."""

    PELVIS = (-np.pi / 2, np.pi / 2)  # the model's own range for pelvis tilt and list

    def __init__(self, body, lengths, yaw0, shared, device):
        n = int(sum(lengths))
        q0 = np.where((body.lo <= 0.05) & (body.hi >= 0.05), 0.05, (body.lo + body.hi) / 2)  # just off a straight elbow or knee
        self.lo, self.hi = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (body.lo, body.hi))
        self.u = torch.tensor(np.tile(_unbounded(q0, body.lo, body.hi), (n, 1)), dtype=torch.float32, device=device, requires_grad=True)
        self.yaw = torch.tensor(np.repeat(yaw0, lengths), dtype=torch.float32, device=device, requires_grad=True)
        self.lean = torch.zeros(n, 2, device=device, requires_grad=True)
        self.logk = torch.full((len(lengths),), float(np.log(1.9)), device=device, requires_grad=True)  # torso lengths per metre
        self.seq = torch.as_tensor(np.repeat(np.arange(len(lengths)), lengths), device=device)
        self.shared = shared
        self.dyaw = self.logk2 = None  # a second view of each sequence, if any: how far it is turned, and its scale
        self.cam = None  # a camera of its own per sequence, if any: tilt (radians) and log distance

    def own_cameras(self):
        """Each sequence gets its own camera, starting from the shared one: for windows filmed by unknown cameras."""
        s = self.shared
        start = [float(s["tilt"]), float(s["log_dist"]) if "log_dist" in s else float(np.log(3.0))]
        self.cam = torch.tensor([start] * len(self.logk), dtype=torch.float32, device=self.u.device, requires_grad=True)

    def add_second_view(self, dyaw0):
        """Each sequence is also seen by a second camera, dyaw0 radians further round the person."""
        self.dyaw = torch.tensor(np.asarray(dyaw0, np.float32), device=self.u.device, requires_grad=True)
        self.logk2 = self.logk.detach().clone().requires_grad_(True)

    def params(self):
        return [self.u, self.yaw, self.lean, self.logk] + ([self.dyaw, self.logk2] if self.dyaw is not None else []) + \
            ([self.cam] if self.cam is not None else [])

    def angles(self):
        return _bounded(self.u, self.lo, self.hi)

    def render(self, body, yaw_turn=0.0, rows=None, balance=False, second=False):
        """The fitted motion in image coordinates (torso lengths), optionally turned by yaw_turn radians. With balance,
        also how far each frame's centre of mass is outside the feet (m). second: as the second camera sees it."""
        rows = slice(None) if rows is None else rows
        logk = self.logk2 if second else self.logk
        if second:
            yaw_turn = yaw_turn + self.dyaw[self.seq[rows]]
        tilt = dist = None
        if self.cam is not None:
            cam = self.cam[self.seq[rows]]
            tilt, dist = cam[:, 0], torch.exp(cam[:, 1])
        return render_states(body, self.shared, self.angles()[rows], self.lean[rows], self.yaw[rows] + yaw_turn, logk[self.seq[rows]], tilt, dist, balance)


def render_states(body, shared, q, lean_raw, yaw, logk, tilt=None, dist=None, balance=False):
    """Joint angles q (N, coords), unbounded pelvis lean (N, 2), facing yaw (N,) and image scale logk (N,) -> image (N, 17, 2)
    in torso lengths. The camera is the shared one unless tilt and dist (per frame) are given. With balance, also how far
    each frame's centre of mass is outside the feet (m)."""
    lean = _bounded(lean_raw, *Fit.PELVIS)
    root = root_rotation(yaw, lean[:, 0], lean[:, 1])
    scale = torch.exp(shared["log_scale"])
    world = body.forward(q, root, scale)
    p = body.points(q, root, scale, shared["face"], world)
    if tilt is None:
        tilt, dist = shared["tilt"], (torch.exp(shared["log_dist"]) if "log_dist" in shared else None)
    img = project(p, tilt, dist) * torch.exp(logk)[:, None, None]
    return (img, body.balance(world, scale, root)) if balance else img


def new_shared(device, tilt_deg=13.0, fit_tilt=False, dist=3.0):
    """Bone sizes, face points and the camera, shared by all takes of one person and camera. The tilt stays at the value
    measured from side views (results/samples/edit/camera_measure.txt) unless fit_tilt: fitted, it trades against the
    body leaning towards the camera. The camera distance is fitted (dist None: no perspective). Without perspective
    the fit makes the legs look shorter, as a high camera shows them, by tipping them away from the camera."""
    out = dict(log_scale=torch.zeros(6, device=device, requires_grad=True),
               face=torch.tensor(FACE0, dtype=torch.float32, device=device, requires_grad=True),
               tilt=torch.tensor(np.radians(tilt_deg), dtype=torch.float32, device=device, requires_grad=fit_tilt))
    if dist:
        out["log_dist"] = torch.tensor(np.log(dist), dtype=torch.float32, device=device, requires_grad=True)
    return out


def _huber(r, delta=0.1):
    return torch.where(r < delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))


def pose_prior(angles, lean, dims, coords):
    """A prior on the body's pose for what a view cannot see: dims {name: frames (rows)} picks, for each angle (a coord
    name, or "lean" / "side" for the pelvis), the frames that see it well. Returns (mean, inverse variance) over the
    motion vector [angles, bounded lean (2)], zero weight on dims not given."""
    m = np.concatenate([angles, np.asarray(Fit.PELVIS[0] + (Fit.PELVIS[1] - Fit.PELVIS[0]) / (1 + np.exp(-lean)))], 1)
    names = list(coords) + ["lean", "side"]
    mean, iv = np.zeros(len(names), np.float32), np.zeros(len(names), np.float32)
    for n, rows in dims.items():
        i = names.index(n)
        mean[i], iv[i] = m[rows, i].mean(), 1 / max(m[rows, i].var(), 1e-4)
    return mean, iv


def _run(body, f, obs, w, steps, smooth, lr, fit_shared, balance, obs2=None, w2=None, prior=None):
    """prior: (mean, inverse variance, weight) on [angles, bounded lean], from pose_prior."""
    params = f.params() + ([v for v in f.shared.values() if v.requires_grad] if fit_shared else [])
    opt = torch.optim.Adam(params, lr=lr)
    inner = f.seq[2:] == f.seq[:-2]  # second differences never cross from one sequence to the next
    n = len(f.logk)
    for _ in range(steps):
        opt.zero_grad()
        img, off = f.render(body, balance=True)
        err = (_huber(torch.linalg.norm(img - obs, dim=-1)) * w).sum(1)
        seen = w.sum(1)
        if obs2 is not None:
            err = err + (_huber(torch.linalg.norm(f.render(body, second=True) - obs2, dim=-1)) * w2).sum(1)
            seen = seen + w2.sum(1)
        per_seq = torch.zeros(n, device=obs.device).index_add_(0, f.seq, err) / torch.zeros(n, device=obs.device).index_add_(0, f.seq, seen).clamp_min(1e-6)
        motion = torch.cat([f.angles(), f.yaw[:, None], f.lean], 1)
        acc = (motion[2:] - 2 * motion[1:-1] + motion[:-2])[inner]
        scale_dev = f.shared["log_scale"] ** 2
        loss = per_seq.sum() + (smooth * (acc ** 2).sum(1).mean() + balance * (off ** 2).mean()) * len(f.logk) + 0.002 * scale_dev.sum()
        if prior is not None:
            mean, iv, weight = prior
            m = torch.cat([f.angles(), _bounded(f.lean, *Fit.PELVIS)], 1)
            loss = loss + weight * (((m - mean) ** 2) * iv).sum(1).mean() * len(f.logk)
        loss.backward()
        opt.step()
        with torch.no_grad():
            f.shared["log_scale"].clamp_(np.log(0.6), np.log(1.5))
            f.shared["tilt"].clamp_(0.0, np.radians(40))
            if "log_dist" in f.shared:
                f.shared["log_dist"].clamp_(np.log(1.2), np.log(30.0))
            if f.cam is not None:
                f.cam[:, 0].clamp_(0.0, np.radians(40))
                f.cam[:, 1].clamp_(np.log(1.2), np.log(30.0))
    return per_seq.detach()


def fit_takes(body, obs, weight, yaw0, starts=(0.0, 90.0, 180.0, 270.0), steps=(400, 1500), smooth=0.05, lr=0.03, device="cuda:0", shared=None, tilt_deg=13.0, balance=0.1,
              own_camera=False, prior=None):
    """Fits sequences into the body. obs: list of (F, 17, 2) body coordinates (hip centre, torso lengths, y down);
    weight: list of (F, 17), 0 where a joint was not seen; yaw0: a first guess of each one's facing (degrees).
    Each sequence is first tried from several facings, and the best is kept. With shared given (bone sizes, face points,
    camera tilt from an earlier fit) those stay fixed. own_camera: each sequence also fits its own camera tilt and
    distance. prior: (mean, inverse variance) from pose_prior, and its weight: pulls what the view cannot see towards
    what other views of the person saw. Returns (Fit, per-sequence error, shared)."""
    fixed = shared is not None
    if prior is not None:
        prior = (torch.as_tensor(prior[0], device=device), torch.as_tensor(prior[1], device=device), float(prior[2]))
    shared = shared or new_shared(device, tilt_deg)
    lengths = np.array([len(o) for o in obs])
    o = torch.as_tensor(np.concatenate(obs), dtype=torch.float32, device=device)
    w = torch.as_tensor(np.concatenate(weight), dtype=torch.float32, device=device)

    tries = Fit(body, np.tile(lengths, len(starts)), np.concatenate([np.radians(np.asarray(yaw0) + s) for s in starts]), shared, device)
    if own_camera:
        tries.own_cameras()
    err = _run(body, tries, o.repeat(len(starts), 1, 1), w.repeat(len(starts), 1), steps[0], smooth, lr, not fixed, balance, prior=prior)
    best = err.view(len(starts), len(obs)).argmin(0).cpu().numpy()

    f = Fit(body, lengths, np.zeros(len(obs)), shared, device)
    if own_camera:
        f.own_cameras()
    offsets = np.r_[0, np.cumsum(np.tile(lengths, len(starts)))]
    rows = np.concatenate([np.arange(offsets[b * len(obs) + i], offsets[b * len(obs) + i + 1]) for i, b in enumerate(best)])
    pick = best * len(obs) + np.arange(len(obs))
    with torch.no_grad():
        for mine, theirs in zip(f.params()[:3], tries.params()[:3]):
            mine.copy_(theirs[rows])
        f.logk.copy_(tries.logk[pick])
        if own_camera:
            f.cam.copy_(tries.cam[pick])
    err = _run(body, f, o, w, steps[1], smooth, lr, not fixed, balance, prior=prior)
    return f, err.cpu().numpy(), shared


def fit_two_views(body, obs_a, w_a, obs_b, w_b, yaw0, dyaw0, shared, steps=1500, smooth=0.05, lr=0.03, balance=0.1, device="cuda:0"):
    """One motion seen from two directions: obs_a[i] and obs_b[i] are (T, 17, 2) views of sequence i, the second
    dyaw0[i] radians further round (from single-view fits). Depth that one view cannot see, the other does. yaw0: the
    facing of view a (radians), per sequence. Bone sizes, face and camera come from shared and stay fixed."""
    lengths = np.array([len(o) for o in obs_a])
    t = lambda x: torch.as_tensor(np.concatenate(x), dtype=torch.float32, device=device)
    f = Fit(body, lengths, np.asarray(yaw0), shared, device)
    f.add_second_view(dyaw0)
    err = _run(body, f, t(obs_a), t(w_a), steps, smooth, lr, False, balance, t(obs_b), t(w_b))
    return f, err.cpu().numpy()
