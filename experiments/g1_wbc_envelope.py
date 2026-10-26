"""Measure what the pretrained Decoupled WBC can track and survive when its posture inputs move.

    python experiments/g1_wbc_envelope.py experiments/configs/g1_wbc_envelope_v2.json [--shard i/n]
    python experiments/g1_wbc_envelope.py experiments/configs/g1_wbc_envelope_v2.json --summarize

Protocol (frozen in the config before the run): the free-standing G1 (dream/sim/g1_mobile.py)
stands at zero velocity while the base-height and torso rpy commands (WBCLocomotion.set_posture)
move at a fixed rate. Cell kinds: ``joint`` (height x pitch x arm pose, ramp-hold-return, repeated
cycles), ``probe`` (roll, yaw, negative pitch), ``transition`` (multi-segment sequences, e.g.
0.74 -> low -> mid -> 0.74, pitch changes while crouched) and ``push`` (lateral and fore-aft
pelvis pushes at the low postures, recovery measured). Every cell runs the same seeds.

Collapse versus an intended crouch is NOT decided from pelvis height. Each trial records,
separately: tilt latch, undesired ground contact (any non-foot robot geom on the floor), loss of
foot support, tracking error against the command, joint-limit contact, CoM margin to the support
polygon, and the torque ratio. Tracking of the torso rpy command uses the upstream waist-chain
definition (torso_link relative to the pelvis, g1_wbc.py docstring) on commanded axes only; the
world-frame torso attitude is recorded but is not a gate. Ground truth (poses, contacts, CoM,
actuator commands) is read here only, as the evaluator.

``envelope.json`` is a set of measured cells with counts and Wilson 95% intervals, not a safety
claim. Commands outside g1_wbc.HEIGHT_RANGE / RPY_RANGE are rejected by the controller and are
recorded as rejections, not run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.robot import g1_wbc  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1  # noqa: E402

TICK = g1_wbc.DT_PHYSICS * g1_wbc.DECIMATION  # 0.02 s
GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}
DIRS = {"fwd": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1)}


# -- protocol ----------------------------------------------------------------------------------
def conditions(cfg: dict) -> list[dict]:
    """Every cell of the frozen protocol, in a fixed order."""
    nom, out = cfg["nominal_height"], []

    def add(kind, name, arms, segments, cycles, push=None, arm_rate=None):
        out.append(
            {
                "arm_rate": arm_rate,
                "name": name,
                "kind": kind,
                "arms": arms,
                "segments": [{"height": h, "rpy": list(r)} for h, r in segments],
                "cycles": cycles,
                "push": push,
            }
        )

    for arms in cfg["arm_poses"]:
        for h in cfg["joint_heights"]:
            for p in cfg["joint_pitches"]:
                name = f"joint_{arms}_h{h:.2f}_p{p:+.1f}"
                add("joint", name, arms, [(h, (0, p, 0))], cfg["cycles"])
    for r in cfg["probe_rolls"]:
        add("probe", f"probe_roll{r:+.1f}", "neutral", [(nom, (r, 0, 0))], cfg["cycles"])
    for y in cfg["probe_yaws"]:
        add("probe", f"probe_yaw{y:+.1f}", "neutral", [(nom, (0, 0, y))], cfg["cycles"])
    for p in cfg["probe_pitches"]:
        add("probe", f"probe_pitch{p:+.1f}", "neutral", [(nom, (0, p, 0))], cfg["cycles"])
    for t in cfg["transitions"]:
        add("transition", "trans_" + t["name"], t["arms"], [(s[0], s[1]) for s in t["segments"]], 1)
    for h in cfg["push_heights"]:
        for d in DIRS:
            for n in cfg["push_newtons"]:
                push = {"dir": d, "newtons": n}
                add("push", f"push_h{h:.2f}_{d}_{n:g}N", "neutral", [(h, (0, 0, 0))], 1, push)
    ar = cfg.get("arm_rate")
    if ar:  # appended last so earlier cells keep their trial files
        for h in ar["heights"]:
            for rate in ar["rates"]:
                add(
                    "arm_rate",
                    f"armrate_h{h:.2f}_r{rate:.1f}",
                    "neutral",
                    [(h, (0, 0, 0))],
                    1,
                    None,
                    rate,
                )
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson 95% score interval for k successes in n trials."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


# -- evaluator-side measurement ----------------------------------------------------------------
def rpy_of(R: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll) (Pinocchio matrixToRpy)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    return np.array([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


def yaw_matrix(R: np.ndarray) -> np.ndarray:
    y = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Probe:
    """Samples every criterion once per policy tick."""

    def __init__(self, env: G1MobileEnv, cfg: dict):
        m = env.model
        self.env, self.cfg = env, cfg
        self.pelvis, self.torso = m.body("pelvis").id, m.body("torso_link").id
        self.feet = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
        self.floor = m.geom("floor").id
        joints = m.actuator_trnid[:, 0]
        self.limit = np.abs(m.jnt_actfrcrange[joints]).max(axis=1)
        self.jlo, self.jhi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        self.jnames = [m.joint(j).name for j in joints]
        self.rows: list[dict] = []
        self.ctrl = None  # set by run_trial: the controller whose arm reference is tracked

    def contacts(self) -> tuple[list[bool], list[str], list[np.ndarray]]:
        d, m = self.env.data, self.env.model
        foot, bad, pts = [False, False], set(), []
        for c in d.contact[: d.ncon]:
            g = (c.geom1, c.geom2)
            if self.floor not in g:
                continue
            body = int(m.geom_bodyid[g[1] if g[0] == self.floor else g[0]])
            if body in self.feet:
                foot[self.feet.index(body)] = True
                pts.append(c.pos[:2].copy())
            else:
                bad.add(m.body(body).name)
        return foot, sorted(bad), pts

    def margin(self, pts: list[np.ndarray]) -> float | None:
        """Signed horizontal distance (m) from the CoM to the support polygon (convex hull of
        foot-floor contacts); positive inside, None with fewer than 3 contact points."""
        if len(pts) < 3:
            return None
        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = ConvexHull(np.array(pts))
        except QhullError:
            return None
        com = self.env.data.subtree_com[1][:2]
        return float(np.min(-(hull.equations[:, :2] @ com + hull.equations[:, 2])))

    def sample(self, t: float, seg: str, push_on: bool) -> None:
        env, d = self.env, self.env.data
        foot, bad, pts = self.contacts()
        Rp, Rt = d.xmat[self.pelvis].reshape(3, 3), d.xmat[self.torso].reshape(3, 3)
        q = d.qpos[env.qadr]
        m = self.cfg["joint_limit_margin_rad"]
        near = (q - self.jlo < m) | (self.jhi - q < m)
        ratio = np.abs(d.ctrl) / self.limit
        arm_err = float(np.max(np.abs(q[15:] - self.ctrl.arm_target)))
        self.rows.append(
            {
                "arm_err": arm_err,
                "t": t,
                "seg": seg,
                "push": push_on,
                "z": float(d.xpos[self.pelvis][2]),
                "rpy_world": rpy_of(yaw_matrix(Rp).T @ Rt).tolist(),  # recorded, not a gate
                "rpy_waist": rpy_of(Rp.T @ Rt).tolist(),  # upstream definition: the gate
                "ratio": {k: float(ratio[s].max()) for k, s in GROUPS.items()},
                "margin": self.margin(pts),
                "feet": foot,
                "bad": bad,
                "limit": [self.jnames[i] for i in np.flatnonzero(near)],
                "feet_xy": [d.xpos[b][:2].tolist() for b in self.feet],
            }
        )


def longest(flags: list[bool]) -> float:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best * TICK


# -- one trial ---------------------------------------------------------------------------------
def ramp_time(cfg: dict, h0: float, r0, h1: float, r1) -> float:
    dr = np.max(np.abs(np.asarray(r1, float) - np.asarray(r0, float)))
    return max(abs(h1 - h0) / cfg["rate_height"], float(dr) / cfg["rate_rpy"])


def run_trial(cfg: dict, cond: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env = G1MobileEnv(build_mobile_g1(head_camera=None), rng, head_camera=None)
    env.reset(0.0, 0.0, 0.0)
    env.data.qpos[env.qadr] += rng.normal(scale=cfg["joint_noise"], size=len(env.qadr))
    mujoco.mj_forward(env.model, env.data)
    ctrl = WBCLocomotion.from_cache()
    probe = Probe(env, cfg)
    probe.ctrl = ctrl
    nom_h = cfg["nominal_height"]
    ctrl.set_posture(
        nom_h, (0, 0, 0), max_height_rate=cfg["rate_height"], max_rpy_rate=cfg["rate_rpy"]
    )
    collapse: dict = {"stopped": None, "contact": None}
    window: list[float] = []  # push window [start, end] in trial time
    t_start = env.time

    def run(seconds: float, seg: str, push=None) -> bool:
        for i in range(round(seconds / TICK)):
            on = push is not None and push["t0"] <= i * TICK < push["t0"] + cfg["push_s"]
            if on:
                vx, vy = DIRS[push["dir"]]
                env.data.xfrc_applied[probe.pelvis, :3] = [
                    vx * push["newtons"],
                    vy * push["newtons"],
                    0,
                ]
            env.step(ctrl)
            env.data.xfrc_applied[probe.pelvis, :3] = 0
            probe.sample(env.time - t_start, seg, on)
            if on:
                window[:] = [window[0] if window else probe.rows[-1]["t"], probe.rows[-1]["t"]]
            if ctrl.stopped:
                collapse["stopped"] = ctrl.stopped
                return False
            n = round(cfg["abort_contact_s"] / TICK)
            recent = probe.rows[-n:]
            if len(recent) == n and all(r["bad"] for r in recent):
                collapse["contact"] = recent[-1]["bad"]
                return False
        return True

    ok = run(cfg["settle_s"], "settle")
    if ok:
        ctrl.arm_target = np.array(cfg["arm_poses"][cond["arms"]], float)
        ok = run(cfg["arms_s"], "arms")
    base = [r for r in probe.rows if r["seg"] in ("settle", "arms")][-25:]
    nominal = {
        "z": float(np.mean([r["z"] for r in base])),
        "rpy_waist": np.mean([r["rpy_waist"] for r in base], axis=0).tolist(),
        "rpy_world": np.mean([r["rpy_world"] for r in base], axis=0).tolist(),
        "feet_xy": base[-1]["feet_xy"],
    }
    prev_h, prev_r = nom_h, [0.0, 0.0, 0.0]
    plan = []  # (tag, command, ramp_s, hold_s, push)
    for cyc in range(cond["cycles"]):
        for i, s in enumerate(cond["segments"]):
            ramp = ramp_time(cfg, prev_h, prev_r, s["height"], s["rpy"])
            push = None
            if cond["push"]:
                push = {**cond["push"], "t0": ramp + cfg["push_after_s"]}
            plan.append((f"{cyc}:{i}", s, ramp, cfg["hold_s"], push))
            prev_h, prev_r = s["height"], s["rpy"]
        back = {"height": nom_h, "rpy": [0.0, 0.0, 0.0]}
        ramp = ramp_time(cfg, prev_h, prev_r, nom_h, back["rpy"])
        plan.append((f"{cyc}:back", back, ramp, cfg["return_s"], None))
        prev_h, prev_r = nom_h, back["rpy"]
    for tag, s, ramp, hold, push in plan:
        if not ok:
            break
        ctrl.set_posture(s["height"], s["rpy"])
        ok = run(ramp + hold, tag, push)
        if ok and cond["arm_rate"] and tag == "0:0":
            ok = arm_motion(cfg, cond["arm_rate"], ctrl, run)
    return {
        "condition": cond,
        "seed": seed,
        "collapse_stop": collapse,
        "nominal": nominal,
        "plan": [{"seg": t, "ramp_s": r, "hold_s": h, "cmd": s} for t, s, r, h, _ in plan],
        "push_window": window,
        "criteria": criteria(cfg, cond, probe.rows, nominal, plan, window, collapse),
    }


def arm_motion(cfg: dict, rate: float, ctrl, run) -> bool:
    """Both arms neutral -> pose -> neutral, each joint reference moving at most ``rate`` rad/s.
    Plain WBCLocomotion arm PD (no gravity feed-forward): the validated controller as it is."""
    ar = cfg["arm_rate"]
    pose = np.array(cfg["arm_poses"][ar["pose"]], float)
    span = float(np.max(np.abs(pose)))
    for tag, a, b in (("armout", 0.0, 1.0), ("armback", 1.0, 0.0)):
        n = max(1, round(span / rate / TICK))
        for i in range(n):
            ctrl.arm_target = pose * (a + (b - a) * (i + 1) / n)
            if not run(TICK, tag):
                return False
        ctrl.arm_target = pose * b
        if not run(ar["hold_s"], "armhold" if b == 1.0 else "armend"):
            return False
    return True


def criteria(cfg, cond, rows, nominal, plan, window, collapse) -> dict:
    """Each criterion recorded separately; ``fails`` lists those outside the gates."""
    g = cfg["gates"]
    rows = [r for r in rows if r["seg"] != "settle"]  # the initial drop is scene setup
    zone = (window[0], window[1] + cfg["push_recover_s"]) if window else None
    calm = [r for r in rows if zone is None or not zone[0] <= r["t"] <= zone[1]]
    tilt = collapse["stopped"]
    bad = [bool(r["bad"]) for r in rows]
    contact = {
        "max_continuous_s": longest(bad),
        "total_s": sum(bad) * TICK,
        "bodies": sorted({n for r in rows for n in r["bad"]}),
    }
    lost = [not (r["feet"][0] and r["feet"][1]) for r in calm]
    support = {"max_continuous_s": longest(lost), "total_s": sum(lost) * TICK}
    lim = [bool(r["limit"]) for r in rows]
    limit = {"total_s": sum(lim) * TICK, "joints": sorted({j for r in rows for j in r["limit"]})}
    margins = [r["margin"] for r in calm if r["margin"] is not None]
    com = {"min_m": float(min(margins)) if margins else None}
    torque = {k: float(max((r["ratio"][k] for r in rows), default=0)) for k in GROUPS}

    by_seg: dict[str, list[dict]] = {}
    for r in rows:
        by_seg.setdefault(r["seg"], []).append(r)
    segs = []
    for tag, s, ramp, hold, _ in plan:
        rs = by_seg.get(tag, [])
        if len(rs) * TICK < ramp + hold - 1e-6:
            continue  # collapse before the segment ended
        last = rs[-round(1.0 / TICK) :]
        z = float(np.mean([r["z"] for r in last]))
        wa = np.mean([r["rpy_waist"] for r in last], axis=0)
        cmd = np.array(s["rpy"], float)
        used = np.abs(cmd) > 1e-9
        segs.append(
            {
                "seg": tag,
                "back": tag.endswith("back"),
                "height_cmd": s["height"],
                "height_achieved": z,
                "height_error": z - s["height"],
                "rpy_cmd": cmd.tolist(),
                "rpy_waist_achieved": wa.tolist(),
                "rpy_waist_error": (wa - cmd).tolist(),
                "rpy_waist_error_commanded_max": float(np.max(np.abs((wa - cmd)[used])))
                if used.any()
                else 0.0,
                "rpy_world_achieved": np.mean([r["rpy_world"] for r in last], axis=0).tolist(),
            }
        )
    out_segs = [s for s in segs if not s["back"]]
    n_out = sum(not p[0].endswith("back") for p in plan)
    n_back = len(plan) - n_out
    back_segs = [s for s in segs if s["back"]]
    complete = len(out_segs) == n_out
    height_err = max((abs(s["height_error"]) for s in out_segs), default=None)
    rpy_err = max((s["rpy_waist_error_commanded_max"] for s in out_segs), default=None)
    ret = None
    if len(back_segs) == n_back and back_segs:
        b = back_segs[-1]
        ret = {
            "height_error": b["height_achieved"] - nominal["z"],
            "rpy_waist_error": float(
                np.max(np.abs(np.array(b["rpy_waist_achieved"]) - nominal["rpy_waist"]))
            ),
        }
    f0 = np.array(nominal["feet_xy"])
    slip = max(
        (
            float(np.linalg.norm(np.array(r["feet_xy"]) - f0, axis=1).max())
            for r in calm
            if r["seg"] not in ("settle", "arms")
        ),
        default=0.0,
    )
    recovery = None
    if window:  # first time after the push the height stays within 0.02 m of its final value
        post = [r for r in rows if r["t"] > window[1]]
        for i in range(len(post)):
            if all(abs(q["z"] - post[-1]["z"]) < 0.02 for q in post[i:]):
                recovery = post[i]["t"] - window[1]
                break
    arm = None
    if cond["arm_rate"]:
        moving = [r["arm_err"] for r in rows if r["seg"] in ("armout", "armback")]
        settle = []
        for tag in ("armhold", "armend"):
            rs = [r["arm_err"] for r in rows if r["seg"] == tag]
            if len(rs) * TICK >= cfg["arm_rate"]["hold_s"] - 1e-6:
                settle.append(float(np.mean(rs[-round(1.0 / TICK) :])))
        z0 = by_seg["0:0"][-1]["z"] if by_seg.get("0:0") else None
        zs = [r["z"] for r in rows if r["seg"].startswith("arm")]
        arm = {
            "lag_max_rad": float(max(moving)) if moving else None,
            "settle_err_rad": settle,
            "complete": len(settle) == 2,
            "height_drift_max_m": float(max(abs(z - z0) for z in zs)) if zs and z0 else None,
        }
    fails = []
    if arm and (not arm["complete"] or max(arm["settle_err_rad"]) > g["arm_settle_tol_rad"]):
        fails.append("arm_settle")
    if arm and (arm["height_drift_max_m"] is None or arm["height_drift_max_m"] > g["height_tol_m"]):
        fails.append("arm_height_drift")
    if tilt:
        fails.append("tilt_latch")
    if contact["max_continuous_s"] > g["undesired_contact_max_s"]:
        fails.append("undesired_ground_contact")
    if support["max_continuous_s"] > g["foot_loss_max_s"]:
        fails.append("foot_support_loss")
    if not complete or height_err is None or height_err > g["height_tol_m"]:
        fails.append("height_tracking")
    if not complete or rpy_err is None or rpy_err > g["rpy_tol_rad"]:
        fails.append("rpy_tracking")
    if limit["total_s"] > g["joint_limit_max_s"]:
        fails.append("joint_limit_contact")
    if com["min_m"] is None or com["min_m"] < g["com_margin_min_m"]:
        fails.append("com_margin")
    if max(torque.values()) >= g["torque_ratio_max"]:
        fails.append("torque_ratio")
    if not cond["push"] and slip > g["feet_slip_max_m"]:
        fails.append("feet_slip")
    if (
        ret is None
        or abs(ret["height_error"]) > g["return_height_tol_m"]
        or ret["rpy_waist_error"] > g["return_rpy_tol_rad"]
    ):
        fails.append("return_to_nominal")
    if cond["push"] and recovery is None:
        fails.append("push_recovery")
    return {
        "tilt_latch": tilt,
        "undesired_ground_contact": contact,
        "foot_support": support,
        "joint_limit": limit,
        "com_margin": com,
        "torque_ratio": torque,
        "height_error_max": height_err,
        "rpy_waist_error_commanded_max": rpy_err,
        "feet_slip_max_m": slip,
        "push_recovery_s": recovery,
        "return": ret,
        "segments": segs,
        "arm_motion": arm,
        "fails": fails,
        "stable": not fails,
    }


# -- summary -----------------------------------------------------------------------------------
def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def trial_path(out: pathlib.Path, cond: dict, seed: int) -> pathlib.Path:
    return out / "trials" / f"{cond['name']}_s{seed}.json"


def arm_rate_max(cfg: dict, cells: list[dict]) -> dict | None:
    """Highest arm speed (rad/s) at which every trial passes, at each height and at all heights
    (the rates below it must pass too). Plain arm PD, no gravity feed-forward."""
    ar = cfg.get("arm_rate")
    if not ar:
        return None
    rates, res = sorted(ar["rates"]), {"by_height": {}}
    for h in ar["heights"]:
        best = None
        for r in rates:
            c = next(x for x in cells if x["name"] == f"armrate_h{h:.2f}_r{r:.1f}")
            if not (c["complete"] and c["n_stable"] == c["n_trials"]):
                break
            best = r
        res["by_height"][f"{h:.2f}"] = best
    vals = list(res["by_height"].values())
    res["all_heights_rad_s"] = None if None in vals else min(vals)
    # Post-hoc diagnostic (added after a one-trial smoke showed a static gravity sag of about
    # 0.1 rad in the held reach): same test with the arm_settle gate waived.
    res["by_height_ignoring_arm_settle"] = {}
    for h in ar["heights"]:
        best = None
        for r in rates:
            c = next(x for x in cells if x["name"] == f"armrate_h{h:.2f}_r{r:.1f}")
            others = set(c["fail_counts"]) - {"arm_settle"}
            if not c["complete"] or others:
                break
            best = r
        res["by_height_ignoring_arm_settle"][f"{h:.2f}"] = best
    v2 = list(res["by_height_ignoring_arm_settle"].values())
    res["all_heights_ignoring_arm_settle_rad_s"] = None if None in v2 else min(v2)
    res["pose"] = ar["pose"]
    res["controller"] = "plain WBCLocomotion arm PD, no gravity feed-forward"
    res["cells"] = {
        c["name"]: {"n_trials": c["n_trials"], "n_stable": c["n_stable"], "wilson95": c["wilson95"]}
        for c in cells
        if c["kind"] == "arm_rate"
    }
    return res


def summarize(cfg: dict, out: pathlib.Path) -> None:
    cells = []
    for c in conditions(cfg):
        paths = [trial_path(out, c, s) for s in cfg["seeds"]]
        crit = [json.loads(p.read_text())["criteria"] for p in paths if p.exists()]
        n, k = len(crit), sum(x["stable"] for x in crit)
        fail_counts: dict[str, int] = {}
        for x in crit:
            for f in x["fails"]:
                fail_counts[f] = fail_counts.get(f, 0) + 1

        def worst(f, crit=crit):
            v = [f(x) for x in crit]
            v = [t for t in v if t is not None]
            return max(v) if v else None

        margins = [x["com_margin"]["min_m"] for x in crit if x["com_margin"]["min_m"] is not None]
        s0 = c["segments"][0]
        cells.append(
            {
                "name": c["name"],
                "kind": c["kind"],
                "arms": c["arms"],
                "segments": c["segments"],
                "push": c["push"],
                "height": s0["height"],
                "pitch": s0["rpy"][1],
                "complete": n == len(cfg["seeds"]),
                "n_trials": n,
                "n_stable": k,
                "wilson95": wilson(k, n),
                "fail_counts": fail_counts,
                "worst": {
                    "height_error_m": worst(lambda x: x["height_error_max"]),
                    "rpy_waist_error_rad": worst(lambda x: x["rpy_waist_error_commanded_max"]),
                    "torque_ratio": worst(lambda x: max(x["torque_ratio"].values())),
                    "torque_ratio_by_group": {
                        g: worst(lambda x, g=g: x["torque_ratio"][g]) for g in GROUPS
                    },
                    "com_margin_min_m": min(margins, default=None),
                    "foot_loss_s": worst(lambda x: x["foot_support"]["max_continuous_s"]),
                    "undesired_contact_s": worst(
                        lambda x: x["undesired_ground_contact"]["max_continuous_s"]
                    ),
                    "joint_limit_s": worst(lambda x: x["joint_limit"]["total_s"]),
                    "feet_slip_m": worst(lambda x: x["feet_slip_max_m"]),
                    "push_recovery_s": worst(lambda x: x["push_recovery_s"]),
                    "arm_lag_max_rad": worst(
                        lambda x: x["arm_motion"]["lag_max_rad"] if x["arm_motion"] else None
                    ),
                    "tilt_latches": sum(x["tilt_latch"] is not None for x in crit),
                },
            }
        )
    rejected = []
    for p in cfg["rejected_probe_pitches"]:
        try:
            WBCLocomotion.from_cache().set_posture(cfg["nominal_height"], (0, p, 0))
            rejected.append({"pitch": p, "rejected": False})
        except ValueError as e:
            rejected.append({"pitch": p, "rejected": True, "reason": str(e)})
    (out / "summary.json").write_text(
        json.dumps({"cells": cells, "rejected_commands": rejected}, indent=1)
    )
    grid = {}
    for arms in cfg["arm_poses"]:
        grid[arms] = {
            "axes": {"height_m": cfg["joint_heights"], "pitch_rad": cfg["joint_pitches"]},
            "cells": {
                f"h{c['height']:.2f}_p{c['pitch']:+.1f}": {
                    "n_trials": c["n_trials"],
                    "n_stable": c["n_stable"],
                    "wilson95": c["wilson95"],
                    "fail_counts": c["fail_counts"],
                    "passes": c["complete"] and c["n_stable"] == c["n_trials"],
                }
                for c in cells
                if c["kind"] == "joint" and c["arms"] == arms
            },
        }
    envelope = {
        "protocol": {
            "config": cfg["name"],
            "seeds": cfg["seeds"],
            "cycles_per_trial": cfg["cycles"],
            "gates": cfg["gates"],
            "rates": {"height_m_s": cfg["rate_height"], "rpy_rad_s": cfg["rate_rpy"]},
            "push_s": cfg["push_s"],
            "arm_poses_rad": cfg["arm_poses"],
        },
        "meaning": (
            "Measured cells only. A cell passes when every trial of it met every pre-registered "
            "gate (no tilt latch, no sustained undesired ground contact, foot support kept, "
            "height and commanded-rpy tracking, joint-limit contact, CoM margin, torque ratio, "
            "slip, return to nominal; push cells also recover). The counts and Wilson 95% "
            "intervals are the evidence; with 5 trials a perfect cell has a lower bound of about "
            "0.57. Nothing here is labelled safe."
        ),
        "input_bound_m": list(g1_wbc.HEIGHT_RANGE),
        "interpolation_rule": (
            "A requested (height, pitch) with a given arm pose may be treated as inside the "
            "envelope only if it lies within the convex region spanned by passing cells of that "
            "arm pose AND every grid cell neighbouring the request (the corners of the grid "
            "rectangle that contains it) passes. Otherwise it is unmeasured and not executable "
            "autonomously. Other arm poses, roll, yaw and combinations not in the grid are "
            "unmeasured."
        ),
        "roll": (
            "Excluded from the executable region: roll is weakly tracked (see probe cells); "
            "executable roll is fixed at 0."
        ),
        "world_frame_torso_attitude": (
            "recorded per segment in the trial files (rpy_world_achieved); not a gate"
        ),
        "arm_rate_max": arm_rate_max(cfg, cells),
        "grid": grid,
        "other_cells": [c for c in cells if c["kind"] != "joint"],
        "rejected_by_input_bound": rejected,
    }
    (out / "envelope.json").write_text(json.dumps(envelope, indent=1))
    for c in cells:
        print(
            c["name"],
            f"{c['n_stable']}/{c['n_trials']}",
            round(c["wilson95"][0], 2),
            c["fail_counts"],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 seed, a few cells; writes *_smoke")
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text())
    if a.smoke:
        cfg = {**cfg, **cfg["smoke"], "name": cfg["name"] + "_smoke"}
    out = ROOT / "results" / cfg["name"]
    if a.summarize:
        summarize(cfg, out)
        return
    (out / "trials").mkdir(parents=True, exist_ok=True)
    i, n = map(int, a.shard.split("/"))
    jobs = [(c, s) for c in conditions(cfg) for s in cfg["seeds"]][i::n]
    meta = out / "meta.json"
    if not meta.exists():
        import onnxruntime

        meta.write_text(
            json.dumps(
                {
                    "git_sha": git_sha(),
                    "config": cfg,
                    "python": platform.python_version(),
                    "mujoco": mujoco.__version__,
                    "numpy": np.__version__,
                    "onnxruntime": onnxruntime.__version__,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=1,
            )
        )
    for c, s in jobs:
        p = trial_path(out, c, s)
        if p.exists():
            continue
        t = time.time()
        r = run_trial(cfg, c, s)
        r["wall_s"] = time.time() - t
        p.write_text(json.dumps(r, indent=1))
        k = r["criteria"]
        status = "stable" if k["stable"] else "FAIL " + ",".join(k["fails"])
        print(c["name"], s, status, f"{r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
