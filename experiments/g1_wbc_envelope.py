"""Measure what the pretrained Decoupled WBC can actually track when its posture inputs move.

    python experiments/g1_wbc_envelope.py experiments/configs/g1_wbc_envelope.json [--shard i/n]
    python experiments/g1_wbc_envelope.py experiments/configs/g1_wbc_envelope.json --summarize

The free-standing G1 (dream/sim/g1_mobile.py) stands at zero velocity while the base-height and
torso roll/pitch/yaw commands (WBCLocomotion.set_posture, upstream contract in g1_wbc.py) are
ramped from nominal at a fixed rate, held, and returned. Optional condition: both arms held
forward (arm_target) to load the centre of mass. Ground truth (pelvis and torso pose, contacts,
CoM, actuator commands) is read here only, as the evaluator. The summary applies the gates
pre-registered in the config and writes ``envelope.json``, the measured feasible region the
posture planner reads as its frozen calibration bound. Commands outside the safety bound
(g1_wbc.HEIGHT_RANGE / RPY_RANGE) are rejected by the controller; such conditions are recorded
as rejections, not run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.robot import g1_wbc  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1  # noqa: E402

TICK = g1_wbc.DT_PHYSICS * g1_wbc.DECIMATION  # 0.02 s
GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}


def conditions(cfg: dict) -> list[dict]:
    """Every (name, height, rpy, arms) condition of the sweep, in a fixed order."""
    nom = cfg["nominal_height"]
    out = []

    def add(h, rpy, arms):
        tag = f"h{h:.2f}_r{rpy[0]:+.2f}_p{rpy[1]:+.2f}_y{rpy[2]:+.2f}" + ("_arms" if arms else "")
        out.append({"name": tag, "height": h, "rpy": list(rpy), "arms": arms})

    for h in cfg["heights"]:
        add(h, (0, 0, 0), False)
    for p in cfg["pitches"]:
        if p != 0:
            add(nom, (0, p, 0), False)
    for r in cfg["rolls"]:
        add(nom, (r, 0, 0), False)
    for y in cfg["yaws"]:
        add(nom, (0, 0, y), False)
    for h in cfg["arms_heights"]:
        add(h, (0, 0, 0), True)
    for p in cfg["arms_pitches"]:
        add(nom, (0, p, 0), True)
    return out


def rpy_of(R: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll) (Pinocchio matrixToRpy)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
    return np.array([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


def yaw_matrix(R: np.ndarray) -> np.ndarray:
    y = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Probe:
    """Evaluator-side measurements, sampled once per policy tick."""

    def __init__(self, env: G1MobileEnv):
        m = env.model
        self.env = env
        self.pelvis, self.torso = m.body("pelvis").id, m.body("torso_link").id
        self.feet = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]
        self.floor = m.geom("floor").id
        self.limit = np.abs(m.jnt_actfrcrange[env.model.actuator_trnid[:, 0]]).max(axis=1)
        self.rows: list[dict] = []

    def support_margin(self) -> float | None:
        """Signed horizontal distance (m) from the CoM to the support polygon (convex hull of
        floor contacts); positive inside, None with fewer than 3 contact points."""
        d, m = self.env.data, self.env.model
        pts = []
        for c in d.contact[: d.ncon]:
            g = (c.geom1, c.geom2)
            if self.floor in g:
                other = g[1] if g[0] == self.floor else g[0]
                if m.geom_bodyid[other] in self.feet:
                    pts.append(c.pos[:2].copy())
        if len(pts) < 3:
            return None
        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = ConvexHull(np.array(pts))
        except QhullError:
            return None
        com = d.subtree_com[1][:2]
        # hull.equations: n . x + b <= 0 inside; margin = min over facets of -(n . x + b)
        return float(np.min(-(hull.equations[:, :2] @ com + hull.equations[:, 2])))

    def sample(self, t: float, phase: str) -> None:
        d = self.env.data
        Rp, Rt = d.xmat[self.pelvis].reshape(3, 3), d.xmat[self.torso].reshape(3, 3)
        ratio = np.abs(d.ctrl) / self.limit
        self.rows.append(
            {
                "t": t,
                "phase": phase,
                "z": float(d.xpos[self.pelvis][2]),
                "torso_z": float(d.xpos[self.torso][2]),
                "rpy_world": rpy_of(yaw_matrix(Rp).T @ Rt).tolist(),  # gravity-aligned heading
                "rpy_waist": rpy_of(Rp.T @ Rt).tolist(),  # upstream FK definition (waist chain)
                "ratio": {k: float(ratio[s].max()) for k, s in GROUPS.items()},
                "margin": self.support_margin(),
                "feet_xy": [d.xpos[b][:2].tolist() for b in self.feet],
                "hand_x": [float(d.xpos[self.env.model.body(n).id][0] - d.xpos[self.pelvis][0])
                           for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")],
            }
        )  # fmt: skip


def run_trial(cfg: dict, cond: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env = G1MobileEnv(build_mobile_g1(head_camera=None), rng, head_camera=None)
    env.reset(0.0, 0.0, 0.0)
    env.data.qpos[env.qadr] += rng.normal(scale=cfg["joint_noise"], size=len(env.qadr))
    mujoco.mj_forward(env.model, env.data)
    ctrl = WBCLocomotion.from_cache()
    probe = Probe(env)
    nom_h, nom_rpy = cfg["nominal_height"], np.zeros(3)
    rate_h, rate_r = cfg["rate_height"], cfg["rate_rpy"]
    ctrl.set_posture(nom_h, nom_rpy, max_height_rate=rate_h, max_rpy_rate=rate_r)
    fell = None
    t0 = [0.0]

    def run(seconds: float, phase: str) -> bool:
        nonlocal fell
        for _ in range(round(seconds / TICK)):
            env.step(ctrl)
            probe.sample(env.time - t0[0], phase)
            low = probe.rows[-1]["torso_z"] < cfg["fall_torso_z"]
            if ctrl.stopped or low:
                fell = ctrl.stopped or f"torso z {probe.rows[-1]['torso_z']:.2f} m"
                return False
        return True

    ok = run(cfg["settle_s"], "settle")
    if ok and cond["arms"]:
        ctrl.arm_target = np.array(cfg["arms_forward"], float)
        ok = run(cfg["arms_s"], "arms")
    base = [r for r in probe.rows if r["phase"] in ("settle", "arms")][-25:]
    nominal = {
        "z": float(np.mean([r["z"] for r in base])),
        "rpy_world": np.mean([r["rpy_world"] for r in base], axis=0).tolist(),
    }
    ramp_out = max(
        abs(cond["height"] - nom_h) / rate_h, float(np.max(np.abs(cond["rpy"]))) / rate_r
    )
    if ok:
        ctrl.set_posture(cond["height"], cond["rpy"])
        ok = run(ramp_out + cfg["hold_s"], "out")
    if ok:
        ctrl.set_posture(nom_h, nom_rpy)
        ok = run(ramp_out + cfg["return_s"], "back")
    return {
        "condition": cond,
        "seed": seed,
        "fell": fell,
        "ramp_s": ramp_out,
        "nominal": nominal,
        "metrics": metrics(cfg, cond, probe.rows, nominal, ramp_out) if not fell else None,
        "rows_fall": probe.rows[-5:] if fell else None,
    }


def metrics(cfg: dict, cond: dict, rows: list[dict], nominal: dict, ramp_s: float) -> dict:
    out = [r for r in rows if r["phase"] == "out"]
    hold = [r for r in out if r["t"] >= out[0]["t"] + ramp_s]  # after the commanded ramp ends
    last = hold[-int(1.0 / TICK) :]  # last 1 s of the hold: steady state
    back = [r for r in rows if r["phase"] == "back"]
    tail = back[-int(1.0 / TICK) :]
    cmd_rpy = np.array(cond["rpy"])
    z = np.mean([r["z"] for r in last])
    rw = np.mean([r["rpy_world"] for r in last], axis=0)
    rt = np.mean([r["rpy_waist"] for r in last], axis=0)
    z_final = z
    settle = None
    for i in range(len(hold)):  # first time the height stays within 0.02 m of its final value
        if all(abs(q["z"] - z_final) < 0.02 for q in hold[i:]):
            settle = hold[i]["t"] - hold[0]["t"]
            break
    margins = [r["margin"] for r in out if r["margin"] is not None]
    f0 = np.array(hold[0]["feet_xy"])
    slip = max(float(np.linalg.norm(np.array(r["feet_xy"]) - f0, axis=1).max()) for r in hold)
    z_back = np.mean([r["z"] for r in tail])
    rb = np.mean([r["rpy_world"] for r in tail], axis=0)
    return {
        "height_cmd": cond["height"],
        "height_achieved": float(z),
        "height_error": float(z - cond["height"]),
        "height_error_vs_nominal_offset": float(
            z - cond["height"] - (nominal["z"] - cfg["nominal_height"])
        ),
        "rpy_cmd": cmd_rpy.tolist(),
        "rpy_world_achieved": rw.tolist(),
        "rpy_world_error": (rw - cmd_rpy).tolist(),
        "rpy_world_max_abs_error": float(np.max(np.abs(rw - cmd_rpy))),
        "rpy_waist_achieved": rt.tolist(),
        "rpy_waist_max_abs_error": float(np.max(np.abs(rt - cmd_rpy))),
        "torque_ratio_max": {k: float(max(r["ratio"][k] for r in out)) for k in GROUPS},
        "com_margin_min": float(min(margins)) if margins else None,
        "com_margin_mean_hold": float(
            np.mean([r["margin"] for r in hold if r["margin"] is not None])
        )
        if any(r["margin"] is not None for r in hold)
        else None,
        "feet_slip_max": slip,
        "settle_s": settle,
        "return_height_error": float(z_back - nominal["z"]),
        "return_rpy_error": float(np.max(np.abs(rb - np.array(nominal["rpy_world"])))),
        "min_torso_z": float(min(r["torso_z"] for r in rows)),
        "max_hand_forward_x": float(max(max(r["hand_x"]) for r in rows)),
    }


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def trial_path(out: pathlib.Path, cond: dict, seed: int) -> pathlib.Path:
    return out / "trials" / f"{cond['name']}_s{seed}.json"


def passes(cfg: dict, trials: list[dict]) -> dict:
    """Pre-registered gates on all seeds of one condition."""
    g = cfg["gates"]
    ms = [t["metrics"] for t in trials if t["metrics"]]
    falls = sum(t["fell"] is not None for t in trials)
    res = {"n": len(trials), "falls": falls}
    if falls or len(ms) != len(trials):
        res["feasible"] = res["feasible_any_attitude"] = False
        return res
    res["height_error_max"] = max(abs(m["height_error"]) for m in ms)
    res["rpy_error_max"] = max(m["rpy_world_max_abs_error"] for m in ms)
    res["torque_ratio_max"] = max(max(m["torque_ratio_max"].values()) for m in ms)
    res["torque_ratio_by_group"] = {k: max(m["torque_ratio_max"][k] for m in ms) for k in GROUPS}
    margins = [m["com_margin_min"] for m in ms if m["com_margin_min"] is not None]
    res["com_margin_min"] = min(margins) if margins else None
    res["feet_slip_max"] = max(m["feet_slip_max"] for m in ms)
    res["return_ok"] = all(
        abs(m["return_height_error"]) < g["return_height_tol_m"]
        and m["return_rpy_error"] < g["return_rpy_tol_rad"]
        for m in ms
    )
    res["gate_failures"] = [
        name
        for name, bad in (
            ("height_error", res["height_error_max"] > g["height_tol_m"]),
            ("rpy_error", res["rpy_error_max"] > g["rpy_tol_rad"]),
            ("torque_ratio", res["torque_ratio_max"] >= g["torque_ratio_max"]),
            (
                "com_margin",
                res["com_margin_min"] is None or res["com_margin_min"] < g["com_margin_min_m"],
            ),
            ("feet_slip", res["feet_slip_max"] > g["feet_slip_max_m"]),
            ("return", not res["return_ok"]),
        )
        if bad
    ]
    res["feasible"] = not res["gate_failures"]
    # Same gates without the torso-attitude one: the policy may squat correctly (height gate)
    # while its torso leans; a planner that tolerates the lean can use the wider region.
    res["feasible_any_attitude"] = res["gate_failures"] in ([], ["rpy_error"])
    return res


def contiguous(vals: list[float], ok: dict[float, bool], nominal: float) -> list[float]:
    """Feasible interval around the nominal value: walk outward until the first infeasible."""
    lo = hi = nominal
    for v in sorted(x for x in vals if x < nominal)[::-1]:
        if not ok.get(v, False):
            break
        lo = v
    for v in sorted(x for x in vals if x > nominal):
        if not ok.get(v, False):
            break
        hi = v
    return [lo, hi]


def summarize(cfg: dict, out: pathlib.Path) -> None:
    conds = conditions(cfg)
    summary, by_name = {}, {}
    for c in conds:
        ts = [
            json.loads(trial_path(out, c, s).read_text())
            for s in cfg["seeds"]
            if trial_path(out, c, s).exists()
        ]
        if len(ts) != len(cfg["seeds"]):
            summary[c["name"]] = {"condition": c, "incomplete": True, "n": len(ts)}
            continue
        summary[c["name"]] = {"condition": c, **passes(cfg, ts)}
        by_name[c["name"]] = summary[c["name"]]
    rejected = []
    for p in cfg["rejected_probe_pitches"]:
        try:
            WBCLocomotion.from_cache().set_posture(cfg["nominal_height"], (0, p, 0))
            rejected.append({"pitch": p, "rejected": False})
        except ValueError as e:
            rejected.append({"pitch": p, "rejected": True, "reason": str(e)})
    (out / "summary.json").write_text(
        json.dumps({"conditions": summary, "rejected_commands": rejected}, indent=1)
    )

    def feas(arms: bool, key: str, idx=None, flag: str = "feasible"):
        d = {}
        for c in conds:
            s = by_name.get(c["name"])
            if s is None or c["arms"] != arms:
                continue
            others = [v for j, v in enumerate(c["rpy"]) if j != idx]
            if key == "height":
                if all(v == 0 for v in c["rpy"]):
                    d[c["height"]] = s[flag]
            elif c["height"] == cfg["nominal_height"] and not any(others):
                d[c["rpy"][idx]] = s["feasible"]
        return d

    env = {"gates": cfg["gates"], "arms_forward": cfg["arms_forward"], "seeds": cfg["seeds"]}
    for arms, tag in ((False, "standing"), (True, "arms_extended")):
        fh = feas(arms, "height")
        fa = feas(arms, "height", flag="feasible_any_attitude")
        region = {
            "height_m": contiguous(list(fh), fh, cfg["nominal_height"]) if fh else None,
            "height_m_any_torso_attitude": contiguous(list(fa), fa, cfg["nominal_height"])
            if fa
            else None,
            "height_tested": {f"{k:.2f}": v for k, v in sorted(fh.items())},
        }
        for i, nm in enumerate(("roll", "pitch", "yaw")):
            fr = feas(arms, "rpy", i)
            fr[0.0] = fr.get(0.0, fh.get(cfg["nominal_height"], False))
            region[nm + "_rad"] = contiguous(list(fr), fr, 0.0)
            region[nm + "_tested"] = {f"{k:+.2f}": v for k, v in sorted(fr.items())}
        env[tag] = region
    env["rejected_by_safety_bound"] = rejected
    env["note"] = (
        "Intervals are the contiguous feasible range around nominal. Height and rpy are "
        "swept independently (no joint height x pitch sweep): a planner must not assume the "
        "product of the intervals is feasible."
    )
    (out / "envelope.json").write_text(json.dumps(env, indent=1))
    print(
        json.dumps({k: v for k, v in env.items() if k in ("standing", "arms_extended")}, indent=1)
    )
    for n, s in summary.items():
        print(n, "feasible" if s.get("feasible") else s.get("gate_failures", s.get("falls", "n/a")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 heights, standing only")
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text())
    out = ROOT / "results" / cfg["name"]
    if a.summarize:
        summarize(cfg, out)
        return
    if a.smoke:
        small = {"pitches": [0], "rolls": [], "yaws": [], "arms_heights": [], "arms_pitches": []}
        h = cfg["heights"]
        cfg = {**cfg, **small, "seeds": cfg["seeds"][:1], "heights": h[:1] + h[5:6]}
        out = ROOT / "results" / (cfg["name"] + "_smoke")
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
        m = r["metrics"]
        print(c["name"], s, "FELL " + r["fell"] if r["fell"] else
              f"dz={m['height_error']:+.3f} rpy_err={m['rpy_world_max_abs_error']:.3f} "
              f"tau={max(m['torque_ratio_max'].values()):.2f} margin={m['com_margin_min']}",
              f"{r['wall_s']:.0f}s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
