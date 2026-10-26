"""Measure the physically validated envelope of the SONIC backend with the frozen WBC protocol.

    python experiments/sonic_envelope.py experiments/configs/sonic_envelope.json --make-refs
    python experiments/sonic_envelope.py experiments/configs/sonic_envelope.json --driver
    python experiments/sonic_envelope.py experiments/configs/sonic_envelope.json --summarize
    python experiments/sonic_envelope.py experiments/configs/sonic_envelope.json --smoke --driver

Same protocol and gates as experiments/g1_wbc_envelope.py (config g1_wbc_envelope_v2): the Probe
sampler, the ``criteria`` gate function and the Wilson interval are IMPORTED from it, not copied.
The body is DREAM's 43-actuator G1 (``build_mobile_g1(hands=True)``, fingers open) driven by
``SonicBackend`` / ``SonicController`` (dream/robot/sonic.py), an unofficial Python/onnxruntime
port of gear_sonic_deploy that is not verified against the reference C++ binary.

Differences from the WBC runner, all stated in the config: (1) SONIC chooses its own motion
timing, so a posture segment is ``backend.execute`` (the planner motion plus the backend's 1 s
settle) followed by the 4 s hold; (2) intended knee contact in kneel segments is recorded
separately and is not "undesired ground contact"; (3) the tilt latch is the WBC latch value
(g1_wbc.FALL_TILT) applied by this evaluator; (4) ``joint_reference`` cells.
Ground truth (contacts, CoM, poses) is read here only, as the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
import time
import traceback

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.robot import g1_wbc, sonic  # noqa: E402
from dream.robot.wholebody import (  # noqa: E402
    ControllerCapabilities,
    StopEvent,
    WholeBodyTarget,
)
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1  # noqa: E402
from experiments.g1_wbc_envelope import (  # noqa: E402  (the WBC protocol's own functions)
    DIRS,
    GROUPS,
    TICK,
    Probe,
    criteria,
    wilson,
)

KNEE_BODIES = ("left_knee_link", "right_knee_link")
JOINT_GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}


# -- protocol ----------------------------------------------------------------------------------
def conditions(cfg: dict) -> list[dict]:
    """Every cell of the frozen protocol, in a fixed order."""
    nom, out = cfg["nominal_height"], []

    def add(kind, name, segments, cycles, push=None, ref=None):
        out.append(
            {
                "arm_rate": None,
                "name": name,
                "kind": kind,
                "arms": "planner",
                "segments": [{"height": h, "rpy": list(r)} for h, r in segments],
                "cycles": cycles,
                "push": push,
                "ref": ref,
            }
        )

    for h in cfg["static_heights"]:
        add("static", f"static_h{h:.2f}", [(h, (0, 0, 0))], cfg["cycles"])
    for t in cfg["transitions"]:
        add("transition", "trans_" + t["name"], [(h, (0, 0, 0)) for h in t["heights"]], 1)
    for h in cfg["push_heights"]:
        for d in DIRS:
            for n in cfg["push_newtons"]:
                add(
                    "push",
                    f"push_h{h:.2f}_{d}_{n:g}N",
                    [(h, (0, 0, 0))],
                    1,
                    {"dir": d, "newtons": n},
                )
    for r in cfg.get("references", []):
        add(
            "joint_reference",
            f"ref_h{r['height']:.2f}",
            [(r["pelvis_height"], tuple(r["torso_rpy"]))],
            1,
            None,
            r,
        )
    return out


def kneel_height(h: float) -> bool:
    """Planner kneelTwoLeg request (sonic.plan_posture): below the squat range."""
    return h < sonic.SQUAT_HEIGHT_RANGE[0] - 1e-9


# -- evaluator-side sampling -------------------------------------------------------------------
class SonicProbe(Probe):
    """The WBC Probe on the 43-actuator model, plus the separately recorded knee contact."""

    def __init__(self, env: G1MobileEnv, cfg: dict, grip: dict, arm: str | None):
        super().__init__(env, cfg)
        assert np.array_equal(env.body_actuators, np.arange(29))  # body actuators first
        self.jlo, self.jhi = self.jlo[:29], self.jhi[:29]  # limit stays 43 long: it divides d.ctrl
        self.jnames = self.jnames[:29]
        m = env.model
        self.knees = [m.body(n).id for n in KNEE_BODIES]
        self.kneel_ok = False
        self._knee: tuple[list[bool], list[np.ndarray]] = ([False, False], [])
        self.grip, self.arm = grip, arm
        self.wrist = None if arm is None else m.body(f"{arm}_wrist_yaw_link").id

    def contacts(self):
        d, m = self.env.data, self.env.model
        foot, bad, pts = [False, False], set(), []
        knee, knee_pts = [False, False], []
        for c in d.contact[: d.ncon]:
            g = (c.geom1, c.geom2)
            if self.floor not in g:
                continue
            body = int(m.geom_bodyid[g[1] if g[0] == self.floor else g[0]])
            if body in self.feet:
                foot[self.feet.index(body)] = True
                pts.append(c.pos[:2].copy())
            elif body in self.knees and self.kneel_ok:
                knee[self.knees.index(body)] = True
                knee_pts.append(c.pos[:2].copy())
            else:
                bad.add(m.body(body).name)
        self._knee = (knee, knee_pts)
        return foot, sorted(bad), pts

    def sample(self, t: float, seg: str, push_on: bool) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            super().sample(t, seg, push_on)
        d, env = self.env.data, self.env
        row = self.rows[-1]
        foot, _, pts = self.contacts()  # recomputes the same tick; also fills self._knee
        knee, knee_pts = self._knee
        row["knee"] = knee
        row["feet_adj"] = [bool(f or k) for f, k in zip(foot, knee, strict=True)]
        row["margin_adj"] = self.margin(pts + knee_pts) if knee_pts else row["margin"]
        row["q"] = d.qpos[env.qadr].tolist()
        row["quat"] = d.qpos[3:7].tolist()
        row["tilt"] = float(g1_wbc.tilt(d.qpos[3:7]))
        if self.wrist is not None:
            row["hand_w"] = (
                d.xpos[self.wrist] + d.xmat[self.wrist].reshape(3, 3) @ self.grip[self.arm]
            ).tolist()


class EvalMonitor:
    """Samples every criterion once per policy tick; stops on the tilt latch and on sustained
    undesired contact (the WBC runner's abort rule)."""

    def __init__(self, env, probe: SonicProbe, cfg: dict):
        self.env, self.probe, self.cfg = env, probe, cfg
        self.t_start = env.time
        self.seg, self.push_on = "settle", False
        self.collapse: dict = {"stopped": None, "contact": None}
        self.window: list[float] = []

    def __call__(self, obs):
        pr = self.probe
        pr.sample(self.env.time - self.t_start, self.seg, self.push_on)
        if self.push_on:
            t = pr.rows[-1]["t"]
            self.window[:] = [self.window[0] if self.window else t, t]
        if g1_wbc.tilt(obs.prop.quat) > g1_wbc.FALL_TILT:
            self.collapse["stopped"] = (
                f"tilt latch: pelvis tilt {np.degrees(g1_wbc.tilt(obs.prop.quat)):.0f} deg"
            )
            return StopEvent(obs.t, self.collapse["stopped"])
        n = round(self.cfg["abort_contact_s"] / TICK)
        recent = pr.rows[-n:]
        if len(recent) == n and all(r["bad"] for r in recent):
            self.collapse["contact"] = recent[-1]["bad"]
            return StopEvent(obs.t, "undesired ground contact")
        return None


# -- capabilities for the measurement (not the product limits) ----------------------------------
def measuring_capabilities(cfg: dict) -> ControllerCapabilities:
    lo, hi = cfg["request_height_range"]
    tol = sonic.RPY_TOLERANCE
    return ControllerCapabilities(
        "gear_sonic",
        frozenset({"posture", "joint_reference"}),
        (lo, hi),
        (-tol,) * 3,
        (tol,) * 3,
        measured=True,
        source="sonic_envelope measurement: full planner request range, not a limit",
        note=sonic.LABEL,
    )


_SESSIONS: sonic.SonicPolicy | None = None


def new_policy() -> sonic.SonicPolicy:
    """A fresh policy state on shared onnxruntime sessions (loaded once per process)."""
    global _SESSIONS
    if _SESSIONS is None:
        _SESSIONS = sonic.SonicPolicy.from_cache()
        return _SESSIONS
    s = _SESSIONS
    return sonic.SonicPolicy(s.enc, s.dec, s.pl_sess)


# -- references ----------------------------------------------------------------------------------
def make_references(cfg: dict) -> list[dict]:
    """Offline whole-body reach references from PosturePlanner (import only), wide hypothetical
    bounds, no obstacles, stance fixed at the robot's start pose."""
    from dream.robot.g1_posture import PlannerConfig, PosturePlanner

    rc = cfg["reference_generation"]
    caps = ControllerCapabilities.unmeasured(
        "hypothetical_wide",
        tuple(rc["pelvis_height_range"]),
        (0.0, rc["pitch_range"][0], 0.0),
        (0.0, rc["pitch_range"][1], 0.0),
    )
    planner = PosturePlanner(caps, PlannerConfig())
    qadr = planner.pm.qadr
    out = []
    for h in rc["target_heights"]:
        chosen = None
        tried = []
        for x, y in rc["target_xy_candidates"]:
            res = planner.plan(np.array([x, y, h]), stances=[tuple(rc["stance"])])
            tried.append({"xy": [x, y], "feasible": res.feasible, "reasons": res.reasons()})
            if res.feasible:
                chosen = (x, y, res.plans[0])
                break
        if chosen is None:
            out.append({"height": h, "feasible": False, "tried": tried})
            continue
        x, y, p = chosen
        q = p.target.q
        q_body = q[qadr]
        q0 = planner.pm.q_nominal[qadr]
        w, qx, qy, qz = q[3:7]
        pitch = float(np.arcsin(np.clip(2 * (w * qy - qz * qx), -1, 1)))
        out.append(
            {
                "height": h,
                "feasible": True,
                "target": [x, y, h],
                "arm": p.arm,
                "stance": list(p.stance),
                "q_full": q.tolist(),
                "q_body": q_body.tolist(),
                "q_stand_body": q0.tolist(),
                "pelvis_height": p.pelvis_height,
                "torso_rpy": list(p.torso_rpy),
                "ref_pelvis_pitch_rad": pitch,
                "ref_pelvis_tilt_rad": p.metrics["pelvis_tilt"],
                "hand_residual_m": p.metrics["hand_residual"],
                "planner_metrics": p.metrics,
                "tried": tried,
            }
        )
    return out


def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def reference_frames(ref: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Times (n,) and body q (n, 29): minimum-jerk from the standing q to the reference q over
    ``approach_s``, then held for ``hold_s``."""
    rc = cfg["reference_execution"]
    q0, q1 = np.array(ref["q_stand_body"]), np.array(ref["q_body"])
    t = np.arange(0.0, rc["approach_s"] + rc["hold_s"] + 1e-9, 0.02)
    s = min_jerk(np.clip(t / rc["approach_s"], 0, 1))
    return t, q0[None] + s[:, None] * (q1 - q0)[None]


# -- one trial -----------------------------------------------------------------------------------
class Trial:
    def __init__(self, cfg: dict, cond: dict, seed: int):
        self.cfg, self.cond, self.seed = cfg, cond, seed
        self.rng = np.random.default_rng(seed)
        self.env = G1MobileEnv(build_mobile_g1(head_camera=None, hands=True), self.rng,
                               head_camera=None)  # fmt: skip
        names = [self.env.model.joint(j).name for j in self.env.model.actuator_trnid[self.env.body_actuators, 0]]
        assert tuple(names) == sonic.BODY_JOINT_NAMES
        self.env.reset(0.0, 0.0, 0.0)
        self.env.data.qpos[self.env.qadr] += self.rng.normal(
            scale=cfg["joint_noise"], size=len(self.env.qadr)
        )
        mujoco.mj_forward(self.env.model, self.env.data)
        from dream.robot.g1_posture import _read_anatomy

        self.grip = _read_anatomy().grip
        ref = cond.get("ref")
        self.probe = SonicProbe(self.env, cfg, self.grip, ref["arm"] if ref else None)
        self.probe.ctrl = type("ArmShim", (), {"arm_target": np.zeros(14)})()  # arm gate unused
        self.mon = EvalMonitor(self.env, self.probe, cfg)
        self.backend = sonic.SonicBackend(
            self.env, new_policy(), measuring_capabilities(cfg), arms="report"
        )
        self.reports: list[dict] = []
        self.ok = True

    # tick helper for holds (the same order as SonicBackend._step)
    def hold(self, seconds: float, push: dict | None = None) -> None:
        env, b = self.env, self.backend
        t0 = env.time
        for i in range(round(seconds / TICK)):
            if not self.ok:
                return
            on = push is not None and push["t0"] <= i * TICK < push["t0"] + self.cfg["push_s"]
            self.mon.push_on = on
            if on:
                vx, vy = DIRS[push["dir"]]
                env.data.xfrc_applied[self.probe.pelvis, :3] = [
                    vx * push["newtons"],
                    vy * push["newtons"],
                    0,
                ]
            env.step(b.controller)
            env.data.xfrc_applied[self.probe.pelvis, :3] = 0
            ev = self.mon(b.observe())
            self.mon.push_on = False
            if ev is not None:
                b.hold(ev)
                self.ok = False
                return
        assert abs(env.time - t0 - seconds) < 1e-6 or not self.ok

    def execute(self, target: WholeBodyTarget) -> None:
        rep = self.backend.execute(target, self.mon)
        self.reports.append(
            {
                "requested": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                              for k, v in rep.requested.items() if k not in ("arm",)},
                "completed": rep.completed,
                "refused": rep.refused,
                "stops": [(s.t, s.reason) for s in rep.stops],
                "ticks": len(rep.trace),
            }
        )  # fmt: skip
        if rep.refused or rep.stops or not rep.completed:
            self.ok = False

    def posture(self, height: float) -> WholeBodyTarget:
        return WholeBodyTarget(np.zeros(36), (0.0, 0.0, 0.0), (), height, (0.0, 0.0, 0.0),
                               np.zeros(14))  # fmt: skip


def run_trial(cfg: dict, cond: dict, seed: int) -> dict:
    T = Trial(cfg, cond, seed)
    nom_h, mon, probe = cfg["nominal_height"], T.mon, T.probe
    mon.seg = "settle"
    T.execute(T.posture(nom_h))
    T.hold(cfg["settle_s"])
    base = [r for r in probe.rows if r["seg"] == "settle"][-25:]
    nominal = {
        "z": float(np.mean([r["z"] for r in base])),
        "rpy_waist": np.mean([r["rpy_waist"] for r in base], axis=0).tolist(),
        "rpy_world": np.mean([r["rpy_world"] for r in base], axis=0).tolist(),
        "feet_xy": base[-1]["feet_xy"],
    }
    plan, done = [], {}
    prev_h = nom_h
    ref_info = None
    for cyc in range(cond["cycles"]):
        steps = [(f"{cyc}:{i}", s) for i, s in enumerate(cond["segments"])]
        steps.append((f"{cyc}:back", {"height": nom_h, "rpy": [0.0, 0.0, 0.0]}))
        for tag, s in steps:
            if not T.ok:
                plan.append((tag, s, 1e6, 0.0, None))
                continue
            back = tag.endswith("back")
            mon.seg = tag
            probe.kneel_ok = kneel_height(s["height"]) or kneel_height(prev_h)
            hold_s = cfg["return_s"] if back else cfg["hold_s"]
            push = None
            if cond["push"] and not back:
                push = {**cond["push"], "t0": cfg["push_after_s"]}
            n0 = len(probe.rows)
            if cond["kind"] == "joint_reference" and not back:
                probe.kneel_ok = False
                ref = cond["ref"]
                times, frames = reference_frames(ref, cfg)
                rt = tuple((float(t), q) for t, q in zip(times, frames, strict=True))
                tgt = WholeBodyTarget(
                    np.zeros(36), (0.0, 0.0, 0.0), (), s["height"], tuple(s["rpy"]),
                    np.zeros(14), joint_reference=rt, joint_names=sonic.BODY_JOINT_NAMES,
                )  # fmt: skip
                t_exec0 = T.env.time
                T.execute(tgt)
                ref_info = {"t_exec0": t_exec0, "blend_s": T.reports[-1]["requested"].get("blend_s", 0.0),
                            "n0": n0}  # fmt: skip
            else:
                T.execute(T.posture(s["height"]))
                T.hold(hold_s, push)
            ntick = len(probe.rows) - n0
            if T.ok:
                plan.append((tag, s, max(0.0, ntick * TICK - hold_s), hold_s, push))
            else:
                plan.append((tag, s, 1e6, hold_s, push))
            prev_h = s["height"]
            done[tag] = T.ok
    collapse, window = mon.collapse, mon.window
    rows = probe.rows
    crit = criteria(cfg, cond, rows, nominal, plan, window, collapse)
    adj_rows = [{**r, "feet": r["feet_adj"], "margin": r["margin_adj"]} for r in rows]
    crit_adj = criteria(cfg, cond, adj_rows, nominal, plan, window, collapse)
    extra = extras(cfg, cond, rows, T, ref_info)
    if cond["kind"] == "joint_reference":
        for c in (crit, crit_adj):
            if extra["reference"].get("hand_error_fk_m") is None or (
                extra["reference"]["hand_error_fk_m"] > cfg["gates"]["hand_error_fk_m"]
            ):
                c["fails"].append("hand_error")
                c["stable"] = False
    return {
        "condition": {k: v for k, v in cond.items() if k != "ref"},
        "seed": seed,
        "collapse_stop": collapse,
        "nominal": nominal,
        "plan": [
            {"seg": t, "ramp_s": r, "hold_s": h, "cmd": s, "done": done.get(t, False)}
            for t, s, r, h, _ in plan
        ],
        "push_window": window,
        "criteria": crit,
        "criteria_kneel_support": {"fails": crit_adj["fails"], "stable": crit_adj["stable"]},
        "extras": extra,
        "reports": T.reports,
        "latency_ms": T.backend.policy.timer.summary(),
    }


def extras(cfg, cond, rows, T: Trial, ref_info) -> dict:
    rows_ = [r for r in rows if r["seg"] != "settle"]
    knee = [any(r["knee"]) for r in rows_]
    from experiments.g1_wbc_envelope import longest

    out = {
        "intended_knee_contact": {
            "max_continuous_s": longest(knee),
            "total_s": sum(knee) * TICK,
            "bodies_note": "knee links, recorded only while a kneel segment is active",
        },
        "tilt_max_rad": float(max((r["tilt"] for r in rows_), default=0.0)),
        "pelvis_z_final": rows_[-1]["z"] if rows_ else None,
        "reference": {},
    }
    if cond["kind"] != "joint_reference" or ref_info is None:
        return out
    ref = cond["ref"]
    seg = [r for r in rows_ if r["seg"] == "0:0"]
    rc = cfg["reference_execution"]
    t0 = ref_info["t_exec0"] + ref_info["blend_s"]
    times, frames = reference_frames(ref, cfg)
    ref_end_t = t0 + rc["approach_s"] + rc["hold_s"]
    # alignment of the reference clock to the trace is within one tick (documented)
    dyn = {g: 0.0 for g in JOINT_GROUPS}
    for r in seg:
        tr = r["t"] - ref_info["t_exec0"] - ref_info["blend_s"]
        if tr < 0 or tr > rc["approach_s"] + rc["hold_s"]:
            continue
        i = min(int(round(tr / 0.02)), len(frames) - 1)
        e = np.abs(np.array(r["q"]) - frames[i])
        for g, sl in JOINT_GROUPS.items():
            dyn[g] = max(dyn[g], float(e[sl].max()))
    # final hold: the last second of the 3 s hold (before the backend's extra 1 s settle)
    fin = [r for r in seg if ref_end_t - 1.0 < r["t"] <= ref_end_t] or seg[-25:]
    q_ref = np.array(ref["q_body"])
    q_ach = np.mean([r["q"] for r in fin], axis=0)
    err = np.abs(q_ach - q_ref)
    final = {
        g: {"max_abs_rad": float(err[sl].max()), "rms_rad": float(np.sqrt(np.mean(err[sl] ** 2)))}
        for g, sl in JOINT_GROUPS.items()
    }
    quat = np.mean([r["quat"] for r in fin], axis=0)
    quat = quat / np.linalg.norm(quat)
    pm = _planning_model()
    ks_ref = pm.kinematic_state(np.array(ref["q_full"][3:7]), q_ref)
    ks_ach = pm.kinematic_state(quat, q_ach)
    ks_ach_up = pm.kinematic_state(np.array([1.0, 0, 0, 0]), q_ach)
    ks_ref_up = pm.kinematic_state(np.array([1.0, 0, 0, 0]), q_ref)
    arm = ref["arm"]
    hand_fk = float(np.linalg.norm(ks_ach["hands"][arm] - ks_ref["hands"][arm]))
    hand_fk_joints = float(np.linalg.norm(ks_ach_up["hands"][arm] - ks_ref_up["hands"][arm]))
    hand_w = np.mean([r["hand_w"] for r in fin], axis=0)
    w, x, y, z = quat
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    out["reference"] = {
        "arm": arm,
        "target": ref["target"],
        "joint_error_final": final,
        "joint_error_dynamic_max_rad": dyn,
        "joint_tracking_blend_s": ref_info["blend_s"],
        "hand_error_fk_m": hand_fk,
        "hand_error_fk_joints_only_m": hand_fk_joints,
        "hand_error_fk_both_arms_m": {
            s: float(np.linalg.norm(ks_ach["hands"][s] - ks_ref["hands"][s])) for s in ks_ref["hands"]
        },
        "hand_error_world_to_target_m": float(np.linalg.norm(hand_w - np.array(ref["target"]))),
        "hand_world_achieved": hand_w.tolist(),
        "pelvis_height_ref_m": ref["pelvis_height"],
        "pelvis_height_achieved_m": float(np.mean([r["z"] for r in fin])),
        "pelvis_pitch_ref_rad": ref["ref_pelvis_pitch_rad"],
        "pelvis_pitch_achieved_rad": pitch,
        "pelvis_pitch_mismatch_rad": pitch - ref["ref_pelvis_pitch_rad"],
        "reference_com_margin_m": ks_ref["com_margin"],
    }
    return out


_PM = None


def _planning_model():
    global _PM
    if _PM is None:
        from dream.robot.g1_posture import PlanningModel

        _PM = PlanningModel()
    return _PM


# -- summary -----------------------------------------------------------------------------------
def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def trial_path(out: pathlib.Path, cond: dict, seed: int) -> pathlib.Path:
    return out / "trials" / f"{cond['name']}_s{seed}.json"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(cfg: dict, out: pathlib.Path) -> None:
    cells = []
    for c in conditions(cfg):
        paths = [trial_path(out, c, s) for s in cfg["seeds"]]
        trials = [json.loads(p.read_text()) for p in paths if p.exists()]
        crit = [t["criteria"] for t in trials]
        n, k = len(crit), sum(x["stable"] for x in crit)
        fail_counts: dict[str, int] = {}
        for x in crit:
            for f in x["fails"]:
                fail_counts[f] = fail_counts.get(f, 0) + 1
        fail_adj: dict[str, int] = {}
        for t in trials:
            for f in t["criteria_kneel_support"]["fails"]:
                fail_adj[f] = fail_adj.get(f, 0) + 1
        k_adj = sum(t["criteria_kneel_support"]["stable"] for t in trials)

        def worst(f, crit=crit):
            v = [f(x) for x in crit]
            v = [t for t in v if t is not None]
            return max(v) if v else None

        margins = [x["com_margin"]["min_m"] for x in crit if x["com_margin"]["min_m"] is not None]
        s0 = c["segments"][0]
        cell = {
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
            "kneel_support_adjusted": {
                "n_stable": k_adj, "wilson95": wilson(k_adj, n), "fail_counts": fail_adj,
            },  # fmt: skip
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
                "tilt_latches": sum(x["tilt_latch"] is not None for x in crit),
                "intended_knee_contact_s": max(
                    (t["extras"]["intended_knee_contact"]["total_s"] for t in trials), default=None
                ),
                "tilt_max_rad": max((t["extras"]["tilt_max_rad"] for t in trials), default=None),
            },
        }
        if c["kind"] == "joint_reference":
            refs = [t["extras"]["reference"] for t in trials if t["extras"]["reference"]]

            def agg(f, refs=refs):
                v = [f(r) for r in refs]
                return {"max": max(v), "mean": float(np.mean(v))} if v else None

            cell["reference"] = {
                "target": c["ref"]["target"] if c.get("ref") else None,
                "n_with_metrics": len(refs),
                "joint_error_final_max_rad": {
                    g: agg(lambda r, g=g: r["joint_error_final"][g]["max_abs_rad"])
                    for g in JOINT_GROUPS
                },
                "joint_error_dynamic_max_rad": {
                    g: agg(lambda r, g=g: r["joint_error_dynamic_max_rad"][g]) for g in JOINT_GROUPS
                },
                "hand_error_fk_m": agg(lambda r: r["hand_error_fk_m"]),
                "hand_error_fk_joints_only_m": agg(lambda r: r["hand_error_fk_joints_only_m"]),
                "hand_error_world_to_target_m": agg(lambda r: r["hand_error_world_to_target_m"]),
                "pelvis_height_error_m": agg(
                    lambda r: abs(r["pelvis_height_achieved_m"] - r["pelvis_height_ref_m"])
                ),
                "pelvis_pitch_mismatch_rad": agg(lambda r: abs(r["pelvis_pitch_mismatch_rad"])),
            }
        cells.append(cell)
    (out / "summary.json").write_text(json.dumps({"cells": cells}, indent=1))
    write_envelope(cfg, out, cells)
    for c in cells:
        print(c["name"], f"{c['n_stable']}/{c['n_trials']}", round(c["wilson95"][0], 2),
              c["fail_counts"])  # fmt: skip


def write_envelope(cfg: dict, out: pathlib.Path, cells: list[dict]) -> None:
    static = [c for c in cells if c["kind"] == "static"]
    heights = sorted((c["height"] for c in static), reverse=True)
    passes = {c["height"]: c["complete"] and c["n_stable"] == c["n_trials"] for c in static}
    # height_m rule (pre-registered): the longest contiguous run, in commanded height, of passing
    # static cells; ties go to the run that contains the highest height.
    runs, cur = [], []
    for h in heights:
        if passes[h]:
            cur.append(h)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    best = max(runs, key=lambda r: (len(r), r[0]), default=[])
    height_m = [min(best), max(best)] if best else [1.0, 0.0]  # empty: nothing is executable
    refs = [c for c in cells if c["kind"] == "joint_reference"]
    rule = cfg["joint_reference_rule"]
    ref_ok = bool(refs) and len(refs) == len(cfg["references"]) and all(
        c["complete"] and c["n_stable"] == c["n_trials"] for c in refs
    )
    kinds = []
    if best:
        kinds.append("posture")
    if ref_ok:
        kinds.append("joint_reference")
    grid = {
        "planner_arms": {
            "axes": {"height_m": cfg["static_heights"], "pitch_rad": [0.0]},
            "cells": {
                f"h{c['height']:.2f}_p{c['pitch']:+.1f}": {
                    "n_trials": c["n_trials"],
                    "n_stable": c["n_stable"],
                    "wilson95": c["wilson95"],
                    "fail_counts": c["fail_counts"],
                    "passes": c["complete"] and c["n_stable"] == c["n_trials"],
                }
                for c in static
            },
        }
    }
    envelope = {
        "protocol": {
            "config": cfg["name"],
            "seeds": cfg["seeds"],
            "cycles_per_trial": cfg["cycles"],
            "gates": cfg["gates"],
            "rates": {"height_m_s": None, "rpy_rad_s": None},
            "push_s": cfg["push_s"],
            "arm_poses_rad": None,
        },
        "meaning": (
            "Measured cells only, same frozen protocol and gates as results/g1_wbc_envelope_v2. "
            "A cell passes when every trial met every gate (no tilt latch, no sustained undesired "
            "ground contact, foot support kept, height tracking, joint-limit contact, CoM margin, "
            "torque ratio, slip, return to nominal; push cells also recover; joint_reference "
            "cells also keep hand FK error within the stated bound). Counts and Wilson 95% "
            "intervals are the evidence; with 5 trials a perfect cell has a lower bound of about "
            "0.57. Nothing here is labelled safe. " + sonic.LABEL + "."
        ),
        "input_bound_m": list(cfg["request_height_range"]),
        "height_m": height_m,
        "height_m_rule": cfg["height_m_rule"],
        "kinds": kinds,
        "joint_reference_rule": rule,
        "joint_reference_rule_passed": ref_ok,
        "interpolation_rule": cfg["interpolation_rule"],
        "roll": "Not commandable in SONIC planner mode; executable roll is fixed at 0.",
        "world_frame_torso_attitude": "recorded per segment in the trial files; not a gate",
        "arm_rate_max": None,
        "controller": sonic.LABEL,
        "grid": grid,
        "other_cells": [c for c in cells if c["kind"] != "static"],
        "rejected_by_input_bound": [],
    }
    (out / "envelope.json").write_text(json.dumps(envelope, indent=1))


# -- driver --------------------------------------------------------------------------------------
def write_meta(cfg: dict, out: pathlib.Path) -> None:
    meta = out / "meta.json"
    if meta.exists():
        return
    import onnxruntime

    cache = sonic.policy_cache()
    meta.write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "git_dirty": bool(
                    subprocess.run(["git", "status", "--porcelain", "--", "dream", "experiments",
                                    "tests"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
                ),  # fmt: skip
                "controller_label": sonic.LABEL,
                "upstream": "NVlabs/GR00T-WholeBodyControl 45c03facbf848e0c743091b7d9c17952096ff908",
                "weights_sha256": {
                    f: sha256(cache / f)
                    for f in ("model_encoder.onnx", "model_decoder.onnx", "planner_sonic.onnx")
                },
                "config": cfg,
                "seeds": cfg["seeds"],
                "python": platform.python_version(),
                "mujoco": mujoco.__version__,
                "numpy": np.__version__,
                "onnxruntime": onnxruntime.__version__,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=1,
        )
    )


def one(cfg: dict, out: pathlib.Path, name: str, seed: int) -> int:
    cond = next(c for c in conditions(cfg) if c["name"] == name)
    t = time.time()
    try:
        r = run_trial(cfg, cond, seed)
    except (MemoryError, OSError) as e:  # environment (paging file), not a result: retry later
        print("ENV-ERROR", name, seed, repr(e), flush=True)
        return 75
    except Exception:  # noqa: BLE001  a crash of the port is a measured failure, recorded
        r = {
            "condition": {k: v for k, v in cond.items() if k != "ref"},
            "seed": seed,
            "error": traceback.format_exc(),
            "criteria": {"stable": False, "fails": ["exception"], "tilt_latch": None,
                         "height_error_max": None, "rpy_waist_error_commanded_max": None,
                         "torque_ratio": {g: 0.0 for g in GROUPS},
                         "com_margin": {"min_m": None},
                         "foot_support": {"max_continuous_s": None},
                         "undesired_ground_contact": {"max_continuous_s": None},
                         "joint_limit": {"total_s": None}, "feet_slip_max_m": None,
                         "push_recovery_s": None},
            "criteria_kneel_support": {"fails": ["exception"], "stable": False},
            "extras": {"intended_knee_contact": {"total_s": 0.0}, "tilt_max_rad": 0.0,
                       "reference": {}},
        }  # fmt: skip
    r["wall_s"] = time.time() - t
    trial_path(out, cond, seed).write_text(json.dumps(r, indent=1))
    k = r["criteria"]
    print(name, seed, "stable" if k["stable"] else "FAIL " + ",".join(k["fails"]),
          f"{r['wall_s']:.0f}s", flush=True)  # fmt: skip
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--make-refs", action="store_true")
    ap.add_argument("--driver", action="store_true", help="one subprocess per trial, resumable")
    ap.add_argument("--one", nargs=2, metavar=("CELL", "SEED"))
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.config).read_text())
    if a.make_refs:
        refs = make_references(cfg)
        p = ROOT / cfg["references_file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(refs, indent=1))
        for r in refs:
            print(r["height"], r["feasible"], r.get("target"), r.get("arm"),
                  r.get("pelvis_height"), r.get("ref_pelvis_pitch_rad"), r.get("hand_residual_m"))  # fmt: skip
        return
    if "references_file" in cfg:
        refs = json.loads((ROOT / cfg["references_file"]).read_text())
        cfg["references"] = [r for r in refs if r["feasible"]]
        cfg["references_infeasible"] = [r["height"] for r in refs if not r["feasible"]]
    if a.smoke:
        cfg = {**cfg, **cfg["smoke"], "name": cfg["name"] + "_smoke"}
        keep = set(cfg["smoke"]["cells"])
    else:
        keep = None
    out = ROOT / "results" / cfg["name"]
    if a.summarize:
        summarize(cfg, out)
        return
    (out / "trials").mkdir(parents=True, exist_ok=True)
    if a.one:
        sys.exit(one(cfg, out, a.one[0], int(a.one[1])))
    write_meta(cfg, out)
    jobs = [
        (c, s)
        for c in conditions(cfg)
        if keep is None or c["name"] in keep
        for s in cfg["seeds"]
    ]
    for c, s in jobs:
        if trial_path(out, c, s).exists():
            continue
        for attempt in range(6):
            cmd = [sys.executable, str(pathlib.Path(__file__)), a.config, "--one", c["name"], str(s)]
            if a.smoke:
                cmd.insert(3, "--smoke")
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            if rc == 0 and trial_path(out, c, s).exists():
                break
            print("retry", c["name"], s, "rc", rc, flush=True)
            time.sleep(60)
    summarize(cfg, out)


if __name__ == "__main__":
    main()
