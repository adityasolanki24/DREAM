"""Active Physical Discovery executed by the arm, and opening along what it learned.

``investigate`` is the body of the ``investigate_physics`` tool. It grasps the handle of an
articulated object whose mechanism is unknown, then repeats: choose a probe (expected
information gain, or random for the baseline), predict its outcome, execute it as a compliant
force-capped drag, measure the settled handle pose, update the belief. It stops when the
mechanism is identified, the model class is inadequate, a safety condition trips, or the
budget runs out. The belief, prior, predictive model and probe selection are the gantry's
(``dream/agent``), unchanged except for the robot's force law: the Panda's admittance caps
the spring force by norm (``ProbeController(norm_clip=True)``).

``open_learned`` opens the object in short closed-loop segments along the learned constraint.
After each segment it re-observes, checks the handle stayed on the believed constraint
(verification of the model during use), and stops when progress stalls at a limit.
``open_fixed_pull`` is the no-discovery baseline: always pull straight out of the visible
front, the motion that opens a drawer.

Nothing here reads simulator state. The mechanism is known only through probe outcomes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import dream.skills.articulation as art
from dream.agent.belief import MIN_EVIDENCE_PROBES, Belief, Prior, make_record
from dream.agent.exploration import (
    EIGSelector,
    RandomSelector,
    candidate_probes,
    predict_for_probes,
)
from dream.agent.model import (
    FIXED,
    PRISMATIC,
    REVOLUTE,
    TYPE_NAMES,
    ProbeController,
    relative_rotvec,
)
from dream.agent.operate import constraint_deviation, plan_segment
from dream.skills.manipulation import SkillResult
from dream.types import Embodiment, PartGeometry, Perception

CONFIDENCE = 0.95  # posterior probability of the most probable mechanism class
AXIS_AGREEMENT = 0.98  # |weighted mean axis| over the class's particles (1 = all agree)
N_PARTICLES = 600
# chi2/dof above which Panda probe data count as unexplained. Calibrated like the gantry's
# threshold, on in-class development scenes only (experiments/calibrate_panda_adequacy.py).
PANDA_ADEQUACY_THRESHOLD = 0.28  # placeholder until calibration
PREDICTION_SAMPLES = 128
SEGMENT = {PRISMATIC: 0.04, REVOLUTE: 0.25}  # m, rad of joint travel per opening segment
MAX_SEGMENTS = 10
STALL_FRACTION = 0.25  # a segment that advances less than this fraction has hit a limit
DEVIATION_TOL = 0.02  # m off the believed constraint during opening: the model is wrong
GRASP_MIN_WIDTH = 0.6 * 0.025  # m: fingers closing below this have lost the knob


@dataclass(frozen=True)
class SafetyConstraints:
    """Limits the investigation must respect, on top of the embodiment's own safety layer."""

    max_force: float = 15.0  # N: no probe may command more (the admittance caps it there)
    max_slip: float = 0.015  # m: change in the hand-handle offset that counts as losing grip
    max_handle_travel: float = 0.3  # m from the handle's initial position


DEFAULT_SAFETY = SafetyConstraints()


def probe_controller() -> ProbeController:
    """The force law of probes executed through ``CompliantMode`` (norm-capped spring)."""
    return ProbeController(art.PROBE_STIFFNESS, norm_clip=True)


@dataclass
class Investigation:
    """Result of ``investigate``. ``belief`` stays with the session for later operation."""

    object: str
    status: str  # identified | insufficient_information | model_inadequate | unsafe | failed
    reason: str
    mechanism: str  # most probable class, or "unknown" when the model class is inadequate
    type_probs: dict
    confidence: float
    constraint: dict  # axis, axis_concentration, origin (revolute), friction, upper
    chi2_per_dof: float
    model_adequate: bool
    prediction_error_m: float  # mean pre-action prediction error of the probes
    interactions: int  # probes executed
    robot_time_s: float
    opening_supported: bool
    steps: list[dict] = field(default_factory=list)
    belief: Belief | None = field(default=None, repr=False)

    def summary(self) -> dict:
        """What a planner sees (no particles, no step logs)."""
        keys = (
            "status",
            "reason",
            "mechanism",
            "type_probs",
            "confidence",
            "constraint",
            "chi2_per_dof",
            "model_adequate",
            "prediction_error_m",
            "interactions",
            "robot_time_s",
            "opening_supported",
        )
        return {k: getattr(self, k) for k in keys}

    def to_dict(self) -> dict:
        return {"object": self.object, **self.summary(), "steps": self.steps}


def geometry_from(obs) -> PartGeometry:
    """Visible geometry for the prior: the box extent and the closed-state handle position."""
    return PartGeometry(
        panel_center=np.asarray(obs.box_centre, float),
        panel_half_extents=np.asarray(obs.box_half, float),
        handle_pos=np.asarray(obs.handle_pos, float),
    )


def make_selector(strategy: str, probes):
    if strategy == "eig":
        return EIGSelector(probes)
    if strategy == "random":
        return RandomSelector(probes)
    raise ValueError(f"unknown exploration strategy {strategy!r}")


def investigate(
    robot: Embodiment,
    perception: Perception,
    name: str,
    budget: int,
    rng: np.random.Generator,
    strategy: str = "eig",
    safety: SafetyConstraints = DEFAULT_SAFETY,
    n_particles: int = N_PARTICLES,
    on_step=None,
    adequacy_threshold: float = PANDA_ADEQUACY_THRESHOLD,
    stop_early: bool = True,
) -> Investigation:
    """Identify the mechanism behind ``name``'s handle by probing it (see module docstring).

    ``on_step(step, belief)`` is called after every belief update, for live displays. The handle
    stays gripped when an opening strategy is supported, and is released otherwise.
    """
    t_start = robot.state().time
    obs0 = art.find_articulated(perception, name)
    if obs0 is None:
        return _failed(name, f"no articulated object '{name}' observed")
    # The closed-state reference must be read before touching the handle.
    ref = art.observe_handle_averaged(perception, name)
    geometry = geometry_from(ref)
    belief = Belief(
        geometry,
        rng,
        n_particles=n_particles,
        prior=Prior(geometry, origin_around_box=True),
        controller=probe_controller(),
        adequacy_threshold=adequacy_threshold,
    )
    probes = [p for p in candidate_probes() if p.max_force <= safety.max_force]
    selector = make_selector(strategy, probes)
    g = art.grasp_handle(robot, perception, name)
    if not g.ok:
        return _failed(name, f"could not grasp the handle: {g.reason}", belief)
    width0 = robot.state().gripper_width
    steps: list[dict] = []
    status, reason = "insufficient_information", f"budget of {budget} probes spent"
    for t in range(1, budget + 1):
        handle_now = steps[-1]["_handle_after"] if steps else ref.handle_pos
        probe, info = selector.select(belief, handle_now, handle_now, rng)
        hyps = belief.sample(PREDICTION_SAMPLES, rng)
        pred = predict_for_probes(
            hyps, belief.h0, handle_now, handle_now, [probe], belief.controller
        )
        pred_pos = pred[0][0][:, :3].mean(axis=0)
        t0 = time.perf_counter()
        o = art.probe(robot, perception, name, probe.direction, probe.distance, probe.max_force)
        rotvec = relative_rotvec(ref.handle_quat, o.handle_quat_after)
        # The spring acts at the grasp point, which moves rigidly with the knob, so the handle
        # position before the probe stands in for the end-effector position in the model.
        record = make_record(
            o.handle_before,
            o.handle_before,
            probe.direction,
            probe.distance,
            probe.max_force,
            o.handle_after,
            rotvec,
        )
        belief.update(record)
        fit = belief.fit_check()
        step = {
            "step": t,
            "probe": {
                "direction": probe.direction.round(4).tolist(),
                "distance": probe.distance,
                "max_force": probe.max_force,
            },
            "chosen_eig": info.get("eig"),
            "predicted_handle": pred_pos.round(5).tolist(),
            "observed_handle": o.handle_after.round(5).tolist(),
            "predicted_displacement_m": float(np.linalg.norm(pred_pos - o.handle_before)),
            "observed_displacement_m": float(np.linalg.norm(o.handle_after - o.handle_before)),
            "observed_rotation_rad": float(np.linalg.norm(rotvec)),
            "pre_action_pred_err_m": float(np.linalg.norm(pred_pos - o.handle_after)),
            "force_n": float(np.linalg.norm(o.force_after)),
            "slip_m": o.slip,
            "type_probs": dict(zip(TYPE_NAMES, belief.type_probs().round(4).tolist(), strict=True)),
            "chi2_per_dof": fit["chi2_per_dof"],
            "probe_wall_s": time.perf_counter() - t0,
            "_handle_after": o.handle_after,
        }
        steps.append(step)
        if on_step is not None:
            on_step({k: v for k, v in step.items() if not k.startswith("_")}, belief)
        unsafe = _unsafe(o, robot, ref, width0, safety)
        if unsafe:
            status, reason = "unsafe", unsafe
            break
        if not stop_early:
            continue
        if not fit["model_adequate"]:
            status = "model_inadequate"
            reason = f"no hypothesis explains the probes (chi2/dof {fit['chi2_per_dof']:.2f})"
            break
        if t >= MIN_EVIDENCE_PROBES and _confident(belief):
            status, reason = "identified", ""
            break
    inv = _result(name, status, reason, belief, steps, robot.state().time - t_start)
    if not inv.opening_supported:
        art.release_handle(robot)
    return inv


def _confident(belief: Belief) -> bool:
    p = belief.type_probs()
    k = int(np.argmax(p))
    if p[k] < CONFIDENCE:
        return False
    return k == FIXED or float(belief.estimate(k)["axis_concentration"]) >= AXIS_AGREEMENT


def _unsafe(o, robot: Embodiment, ref, width0: float, safety: SafetyConstraints) -> str:
    if not o.ok:
        return f"protective stop: {o.reason}"
    if o.slip > safety.max_slip:
        return f"handle slipped {o.slip * 1000:.0f} mm in the grip"
    if robot.state().gripper_width < min(GRASP_MIN_WIDTH, width0 - 0.008):
        return "grasp on the handle lost"
    if np.linalg.norm(o.handle_after - ref.handle_pos) > safety.max_handle_travel:
        return "handle moved beyond the allowed travel"
    return ""


def _result(name, status, reason, belief: Belief, steps, robot_time) -> Investigation:
    fit = belief.fit_check()
    p = belief.type_probs()
    k = int(np.argmax(p))
    constraint: dict = {}
    if k != FIXED:
        est = belief.estimate(k)
        constraint = {key: np.round(v, 4).tolist() for key, v in est.items()}
    adequate = bool(fit["model_adequate"])
    errs = [s["pre_action_pred_err_m"] for s in steps]
    return Investigation(
        object=name,
        status=status,
        reason=reason,
        mechanism=TYPE_NAMES[k] if adequate else "unknown",
        type_probs=dict(zip(TYPE_NAMES, p.round(4).tolist(), strict=True)),
        confidence=float(p[k]),
        constraint=constraint,
        chi2_per_dof=float(fit["chi2_per_dof"]),
        model_adequate=adequate,
        prediction_error_m=float(np.mean(errs)) if errs else float("nan"),
        interactions=len(steps),
        robot_time_s=float(robot_time),
        opening_supported=status == "identified" and k != FIXED,
        steps=[{k2: v for k2, v in s.items() if not k2.startswith("_")} for s in steps],
        belief=belief,
    )


def _failed(name: str, reason: str, belief: Belief | None = None) -> Investigation:
    return Investigation(
        object=name,
        status="failed",
        reason=reason,
        mechanism="unknown",
        type_probs={},
        confidence=0.0,
        constraint={},
        chi2_per_dof=float("nan"),
        model_adequate=True,
        prediction_error_m=float("nan"),
        interactions=0,
        robot_time_s=0.0,
        opening_supported=False,
        belief=belief,
    )


def _ensure_grasped(robot: Embodiment, perception: Perception, name: str) -> SkillResult:
    """Grasp the handle unless the fingers are already closed on something near it."""
    st = robot.state()
    obs = art.find_articulated(perception, name)
    if obs is None:
        return SkillResult(False, f"no articulated object '{name}' observed")
    near = np.linalg.norm(obs.handle_pos - st.tcp_pos) < 0.02
    if near and st.gripper_command < 0.005 and st.gripper_width > GRASP_MIN_WIDTH:
        return SkillResult(True, "")
    return art.grasp_handle(robot, perception, name)


def open_learned(
    robot: Embodiment,
    perception: Perception,
    name: str,
    inv: Investigation,
    safety: SafetyConstraints = DEFAULT_SAFETY,
) -> SkillResult:
    """Open along the learned constraint in closed-loop segments, then release the handle."""
    belief = inv.belief
    if belief is None or not inv.opening_supported:
        return SkillResult(False, "no supported opening strategy for this object")
    kind = belief.map_type()
    t0 = robot.state().time
    g = _ensure_grasped(robot, perception, name)
    if not g.ok:
        return SkillResult(False, f"could not grasp the handle: {g.reason}")
    width0 = robot.state().gripper_width
    log, outcome = [], SkillResult(True, "")
    for _ in range(MAX_SEGMENTS):
        st = robot.state()
        now = art.observe_handle_averaged(perception, name).handle_pos
        planned = plan_segment(belief, now, SEGMENT[kind])
        if planned is None:
            outcome = SkillResult(False, "belief changed to 'fixed'")
            break
        targets, q_now, q_goal = planned
        offset = st.tcp_pos - now  # the grasp offset is measured, not assumed
        r = art.pull_along(robot, [x + offset for x in targets], safety.max_force)
        after = art.observe_handle_averaged(perception, name).handle_pos
        q_after, deviation = constraint_deviation(belief, after)
        log.append({"q_before": q_now, "q_after": q_after, "deviation_m": deviation})
        if not r.ok:
            outcome = SkillResult(False, r.reason)
            break
        if robot.state().gripper_width < min(GRASP_MIN_WIDTH, width0 - 0.008):
            outcome = SkillResult(False, "grasp on the handle lost while opening")
            break
        if deviation > DEVIATION_TOL:
            outcome = SkillResult(
                False,
                f"handle left the learned constraint by {deviation * 1000:.0f} mm: "
                "the model is wrong for this object",
            )
            break
        if q_after - q_now < STALL_FRACTION * (q_goal - q_now):
            break  # at the opening limit (or blocked): stop pulling
    released = art.release_handle(robot)
    details = {
        "segments": log,
        "q_reached": log[-1]["q_after"] if log else 0.0,
        "max_deviation_m": max((s["deviation_m"] for s in log), default=0.0),
        "release_ok": released.ok,
    }
    outcome.details.update(details)
    outcome.sim_time = robot.state().time - t0
    return outcome


def open_fixed_pull(
    robot: Embodiment,
    perception: Perception,
    name: str,
    safety: SafetyConstraints = DEFAULT_SAFETY,
    distance: float = 0.22,
) -> SkillResult:
    """Baseline without discovery: pull straight out of the visible front face, then release.

    The front is where the handle sticks out of the box (horizontal). This is the right motion
    for a drawer and the wrong one for anything else; it is a fixed strategy, not a model.
    """
    t0 = robot.state().time
    obs = art.find_articulated(perception, name)
    if obs is None:
        return SkillResult(False, f"no articulated object '{name}' observed")
    front = (obs.handle_pos - obs.box_centre) * [1, 1, 0]
    front /= np.linalg.norm(front)
    g = _ensure_grasped(robot, perception, name)
    if not g.ok:
        return SkillResult(False, f"could not grasp the handle: {g.reason}")
    start = robot.state().tcp_pos
    waypoints = [start + front * distance * i / 30 for i in range(1, 31)]
    r = art.pull_along(robot, waypoints, safety.max_force)
    after = art.observe_handle_averaged(perception, name).handle_pos
    released = art.release_handle(robot)
    r.details.update(
        {
            "handle_travel_m": float(np.linalg.norm(after - obs.handle_pos)),
            "release_ok": released.ok,
        }
    )
    r.sim_time = robot.state().time - t0
    return r
