"""Contact-aware feasibility gate: the real controller executes a ``WholeBodyTarget`` on a
PLANNING COPY of the robot model before the plan is allowed near the real robot (PLAN C5).

The copy holds the robot, the floor of the robot MJCF and the PERCEIVED obstacle boxes as
geoms. It never contains scene truth; the caller's ``backend_factory(model, stance)`` builds a
physics owner and controller on it and returns a ``WholeBodyBackend`` that also offers
``physics() -> (model, data)`` (``WBCBackend`` does). Measured over the run, from actual
contacts: requested joint torque against actuator limits, foot normal force and slip, CoM
margin to the support polygon of the feet that carry load, pelvis tilt and tilt rate, hand
error after settling, and undesired contacts (any non-foot robot geom touching the floor or a
perceived box, except the hand on a box marked ``hand_may_touch``).

CLOSED LOOP (W6). The real robot does not stop after the open-loop execution: it corrects the hand
against fresh observations (dream/robot/g1_correction.py) and then advances to the contact pose.
The rollout therefore runs the SAME procedure on the planning copy (execute the first stage,
correct, advance, correct) under the same supervisor limits, with the perceived point as the
target the copy "observes" (the copy has no scene, so the point is the plan's own). The gate
is on the FINAL predicted hand error after the corrections, not on the open-loop error (a 4 cm
open-loop screen refused 21 of 35 rollouts that the real correction later brought to about 3
cm), and on the peak torque and support measured over the whole procedure. The measured peak
torque replaces the static estimate in the ranking (``RolloutResult.ranking``).

Passing a rollout makes a plan eligible for real execution; it does not count as success.
Only the monitored real execution does. Rollouts cost roughly the simulated duration of the
motion, so only the best ``TOP_K`` ranked plans are verified and wall-clock timings are reported.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from dream.robot.g1_correction import correct_reach
from dream.robot.g1_posture import (
    ARMS,
    HAND_TOLERANCE,
    W_TORQUE,
    PerceivedBox,
    PlanningModel,
    PosturePlan,
    _hull,
    _load_spec,
    polygon_margin,
)
from dream.robot.g1_supervisor import SafetyLimits, WholeBodySupervisor
from dream.robot.g1_wbc import DT_PHYSICS, tilt
from dream.robot.wholebody import TICK, Observation, StopEvent, WholeBodyTarget

TOP_K = 3  # plans verified per request; each rollout costs about the motion's simulated time
# COARSE SCREEN, an assumption: not a success criterion (success is decided by the evaluator;
# closed-loop correction in g1_correction.py removes the residual).
HAND_ERROR_MAX = 0.04  # m after settling, OPEN-LOOP (v1 screen; used when no planner is given)
# Closed-loop gate (v2): the predicted hand error after the corrections must be inside the task
# tolerance, the same 3 cm the evaluator allows (an assumption: it is the claim the real
# procedure must be able to meet, not a calibrated quantity).
FINAL_HAND_ERROR_MAX = 0.03  # m
W_FINAL_ERROR = 0.5  # ranking: per hand-tolerance of predicted final error
TORQUE_RATIO_MAX = 0.9  # fraction of actuator limit, the gate W1 pre-registered for its envelope
SLIP_MAX = 0.10  # m of foot travel, W1's feet_slip_max
COM_MARGIN_MIN = 0.0  # m, W1's com_margin_min
# The margin must stay at or above COM_MARGIN_MIN except for transients: the CoM may leave the
# polygon of force-carrying contacts for a tick or two while the heel unloads in a lean and the
# robot is dynamically balanced (measured on development rollouts: one tick at -0.04 m, then four
# loaded contacts per foot again and a settled stance). A margin below the minimum for longer than
# this is a failure.
COM_MARGIN_MAX_VIOLATION_S = 0.1
FOOT_FORCE_MIN = 1.0  # N of normal force for a foot contact to count as load-carrying
FINAL_TICKS = 10  # hand error is the mean over the last ticks of the settle


@dataclass
class RolloutResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0  # wall clock
    simulated: float = 0.0  # simulated seconds
    stops: list[StopEvent] = field(default_factory=list)
    metrics_text: dict[str, str] = field(default_factory=dict)
    # the plan's score with the MEASURED peak torque in place of the static proxy
    ranking: float = math.nan


def longest_violation(margins, minimum: float = COM_MARGIN_MIN) -> int:
    """Longest run of consecutive ticks with the margin below ``minimum``."""
    best = run = 0
    for m in margins:
        run = run + 1 if m < minimum else 0
        best = max(best, run)
    return best


SETTLE_TICKS = 75  # 1.5 s of standing after the planning copy is reset


def settle(backend, ticks: int = SETTLE_TICKS) -> None:
    """Stand at the current command for ``ticks`` policy steps. ``backend.follow([command])`` was
    meant to do this but a path from the command to itself is empty, so nothing ran and the first
    ticks of the motion saw feet not yet loaded: 38 W5 rollouts stopped with "foot support lost"
    at 0.1-0.2 s simulated, before the plan had moved."""
    c, cmd = backend.controller, backend.command()
    for _ in range(ticks):
        c.set_command(0.0, 0.0, 0.0)
        c.set_posture(cmd.height, cmd.rpy)
        c.arm_target = np.array(cmd.arm, float)
        backend.env.step(c)


def build_rollout_model(boxes: list[PerceivedBox]) -> mujoco.MjModel:
    """Robot MJCF (physics step as the controller expects) plus perceived boxes as static
    collision geoms named ``obstacle_i``."""
    spec = _load_spec()
    spec.option.timestep = DT_PHYSICS
    for i, b in enumerate(boxes):
        spec.worldbody.add_geom(
            name=f"obstacle_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(b.center),
            size=list(b.half_extents), quat=[math.cos(b.yaw / 2), 0, 0, math.sin(b.yaw / 2)],
            contype=1, conaffinity=1,
        )  # fmt: skip
    return spec.compile()


class _Meter:
    """Per-tick measurements from the planning copy's own contacts (a monitor wrapper)."""

    def __init__(self, backend, pm: PlanningModel, target: WholeBodyTarget, boxes, supervisor):
        self.model, self.data = backend.physics()
        self.pm, self.supervisor, self.boxes = pm, supervisor, boxes
        m = self.model
        self.floor = m.geom("floor").id
        self.foot_bodies = {s: m.body(f"{s}_ankle_roll_link").id for s in ARMS}
        self.obstacles = {
            m.geom(f"obstacle_{i}").id: b for i, b in enumerate(boxes)
        }  # geom id -> box
        self.hand_bodies = {
            m.body(f"{s}_{n}").id for s in ARMS for n in (
                "wrist_yaw_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
                "hand_middle_0_link", "hand_middle_1_link", "hand_index_0_link",
                "hand_index_1_link")
        }  # fmt: skip
        goal = target.hands[0]
        self.arm, self.point = goal.arm, np.asarray(goal.position)
        wrist = m.body(f"{self.arm}_wrist_yaw_link").id
        self.wrist, self.grip = wrist, pm.anatomy.grip[self.arm]
        self.foot0 = {s: self.data.xpos[b].copy() for s, b in self.foot_bodies.items()}
        self.peak_joint = 0
        self.peak = dict.fromkeys(
            ("torque_ratio", "tilt", "tilt_rate", "slip", "foot_force", "undesired"), 0.0
        )
        self.min_margin = math.inf
        self.margins: list[float] = []
        self.hand_errors: list[float] = []
        self.dt = 0.0

    def hand(self) -> np.ndarray:
        d = self.data
        return d.xpos[self.wrist] + d.xmat[self.wrist].reshape(3, 3) @ self.grip

    def __call__(self, obs: Observation) -> StopEvent | None:
        m, d = self.model, self.data
        pts, force, undesired = [], dict.fromkeys(ARMS, 0.0), 0
        f6 = np.zeros(6)
        body_of = m.geom_bodyid
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            robot, other = (g1, g2) if m.body_rootid[body_of[g1]] != 0 else (g2, g1)
            if m.body_rootid[body_of[robot]] == 0:
                continue  # world-world cannot happen; guard
            if m.body_rootid[body_of[other]] != 0:
                continue  # robot self contact is not measured here
            mujoco.mj_contactForce(m, d, i, f6)
            b = int(body_of[robot])
            foot = next((s for s, fb in self.foot_bodies.items() if fb == b), None)
            if foot is not None and other == self.floor:
                force[foot] += float(f6[0])
                if f6[0] > FOOT_FORCE_MIN / 4:
                    pts.append(c.pos[:2].copy())
                continue
            box = self.obstacles.get(other)
            if box is not None and box.hand_may_touch and b in self.hand_bodies:
                continue
            if other == self.floor or box is not None:
                undesired += 1
        margin = -1.0
        if len(pts) >= 3:
            margin = polygon_margin(d.subtree_com[1][:2], _hull(np.array(pts)))
        self.min_margin = min(self.min_margin, margin)
        self.margins.append(margin)
        slip = max(
            float(np.linalg.norm(d.xpos[b][:2] - self.foot0[s][:2]))
            for s, b in self.foot_bodies.items()
        )
        p = obs.prop
        feet = tuple(force[s] > FOOT_FORCE_MIN for s in ARMS)
        self.peak["slip"] = max(self.peak["slip"], slip)
        self.peak["foot_force"] = max(self.peak["foot_force"], sum(force.values()))
        ratio = np.abs(obs.tau) / self.pm.torque_limit
        if float(ratio.max()) > self.peak["torque_ratio"]:
            self.peak["torque_ratio"], self.peak_joint = float(ratio.max()), int(ratio.argmax())
        self.peak["tilt"] = max(self.peak["tilt"], tilt(p.quat))
        self.peak["tilt_rate"] = max(self.peak["tilt_rate"], float(np.linalg.norm(p.gyro[:2])))
        self.peak["undesired"] += undesired
        self.hand_errors.append(float(np.linalg.norm(self.hand() - self.point)))
        return self.supervisor(replace(obs, feet_contact=feet))

    def measure(self, obs: Observation) -> dict[str, object]:
        """The supervisor's row (so the correction can read the measured hand and torque)."""
        return self.supervisor.measure(obs)


def verify_by_rollout(
    target: WholeBodyTarget,
    backend_factory,
    perceived_obstacles=(),
    planning_model: PlanningModel | None = None,
    safety: SafetyLimits | None = None,
    procedure: Procedure | None = None,
    plan: PosturePlan | None = None,
) -> RolloutResult:
    """Execute ``target`` with the real controller on a planning copy and judge the measurements.

    ``backend_factory(model, stance)`` returns a backend whose robot stands at ``stance`` on
    ``model`` (robot + floor + perceived boxes). The backend must be executable (measured
    capabilities); an unmeasured backend fails the rollout with its refusal reason.

    Without ``procedure`` the rollout is the open-loop v1 screen. With one, the copy runs the real
    robot's procedure after the first execution: ``procedure.stages`` is a list of
    ``(plan, point, tolerance)``; each stage is corrected with ``correct_reach`` toward ``point``
    (the first stage is the one executed; later stages advance from the previous one). The gate
    then uses the FINAL predicted hand error (against the last stage's point)."""
    t0 = time.perf_counter()
    boxes = list(perceived_obstacles)
    pm = planning_model or PlanningModel()
    backend = backend_factory(build_rollout_model(boxes), target.stance)
    result = RolloutResult(False)
    settle(backend)  # stand still at the nominal command before the motion
    goal = target.hands[0]
    last_point = goal.position
    if procedure is not None:
        last_point = tuple(float(v) for v in procedure.stages[-1][1])
    sup = WholeBodySupervisor(pm, safety, (goal.arm, last_point))
    meter = _Meter(backend, pm, target, boxes, sup)
    if procedure is not None:
        meter.point = np.asarray(last_point, float)
    start = backend.env.time
    report = backend.execute(target, monitor=meter)
    stops, completed, refused = list(report.stops), report.completed, report.refused
    final = None
    if procedure is not None and completed and not stops:
        executed = None
        for stage_plan, point, tol in procedure.stages:
            pt = np.asarray(point, float)
            corr = correct_reach(
                backend, meter, procedure.planner, stage_plan, lambda pt=pt: pt, boxes,
                tolerance=tol, executed=executed, approach_axis=procedure.approach,
            )  # fmt: skip
            stops += corr.stops
            executed = corr.final_target or executed or stage_plan.target
            if corr.stops or corr.refused or not corr.completed:
                completed = False
                refused = refused or corr.refused
                break
            final = corr.iterations[-1]["error"] if corr.iterations else None
    result.simulated = backend.env.time - start
    result.stops = stops
    if refused:
        result.reasons.append(f"refused: {refused}")
    if stops:
        result.reasons.append(f"stop: {stops[0].reason}")
        detail = stops[0].detail
        if "joint_name" in detail:
            result.reasons[-1] += f" ({detail['joint_name']})"
    errs = meter.hand_errors
    open_loop = float(np.mean(errs[-FINAL_TICKS:])) if errs else math.nan
    hand = open_loop if final is None else float(final)
    limit = HAND_ERROR_MAX if procedure is None else FINAL_HAND_ERROR_MAX
    peak_name = sup.joint_names[meter.peak_joint]
    result.metrics = {
        "hand_error": hand,
        "hand_error_open_loop": open_loop,
        "peak_torque_ratio": meter.peak["torque_ratio"],
        "peak_torque_joint": float(meter.peak_joint),
        "peak_tilt": meter.peak["tilt"],
        "peak_tilt_rate": meter.peak["tilt_rate"],
        "max_foot_slip": meter.peak["slip"],
        "min_com_margin": meter.min_margin,
        "com_margin_violation_s": longest_violation(meter.margins) * TICK,
        "undesired_contact_ticks": meter.peak["undesired"],
    }
    result.metrics_text = {"peak_torque_joint_name": peak_name}
    checks = (
        (not hand <= limit, "hand_error"),
        (meter.peak["torque_ratio"] > TORQUE_RATIO_MAX, "torque_ratio"),
        (meter.peak["slip"] > SLIP_MAX, "foot_slip"),
        (longest_violation(meter.margins) * TICK > COM_MARGIN_MAX_VIOLATION_S, "com_margin"),
        (meter.peak["undesired"] > 0, "undesired_contact"),
    )
    result.reasons += [name for bad, name in checks if bad and completed]
    result.passed = completed and not result.reasons
    if plan is not None:
        static = float(plan.metrics.get("torque_ratio", 0.0))
        result.ranking = (
            plan.score + W_TORQUE * (meter.peak["torque_ratio"] - static)
            + W_FINAL_ERROR * (0.0 if math.isnan(hand) else hand) / HAND_TOLERANCE
        )  # fmt: skip
    result.seconds = time.perf_counter() - t0
    return result


@dataclass
class Procedure:
    """What the real robot does after the first execution (see the module docstring)."""

    planner: object  # PosturePlanner (for the correction's IK)
    stages: list  # [(plan, point, tolerance)]
    approach: object = None  # preferred finger direction for the corrections


def verify_top_k(
    plans: list[PosturePlan],
    backend_factory,
    perceived_obstacles=(),
    k: int = TOP_K,
    planning_model: PlanningModel | None = None,
):
    """Roll out the best ``k`` ranked plans in order until one passes (open-loop v1 screen).

    Returns (chosen plan or None, [(plan, RolloutResult)]). Timings are in each result."""
    pm = planning_model or PlanningModel()
    tried = []
    for plan in plans[:k]:
        r = verify_by_rollout(plan.target, backend_factory, perceived_obstacles, pm, plan=plan)
        tried.append((plan, r))
        if r.passed:
            return plan, tried
    return None, tried
