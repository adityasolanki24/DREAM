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

from dream.robot.g1_posture import (
    ARMS,
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
HAND_ERROR_MAX = 0.04  # m after settling; looser than the 2 cm IK residual because the
# controller tracks the posture with a finite error (assumed, to be tightened with data)
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


def longest_violation(margins, minimum: float = COM_MARGIN_MIN) -> int:
    """Longest run of consecutive ticks with the margin below ``minimum``."""
    best = run = 0
    for m in margins:
        run = run + 1 if m < minimum else 0
        best = max(best, run)
    return best


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
        self.peak["torque_ratio"] = max(
            self.peak["torque_ratio"], float(np.max(np.abs(obs.tau) / self.pm.torque_limit))
        )
        self.peak["tilt"] = max(self.peak["tilt"], tilt(p.quat))
        self.peak["tilt_rate"] = max(self.peak["tilt_rate"], float(np.linalg.norm(p.gyro[:2])))
        self.peak["undesired"] += undesired
        self.hand_errors.append(float(np.linalg.norm(self.hand() - self.point)))
        return self.supervisor(replace(obs, feet_contact=feet))

    def measure(self, obs: Observation) -> dict[str, object]:
        return {"hand_error": self.hand_errors[-1] if self.hand_errors else math.nan}


def verify_by_rollout(
    target: WholeBodyTarget,
    backend_factory,
    perceived_obstacles=(),
    planning_model: PlanningModel | None = None,
    safety: SafetyLimits | None = None,
) -> RolloutResult:
    """Execute ``target`` with the real controller on a planning copy and judge the measurements.

    ``backend_factory(model, stance)`` returns a backend whose robot stands at ``stance`` on
    ``model`` (robot + floor + perceived boxes). The backend must be executable (measured
    capabilities); an unmeasured backend fails the rollout with its refusal reason."""
    t0 = time.perf_counter()
    boxes = list(perceived_obstacles)
    pm = planning_model or PlanningModel()
    backend = backend_factory(build_rollout_model(boxes), target.stance)
    result = RolloutResult(False)
    backend.follow([backend.command()])  # settle at nominal before the motion
    goal = target.hands[0]
    sup = WholeBodySupervisor(pm, safety, (goal.arm, goal.position))
    meter = _Meter(backend, pm, target, boxes, sup)
    start = backend.env.time
    report = backend.execute(target, monitor=meter)
    result.simulated = backend.env.time - start
    result.stops = report.stops
    if report.refused:
        result.reasons.append(f"refused: {report.refused}")
    if report.stops:
        result.reasons.append(f"stop: {report.stops[0].reason}")
    errs = meter.hand_errors
    hand = float(np.mean(errs[-FINAL_TICKS:])) if errs else math.nan
    result.metrics = {
        "hand_error": hand,
        "peak_torque_ratio": meter.peak["torque_ratio"],
        "peak_tilt": meter.peak["tilt"],
        "peak_tilt_rate": meter.peak["tilt_rate"],
        "max_foot_slip": meter.peak["slip"],
        "min_com_margin": meter.min_margin,
        "com_margin_violation_s": longest_violation(meter.margins) * TICK,
        "undesired_contact_ticks": meter.peak["undesired"],
    }
    checks = (
        (not hand <= HAND_ERROR_MAX, "hand_error"),
        (meter.peak["torque_ratio"] > TORQUE_RATIO_MAX, "torque_ratio"),
        (meter.peak["slip"] > SLIP_MAX, "foot_slip"),
        (longest_violation(meter.margins) * TICK > COM_MARGIN_MAX_VIOLATION_S, "com_margin"),
        (meter.peak["undesired"] > 0, "undesired_contact"),
    )
    result.reasons += [name for bad, name in checks if bad and report.completed]
    result.passed = report.completed and not result.reasons
    result.seconds = time.perf_counter() - t0
    return result


def verify_top_k(
    plans: list[PosturePlan],
    backend_factory,
    perceived_obstacles=(),
    k: int = TOP_K,
    planning_model: PlanningModel | None = None,
):
    """Roll out the best ``k`` ranked plans in order until one passes.

    Returns (chosen plan or None, [(plan, RolloutResult)]). Timings are in each result."""
    pm = planning_model or PlanningModel()
    tried = []
    for plan in plans[:k]:
        r = verify_by_rollout(plan.target, backend_factory, perceived_obstacles, pm)
        tried.append((plan, r))
        if r.passed:
            return plan, tried
    return None, tried
