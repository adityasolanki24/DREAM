"""Safety supervisor for whole-body posture execution: a monitor that a ``WholeBodyBackend``
calls on every 50 Hz tick, plus an explicit, validated recovery.

Checks per tick, from the robot's own sensors only (IMU, joint encoders, requested torques) and
optional environment-adapter callbacks carried in the ``Observation`` (contact force, per-foot
contact; the adapter may use force sensors, the supervisor never reads scene state):
- joint position within the model range (minus a margin) and joint velocity limit;
- requested |torque| against the actuator limit for several consecutive ticks;
- pelvis tilt and tilt rate;
- optional external contact-force limit; optional loss of foot support.

A violation LATCHES a protective stop and is returned to the backend, which HOLDS: posture and
arm references freeze at the current state, velocity commands go to zero, the balance policy
keeps running (PLAN C4). The supervisor does not stand the robot up. ``recover`` is a separate
explicit action allowed only in a validated support state (both feet in contact from the supplied
callback, CoM inside the support polygon, tilt and tilt rate small); it reverses the recorded
command trajectory at a reduced rate, still monitored.

The trace records, per tick, commanded versus achieved pelvis height, torso rpy and hand
positions (achieved from forward kinematics of the encoders and IMU), CoM, torque ratio, tilt
and stop events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dream.robot.g1_posture import ARMS, PlanningModel
from dream.robot.g1_wbc import tilt
from dream.robot.wholebody import ExecutionReport, Observation, StopEvent, yaw_matrix

# Thresholds are DREAM safety choices (assumptions, not vendor figures), applied on top of the
# model's own joint ranges and actuator force limits.
JOINT_MARGIN = 0.02  # rad kept inside the joint range
JOINT_VELOCITY = 10.0  # rad/s; Unitree publishes higher limits, the 50 Hz policy stays far below
TORQUE_FRACTION = 1.0  # requested torque / actuator limit
TORQUE_TICKS = 5  # consecutive ticks over the limit before a stop (0.1 s)
TILT_LIMIT = 0.30  # rad; the WBC fall latch is 0.6
TILT_RATE_LIMIT = 1.5  # rad/s, roll/pitch gyro magnitude
SUPPORT_TICKS = 5  # consecutive ticks without both feet in contact
RECOVER_TILT = 0.10  # rad
RECOVER_TILT_RATE = 0.30  # rad/s
RECOVER_RATE_SCALE = 0.5  # recovery runs at half the backend's rate limits
RECOVER_WAYPOINT_STRIDE = 5  # ticks between replayed waypoints


@dataclass(frozen=True)
class SafetyLimits:
    joint_margin: float = JOINT_MARGIN
    joint_velocity: float = JOINT_VELOCITY
    torque_fraction: float = TORQUE_FRACTION
    torque_ticks: int = TORQUE_TICKS
    tilt: float = TILT_LIMIT
    tilt_rate: float = TILT_RATE_LIMIT
    contact_force: float | None = None  # N; None disables
    require_support: bool = True  # stop on lost foot support when feet_contact is supplied
    support_ticks: int = SUPPORT_TICKS


@dataclass
class RecoveryResult:
    ok: bool
    refused: str | None = None
    report: ExecutionReport | None = None


class WholeBodySupervisor:
    """Callable monitor (``Monitor`` protocol) with a trace; ``measure`` feeds the backend's."""

    def __init__(
        self,
        planning_model: PlanningModel,
        limits: SafetyLimits | None = None,
        hand_goal: tuple[str, tuple[float, float, float]] | None = None,
    ):
        self.pm = planning_model
        self.limits = limits or SafetyLimits()
        self.hand_goal = hand_goal  # (arm, point) to report the hand error against
        self.latched: StopEvent | None = None
        self.stops: list[StopEvent] = []
        self.trace: list[dict[str, object]] = []
        self._over = 0
        self._no_support = 0
        self._last: dict[str, object] = {}

    def reset(self) -> None:
        self.latched, self._over, self._no_support = None, 0, 0

    # -- measurement ---------------------------------------------------------------------
    def measure(self, obs: Observation) -> dict[str, object]:
        p = obs.prop
        state = self.pm.kinematic_state(p.quat, p.q, obs.feet_contact or (True, True))
        rot = yaw_matrix(float(obs.pose[2]))
        base = np.array([obs.pose[0], obs.pose[1], 0.0])
        hands = {s: base + rot @ state["hands"][s] for s in ARMS}
        ratio = float(np.max(np.abs(obs.tau) / self.pm.torque_limit))
        row: dict[str, object] = {
            "ach_height": state["height"],
            "ach_rpy": tuple(float(v) for v in state["rpy"]),
            "ach_hand_left": tuple(float(v) for v in hands["left"]),
            "ach_hand_right": tuple(float(v) for v in hands["right"]),
            "com_margin": state["com_margin"],
            "torque_ratio": ratio,
            "tilt": float(tilt(p.quat)),
            "tilt_rate": float(np.linalg.norm(p.gyro[:2])),
        }
        if self.hand_goal is not None:
            arm, point = self.hand_goal
            row["hand_error"] = float(np.linalg.norm(hands[arm] - np.asarray(point)))
        self._last = row
        self.trace.append({"t": obs.t, **row})
        return row

    # -- monitoring ----------------------------------------------------------------------
    def __call__(self, obs: Observation) -> StopEvent | None:
        if self.latched is not None:
            return self.latched
        lim, p, pm = self.limits, obs.prop, self.pm
        q = np.asarray(p.q)
        lo, hi = pm.joint_range[:, 0] + lim.joint_margin, pm.joint_range[:, 1] - lim.joint_margin
        bad = np.flatnonzero((q < lo) | (q > hi))
        if bad.size:
            return self._stop(obs, "joint position limit", {"joint": float(bad[0])})
        fast = np.flatnonzero(np.abs(p.dq) > lim.joint_velocity)
        if fast.size:
            return self._stop(obs, "joint velocity limit", {"joint": float(fast[0])})
        ratio = np.abs(obs.tau) / pm.torque_limit
        self._over = self._over + 1 if ratio.max() > lim.torque_fraction else 0
        if self._over >= lim.torque_ticks:
            return self._stop(obs, "torque limit", {"ratio": float(ratio.max())})
        a, rate = tilt(p.quat), float(np.linalg.norm(p.gyro[:2]))
        if a > lim.tilt:
            return self._stop(obs, "pelvis tilt", {"tilt": a})
        if rate > lim.tilt_rate:
            return self._stop(obs, "pelvis tilt rate", {"rate": rate})
        if lim.contact_force is not None and (obs.contact_force or 0.0) > lim.contact_force:
            return self._stop(obs, "contact force", {"force": float(obs.contact_force)})
        if lim.require_support and obs.feet_contact is not None:
            self._no_support = 0 if all(obs.feet_contact) else self._no_support + 1
            if self._no_support >= lim.support_ticks:
                return self._stop(obs, "foot support lost", {})
        return None

    def _stop(self, obs: Observation, reason: str, detail: dict[str, float]) -> StopEvent:
        self.latched = StopEvent(obs.t, reason, detail)
        self.stops.append(self.latched)
        return self.latched

    # -- recovery ------------------------------------------------------------------------
    def support_state(self, obs: Observation) -> tuple[bool, str]:
        """Is the support state valid for recovery? Needs per-foot contact from the adapter."""
        if obs.feet_contact is None:
            return False, "support unverifiable: no foot contact signal"
        if not all(obs.feet_contact):
            return False, "support invalid: a foot is not in contact"
        p = obs.prop
        a, rate = tilt(p.quat), float(np.linalg.norm(p.gyro[:2]))
        if a > RECOVER_TILT or rate > RECOVER_TILT_RATE:
            return False, f"support invalid: tilt {a:.2f} rad, rate {rate:.2f} rad/s"
        state = self.pm.kinematic_state(p.quat, p.q, obs.feet_contact)
        if state["com_margin"] <= 0.0:
            return False, f"support invalid: CoM margin {state['com_margin']:.3f} m"
        return True, ""

    def recover(self, backend) -> RecoveryResult:
        """Return to the start of the recorded trajectory by reversing it at a reduced rate.

        Refuses unless the support state is valid; never stands up a fallen robot (the backend's
        fall latch stays in force)."""
        if getattr(backend.controller, "stopped", None):
            return RecoveryResult(False, "controller fall latch is set")
        ok, why = self.support_state(backend.observe())
        if not ok:
            return RecoveryResult(False, why)
        history = backend.history
        waypoints = history[::-1][::RECOVER_WAYPOINT_STRIDE]
        if waypoints[-1] is not history[0]:
            waypoints.append(history[0])
        backend.release_hold()
        self.reset()
        report = backend.follow(waypoints, monitor=self, rate_scale=RECOVER_RATE_SCALE)
        return RecoveryResult(report.completed, None if report.completed else "stopped", report)


def summary(trace: list[dict[str, object]]) -> dict[str, float]:
    """Peaks over a trace, for reports."""

    def peak(key, fn=max):
        vals = [r[key] for r in trace if key in r]
        return float(fn(vals)) if vals else math.nan

    return {
        "peak_torque_ratio": peak("torque_ratio"),
        "peak_tilt": peak("tilt"),
        "peak_tilt_rate": peak("tilt_rate"),
        "min_com_margin": peak("com_margin", min),
    }
