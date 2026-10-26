"""Controller-independent whole-body interface: what a planner asks for and what a backend reports.

A planner emits a ``WholeBodyTarget``: the full robot configuration ``q`` (qpos of the G1 MJCF:
floating base 7 + 29 joints in the model's joint order), task-space goals (hand points, the
stance) and the pelvis height and torso rpy DERIVED from ``q`` (the quantities the pretrained
Decoupled WBC accepts). A ``WholeBodyBackend`` states what it can execute (``capabilities()``:
supported request kinds, MEASURED limits, rates) and executes a target under a safety monitor,
returning an ``ExecutionReport`` of requested versus achieved values, stop events and a trace.
``WBCBackend`` wraps NVIDIA's Decoupled WBC (dream/robot/g1_wbc.py); a SONIC backend would
consume joint-position reference frames from the same ``WholeBodyTarget`` and publish its own
capabilities. Planning and task code reads limits only from ``capabilities()``.

Limits are measured, not assumed (PLAN C1): ``ControllerCapabilities.from_envelope_json`` reads
results/g1_wbc_envelope/envelope.json; without it the capabilities are "unmeasured: not
executable" and ``execute`` refuses, while planning (tests) still works against stated bounds.

A protective stop is a HOLD (PLAN C4): posture and arm references freeze at the current state,
velocity commands go to zero and the balance policy keeps running; the damping-only latch of
the WBC applies only after its own fall detection. Recovery is a separate explicit action.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from dream.robot import g1_wbc
from dream.robot.g1_wbc import DECIMATION, DT_PHYSICS, N_ACTION, N_BODY, Proprioception

TICK = DT_PHYSICS * DECIMATION  # 0.02 s, the 50 Hz policy step
N_ARM = N_BODY - N_ACTION  # 14: left 7 then right 7
ENVELOPE_JSON = Path(__file__).resolve().parents[2] / "results/g1_wbc_envelope/envelope.json"
# Region of the measured envelope used for reaching: both arms forward, the loaded case.
ENVELOPE_REGION = "arms_extended"
# Command rates. Height 0.1 m/s and rpy 0.3 rad/s are the rates W1's envelope runs were measured
# at (experiments/configs/g1_wbc_envelope.json). The arm rate is a DREAM assumption (arm PD
# kp 100 follows well below this); it is a rate limit, not a measured tracking bound.
HEIGHT_RATE, RPY_RATE, ARM_RATE = 0.1, 0.3, 0.8
SETTLE_SECONDS = 0.6  # monitored hold after the last waypoint


def yaw_matrix(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def torso_rpy_from_matrices(r_pelvis: np.ndarray, r_torso: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) of torso_link relative to the yaw-only pelvis frame.

    Upstream (g1_decoupled_whole_body_policy.py): R_yaw(pelvis)^T R_torso with the lower body at
    zero, i.e. the waist chain's own orientation, converted by Pinocchio matrixToRpy
    (R = Rz(yaw) Ry(pitch) Rx(roll)). Hence this uses the pelvis-relative rotation
    R_pelvis^T R_torso, which equals the waist chain whatever the pelvis tilt."""
    r = np.asarray(r_pelvis).T @ np.asarray(r_torso)
    pitch = -math.asin(float(np.clip(r[2, 0], -1.0, 1.0)))
    return np.array([math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])])


@dataclass(frozen=True)
class HandGoal:
    arm: str  # "left" or "right"
    position: tuple[float, float, float]  # grip point, odometry/world frame
    approach_axis: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class WholeBodyTarget:
    q: np.ndarray  # full qpos of the G1 MJCF (36 values)
    stance: tuple[float, float, float]  # x, y, yaw the feet are placed for
    hands: tuple[HandGoal, ...]
    pelvis_height: float  # derived from q
    torso_rpy: tuple[float, float, float]  # derived from q (torso_rpy_from_matrices)
    arm_joints: np.ndarray  # derived from q: 14 targets, left 7 then right 7


@dataclass(frozen=True)
class PostureCommand:
    """One applied command of the WBC backend: height, torso rpy, 14 arm joint targets."""

    height: float
    rpy: tuple[float, float, float]
    arm: np.ndarray


@dataclass(frozen=True)
class ControllerCapabilities:
    name: str
    request_kinds: frozenset[str]
    height_range: tuple[float, float]
    torso_rpy_min: tuple[float, float, float]
    torso_rpy_max: tuple[float, float, float]
    height_rate: float = HEIGHT_RATE
    rpy_rate: float = RPY_RATE
    arm_rate: float = ARM_RATE
    # rad; the commands carry no pelvis tilt, so plans stay near upright (assumed, not measured)
    max_pelvis_tilt: float = 0.15
    measured: bool = False
    source: str = "unmeasured: not executable"
    note: str = ""

    @property
    def executable(self) -> bool:
        return self.measured

    def supports(self, *kinds: str) -> bool:
        return set(kinds) <= self.request_kinds

    def violations(self, height: float, rpy) -> list[str]:
        out = []
        if not self.height_range[0] - 1e-6 <= height <= self.height_range[1] + 1e-6:
            out.append(f"height {height:.3f} outside {self.height_range}")
        r = np.asarray(rpy, float)
        if np.any(r < np.array(self.torso_rpy_min) - 1e-6) or np.any(
            r > np.array(self.torso_rpy_max) + 1e-6
        ):
            out.append(f"torso rpy {np.round(r, 3).tolist()} outside measured range")
        return out

    @classmethod
    def unmeasured(cls, name, height_range, rpy_min, rpy_max, kinds=None, **kw):
        """Stated (not measured) bounds: planning may use them, execution refuses."""
        return cls(name, frozenset(kinds or WBC_KINDS), tuple(height_range), tuple(rpy_min),
                   tuple(rpy_max), **kw)  # fmt: skip

    @classmethod
    def from_envelope_json(cls, path: str | Path = ENVELOPE_JSON, region: str = ENVELOPE_REGION):
        """Measured limits (W1). Intervals are per-axis (swept independently), so the product
        of intervals is NOT guaranteed: every plan still needs a rollout and a monitored run."""
        p = Path(path)
        bounds = (g1_wbc.HEIGHT_RANGE, g1_wbc.RPY_RANGE)
        if not p.exists():
            return cls.unmeasured(
                "decoupled_wbc", bounds[0], bounds[1][:, 0], bounds[1][:, 1],
                note="no envelope file: input bounds only",
            )  # fmt: skip
        raw = json.loads(p.read_text())[region]
        axes = [raw["roll_rad"], raw["pitch_rad"], raw["yaw_rad"]]
        return cls(
            "decoupled_wbc", WBC_KINDS, tuple(raw["height_m"]),
            tuple(a[0] for a in axes), tuple(a[1] for a in axes),
            measured=True, source=str(p),
            note=f"region {region}; per-axis intervals, product not validated",
        )  # fmt: skip


WBC_KINDS = frozenset({"pelvis_height", "torso_rpy", "arm_joint_positions", "hold"})


@dataclass(frozen=True)
class StopEvent:
    t: float
    reason: str
    detail: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """What a monitor may see: the robot's own sensors and the applied command."""

    t: float
    prop: Proprioception
    tau: np.ndarray  # 29 requested joint torques
    pose: np.ndarray  # odometry (x, y, yaw)
    command: PostureCommand
    contact_force: float | None = None  # N, largest external contact (environment callback)
    feet_contact: tuple[bool, bool] | None = None  # (left, right) from foot force sensing


@dataclass
class ExecutionReport:
    completed: bool
    requested: dict[str, object] = field(default_factory=dict)
    achieved: dict[str, object] = field(default_factory=dict)
    stops: list[StopEvent] = field(default_factory=list)
    trace: list[dict[str, object]] = field(default_factory=list)
    refused: str | None = None  # why the backend did not start


class Monitor(Protocol):
    def __call__(self, obs: Observation) -> StopEvent | None: ...


class WholeBodyBackend(Protocol):
    def capabilities(self) -> ControllerCapabilities: ...

    def execute(
        self, target: WholeBodyTarget, monitor: Monitor | None = None
    ) -> ExecutionReport: ...


def synchronized_path(start: PostureCommand, goal: PostureCommand, caps, scale: float = 1.0):
    """Linear path in command space whose channels arrive together at the slowest channel's
    rate limit (``scale`` < 1 slows it down). Returns the commands after ``start``."""
    dh = abs(goal.height - start.height) / (caps.height_rate * scale)
    dr = float(np.max(np.abs(np.subtract(goal.rpy, start.rpy)))) / (caps.rpy_rate * scale)
    da = float(np.max(np.abs(goal.arm - start.arm))) / (caps.arm_rate * scale)
    n = max(1, math.ceil(max(dh, dr, da) / TICK - 1e-9))
    out = []
    for k in range(1, n + 1):
        a = k / n
        out.append(
            PostureCommand(
                start.height + a * (goal.height - start.height),
                tuple(np.asarray(start.rpy) + a * (np.asarray(goal.rpy) - np.asarray(start.rpy))),
                start.arm + a * (goal.arm - start.arm),
            )
        )
    return out


class WBCBackend:
    """Decoupled-WBC backend over a physics owner (``G1MobileEnv``-like: ``step(controller)``,
    ``proprioception()``, ``odometry()``, ``model``, ``data``) and a ``WBCLocomotion``.

    ``contact_force`` and ``feet_contact`` are environment-adapter callbacks that may use force
    sensors; the backend never reads scene state itself."""

    def __init__(
        self,
        env,
        controller,
        capabilities: ControllerCapabilities | None = None,
        contact_force: Callable[[], float] | None = None,
        feet_contact: Callable[[], tuple[bool, bool]] | None = None,
    ):
        self.env, self.controller = env, controller
        self._caps = capabilities or ControllerCapabilities.from_envelope_json()
        self._contact_force, self._feet_contact = contact_force, feet_contact
        self.history: list[PostureCommand] = [self.command()]
        self.held: StopEvent | None = None

    def capabilities(self) -> ControllerCapabilities:
        return self._caps

    def physics(self):
        return self.env.model, self.env.data

    def command(self) -> PostureCommand:
        c = self.controller
        return PostureCommand(float(c.height), tuple(float(v) for v in c.rpy), c.arm_target.copy())

    def observe(self) -> Observation:
        prop = self.env.proprioception()
        return Observation(
            self.env.time, prop, self.controller.torques(prop), self.env.odometry(),
            self.command(),
            None if self._contact_force is None else float(self._contact_force()),
            None if self._feet_contact is None else tuple(self._feet_contact()),
        )  # fmt: skip

    # -- holding -------------------------------------------------------------------------
    def hold(self, event: StopEvent) -> None:
        """Protective stop: freeze references at the current state, zero velocity, keep the
        balance policy running. Posture goals become the applied command (the ramp stops); arm
        targets become the measured arm joints."""
        c = self.controller
        measured = np.asarray(self.env.proprioception().q)[N_ACTION:].copy()
        c.set_command(0.0, 0.0, 0.0)
        c.set_posture(float(c.height), tuple(c.rpy))
        c.arm_target = measured
        self.held = event

    def release_hold(self) -> None:
        self.held = None

    # -- execution -----------------------------------------------------------------------
    def execute(self, target: WholeBodyTarget, monitor: Monitor | None = None) -> ExecutionReport:
        caps = self._caps
        req = {"height": target.pelvis_height, "rpy": target.torso_rpy, "arm": target.arm_joints}
        if not caps.executable:
            return ExecutionReport(False, req, refused=caps.source)
        if not caps.supports("pelvis_height", "torso_rpy", "arm_joint_positions"):
            return ExecutionReport(False, req, refused="request kind unsupported")
        bad = caps.violations(target.pelvis_height, target.torso_rpy)
        if bad:
            return ExecutionReport(False, req, refused="; ".join(bad))
        goal = PostureCommand(
            target.pelvis_height, tuple(target.torso_rpy), self._arm_feedforward(target)
        )
        return self._run(synchronized_path(self.command(), goal, caps), req, monitor)

    def _arm_feedforward(self, target: WholeBodyTarget) -> np.ndarray:
        """Arm joint targets plus the PD hold's steady-state sag: the controller holds the arms
        by PD (kp ARM_KP), so a static gravity torque tau leaves an error tau / kp. The gravity
        torque is MuJoCo's recursive Newton-Euler at the target configuration (``target.q[7:36]``
        are the 29 body joints in actuator order, the planner's contract)."""
        import mujoco

        if not all(hasattr(self.env, a) for a in ("model", "data", "qadr", "vadr")):
            return np.asarray(target.arm_joints, float)  # no physics model to differentiate
        m = self.env.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.env.data.qpos
        d.qpos[0:7] = target.q[0:7]
        d.qpos[self.env.qadr] = target.q[7 : 7 + N_BODY]
        mujoco.mj_forward(m, d)
        bias = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, bias)
        tau = bias[self.env.vadr[N_ACTION:]]
        return np.asarray(target.arm_joints, float) + tau / g1_wbc.ARM_KP

    def follow(self, waypoints, monitor: Monitor | None = None, rate_scale: float = 1.0):
        """Visit recorded commands in order at ``rate_scale`` x the rate limits."""
        path, here = [], self.command()
        for w in waypoints:
            path += synchronized_path(here, w, self._caps, rate_scale)
            here = w
        last = waypoints[-1]
        req = {"height": last.height, "rpy": last.rpy, "arm": last.arm}
        return self._run(path, req, monitor)

    def _run(self, path, requested, monitor) -> ExecutionReport:
        if self.held is not None:
            return ExecutionReport(False, requested, refused=f"held: {self.held.reason}")
        report = ExecutionReport(False, requested)
        c = self.controller
        settle = [path[-1]] * round(SETTLE_SECONDS / TICK) if path else []
        for cmd in path + settle:
            c.set_command(0.0, 0.0, 0.0)
            c.set_posture(cmd.height, cmd.rpy)
            c.arm_target = np.array(cmd.arm, float)
            self.env.step(c)
            self.history.append(self.command())
            obs = self.observe()
            row: dict[str, object] = {"t": obs.t, "cmd_height": cmd.height, "cmd_rpy": cmd.rpy}
            measure = getattr(monitor, "measure", None)
            if measure is not None:
                row.update(measure(obs))
            report.trace.append(row)
            event = monitor(obs) if monitor is not None else None
            if event is None and c.stopped:
                event = StopEvent(obs.t, str(c.stopped))
            if event is not None:
                report.stops.append(event)
                if not c.stopped:
                    self.hold(event)  # a fallen robot only damps; nothing to hold
                break
        else:
            report.completed = True
        if report.trace:
            report.achieved = {k: v for k, v in report.trace[-1].items() if k.startswith("ach_")}
        return report
