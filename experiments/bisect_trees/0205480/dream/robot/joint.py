"""Separate experimental Panda absolute-joint controller, never a Cartesian fallback.

Uses the existing robot proprioception and force monitor. Own-robot kinematics check a
smooth path; no object pose or simulator contact identity is used to select a motion.
Collision protection is reactive, not a collision-free planning guarantee.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from dream.robot.panda import GRIPPER_MAX_WIDTH, PandaController


@dataclass(frozen=True)
class JointLimits:
    # Conservative prototype engineering limits, fixed before physical validation.
    command_speed_rad_s: float = 0.5
    command_accel_rad_s2: float = 1.0
    measured_speed_rad_s: float = 1.0
    measured_accel_rad_s2: float = 4.0
    link_contact_N: float = 5.0  # stricter than classical 80 N; not a grasp-force limit
    gripper_speed_m_s: float = 0.05  # original DROID gripper.goto speed
    gripper_effort_N: float = 30.0  # conservative prototype cap on the hand's actuator effort
    watchdog_s: float = 5.0


class JointTargetController(PandaController):
    """Rest-to-rest quintic joint setpoints, stretched rather than applied instantaneously.

    The inherited contact/proprioception functions read only the robot's own sensors.
    The default TabletopEnv keeps its classical controller; this adapter is opt-in.
    """

    def __init__(self, model, data, limits=None, *, clock=time.monotonic):
        super().__init__(model, data, limits)
        self.joint_limits = JointLimits()
        self.clock = clock
        self._kin = mujoco.MjData(model)
        self._active = False
        self._deadline = None
        self._last_velocity = data.qvel[self.dadr].copy()
        self.command_velocity = np.zeros(7)
        self.command_acceleration = np.zeros(7)
        self.hold()

    def hold(self) -> None:
        """Cancel active motion; retain completed setpoints during inference.

        Resetting a settled servo to its lagging measured position creates a command
        discontinuity. Active cancellation instead holds the measured position and
        retains measured safety monitoring; it does not continue the unfinished path.
        """
        super().hold()
        if self._active or not hasattr(self, "joint_command"):
            self.joint_command = self.d.qpos[self.qadr].copy()
        self._active = False
        self._deadline = None
        self.command_velocity = np.zeros(7)
        self.command_acceleration = np.zeros(7)

    def set_tcp_target(self, *args, **kwargs):
        raise RuntimeError("joint adapter has no Cartesian fallback")

    def set_compliance(self, *args, **kwargs):
        raise RuntimeError("joint adapter has no compliant-control fallback")

    def _kinematics(self, q):
        self._kin.qpos[self.qadr] = q
        for index in self.finger_q:
            self._kin.qpos[index] = self.d.qpos[index]
        mujoco.mj_kinematics(self.m, self._kin)
        mujoco.mj_comPos(self.m, self._kin)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self._kin, jp, jr, self.tcp)
        return self._kin.site_xpos[self.tcp].copy(), jp[:, self.dadr], jr[:, self.dadr]

    def submit(self, joint_rad: np.ndarray, width_m: float, period_s: float) -> dict:
        """Validate one absolute target. Caller must log the raw model chunk first."""
        self.hold()  # a rejected replacement must not leave the old target moving
        q = np.asarray(joint_rad, dtype=float)
        if (
            q.shape != (7,)
            or not np.isfinite(q).all()
            or np.any(q < self.ctrl_lo)
            or np.any(q > self.ctrl_hi)
        ):
            raise ValueError("invalid absolute joint target")
        if not np.isfinite(width_m) or not 0 <= width_m <= GRIPPER_MAX_WIDTH:
            raise ValueError("invalid physical gripper width")
        if not np.isfinite(period_s) or not 0 < period_s <= 1:
            raise ValueError("invalid nominal sample period")
        if self.protective_stop:
            raise ValueError("robot protective stop latched")
        if np.max(np.abs(self.d.qvel[self.dadr])) > 0.02:
            # Rest-to-rest profiles must start near rest. 0.02 rad/s is the acceptance gate.
            raise ValueError("joint trajectory must start near rest")
        # Preserve servo-command continuity across completed trajectories and holds.
        self._start = self.joint_command.copy()
        self._delta = q - self._start
        lim = self.joint_limits
        # Exact quintic maxima: max ds/du=1.875, max |d²s/du²|=10/sqrt(3).
        duration = max(
            period_s,
            1.875 * np.max(np.abs(self._delta)) / lim.command_speed_rad_s,
            np.sqrt((10 / np.sqrt(3)) * np.max(np.abs(self._delta)) / lim.command_accel_rad_s2),
        )
        for u in np.linspace(0, 1, 65):
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            derivative = 30 * u**2 * (1 - u) ** 2 * self._delta
            pos, jp, jr = self._kinematics(self._start + s * self._delta)
            if np.any(pos < self.limits.workspace_lo) or np.any(pos > self.limits.workspace_hi):
                raise ValueError("joint path leaves TCP workspace")
            # 5% sampling margin; actual measured constraints are checked every physics tick.
            duration = max(
                duration,
                1.05 * np.linalg.norm(jp @ derivative) / self.limits.max_linear_speed,
                1.05 * np.linalg.norm(jr @ derivative) / self.limits.max_angular_speed,
            )
        if duration >= lim.watchdog_s:
            raise ValueError("safe interpolation exceeds watchdog budget")
        self._width_start = self.gripper_width_cmd
        self._width_goal = width_m
        duration = max(duration, 1.875 * abs(width_m - self._width_start) / lim.gripper_speed_m_s)
        self._duration = float(duration)
        self._start_time = float(self.d.time)
        self._deadline = self.clock() + lim.watchdog_s
        self._active = True
        return {
            "controller": "absolute_joint_quintic",
            "nominal_period_s": period_s,
            "duration_s": duration,
            "time_dilation": duration / period_s,
        }

    @property
    def moving(self) -> bool:
        return self._active

    def _stop(self, reason: str) -> None:
        self.protective_stop = True
        self.stop_reason = reason
        self.hold()

    def tick(self) -> None:
        self._monitor_contacts()
        qvel = self.d.qvel[self.dadr].copy()
        accel = (qvel - self._last_velocity) / self.m.opt.timestep
        self._last_velocity = qvel
        pos = self.d.site_xpos[self.tcp]
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.tcp)
        lim = self.joint_limits
        q = self.d.qpos[self.qadr]
        if np.any(q < self.ctrl_lo) or np.any(q > self.ctrl_hi):
            self._stop("joint mode measured position")
        elif abs(self.d.actuator_force[self.grip_act]) > lim.gripper_effort_N:
            self._stop("joint mode gripper effort")
        elif self.peak_contact_force > lim.link_contact_N:
            self._stop("joint mode link contact")
        elif np.any(pos < self.limits.workspace_lo) or np.any(pos > self.limits.workspace_hi):
            self._stop("joint mode workspace")
        elif np.max(np.abs(qvel)) > lim.measured_speed_rad_s:
            self._stop("joint mode measured velocity")
        elif np.max(np.abs(accel)) > lim.measured_accel_rad_s2:
            self._stop("joint mode measured acceleration")
        elif (
            np.linalg.norm(jp[:, self.dadr] @ qvel) > self.limits.max_linear_speed
            or np.linalg.norm(jr[:, self.dadr] @ qvel) > self.limits.max_angular_speed
        ):
            self._stop("joint mode measured TCP speed")
        elif self._deadline is not None and self.clock() > self._deadline:
            self._stop("joint mode watchdog")
        if self.protective_stop:
            self.hold()
        if self._active:
            u = float(np.clip((self.d.time - self._start_time) / self._duration, 0, 1))
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            self.joint_command = self._start + s * self._delta
            self.command_velocity = 30 * u**2 * (1 - u) ** 2 * self._delta / self._duration
            self.command_acceleration = (
                (60 * u - 180 * u**2 + 120 * u**3) * self._delta / self._duration**2
            )
            self.gripper_width_cmd = self._width_start + s * (self._width_goal - self._width_start)
            if u >= 1:
                self._active = False
        self.d.ctrl[self.arm_act] = self.joint_command
        self.d.ctrl[self.grip_act] = 255 * self.gripper_width_cmd / GRIPPER_MAX_WIDTH
