"""Franka Panda embodiment: proprioception, a deterministic Cartesian controller and safety limits.

The controller is standard damped-least-squares differential IK on the TCP site (as in
MuJoCo's and Menagerie's diffik examples), with a null-space pull toward the home posture.
Each tick it turns a TCP pose target into joint position targets for the Menagerie position
servos. Safety is enforced here, below every higher layer, and cannot be bypassed by callers:

- TCP targets are clamped to a workspace box;
- commanded Cartesian speed is capped;
- joint targets are clipped to the actuator control ranges;
- if the contact force on the robot exceeds a limit, the controller latches a protective stop
  and holds the current posture until ``reset_safety`` is called.

This module reads only the robot's own joints, sites and contacts (proprioception, and force
sensing as a real Franka estimates it). It never reads object state.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from dream.types import RobotState

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ROBOT_BODIES = (
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
)  # fmt: skip
GRIPPER_MAX_WIDTH = 0.08  # m, both fingers fully open
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])  # Menagerie "home"
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # TCP z axis pointing at the table (w, x, y, z)


@dataclass(frozen=True)
class SafetyLimits:
    workspace_lo: tuple[float, float, float] = (0.2, -0.45, 0.005)  # m
    workspace_hi: tuple[float, float, float] = (0.8, 0.45, 0.6)  # m
    max_linear_speed: float = 0.25  # m/s
    max_angular_speed: float = 1.0  # rad/s
    max_contact_force: float = 80.0  # N, total normal force on robot links (excludes fingers)
    max_finger_force: float = 40.0  # N, net external force on hand + fingers (grip cancels)


@dataclass(frozen=True)
class Compliance:
    """Cartesian admittance: the hand behaves like a force-capped spring toward its target.

    Commanded pose dynamics: x_cmd' = (clip(K (x_target - x_cmd), F_max) + F_ext) / D. At
    equilibrium the hand pushes with at most F_max and yields wherever the environment pushes back.
    This is the force-capped spring the discovery model assumes, with Panda-specific
    parameters. Rotation follows the same law with torques about the TCP.
    """

    stiffness: float = 300.0  # N/m
    damping: float = 100.0  # N s/m; > dt * contact stiffness / 2 for a stable discrete loop
    force_limit: float = 15.0  # N
    rot_stiffness: float = 1.0  # N m/rad
    rot_damping: float = 0.3  # N m s/rad
    torque_limit: float = 1.0  # N m
    max_lag: float = 0.03  # m: x_cmd never strays further than this from the actual TCP


class PandaController:
    """Owns the arm and gripper actuators of a Panda inside a MuJoCo model."""

    INTEGRATION_DT = 0.05  # s, lead time of the joint target (Menagerie diffik convention)
    DAMPING = 1e-2  # DLS damping (lambda^2)
    MAX_JOINT_SPEED = 1.5  # rad/s, joint-space cap on the IK velocity
    FORCE_FILTER = 0.05  # per-step low-pass coefficient for the external force (~40 ms at 2 ms)
    NULL_GAIN = 0.5

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, limits: SafetyLimits | None = None
    ):
        self.m, self.d = model, data
        self.limits = limits or SafetyLimits()
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_act = np.array([model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.grip_act = model.actuator("actuator8").id
        self.tcp = model.site("tcp").id
        self.finger_q = [model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.robot_bodies = {model.body(b).id for b in ROBOT_BODIES}
        self.hand_bodies = {model.body(b).id for b in ("hand", "left_finger", "right_finger")}
        self.ctrl_lo = model.actuator_ctrlrange[self.arm_act, 0]
        self.ctrl_hi = model.actuator_ctrlrange[self.arm_act, 1]
        self.target_pos: np.ndarray | None = None
        self.target_quat = DOWN_QUAT.copy()
        self.gripper_width_cmd = GRIPPER_MAX_WIDTH
        self.protective_stop = False
        self.stop_reason = ""
        self.peak_contact_force = 0.0
        self.peak_finger_force = 0.0
        self.stop_contact_body: str | None = None
        self.external_force = np.zeros(3)  # N, net force of the world on hand + fingers
        self.filtered_force = np.zeros(3)  # N, first-order low-pass of external_force
        self.external_torque = np.zeros(3)  # N m about the TCP
        self.filtered_torque = np.zeros(3)
        self.compliance: Compliance | None = None
        self.cmd_pos = np.zeros(3)  # admittance state (compliant mode only)
        self.cmd_quat = DOWN_QUAT.copy()

    # ----- commands -------------------------------------------------------------------

    def set_tcp_target(self, pos: np.ndarray, quat: np.ndarray | None = None) -> np.ndarray:
        """Set the TCP goal. Returns the goal actually used after workspace clamping."""
        clamped = np.clip(
            np.asarray(pos, float), self.limits.workspace_lo, self.limits.workspace_hi
        )
        self.target_pos = clamped
        if quat is not None:
            q = np.asarray(quat, float)
            self.target_quat = q / np.linalg.norm(q)
        return clamped

    def set_gripper(self, width: float) -> None:
        self.gripper_width_cmd = float(np.clip(width, 0.0, GRIPPER_MAX_WIDTH))

    def hold(self) -> None:
        """Hold the current TCP pose (a safe state)."""
        self.target_pos = self.d.site_xpos[self.tcp].copy()
        self.target_quat = self._tcp_quat()

    def set_compliance(self, compliance: Compliance | None) -> None:
        """Enter (or leave, with None) compliant mode, starting from the current pose."""
        self.hold()
        self.compliance = compliance
        self.cmd_pos = self.target_pos.copy()
        self.cmd_quat = self.target_quat.copy()

    def reset_safety(self) -> None:
        self.protective_stop = False
        self.stop_reason = ""
        self.hold()

    # ----- per-tick control ------------------------------------------------------------

    def tick(self) -> None:
        """Compute actuator commands for the next physics step."""
        self._monitor_contacts()
        if self.target_pos is None:
            self.hold()
        q = self.d.qpos[self.qadr]
        if self.protective_stop:
            self.d.ctrl[self.arm_act] = q
        else:
            if self.compliance is not None:
                self._admittance_step()
            self.d.ctrl[self.arm_act] = np.clip(
                q + self._ik_step() * self.INTEGRATION_DT, self.ctrl_lo, self.ctrl_hi
            )
        self.d.ctrl[self.grip_act] = 255.0 * self.gripper_width_cmd / GRIPPER_MAX_WIDTH

    def _admittance_step(self) -> None:
        c, dt = self.compliance, self.m.opt.timestep
        spring = _clip_norm(c.stiffness * (self.target_pos - self.cmd_pos), c.force_limit)
        self.cmd_pos = self.cmd_pos + dt * (spring + self.filtered_force) / c.damping
        tcp = self.d.site_xpos[self.tcp]
        self.cmd_pos = tcp + _clip_norm(self.cmd_pos - tcp, c.max_lag)
        rot_spring = _clip_norm(
            c.rot_stiffness * _quat_error(self.target_quat, self.cmd_quat), c.torque_limit
        )
        omega = (rot_spring + self.filtered_torque) / c.rot_damping
        dq = np.zeros(4)
        angle = np.linalg.norm(omega) * dt
        mujoco.mju_axisAngle2Quat(dq, omega / max(np.linalg.norm(omega), 1e-12), angle)
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, dq, self.cmd_quat)
        self.cmd_quat = out / np.linalg.norm(out)

    def _ik_step(self) -> np.ndarray:
        lim = self.limits
        goal_pos, goal_quat = (
            (self.cmd_pos, self.cmd_quat)
            if self.compliance is not None
            else (self.target_pos, self.target_quat)
        )
        err = np.zeros(6)
        err[:3] = goal_pos - self.d.site_xpos[self.tcp]
        err[3:] = _quat_error(goal_quat, self._tcp_quat())
        twist = err / self.INTEGRATION_DT
        v, w = np.linalg.norm(twist[:3]), np.linalg.norm(twist[3:])
        if v > lim.max_linear_speed:
            twist[:3] *= lim.max_linear_speed / v
        if w > lim.max_angular_speed:
            twist[3:] *= lim.max_angular_speed / w
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.tcp)
        jac = np.vstack([jacp, jacr])[:, self.dadr]
        # Damped pseudo-inverse for both the task and the null-space projector: an undamped
        # pinv blows up near the shoulder singularity (joint2 ~ 0 makes joints 1 and 3 coaxial)
        # and made those joints chatter against their torque limits. Constant lambda^2 = 1e-2 was
        # measured against 3e-3, 1e-3 and singular-value-dependent damping on 40 dev scenes:
        # it was the only setting with no failures (others failed 2-6 of 40).
        jinv = jac.T @ np.linalg.inv(jac @ jac.T + self.DAMPING * np.eye(6))
        dq = jinv @ twist
        dq += (np.eye(7) - jinv @ jac) @ (self.NULL_GAIN * (HOME_QPOS - self.d.qpos[self.qadr]))
        peak = np.max(np.abs(dq))
        if peak > self.MAX_JOINT_SPEED:
            dq *= self.MAX_JOINT_SPEED / peak
        return dq

    def _monitor_contacts(self) -> None:
        """Estimate the external force on the hand and latch a protective stop if needed.

        Links: the summed normal-force magnitude of their contacts with the world (> 80 N stops).
        Hand and fingers: the *net* force vector on them from the world, which is what a wrist
        force/torque sensor (or Franka's external-wrench estimate) measures. The two fingers'
        opposing grip forces cancel in it, so a firm grasp is not mistaken for a collision.
        """
        links = 0.0
        hand = np.zeros(3)
        torque = np.zeros(3)
        tcp = self.d.site_xpos[self.tcp]
        f6 = np.zeros(6)
        strongest, strongest_body = 0.0, -1  # the world body pressing hardest on the hand
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            robot1, robot2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if robot1 == robot2:  # robot self-contact is excluded by the model; ignore world-world
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            body = b1 if robot1 else b2
            if body in self.hand_bodies:
                # mj_contactForce is in the contact frame (rows: normal, tangents), and it is the
                # force geom1 exerts on geom2. Convert it to the force on the robot body.
                f_world = c.frame.reshape(3, 3).T @ f6[:3]
                f_robot = -f_world if robot1 else f_world
                hand += f_robot
                torque += np.cross(c.pos - tcp, f_robot)
                if abs(f6[0]) > strongest:
                    strongest, strongest_body = abs(f6[0]), b2 if robot1 else b1
            else:
                links += abs(f6[0])
        self.peak_contact_force = max(self.peak_contact_force, links)
        self.external_force = hand
        self.filtered_force += self.FORCE_FILTER * (hand - self.filtered_force)
        self.external_torque = torque
        self.filtered_torque += self.FORCE_FILTER * (torque - self.filtered_torque)
        fingers = float(np.linalg.norm(hand))
        self.peak_finger_force = max(self.peak_finger_force, fingers)
        if self.protective_stop:
            return
        lim = self.limits
        if links > lim.max_contact_force:
            self.protective_stop = True
            self.stop_reason = f"link contact force {links:.0f} N > {lim.max_contact_force:.0f} N"
        elif fingers > lim.max_finger_force:
            self.protective_stop = True
            self.stop_reason = f"hand external force {fingers:.0f} N > {lim.max_finger_force:.0f} N"
            # Diagnostics for evaluation only; never reported through the Embodiment interface.
            self.stop_contact_body = mujoco.mj_id2name(
                self.m, mujoco.mjtObj.mjOBJ_BODY, strongest_body
            )

    # ----- proprioception -----------------------------------------------------------------

    def _tcp_quat(self) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[self.tcp])
        return q

    def state(self) -> RobotState:
        return RobotState(
            time=float(self.d.time),
            joint_pos=self.d.qpos[self.qadr].copy(),
            joint_vel=self.d.qvel[self.dadr].copy(),
            tcp_pos=self.d.site_xpos[self.tcp].copy(),
            tcp_quat=self._tcp_quat(),
            gripper_width=float(sum(self.d.qpos[a] for a in self.finger_q)),
            gripper_command=self.gripper_width_cmd,
            protective_stop=self.protective_stop,
            stop_reason=self.stop_reason,
            external_force=self.filtered_force.copy(),
        )

    def tracking_error(self) -> tuple[float, float]:
        """(position error m, orientation error rad) between the TCP and its target."""
        if self.target_pos is None:
            return 0.0, 0.0
        pos = float(np.linalg.norm(self.target_pos - self.d.site_xpos[self.tcp]))
        rot = float(np.linalg.norm(_quat_error(self.target_quat, self._tcp_quat())))
        return pos, rot


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * (limit / n) if n > limit else v


def _quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``target``."""
    conj = np.zeros(4)
    mujoco.mju_negQuat(conj, current)
    diff = np.zeros(4)
    mujoco.mju_mulQuat(diff, target, conj)
    if diff[0] < 0:
        diff = -diff
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, diff, 1.0)
    return vel


def yaw_down_quat(yaw: float) -> np.ndarray:
    """TCP pointing down with the gripper's closing direction rotated by ``yaw`` about world z."""
    zq = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, zq, DOWN_QUAT)
    return out
