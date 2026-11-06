"""Supported G1 rev 1.0 + seven-joint Unitree hands; proprioceptive torque PD only.

This is a classical commissioning controller, not a locomotion or grasp policy. Joint and
effort ranges come from the pinned Unitree MJCF. No scene/object state enters control.
"""

from __future__ import annotations

import mujoco
import numpy as np

from dream.types import DexterousState

FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
ARM = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


def hand_names(side: str) -> tuple[str, ...]:
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    return tuple(f"{side}_hand_{name}_joint" for name in FINGERS)


class G1Controller:
    """Rate-limited position setpoints driving upstream effort-limited motors at 1 kHz.

    Gains and speed caps are conservative commissioning choices, not hardware calibration.
    Targets stay inside mechanical limits; realized excursions latch a stop. Reset is a
    simulator initialization operation and is deliberately absent from this controller.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model, self.data = model, data
        self.names = tuple(model.actuator(i).name for i in range(model.nu))
        if len(self.names) != 43 or not set(hand_names("left") + hand_names("right")) <= set(
            self.names
        ):
            raise ValueError("expected 29 body + 14 hand actuators")
        self.joints = model.actuator_trnid[:, 0].copy()
        if len(set(self.joints)) != 43 or any(
            model.joint(j).name != n for j, n in zip(self.joints, self.names, strict=True)
        ):
            raise ValueError("G1 actuator/joint name mismatch")
        self.qadr = model.jnt_qposadr[self.joints].copy()
        self.dadr = model.jnt_dofadr[self.joints].copy()
        self.limits = model.jnt_range[self.joints].copy()
        self.effort = model.jnt_actfrcrange[self.joints].copy()
        if not np.all(model.jnt_actfrclimited[self.joints]) or not np.all(
            model.actuator_gear[:, 0] == 1
        ):
            raise ValueError("expected upstream joint effort limits and unit motor gears")
        hand = np.array(["_hand_" in n for n in self.names])
        self.kp = np.where(hand, 2.0, 80.0)  # Nm/rad, initial commissioning PD
        self.kd = np.where(hand, 0.06, 4.0)  # Nm s/rad
        if (
            not np.allclose(model.actuator_gainprm[:, 0], self.kp)
            or not np.allclose(model.actuator_biasprm[:, 1], -self.kp)
            or not np.allclose(model.actuator_biasprm[:, 2], -self.kd)
            or not np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        ):
            raise ValueError("incompatible G1 servo convention")
        self.speed = np.where(hand, 1.0, 0.4)  # rad/s, setpoint slew caps
        self.target = data.qpos[self.qadr].copy()
        self.reference = self.target.copy()
        self.stop_reason = ""
        self.root = model.body("pelvis").id
        self.palms = {s: model.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}
        self.cartesian_target: tuple[str, np.ndarray, np.ndarray] | None = None
        self.hand_bodies = {
            side: {
                model.body(i).id
                for i in range(model.nbody)
                if model.body(i).name.startswith(side + "_hand_")
            }
            for side in ("left", "right")
        }

    def state(self) -> DexterousState:
        return DexterousState(
            float(self.data.time),
            self.names,
            self.data.qpos[self.qadr].copy(),
            self.data.qvel[self.dadr].copy(),
            self.data.xpos[self.root].copy(),
            self.data.xquat[self.root].copy(),
            bool(self.stop_reason),
            self.stop_reason,
        )

    def palm_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        hand_names(side)
        body = self.palms[side]
        # Upstream palm mesh frame relative to wrist-yaw link; this is robot geometry.
        offset = np.array([0.0415, 0.003 if side == "left" else -0.003, 0.0])
        return (
            self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ offset,
            self.data.xquat[body].copy(),
        )

    def command_joints(self, positions: dict[str, float]) -> None:
        if self.stop_reason:
            raise ValueError(self.stop_reason)
        target = self.target.copy()
        for name, value in positions.items():
            if name not in self.names or not np.isfinite(value):
                raise ValueError("unknown joint or nonfinite target")
            i = self.names.index(name)
            if not self.limits[i, 0] <= value <= self.limits[i, 1]:
                raise ValueError("joint target outside mechanical range")
            target[i] = value
        self.target = target

    def set_hand(self, side: str, closure: float) -> None:
        """A scoped seven-joint synergy, not a general learned grasp controller."""
        names = hand_names(side)
        if not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError("closure must be in [0, 1]")
        # Interior targets avoid deliberately driving into joint stops. Mirrored axes
        # are explicit in Unitree's model; thumb opposition is a separate rotation.
        sign = 1 if side == "left" else -1
        opened = np.array([0, 0, 0.08, -0.08, -0.08, -0.08, -0.08])
        closed = np.array([0, 0.7, 1.2, -1.1, -1.2, -1.1, -1.2])
        values = sign * (opened + closure * (closed - opened))
        self.command_joints(dict(zip(names, values, strict=True)))

    def move_arm(self, side: str, positions: np.ndarray) -> None:
        hand_names(side)
        q = np.asarray(positions, float)
        if q.shape != (7,):
            raise ValueError("arm requires seven joint positions in radians")
        self.command_joints(dict(zip((f"{side}_{n}_joint" for n in ARM), q, strict=True)))
        self.cartesian_target = None

    def grasp_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Centre of the supported finger synergy, from robot kinematics only.

        This replaces the previous scene-side offset with an explicit robot tool frame.
        It describes a supported grasp morphology, not an observed object location.
        """
        pos, quat = self.palm_pose(side)
        offset = np.array([0.09, -0.045 if side == "left" else 0.045, 0.0])
        rot = self.data.xmat[self.palms[side]].reshape(3, 3)
        return pos + rot @ offset, quat

    def finger_forces(self, side: str) -> dict[str, float]:
        """Simulated tactile normal loads on own finger links, no contacted-object IDs.

        As with Panda force sensing, contacts are the simulated sensor source. Object
        identity, pose, mass and joint type are never queried or returned.
        """
        hand_names(side)
        forces = {name: 0.0 for name in ("thumb", "index", "middle")}
        for i, contact in enumerate(self.data.contact):
            own = [
                int(self.model.geom_bodyid[g])
                for g in (contact.geom1, contact.geom2)
                if self.model.geom_bodyid[g] in self.hand_bodies[side]
            ]
            if not own:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            for body in own:
                chain = self.model.body(body).name.split("_hand_")[1].split("_")[0]
                forces[chain] += max(0, float(wrench[0]))
        return forces

    def reach(self, side: str, position: np.ndarray, quaternion: np.ndarray) -> None:
        """World-frame SI target; DLS runs below tools, never accepts scene state."""
        hand_names(side)
        pos, quat = np.asarray(position, float), np.asarray(quaternion, float)
        # Commissioning workspace around the supported torso, deliberately bounded.
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.isfinite(pos).all()
            or not np.isfinite(quat).all()
            or not np.isclose(np.linalg.norm(quat), 1)
            or np.any(pos < [0.20, -0.35, 0.74])
            or np.any(pos > [0.55, 0.35, 1.20])
        ):
            raise ValueError("invalid or out-of-workspace Cartesian target")
        if self.stop_reason:
            raise ValueError(self.stop_reason)
        self.cartesian_target = side, pos.copy(), quat.copy()

    def _reach_step(self) -> None:
        side, goal, quat = self.cartesian_target
        pos, _ = self.grasp_pose(side)
        body = self.palms[side]
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, jr, pos, body)
        ids = np.array([self.names.index(f"{side}_{n}_joint") for n in ARM])
        desired = np.empty(9)
        mujoco.mju_quat2Mat(desired, quat)
        current = self.data.xmat[body].reshape(3, 3)
        rotation_error = 0.5 * sum(
            np.cross(current[:, i], desired.reshape(3, 3)[:, i]) for i in range(3)
        )
        # Scale rotational rows into metres (8 cm tool lever) before DLS, as in
        # DREAM's existing differential IK; damping avoids singular inverses.
        jac = np.vstack([jp[:, self.dadr[ids]], 0.08 * jr[:, self.dadr[ids]]])
        velocity = 4 * np.concatenate([goal - pos, 0.08 * rotation_error])
        linear_norm = np.linalg.norm(velocity[:3])
        velocity[:3] *= min(1, 0.04 / max(linear_norm, 1e-12))  # m/s commissioning cap
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), velocity)
        dq *= min(1, 0.4 / max(np.max(np.abs(dq)), 1e-12))  # preserve arm slew cap
        # 100 ms reference lead; existing rate/effort/range guards still apply.
        self.target[ids] = np.clip(
            self.data.qpos[self.qadr[ids]] + 0.1 * dq, self.limits[ids, 0], self.limits[ids, 1]
        )

    def hold(self) -> None:
        self.cartesian_target = None
        self.target = self.data.qpos[self.qadr].copy()
        self.reference = self.target.copy()

    def hold_for_reasoning(self) -> None:
        """Stop body motion while retaining existing finger references under load.

        Resetting a compressed finger's reference to its measured joint angle removes
        its contact effort. Safety faults still use hold(), and all effort/slew/range
        checks remain active. This method never increases the existing grip command.
        """
        if self.stop_reason:
            self.hold()
            return
        self.cartesian_target = None
        body = np.array(["_hand_" not in name for name in self.names])
        self.target[body] = self.data.qpos[self.qadr[body]]
        self.reference[body] = self.target[body]

    def check_safety(self) -> None:
        """Latch a fault on current measurements, including the final physics step."""
        if self.stop_reason:
            return
        q, dq = self.data.qpos[self.qadr], self.data.qvel[self.dadr]
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            self.stop_reason = "nonfinite proprioception"
        elif np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            self.stop_reason = "measured joint outside mechanical range"
        elif np.any(np.abs(dq) > 20.0):  # rad/s, commissioning overspeed stop
            self.stop_reason = "joint overspeed"

    def tick(self) -> None:
        self.check_safety()
        if self.stop_reason:
            self.hold()
            self.reference = np.clip(self.reference, self.limits[:, 0], self.limits[:, 1])
            self.data.ctrl[:] = np.nan_to_num(
                self.reference + self.data.qfrc_bias[self.dadr] / self.kp
            )
            return
        dt = self.model.opt.timestep
        if self.cartesian_target is not None:
            self._reach_step()
        self.reference += np.clip(self.target - self.reference, -self.speed * dt, self.speed * dt)
        # Affine actuator bias integrates damping implicitly. Gravity compensation
        # is an equivalent reference offset; actuator AND joint effort clamps remain.
        self.data.ctrl[:] = self.reference + self.data.qfrc_bias[self.dadr] / self.kp
