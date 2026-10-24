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

    def hold(self) -> None:
        self.target = self.data.qpos[self.qadr].copy()
        self.reference = self.target.copy()

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
        self.reference += np.clip(self.target - self.reference, -self.speed * dt, self.speed * dt)
        # Affine actuator bias integrates damping implicitly. Gravity compensation
        # is an equivalent reference offset; actuator AND joint effort clamps remain.
        self.data.ctrl[:] = self.reference + self.data.qfrc_bias[self.dadr] / self.kp
