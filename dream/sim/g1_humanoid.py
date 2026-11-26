"""Free-standing G1 behind DREAM's shared humanoid capability interface."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dream.robot.g1_supervisor import WholeBodySupervisor
from dream.robot.g1_wbc import N_ACTION, N_BODY
from dream.robot.wholebody import StopEvent, WBCBackend
from dream.sim.g1_mobile import BODY_JOINTS, HAND_FINGERS, HAND_JOINTS, HAND_SIDES, G1MobileEnv
from dream.types import DexterousState, RGBDFrame


def _side_slice(side: str, width: int) -> slice:
    if side not in HAND_SIDES:
        raise ValueError(f"unknown humanoid side {side!r}")
    start = HAND_SIDES.index(side) * width
    return slice(start, start + width)


class G1Humanoid:
    """Share one environment, WBC, and protective stop across walking and manipulation."""

    def __init__(
        self, env: G1MobileEnv, backend: WBCBackend, supervisor: WholeBodySupervisor
    ) -> None:
        if backend.env is not env or not env.hands:
            raise ValueError("humanoid requires its own WBC backend and both hands")
        self.env, self.backend, self.supervisor = env, backend, supervisor
        finger_joints = env.model.actuator_trnid[env.finger_actuators, 0]
        self._finger_dofs = env.model.jnt_dofadr[finger_joints]

    @property
    def stop_reason(self) -> str:
        held = self.backend.held
        return self.backend.controller.stopped or (held.reason if held else "")

    def state(self) -> DexterousState:
        obs = self.backend.observe()
        prop = obs.prop
        height = self.supervisor.pm.kinematic_state(prop.quat, prop.q)["height"]
        return DexterousState(
            obs.t,
            (*BODY_JOINTS, *HAND_JOINTS),
            np.r_[prop.q, self.env.hand_positions()],
            np.r_[prop.dq, self.env.data.qvel[self._finger_dofs]],
            np.array([*obs.pose[:2], height]),
            prop.quat,
            bool(self.stop_reason),
            self.stop_reason,
        )

    def odometry(self) -> np.ndarray:
        return self.env.odometry()

    def capture(self) -> RGBDFrame:
        return self.env.capture_head()

    def finger_forces(self, side: str) -> dict[str, float]:
        if side not in HAND_SIDES:
            raise ValueError(f"unknown humanoid side {side!r}")
        return self.env.touch(side)

    def finger_positions(self, side: str) -> np.ndarray:
        return self.env.hand_positions()[_side_slice(side, len(HAND_FINGERS))]

    def set_velocity(self, forward: float, lateral: float, turn: float) -> None:
        if not self.stop_reason:
            self.backend.controller.set_command(forward, lateral, turn)

    def move_arm(self, side: str, positions: np.ndarray) -> bool:
        width = (N_BODY - N_ACTION) // len(HAND_SIDES)
        indices = _side_slice(side, width)
        positions = np.asarray(positions, float)
        limits = self.supervisor.pm.joint_range[N_ACTION:][indices]
        if (
            positions.shape != (width,)
            or not np.isfinite(positions).all()
            or np.any(positions < limits[:, 0])
            or np.any(positions > limits[:, 1])
        ):
            raise ValueError(f"{side} arm targets outside joint limits")
        if self.stop_reason or not self.backend.capabilities().executable:
            return False
        start = self.backend.command()
        arm = start.arm.copy()
        arm[indices] = positions
        return self.backend.follow([replace(start, arm=arm)], monitor=self.supervisor).completed

    def set_hand(self, side: str, positions: np.ndarray) -> None:
        width = len(HAND_FINGERS)
        indices = _side_slice(side, width)
        positions = np.asarray(positions, float)
        if positions.shape != (width,) or not np.isfinite(positions).all():
            raise ValueError(f"{side} hand needs {width} finite joint targets")
        if not self.stop_reason:
            targets = self.env.hand_targets.copy()
            targets[indices] = positions
            self.env.set_hand(targets)

    def step(self) -> bool:
        if self.stop_reason:
            return False
        self.env.step(self.backend.controller)
        obs = self.backend.observe()
        stop = self.supervisor(obs)
        if stop and not self.backend.controller.stopped:
            self.backend.hold(stop)
        return not self.stop_reason

    def hold(self, reason: str) -> None:
        if not self.stop_reason:
            self.backend.hold(StopEvent(self.env.time, reason))
