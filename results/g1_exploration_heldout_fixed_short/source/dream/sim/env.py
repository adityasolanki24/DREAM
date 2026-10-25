"""MuJoCo environment: executes robot commands and returns noisy sensor observations.

The environment holds the ground-truth ``MechanismSpec``. Its public interface for the agent
(``geometry``, ``observe``, ``move_ee_to``, ``grasp``, ``execute_probe``) returns only
``dream.types`` objects built from MuJoCo *sensor* readings. Methods prefixed ``privileged_``
exist for evaluation and tests, and the agent must not call them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from dream.sim.mechanisms import EE_HOME, PANEL_CENTER, MechanismSpec, build_mjcf
from dream.types import Observation, PartGeometry, Probe, ProbeResult


@dataclass(frozen=True)
class SensorNoise:
    handle_pos_std: float = 0.001  # m, a good marker or point-cloud tracker
    handle_rot_std: float = np.deg2rad(0.5)  # rad
    ee_pos_std: float = 0.0002  # m, joint encoders
    force_std: float = 0.1  # N


class MechanismEnv:
    """One mechanism plus a force-limited Cartesian gantry gripper."""

    SAMPLE_HZ = 50.0
    GRASP_TOLERANCE = 0.01  # m

    def __init__(
        self,
        spec: MechanismSpec,
        rng: np.random.Generator,
        noise: SensorNoise | None = None,
    ) -> None:
        self._spec = spec
        self._rng = rng
        self.noise = noise or SensorNoise()
        self.mjcf = build_mjcf(spec)
        self.model = mujoco.MjModel.from_xml_string(self.mjcf)
        self.data = mujoco.MjData(self.model)
        self._grasp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
        self._sensor = {
            name: (self.model.sensor(name).adr[0], self.model.sensor(name).dim[0])
            for name in ("handle_pos", "handle_quat", "ee_pos", "frc_x", "frc_y", "frc_z")
        }
        mujoco.mj_forward(self.model, self.data)
        self._initial_handle = self._read("handle_pos").copy()
        self.step_hook = None  # optional callable(model, data) after every physics step

    # ----- agent-facing interface -------------------------------------------------------

    def geometry(self) -> PartGeometry:
        """Visible part geometry (perfect segmentation of the closed panel)."""
        return PartGeometry(
            panel_center=PANEL_CENTER.copy(),
            panel_half_extents=self._spec.panel_half_extents.copy(),
            handle_pos=self._initial_handle.copy(),
        )

    @property
    def grasped(self) -> bool:
        return bool(self.data.eq_active[self._grasp_id])

    def observe(self) -> Observation:
        n = self.noise
        quat = self._read("handle_quat").copy()
        quat = _perturb_quat(quat, self._rng.normal(scale=n.handle_rot_std, size=3))
        force = np.array([self._read(k)[0] for k in ("frc_x", "frc_y", "frc_z")])
        return Observation(
            time=float(self.data.time),
            handle_pos=self._read("handle_pos") + self._rng.normal(scale=n.handle_pos_std, size=3),
            handle_quat=quat,
            ee_pos=self._read("ee_pos") + self._rng.normal(scale=n.ee_pos_std, size=3),
            ee_force=force + self._rng.normal(scale=n.force_std, size=3),
        )

    def move_ee_to(self, target: np.ndarray, duration: float = 1.0, settle: float = 0.5) -> None:
        """Move the end effector in a straight line to a world-frame target."""
        self._set_force_limit(20.0)
        start = self.data.ctrl.copy()
        goal = np.asarray(target, dtype=float) - EE_HOME
        self._ramp(start, goal, duration)
        self._hold(goal, settle)

    def grasp(self) -> bool:
        """Close the gripper. This succeeds only if the end effector is at the handle."""
        mujoco.mj_forward(self.model, self.data)
        tip = self.data.site("ee_tip").xpos
        handle = self.data.site("handle").xpos
        if np.linalg.norm(tip - handle) > self.GRASP_TOLERANCE:
            return False
        self.data.eq_active[self._grasp_id] = 1
        return True

    def release(self) -> None:
        self.data.eq_active[self._grasp_id] = 0

    def execute_probe(self, probe: Probe, settle: float = 1.0) -> ProbeResult:
        """Drag the gripped handle along ``probe.direction`` under a force limit, then relax.

        The outcome is recorded at the end of the loaded hold. Afterwards the commanded target
        snaps to the current end-effector position so the mechanism is left unloaded.
        """
        if not self.grasped:
            raise RuntimeError("execute_probe requires a successful grasp first")
        self._set_force_limit(probe.max_force)
        before = self.observe()
        start = self._ee_local()
        goal = start + probe.direction * probe.distance
        traj = self._ramp(start, goal, probe.duration, record=True)
        traj += self._hold(goal, settle, record=True)
        after = self.observe()
        self._hold(self._ee_local(), 0.3)
        return ProbeResult(probe=probe, before=before, after=after, trajectory=traj)

    def follow_path(
        self, targets: list[np.ndarray], max_force: float, seconds_per_target: float = 0.1
    ) -> Observation:
        """Drag the gripped handle through a sequence of world-frame gripper targets."""
        if not self.grasped:
            raise RuntimeError("follow_path requires a successful grasp first")
        self._set_force_limit(max_force)
        start = self._ee_local()
        for target in targets:
            goal = np.asarray(target, dtype=float) - EE_HOME
            self._ramp(start, goal, seconds_per_target)
            start = goal
        self._hold(start, 0.5)
        return self.observe()

    # ----- privileged (evaluation / tests only) -----------------------------------------

    def privileged_joint_position(self) -> float:
        """True joint coordinate. Evaluation only."""
        if self._spec.joint_type == "fixed":
            return 0.0
        return float(self.data.joint("mech").qpos[0])

    def privileged_spec(self) -> MechanismSpec:
        return self._spec

    def privileged_fork(self, rng: np.random.Generator) -> MechanismEnv:
        """Independent copy of the current physical state, for "what would happen if" scoring."""
        twin = MechanismEnv(self._spec, rng, self.noise)
        twin.data = copy.copy(self.data)
        twin._initial_handle = self._initial_handle.copy()
        return twin

    # ----- internals ------------------------------------------------------------------

    def _read(self, name: str) -> np.ndarray:
        adr, dim = self._sensor[name]
        return self.data.sensordata[adr : adr + dim]

    def _ee_local(self) -> np.ndarray:
        return np.array([self.data.joint(j).qpos[0] for j in ("ee_x", "ee_y", "ee_z")])

    def _set_force_limit(self, max_force: float) -> None:
        self.model.actuator_forcerange[:] = [-max_force, max_force]

    def _ramp(
        self, start: np.ndarray, goal: np.ndarray, duration: float, record: bool = False
    ) -> list[Observation]:
        n_steps = max(1, int(round(duration / self.model.opt.timestep)))
        every = max(1, int(round(1.0 / (self.SAMPLE_HZ * self.model.opt.timestep))))
        out = []
        for i in range(n_steps):
            # Smoothstep profile: zero velocity at both ends, so no impulsive tugs.
            s = (i + 1) / n_steps
            s = s * s * (3 - 2 * s)
            self.data.ctrl[:] = start + s * (goal - start)
            mujoco.mj_step(self.model, self.data)
            if self.step_hook is not None:
                self.step_hook(self.model, self.data)
            if record and i % every == 0:
                out.append(self.observe())
        return out

    def _hold(self, goal: np.ndarray, duration: float, record: bool = False) -> list[Observation]:
        n_steps = max(1, int(round(duration / self.model.opt.timestep)))
        every = max(1, int(round(1.0 / (self.SAMPLE_HZ * self.model.opt.timestep))))
        out = []
        self.data.ctrl[:] = goal
        for i in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            if self.step_hook is not None:
                self.step_hook(self.model, self.data)
            if record and i % every == 0:
                out.append(self.observe())
        return out


def _perturb_quat(q: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, rotvec / max(angle, 1e-12), angle)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, q)
    return out
