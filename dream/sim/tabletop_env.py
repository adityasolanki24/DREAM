"""Tabletop world: physics stepping, the Panda controller and a simulated object tracker.

The object tracker stands in for perception (README status: visual perception is planned). It
reports each cube's position and yaw with Gaussian noise, the same convention as the mechanism
tracker in ``env.py``. It reports nothing about physics: mass and friction, and whether an
object is grasped or inside the tray, must be inferred by decision-making code.

``privileged_*`` methods are for evaluation only.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from dream.robot.panda import HOME_QPOS, PandaController, SafetyLimits
from dream.sim.tabletop import TabletopSpec, build_model
from dream.types import ObjectObservation, RobotState


@dataclass(frozen=True)
class TrackerNoise:
    position_std: float = 0.002  # m
    yaw_std: float = np.deg2rad(2.0)  # rad


class TabletopEnv:
    """One tabletop scene. Physics advances only through ``step`` (one thread at a time)."""

    def __init__(
        self,
        spec: TabletopSpec,
        rng: np.random.Generator,
        noise: TrackerNoise | None = None,
        limits: SafetyLimits | None = None,
    ) -> None:
        self._spec = spec
        self.rng = rng
        self.noise = noise or TrackerNoise()
        self.model = build_model(spec)
        self.data = mujoco.MjData(self.model)
        self.robot = PandaController(self.model, self.data, limits)
        self.lock = threading.RLock()  # guards data between a physics thread and readers
        self.step_hook: Callable | None = None  # e.g. a live viewer sync
        self.background = False  # True while a physics thread advances time (live session)
        self.reset()

    def reset(self) -> None:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self.robot.qadr] = HOME_QPOS
            for i in self.robot.finger_q:
                self.data.qpos[i] = 0.04
            mujoco.mj_forward(self.model, self.data)
            self.robot.hold()
            self.robot.set_gripper(0.08)
            self.step(250)  # let contacts settle

    # ----- time ---------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(n):
                self.robot.tick()
                mujoco.mj_step(self.model, self.data)
                if self.step_hook is not None:
                    self.step_hook(self.model, self.data)

    def wait_until(self, predicate: Callable[[], bool], timeout: float) -> bool:
        """Wait until ``predicate()`` holds or ``timeout`` simulated seconds pass.

        Synchronous mode (tests, benchmarks): advances physics itself. When a background physics
        thread owns the clock (``background = True``, live sessions), it only polls.
        """
        end = self.time + timeout
        if self.background:
            while self.time < end:
                if predicate():
                    return True
                time.sleep(0.002)
            return predicate()
        while self.time < end:
            self.step(10)
            if predicate():
                return True
        return False

    # ----- sensing ----------------------------------------------------------------------

    def robot_state(self) -> RobotState:
        with self.lock:
            return self.robot.state()

    def observe_objects(self) -> list[ObjectObservation]:
        out = []
        with self.lock:
            for obj in self._spec.objects:
                pos = self.data.body(obj.name).xpos.copy()
                yaw = _yaw(self.data.body(obj.name).xquat)
                out.append(
                    ObjectObservation(
                        name=obj.name,
                        color=obj.name,
                        position=pos + self.rng.normal(scale=self.noise.position_std, size=3),
                        yaw=float(yaw + self.rng.normal(scale=self.noise.yaw_std)),
                        size=2 * obj.half_size,
                    )
                )
        return out

    def tray_region(self) -> tuple[np.ndarray, float]:
        """Visible tray geometry: (centre, inner half width). A static fixture the agent can see."""
        t = self._spec.tray
        return np.array([*t.xy, 0.0]), t.inner_half

    def fixtures(self) -> dict[str, tuple[np.ndarray, float]]:
        centre, half = self.tray_region()
        return {"tray": (centre, half)}

    def embodiment(self) -> SimPanda:
        return SimPanda(self)

    # ----- privileged (evaluation / tests only) ---------------------------------------------

    def privileged_object_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def privileged_spec(self) -> TabletopSpec:
        return self._spec


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class SimPanda:
    """The ``Embodiment`` facade over a simulated Panda: no access to the model or scene."""

    def __init__(self, env: TabletopEnv) -> None:
        self._env = env

    def state(self) -> RobotState:
        return self._env.robot_state()

    def move_tcp(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        with self._env.lock:
            return self._env.robot.set_tcp_target(pos, quat)

    def set_gripper(self, width: float) -> None:
        with self._env.lock:
            self._env.robot.set_gripper(width)

    def tracking_error(self) -> tuple[float, float]:
        with self._env.lock:
            return self._env.robot.tracking_error()

    def wait_until(self, predicate: Callable[[], bool], timeout: float) -> bool:
        return self._env.wait_until(predicate, timeout)

    def hold(self) -> None:
        with self._env.lock:
            self._env.robot.hold()
