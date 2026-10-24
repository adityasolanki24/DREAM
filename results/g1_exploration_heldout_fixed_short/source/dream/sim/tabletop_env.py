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

from dream.robot.panda import HOME_QPOS, Compliance, PandaController, SafetyLimits
from dream.sim.camera import CameraSpec, RGBDCamera
from dream.sim.compartment import INNER_HALF
from dream.sim.tabletop import SHOULDER_ARMATURE, TRAY_FLOOR_TOP, TabletopSpec, build_model
from dream.types import (
    ArticulatedObservation,
    CompliantMode,
    ObjectObservation,
    RGBDFrame,
    RobotState,
)


@dataclass(frozen=True)
class TrackerNoise:
    position_std: float = 0.002  # m
    yaw_std: float = np.deg2rad(2.0)  # rad


class TabletopEnv:
    """One tabletop scene. Physics advances only through ``step`` (one thread at a time).

    It also implements ``Perception`` as the *oracle* backend: noisy ground-truth trackers,
    kept for regression, upper-bound comparisons and diagnosing non-perception failures.
    """

    backend = "oracle"

    def __init__(
        self,
        spec: TabletopSpec,
        rng: np.random.Generator,
        noise: TrackerNoise | None = None,
        limits: SafetyLimits | None = None,
        shoulder_armature: float = SHOULDER_ARMATURE,
        camera: CameraSpec | None = None,
        light_scale: float = 1.0,
        camera_seed: int = 0,
        extra_cameras: tuple[CameraSpec, ...] = (),
    ) -> None:
        self._spec = spec
        self.rng = rng
        self.noise = noise or TrackerNoise()
        self.camera_spec = camera or CameraSpec()
        self.extra_camera_specs = tuple(extra_cameras)
        self.model = build_model(
            spec, shoulder_armature, self.camera_spec, light_scale, self.extra_camera_specs
        )
        self._extra_cameras: dict[str, RGBDCamera] = {}
        self._camera: RGBDCamera | None = None  # created on first use
        self.camera_seed = camera_seed
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

    def capture_frame(self) -> RGBDFrame:
        """One RGB-D frame from the fixed camera (sensor data only; see dream/sim/camera.py)."""
        if self._camera is None:
            # Its own noise stream, so the oracle tracker's draws are unchanged by camera use.
            rng = np.random.default_rng([self.camera_seed, 7])
            self._camera = RGBDCamera(self.model, self.camera_spec, rng)
        return self._camera.capture(self.data, self.lock)

    def frame_sources(self) -> list:
        """One capture function per mounted camera (the default camera first)."""
        sources = [self.capture_frame]
        for i, spec in enumerate(self.extra_camera_specs, start=1):
            sources.append(lambda spec=spec, i=i: self._capture_extra(spec, i))
        return sources

    def _capture_extra(self, spec: CameraSpec, i: int) -> RGBDFrame:
        cam = self._extra_cameras.get(spec.name)
        if cam is None:
            cam = RGBDCamera(self.model, spec, np.random.default_rng([self.camera_seed, 7, i]))
            self._extra_cameras[spec.name] = cam
        return cam.capture(self.data, self.lock)

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
        """Visible tray geometry: (floor-top centre, inner half width). A static fixture."""
        t = self._spec.tray
        return np.array([*t.xy, TRAY_FLOOR_TOP]), t.inner_half

    def fixtures(self) -> dict[str, tuple[np.ndarray, float]]:
        """Visible regions objects can be put into: name -> (floor-top centre, half width).

        The compartment appears only while part of its cavity is uncovered (see
        ``_visible_cavity``). A closed compartment has no visible cavity.
        """
        out = {}
        if self._spec.tray is not None:
            out["tray"] = self.tray_region()
        if self._spec.compartment is not None:
            cavity = self._visible_cavity()
            if cavity is not None:
                out["compartment"] = cavity
        return out

    def observe_articulated(self) -> list[ArticulatedObservation]:
        """Objects with a handle: handle pose from a noisy tracker, the visible box, and the
        handle's visible extent (oracle-assisted, like the tracker)."""
        comp = self._spec.compartment
        if comp is None:
            return []
        with self.lock:
            site = self.data.site("handle")
            pos = site.xpos + self.rng.normal(scale=self.noise.position_std, size=3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, site.xmat)
            quat = _perturb_quat(quat, self.rng.normal(scale=self.noise.yaw_std, size=3))
            lo, hi = self._compartment_aabb()
        return [
            ArticulatedObservation(
                "compartment",
                pos,
                quat,
                (lo + hi) / 2,
                (hi - lo) / 2,
                comp.handle_half.copy(),
                position_std=self.noise.position_std,
            )
        ]

    def _compartment_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """World AABB of the compartment's geoms in their current pose (its visible extent)."""
        bodies = {self.model.body("housing").id, self.model.body("compartment_part").id}
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] not in bodies:
                continue
            c = self.data.geom_xpos[g]
            size = self.model.geom_size[g]
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                size = np.full(3, size[0])
            r = np.abs(self.data.geom_xmat[g].reshape(3, 3)) @ size
            lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
        return lo, hi

    def _visible_cavity(self, step: float = 0.01) -> tuple[np.ndarray, float] | None:
        """Largest uncovered square of the cavity, seen from above like a top-down depth camera.

        Vertical rays are cast over a grid around the box (robot geometry ignored, as if several
        viewpoints were available). A cell is open if its ray first hits the cavity floor, or an
        object resting low inside the cavity, and not the roof or lid. Returns (centre, half
        width) of the largest open square, axis-aligned in the world, or None if it is too small.
        """
        comp = self._spec.compartment
        floor_id = self.model.geom("cavity_floor").id
        object_ids = {self.model.geom(f"{o.name}_geom").id for o in self._spec.objects}
        a = INNER_HALF
        xs = np.arange(-a + step / 2, a + 0.26, step)  # include a drawer pulled fully out
        ys = np.arange(-a + step / 2, a, step)
        open_cells = np.zeros((len(xs), len(ys)), bool)
        hit_z = np.zeros((len(xs), len(ys)))
        geomid = np.zeros(1, np.int32)
        groups = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # scene geoms only (robot is 2/3)
        with self.lock:
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    p = comp.to_world(np.array([x, y, 0.0]))
                    p[2] = 0.6
                    dist = mujoco.mj_ray(
                        self.model, self.data, p, np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid
                    )
                    hit_z[i, j] = 0.6 - dist
                    g = int(geomid[0])
                    open_cells[i, j] = dist >= 0 and (
                        g == floor_id or (g in object_ids and hit_z[i, j] < comp.height - 0.01)
                    )
        best, bi, bj = _largest_square(open_cells)
        if best < 3:
            return None
        # The floor height is the depth reading on open floor cells inside the square.
        sq = (slice(bi, bi + best), slice(bj, bj + best))
        floor_z = float(np.median(hit_z[sq][open_cells[sq]]))
        local = np.array([xs[bi] + (best - 1) * step / 2, ys[bj] + (best - 1) * step / 2, 0.0])
        centre = comp.to_world(local)
        centre[2] = floor_z
        half_local = best * step / 2
        # Axis-aligned square inscribed in the rotated one (the agent's regions are axis-aligned).
        half = half_local / (abs(np.cos(comp.yaw)) + abs(np.sin(comp.yaw)))
        return centre, float(half)

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

    def set_compliance(self, mode: CompliantMode | None) -> None:
        c = None
        if mode is not None:
            c = Compliance(stiffness=mode.stiffness, force_limit=mode.force_limit)
        with self._env.lock:
            self._env.robot.set_compliance(c)


def _largest_square(grid: np.ndarray) -> tuple[int, int, int]:
    """Side length (cells) and lower-left index of the largest all-True square (classic DP)."""
    n, m = grid.shape
    dp = np.zeros((n + 1, m + 1), int)
    best, bi, bj = 0, 0, 0
    for i in range(n):
        for j in range(m):
            if grid[i, j]:
                dp[i + 1, j + 1] = 1 + min(dp[i, j + 1], dp[i + 1, j], dp[i, j])
                if dp[i + 1, j + 1] > best:
                    best = dp[i + 1, j + 1]
                    bi, bj = i - best + 1, j - best + 1
    return best, bi, bj


def _perturb_quat(q: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, rotvec / max(angle, 1e-12), angle)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, q)
    return out
