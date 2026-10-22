"""A simulated fixed RGB-D camera: renders colour and depth images from the MuJoCo scene.

This is the camera *driver*, so it lives with the simulator. Everything downstream
(``dream/perception``) receives only ``RGBDFrame`` arrays and the camera's calibration.
No segmentation, geom ids or body poses leave this module.

Sensor model:
- depth noise of a structured-light camera, sigma(z) = 1 mm + 1.5 mm/m^2 * z^2 (about 2.5 mm
  at 1 m), plus quantisation to 1 mm;
- RGB noise of 3/255 per channel;
- NaN where nothing is hit.

The self mask is rendered from the robot's own geometry alone (its kinematic model at the
current joint angles). A pixel belongs to the robot if the robot surface there is the nearest
one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import mujoco
import numpy as np

from dream.types import CameraCalibration, RGBDFrame

CAMERA_NAME = "rgbd"  # the default camera
ROBOT_GEOM_GROUP = 2  # Menagerie Panda visual geoms; scene geoms are group 0
MAX_RANGE = 3.0  # m: beyond this the sensor reports no return


# A second viewpoint from the robot's side. Added only where an observability measurement
# showed it is needed: a lid swung open at the back of a compartment hides the cavity from the
# default camera (README, camera-driven discovery).
SIDE_CAMERA = dict(name="rgbd_side", position=(0.25, 0.65, 0.75), target=(0.6, 0.0, 0.0))


@dataclass(frozen=True)
class CameraSpec:
    """Where the camera is mounted and how it images. Defaults: in front of the workspace,
    facing the robot, about 1 m from the objects."""

    position: tuple[float, float, float] = (1.15, 0.0, 0.75)
    target: tuple[float, float, float] = (0.5, 0.0, 0.0)
    fovy_deg: float = 50.0
    width: int = 480
    height: int = 360
    depth_noise: tuple[float, float] = (0.001, 0.0015)  # sigma = a + b z^2 (m)
    rgb_noise: float = 3.0  # per channel, 0-255
    # Calibration error: the camera reports this much less than its true pose (dx, dy, dz in m,
    # then rotation about world x, y, z in degrees). Perception only ever sees the report.
    calibration_error: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    name: str = CAMERA_NAME

    def xyaxes(self) -> list[float]:
        f = np.subtract(self.target, self.position)
        f /= np.linalg.norm(f)
        x = np.cross(f, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(x, f)
        return [*x.tolist(), *y.tolist()]


def add_camera(world, spec: CameraSpec) -> None:
    """Add the camera to an ``MjSpec`` world body."""
    world.add_camera(
        name=spec.name, pos=list(spec.position), xyaxes=spec.xyaxes(), fovy=spec.fovy_deg
    )


class RGBDCamera:
    """Renders ``RGBDFrame``s. Renderers are per thread (each owns a GL context)."""

    def __init__(self, model: mujoco.MjModel, spec: CameraSpec, rng: np.random.Generator):
        self.model = model
        self.spec = spec
        self.rng = rng
        self._local = threading.local()
        self._scene_opt = mujoco.MjvOption()  # default: scene and robot visual groups
        self._robot_opt = mujoco.MjvOption()
        self._robot_opt.geomgroup[:] = 0
        self._robot_opt.geomgroup[ROBOT_GEOM_GROUP] = 1

    def calibration(self, data: mujoco.MjData) -> CameraCalibration:
        cam = data.cam(self.spec.name)
        err = np.asarray(self.spec.calibration_error, float)
        return CameraCalibration(
            width=self.spec.width,
            height=self.spec.height,
            fovy_deg=float(self.model.cam(self.spec.name).fovy[0]),
            position=cam.xpos - err[:3],
            rotation=_rotation(-np.deg2rad(err[3:])) @ cam.xmat.reshape(3, 3),
        )

    def _renderers(self) -> tuple[mujoco.Renderer, mujoco.Renderer, mujoco.Renderer]:
        """(colour, depth, robot-only depth) renderers; each holds its own abstract scene."""
        if not hasattr(self._local, "renderers"):
            h, w = self.spec.height, self.spec.width
            rs = (mujoco.Renderer(self.model, h, w) for _ in range(3))
            rgb, depth, robot = rs
            depth.enable_depth_rendering()
            robot.enable_depth_rendering()
            self._local.renderers = (rgb, depth, robot)
        return self._local.renderers

    def capture(self, data: mujoco.MjData, lock=None) -> RGBDFrame:
        """Render one frame. Only the (cheap) scene update holds ``lock``; the renders happen
        outside it, so a physics thread is never blocked for the duration of a render."""
        rgb_r, depth_r, robot_r = self._renderers()
        if lock is not None:
            lock.acquire()
        try:
            time = float(data.time)
            calibration = self.calibration(data)
            name = self.spec.name
            rgb_r.update_scene(data, camera=name, scene_option=self._scene_opt)
            depth_r.update_scene(data, camera=name, scene_option=self._scene_opt)
            robot_r.update_scene(data, camera=name, scene_option=self._robot_opt)
        finally:
            if lock is not None:
                lock.release()
        rgb = rgb_r.render().copy()
        depth = depth_r.render().astype(np.float32)
        robot_depth = robot_r.render().astype(np.float32)
        depth[depth > MAX_RANGE] = np.nan
        robot_depth[robot_depth > MAX_RANGE] = np.nan
        # Robot pixels: where the robot's own surface is (within 5 mm) the nearest surface.
        self_mask = np.isfinite(robot_depth) & ~(depth < robot_depth - 0.005)
        a, b = self.spec.depth_noise
        sigma = a + b * np.nan_to_num(depth) ** 2
        noisy = depth + self.rng.normal(size=depth.shape).astype(np.float32) * sigma
        noisy = (np.round(noisy * 1000) / 1000).astype(np.float32)  # 1 mm quantisation
        if self.spec.rgb_noise > 0:
            n = self.rng.normal(scale=self.spec.rgb_noise, size=rgb.shape)
            rgb = np.clip(rgb.astype(np.float32) + n, 0, 255).astype(np.uint8)
        return RGBDFrame(time, rgb, noisy, self_mask, calibration)

    def close(self) -> None:
        for r in getattr(self._local, "renderers", ()):
            r.close()


def _rotation(rv: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (rad)."""
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3)
    k = rv / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
