"""A KINEMATIC SURROGATE mobile base: no legged dynamics, no falls.

For fast development and large-sample policy comparisons only. Headline results use the real
walking G1 (dream/sim/g1_mobile.py). Same ``MobileBase`` protocol, same head camera pose
relative to the floor footprint, same velocity-command limits, same oracle-assisted odometry
noise model, but the "robot" is a 0.25 m radius capsule on a body that slides over the
floor of a ``HomeSpec`` scene and is stopped by walls and furniture.

Walking model: the commanded body-frame velocity is tracked with a first-order lag (time
constant ``LAG_TAU``) and integrated in 0.02 s substeps (the G1 policy period). A substep that
would put the capsule in contact with the scene is not taken; the base stays at its last free
pose, the contact is logged for evaluation (``contact_log``, not part of the protocol), and
``walk`` carries on. ``stop_reason`` is always None.

Odometry is ORACLE-ASSISTED exactly as for ``G1MobileEnv``: true increments with the same
``OdometryNoise``. ``true_pose`` and ``contact_log`` are for evaluators only.
"""

from __future__ import annotations

from dataclasses import replace
from math import cos, exp, sin

import mujoco
import numpy as np

from dream.robot.g1_wbc import MAX_COMMAND
from dream.sim.camera import MAX_RANGE, ROBOT_GEOM_GROUP, CameraSpec, RGBDCamera
from dream.sim.g1_mobile import DEFAULT_ODOMETRY_NOISE, OdometryNoise, wrap
from dream.sim.home import HomeSpec, add_home, semantic_index
from dream.types import RGBDFrame

DT = 0.02  # s, one substep = the G1 policy period (DT_PHYSICS * DECIMATION)
LAG_TAU = 0.2  # s. ASSUMED: approximates the whole-body controller's velocity tracking lag
CAPSULE_RADIUS = 0.25  # m, the G1's torso/arm footprint with margin
CAPSULE_HEIGHT = 1.3  # m
# Head camera relative to the floor point under the pelvis, in the pelvis yaw frame. Computed
# once from build_mobile_g1 forward kinematics at the standing reset pose (pelvis 0.79 m):
# position (0.0760365, 0, 1.264), optical axis pitched 35 degrees down. tests/test_kinematic_base
# compares it against the G1 model again (2 cm, 1 degree).
_PITCH = np.radians(35.0)
_CAM_POS = (0.0760365, 0.0, 1.264)
HEAD_CAMERA = CameraSpec(
    name="head_rgbd",
    position=_CAM_POS,
    target=(_CAM_POS[0] + np.cos(_PITCH), 0.0, _CAM_POS[2] - np.sin(_PITCH)),
    fovy_deg=58.0,
    width=320,
    height=240,
    mount_body="base",
)


def _rz(yaw: float) -> np.ndarray:
    c, s = cos(yaw), sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _local_rotation() -> np.ndarray:
    x, y = np.array(HEAD_CAMERA.xyaxes()).reshape(2, 3)
    return np.stack([x, y, np.cross(x, y)], axis=1)  # MuJoCo camera frame columns


_R_LOCAL = _local_rotation()


def head_camera_pose(x: float, y: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """World (position, rotation) of the head camera for a footprint at (x, y, yaw)."""
    r = _rz(yaw)
    return np.array([x, y, 0.0]) + r @ np.array(_CAM_POS), r @ _R_LOCAL


class _FastCamera(RGBDCamera):
    """``RGBDCamera`` with the same sensor model, minus the robot-only render (the camera sits
    inside the footprint capsule, so the self mask would be empty) and with float32 noise draws.
    Brings a capture from about 23 ms to under 15 ms on the development laptop."""

    def _renderers(self):
        if not hasattr(self._local, "renderers"):
            h, w = self.spec.height, self.spec.width
            rgb, depth = mujoco.Renderer(self.model, h, w), mujoco.Renderer(self.model, h, w)
            depth.enable_depth_rendering()
            self._local.renderers = (rgb, depth)
        return self._local.renderers

    def capture(self, data, lock=None) -> RGBDFrame:
        rgb_r, depth_r = self._renderers()
        name = self.spec.name
        rgb_r.update_scene(data, camera=name, scene_option=self._scene_opt)
        depth_r.update_scene(data, camera=name, scene_option=self._scene_opt)
        rgb = rgb_r.render().copy()
        depth = depth_r.render().astype(np.float32)
        depth[depth > MAX_RANGE] = np.nan
        a, b = self.spec.depth_noise
        sigma = a + b * np.nan_to_num(depth) ** 2
        noisy = depth + self.rng.standard_normal(depth.shape, dtype=np.float32) * sigma
        noisy = (np.round(noisy * 1000) / 1000).astype(np.float32)  # 1 mm quantisation
        n = self.rng.standard_normal(rgb.shape, dtype=np.float32)
        n *= self.spec.rgb_noise
        n += rgb
        rgb = np.clip(n, 0, 255, out=n).astype(np.uint8)
        return RGBDFrame(
            float(data.time), rgb, noisy, np.zeros(depth.shape, bool), self.calibration(data)
        )


def build_kinematic_model(home: HomeSpec) -> mujoco.MjModel:
    """The home plus a floor and the robot body (capsule in ROBOT_GEOM_GROUP + camera)."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30.0, 30.0, 0.1],
        rgba=[0.45, 0.45, 0.47, 1.0], contype=0, conaffinity=0,
    )  # fmt: skip
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.worldbody.add_light(
        pos=[0, 0, 4],
        dir=[0.2, 0.1, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.6, 0.6, 0.6],
    )
    add_home(spec, home)
    # Planar joints, not a mocap body: MuJoCo never collides two bodies welded to the world.
    base = spec.worldbody.add_body(name="base")
    base.add_joint(name="x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    base.add_joint(name="y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    base.add_joint(name="yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    r = CAPSULE_RADIUS
    base.add_geom(
        name="robot_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE, group=ROBOT_GEOM_GROUP,
        fromto=[0, 0, r, 0, 0, CAPSULE_HEIGHT - r], size=[r, 0, 0], rgba=[0.2, 0.3, 0.8, 1.0],
    )  # fmt: skip
    base.add_camera(
        name=HEAD_CAMERA.name, pos=list(HEAD_CAMERA.position), xyaxes=HEAD_CAMERA.xyaxes(),
        fovy=HEAD_CAMERA.fovy_deg,
    )  # fmt: skip
    return spec.compile()


class KinematicBase:
    """``MobileBase`` over a ``HomeSpec`` scene. See the module docstring for what it is not."""

    def __init__(
        self,
        home: HomeSpec,
        rng: np.random.Generator,
        noise: OdometryNoise = DEFAULT_ODOMETRY_NOISE,
        semantic=None,
    ) -> None:
        """``semantic``: a ``SemanticNoise`` to attach an oracle-assisted detector, or None."""
        cam_rng, self._rng, sem_rng = rng.spawn(3)  # independent streams per consumer
        self.home, self.noise = home, noise
        self.model = build_kinematic_model(home)
        self.data = mujoco.MjData(self.model)
        self._capsule = self.model.geom("robot_capsule").id
        self._camera = _FastCamera(self.model, HEAD_CAMERA, cam_rng)
        self._semantic = None
        if semantic is not None:
            from dream.sim.semantic_sensor import SemanticCamera

            self._semantic = SemanticCamera(
                self.model, HEAD_CAMERA, semantic_index(self.model, home), sem_rng, semantic
            )
        self._pose = np.array(home.start, float)
        self._vel = np.zeros(3)
        self._time = 0.0
        self._odom = np.zeros(3)
        self._bias = float(self._rng.normal(scale=noise.gyro_bias_std))
        self._contacts: list[tuple[float, str, tuple[float, float]]] = []
        self._place(self._pose)

    # -- MobileBase --------------------------------------------------------------------
    @property
    def time(self) -> float:
        return self._time

    @property
    def stop_reason(self) -> str | None:
        return None  # a kinematic base never falls or safety-stops

    def odometry(self) -> np.ndarray:
        """(x, y, yaw) in the odometry frame. Oracle-assisted (module docstring)."""
        return self._odom.copy()

    def capture(self) -> RGBDFrame:
        mujoco.mj_forward(self.model, self.data)
        frame = self._camera.capture(self.data)
        x, y, yaw = self._odom
        r = _rz(yaw)
        calibration = replace(
            frame.calibration,
            position=np.array([x, y, 0.0]) + r @ np.array(_CAM_POS),
            rotation=r @ _R_LOCAL,
        )  # the camera's pose in the odometry frame, as G1MobileEnv.capture_head reports it
        detections = None  # no detector ran; an empty tuple would mean it found nothing
        if self._semantic is not None:
            detections = self._semantic.detect(self.data, frame.depth)
        return replace(frame, calibration=calibration, detections=detections)

    def walk(self, forward: float, lateral: float, turn: float, seconds: float) -> bool:
        cmd = np.array([forward, lateral, turn], float)
        if not np.isfinite(cmd).all() or np.any(np.abs(cmd) > MAX_COMMAND + 1e-9):
            raise ValueError(f"velocity command {cmd.tolist()} outside {MAX_COMMAND.tolist()}")
        alpha = 1.0 - exp(-DT / LAG_TAU)
        for _ in range(max(1, round(seconds / DT))):
            self._vel += alpha * (cmd - self._vel)
            self._substep()
            self._time += DT
        return True

    # -- evaluation only ---------------------------------------------------------------
    def true_pose(self) -> np.ndarray:
        """World (x, y, yaw) of the footprint: evaluator only."""
        return self._pose.copy()

    def contact_log(self) -> list[tuple[float, str, tuple[float, float]]]:
        """(time, scene geom name, footprint xy) of every substep blocked by a contact."""
        return list(self._contacts)

    # -- internals ---------------------------------------------------------------------
    def _place(self, pose: np.ndarray) -> None:
        self.data.qpos[:3] = pose

    def _blocked_by(self) -> str | None:
        """Name of a scene geom the capsule penetrates at the placed pose, else None."""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._capsule in (c.geom1, c.geom2) and c.dist < 0:
                other = c.geom2 if c.geom1 == self._capsule else c.geom1
                return self.model.geom(other).name
        return None

    def _substep(self) -> None:
        x, y, yaw = self._pose
        fwd, left, turn = self._vel
        c, s = cos(yaw), sin(yaw)
        dx, dy = (c * fwd - s * left) * DT, (s * fwd + c * left) * DT
        self._place(np.array([x + dx, y + dy, yaw]))
        hit = self._blocked_by()
        if hit is None:
            moved = (dx, dy)
            self._pose[:2] += moved
        else:
            moved = (0.0, 0.0)
            self._vel[:2] = 0.0  # pushing against an obstacle
            self._contacts.append((self._time, hit, (float(x), float(y))))
        dyaw = turn * DT
        self._pose[2] = wrap(yaw + dyaw)
        self._place(self._pose)
        self._integrate_odometry(moved, dyaw, yaw)

    def _integrate_odometry(self, moved, dyaw: float, yaw0: float) -> None:
        """Same noise model as G1MobileEnv._integrate_odometry, applied to the true increment."""
        c, s = cos(yaw0), sin(yaw0)
        fwd, left = c * moved[0] + s * moved[1], -s * moved[0] + c * moved[1]
        n = self.noise
        fwd *= 1 + self._rng.normal(scale=n.translation)
        left *= 1 + self._rng.normal(scale=n.translation)
        dyaw = dyaw * (1 + self._rng.normal(scale=n.rotation)) + self._bias * DT
        ex, ey, eyaw = self._odom
        ce, se = cos(eyaw), sin(eyaw)
        self._odom = np.array(
            [ex + ce * fwd - se * left, ey + se * fwd + ce * left, wrap(eyaw + dyaw)]
        )
