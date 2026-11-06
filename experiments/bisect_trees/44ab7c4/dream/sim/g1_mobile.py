"""A free-standing Unitree G1 walking under NVIDIA's pretrained Decoupled WBC policies.

Separate from the supported-pelvis manipulation testbed (dream/sim/g1.py), which is kept
as it is. The robot model is NVIDIA's g1_gear_wbc.xml, the MJCF paired with the policies
(third_party/gr00t_wbc_g1), unmodified except for the mesh directory and any scene the
caller adds (rooms, furniture, a head camera). Physics runs at the policy's 0.005 s step.

What the robot can know is separated from the simulator here:
- ``proprioception()``: pelvis IMU orientation and gyro, joint encoders.
- ``odometry()``: a planar pose estimate in the robot's own odometry frame (origin where it
  started). It is ORACLE-ASSISTED: true motion increments with accumulating noise and gyro
  bias, a stand-in for legged/visual-inertial odometry, labelled wherever it is used.
- ``capture_head()``: RGB-D frames whose calibration is expressed in the odometry frame
  (estimated pelvis pose composed with the head's forward kinematics), never in world
  coordinates.
``true_pose()`` is for the evaluator and recordings only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import atan2, cos, pi, sin
from pathlib import Path

import mujoco
import numpy as np

from dream.robot.g1_wbc import DECIMATION, DEFAULT, DT_PHYSICS, N_BODY, Proprioception
from dream.sim.camera import ROBOT_GEOM_GROUP, CameraSpec, RGBDCamera
from dream.types import CameraCalibration, RGBDFrame

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "third_party/gr00t_wbc_g1/g1_gear_wbc.xml"
MESHES = ROOT / "third_party/unitree_g1/meshes"  # byte-identical to NVIDIA's (PROVENANCE)
STAND_HEIGHT = 0.79  # m, pelvis height at reset; the policy settles it near 0.745
# Head RGB-D camera on torso_link, near the G1's head camera, pitched 35 degrees down so the
# floor ahead and furniture up to ~1.5 m tall are in view. Mount choice, not a calibration.
HEAD_CAMERA = CameraSpec(
    name="head_rgbd",
    position=(0.08, 0.0, 0.43),
    target=(1.08, 0.0, 0.43 - np.tan(np.radians(35))),
    fovy_deg=58.0,
    width=320,
    height=240,
    mount_body="torso_link",
)


def wrap(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_mobile_g1(
    scene: Callable[[mujoco.MjSpec], None] | None = None,
    head_camera: CameraSpec | None = HEAD_CAMERA,
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ASSET))
    spec.meshdir = str(MESHES)
    spec.option.timestep = DT_PHYSICS
    if head_camera is not None:
        for geom in spec.geoms:  # robot geometry renders the camera's self-filter
            if geom.name != "floor":
                geom.group = ROBOT_GEOM_GROUP
        spec.body(head_camera.mount_body).add_camera(
            name=head_camera.name,
            pos=list(head_camera.position),
            xyaxes=head_camera.xyaxes(),
            fovy=head_camera.fovy_deg,
        )
        # Tracking spectator view for recordings (evidence only, never a robot sensor).
    spec.worldbody.add_camera(
        name="spectator", mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, pos=[-2.6, -2.6, 2.4],
        xyaxes=[0.707, -0.707, 0, 0.42, 0.42, 0.8],
    )  # fmt: skip
    if scene is not None:
        scene(spec)
    return spec.compile()


@dataclass(frozen=True)
class OdometryNoise:
    """Assumed, not calibrated: 2% translation and 2% rotation scale noise per step, and a
    constant gyro bias drawn once per episode with this standard deviation."""

    translation: float = 0.02
    rotation: float = 0.02
    gyro_bias_std: float = 0.002  # rad/s


DEFAULT_ODOMETRY_NOISE = OdometryNoise()


class G1MobileEnv:
    """Synchronous physics owner. ``reset`` is episode initialisation, not an agent action."""

    def __init__(
        self,
        model: mujoco.MjModel,
        rng: np.random.Generator,
        noise: OdometryNoise = DEFAULT_ODOMETRY_NOISE,
        head_camera: CameraSpec | None = HEAD_CAMERA,
    ) -> None:
        self.model, self.rng, self.noise = model, rng, noise
        self.data = mujoco.MjData(model)
        self.head = head_camera
        self._camera = None if head_camera is None else RGBDCamera(model, head_camera, rng)
        if model.joint(0).name != "floating_base_joint" or model.nu != N_BODY:
            raise ValueError("expected the 29-DoF free-base G1 of g1_gear_wbc.xml")
        # Robot joints in actuator (= policy) order; scene joints (doors) come after them.
        joints = [model.actuator(i).trnid[0] for i in range(model.nu)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.vadr = np.array([model.jnt_dofadr[j] for j in joints])
        self.pelvis = model.body("pelvis").id

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [x, y, STAND_HEIGHT]
        self.data.qpos[3:7] = [cos(yaw / 2), 0, 0, sin(yaw / 2)]
        self.data.qpos[self.qadr[: len(DEFAULT)]] = DEFAULT
        mujoco.mj_forward(self.model, self.data)
        self._odom = np.zeros(3)
        self._last_true = self.true_pose()
        self._bias = float(self.rng.normal(scale=self.noise.gyro_bias_std))

    # -- robot sensing -----------------------------------------------------------------
    def proprioception(self) -> Proprioception:
        d = self.data
        return Proprioception(
            d.qpos[3:7].copy(), d.qvel[3:6].copy(), d.qpos[self.qadr], d.qvel[self.vadr]
        )

    def odometry(self) -> np.ndarray:
        """(x, y, yaw) in the odometry frame. Oracle-assisted (see the module docstring)."""
        return self._odom.copy()

    def capture_head(self) -> RGBDFrame:
        if self._camera is None:
            raise ValueError("no head camera mounted")
        frame = self._camera.capture(self.data)
        return replace(frame, calibration=self._odom_calibration(frame.calibration))

    def _odom_calibration(self, world: CameraCalibration) -> CameraCalibration:
        d = self.data
        rp = d.xmat[self.pelvis].reshape(3, 3)
        cam_in_pelvis_r = rp.T @ world.rotation  # head forward kinematics (joint encoders)
        cam_in_pelvis_p = rp.T @ (world.position - d.xpos[self.pelvis])
        true_yaw = _yaw(d.qpos[3:7])
        cy, sy = cos(true_yaw), sin(true_yaw)
        tilt = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]]) @ rp  # IMU roll/pitch only
        x, y, yaw = self._odom
        c, s = cos(yaw), sin(yaw)
        r_est = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ tilt
        p_est = np.array([x, y, d.xpos[self.pelvis][2]])  # height from leg kinematics
        return replace(
            world,
            rotation=r_est @ cam_in_pelvis_r,
            position=p_est + r_est @ cam_in_pelvis_p,
        )

    # -- stepping ----------------------------------------------------------------------
    def step(self, controller) -> None:
        """One 50 Hz policy tick and DECIMATION physics steps of PD torque."""
        controller.tick(self.proprioception())
        for _ in range(DECIMATION):
            self.data.ctrl[:] = controller.torques(self.proprioception())
            mujoco.mj_step(self.model, self.data)
        self._integrate_odometry()

    def _integrate_odometry(self) -> None:
        now = self.true_pose()
        x0, y0, yaw0 = self._last_true
        c, s = cos(yaw0), sin(yaw0)
        dx, dy = now[0] - x0, now[1] - y0
        fwd, left = c * dx + s * dy, -s * dx + c * dy  # body-frame increment
        dyaw = wrap(now[2] - yaw0)
        n = self.noise
        fwd *= 1 + self.rng.normal(scale=n.translation)
        left *= 1 + self.rng.normal(scale=n.translation)
        dyaw = dyaw * (1 + self.rng.normal(scale=n.rotation)) + self._bias * DT_PHYSICS * DECIMATION
        ex, ey, eyaw = self._odom
        ce, se = cos(eyaw), sin(eyaw)
        self._odom = np.array(
            [ex + ce * fwd - se * left, ey + se * fwd + ce * left, wrap(eyaw + dyaw)]
        )
        self._last_true = now

    # -- evaluation only ---------------------------------------------------------------
    def true_pose(self) -> np.ndarray:
        """World (x, y, yaw) of the pelvis: evaluator and recordings only."""
        q = self.data.qpos
        return np.array([q[0], q[1], _yaw(q[3:7])])

    @property
    def time(self) -> float:
        return float(self.data.time)


class G1MobileBase:
    """``MobileBase`` for the walking G1: velocity commands go to the pretrained WBC policy.

    ``on_tick`` (optional) runs after every 50 Hz policy step, e.g. a recorder's sample().
    """

    def __init__(self, env: G1MobileEnv, controller, on_tick=None) -> None:
        self.env, self.controller, self.on_tick = env, controller, on_tick

    @property
    def time(self) -> float:
        return self.env.time

    @property
    def stop_reason(self) -> str | None:
        return self.controller.stopped

    def odometry(self) -> np.ndarray:
        return self.env.odometry()

    def capture(self) -> RGBDFrame:
        return self.env.capture_head()

    def walk(self, forward: float, lateral: float, turn: float, seconds: float) -> bool:
        self.controller.set_command(forward, lateral, turn)
        for _ in range(max(1, round(seconds / (DT_PHYSICS * DECIMATION)))):
            self.env.step(self.controller)
            if self.on_tick:
                self.on_tick()
            if self.controller.stopped:
                return False
        return True
