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
HAND_ASSET = ROOT / "third_party/unitree_g1/g1_29dof_with_hand_rev_1_0.xml"
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


# Dex3-1 finger servos (``hands=True``). The 14 hinges and their limits are read from the
# Unitree MJCF; the body inertias in g1_gear_wbc.xml are already Unitree's, so nothing
# about mass is edited. Servo gains are the ones commissioned on the supported-pelvis G1
# (dream/sim/g1.py). The gains' velocity term is the joint damping (integrated implicitly
# by MuJoCo's default Euler integrator), because the same term as an explicit actuator
# bias is unstable at the WBC's 0.005 s step for finger inertias of ~1e-5 kg m^2
# (dt * kd / inertia ~ 18). The armature (reflected rotor inertia) is 2e-4 kg m^2, the order
# of the finger links' own inertia (measured 2e-5 to 2e-4 from the compiled mass matrix); the
# WBC default class's 0.01 would be 50-500 times the link. No vendor figure is published, so
# the value is an assumption; fingers were measured stable at the default integrator and
# 0.005 s for armatures from 0 to 1e-3 (tests/test_g1_hands.py runs the full-range sweep).
FINGER_KP = 2.0  # N m / rad
FINGER_KD = 0.06  # N m s / rad
FINGER_ARMATURE = 2.0e-4  # kg m^2, assumed (see above)
FINGER_FRICTIONLOSS = 0.0  # the WBC default class's 0.1 N m is large against 1.4 N m servos
HAND_SIDES = ("left", "right")
HAND_FINGERS = (
    "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1",
)  # fmt: skip
HAND_JOINTS = tuple(f"{side}_hand_{f}_joint" for side in HAND_SIDES for f in HAND_FINGERS)
N_HAND = len(HAND_JOINTS)  # 14


_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
_ARM = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
# The 29 joints in the WBC policy's order (legs, waist, arms; g1_wbc.N_BODY). The observation
# and action contract of the policy is expressed in this order and never changes.
BODY_JOINTS = (
    *(f"{s}_{j}_joint" for s in HAND_SIDES for j in _LEG),
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *(f"{s}_{j}_joint" for s in HAND_SIDES for j in _ARM),
)  # fmt: skip
assert len(BODY_JOINTS) == N_BODY


def _has_actuator(model: mujoco.MjModel, joint: str) -> bool:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return jid >= 0 and bool(np.any(model.actuator_trnid[:, 0] == jid))


def _actuators_for(model: mujoco.MjModel, joints: tuple[str, ...]) -> np.ndarray:
    """Actuator index of each named joint, in the given order; every one must exist once."""
    out = []
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        found = np.flatnonzero(model.actuator_trnid[:, 0] == jid) if jid >= 0 else []
        if len(found) != 1:
            raise ValueError(f"expected exactly one actuator on joint {name!r}, found {len(found)}")
        out.append(int(found[0]))
    return np.array(out, int)


def _add_hands(spec: mujoco.MjSpec) -> None:
    """Re-enable the Unitree finger hinges that g1_gear_wbc.xml comments out, plus servos.

    Appended after the 29 body motors, so body actuator indices and the policy order do
    not change. Joint geometry and limits are copied from the Unitree MJCF, never retyped.
    """
    src = mujoco.MjSpec.from_file(str(HAND_ASSET))
    for name in HAND_JOINTS:
        ref = src.joint(name)
        body = spec.body(name.removesuffix("_joint") + "_link")
        joint = body.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=list(ref.pos),
            axis=list(ref.axis),
            range=list(ref.range),
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            actfrcrange=list(ref.actfrcrange),
            actfrclimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            armature=FINGER_ARMATURE,
            damping=FINGER_KD,
            frictionloss=FINGER_FRICTIONLOSS,
        )
        spec.add_actuator(
            name=name,
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[FINGER_KP] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0.0, -FINGER_KP] + [0.0] * 8,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=list(joint.range),
            forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            forcerange=list(ref.actfrcrange),
        )


# Touch sensing of the Dex3 hands. Unitree's Dex3-1 carries 33 tactile (pressure) sensors per hand
# over the palm, finger pads and fingertips (https://unitree.com/Dex3-1; retailers list a sensing
# range of about 10 g to 2.5 kg at 1 kHz). The simulation is an IDEALISED stand-in (an assumption,
# not a model of that sensor): one MuJoCo touch sensor per hand link, summing the normal contact
# force on any geom inside a box site that bounds the link's collision meshes. Sites only: no
# geometry, no mass and no collision are added.
TOUCH_LINKS = ("wrist_yaw", "hand_thumb_2", "hand_middle_0", "hand_middle_1", "hand_index_0",
               "hand_index_1")  # the links with mesh collision geoms; fmt: skip
TOUCH_SITE_MARGIN = 0.003  # m of the site box beyond the link's collision meshes
TOUCH_THRESHOLD = 0.1  # N, about 10 g: a contact the hand reports (assumed, from the range above)


def touch_sensor_name(side: str, link: str) -> str:
    return f"touch_{side}_{link}"


def _add_touch_sensors(spec: mujoco.MjSpec) -> None:
    """Touch-sensor sites on the hand links (see TOUCH_LINKS); sensors only."""
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        for link in TOUCH_LINKS:
            body = model.body(f"{side}_{link}_link")
            pts = []
            for g in np.flatnonzero((model.geom_bodyid == body.id) & (model.geom_contype > 0)):
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = model.geom_dataid[g]
                v = model.mesh_vert[
                    model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
                ]
                pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            if not pts:
                continue
            local = (np.concatenate(pts) - data.xpos[body.id]) @ data.xmat[body.id].reshape(3, 3)
            lo, hi = local.min(axis=0), local.max(axis=0)
            name = touch_sensor_name(side, link)
            spec.body(f"{side}_{link}_link").add_site(
                name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list((lo + hi) / 2),
                size=list((hi - lo) / 2 + TOUCH_SITE_MARGIN), group=5,
            )  # fmt: skip
            spec.add_sensor(
                name=name, type=mujoco.mjtSensor.mjSENS_TOUCH,
                objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name,
            )  # fmt: skip


def wrap(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_mobile_g1(
    scene: Callable[[mujoco.MjSpec], None] | None = None,
    head_camera: CameraSpec | None = HEAD_CAMERA,
    hands: bool = False,
    touch: bool = False,
) -> mujoco.MjModel:
    """``touch=True`` (with hands) adds the touch-sensor sites of ``_add_touch_sensors``.

    ``hands=True`` adds the 14 Dex3 finger hinges and position servos (actuators 29-42).

    The default model is unchanged. Rule 3 class of the hands: physically justified
    calibration (Unitree's own joints, limits and servo gains; armature assumed).
    """
    spec = mujoco.MjSpec.from_file(str(ASSET))
    spec.meshdir = str(MESHES)
    spec.option.timestep = DT_PHYSICS
    if hands:
        _add_hands(spec)
        if touch:
            _add_touch_sensors(spec)
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
        if model.joint(0).name != "floating_base_joint":
            raise ValueError("expected the free-base G1 of g1_gear_wbc.xml")
        # Actuators are found BY JOINT NAME, never by position: the 29 body joints in the
        # WBC policy order, and the 14 finger joints when the model has hands.
        self.body_actuators = _actuators_for(model, BODY_JOINTS)
        self.hands = any(_has_actuator(model, n) for n in HAND_JOINTS)
        self.finger_actuators = (
            _actuators_for(model, HAND_JOINTS) if self.hands else np.zeros(0, int)
        )
        joints = model.actuator_trnid[self.body_actuators, 0]
        self.qadr = model.jnt_qposadr[joints]
        self.vadr = model.jnt_dofadr[joints]
        self.hand_qadr = model.jnt_qposadr[model.actuator_trnid[self.finger_actuators, 0]]
        self.hand_range = model.actuator_ctrlrange[self.finger_actuators].copy()
        self.hand_targets = np.zeros(len(self.finger_actuators))  # open: joint zero
        self.pelvis = model.body("pelvis").id

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [x, y, STAND_HEIGHT]
        self.data.qpos[3:7] = [cos(yaw / 2), 0, 0, sin(yaw / 2)]
        self.data.qpos[self.qadr[: len(DEFAULT)]] = DEFAULT
        self.hand_targets[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._odom = np.zeros(3)
        self._last_true = self.true_pose()
        self._bias = float(self.rng.normal(scale=self.noise.gyro_bias_std))

    def touch(self, side: str) -> dict[str, float]:
        """Normal contact force (N) on each touch-sensing link of one hand; {} if the model has
        none. The robot's own tactile sensing: no contacted-object identity."""
        out = {}
        for link in TOUCH_LINKS:
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, touch_sensor_name(side, link)
            )
            if sid >= 0:
                out[link] = float(self.data.sensordata[self.model.sensor_adr[sid]])
        return out

    def localise(self, pose) -> None:
        """Tell the robot where it is in the plan frame (odometry = ``pose``). A planning copy is
        reset AT its stance but its odometry starts at zero; the correction measures the hand in
        the odometry frame, so without this its error was the stance offset (1.1 m) and the first
        correction step commanded 2.6 rad (W6)."""
        self._odom = np.asarray(pose, float).copy()
        self._last_true = self.true_pose()

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

    # -- hands -------------------------------------------------------------------------
    def set_hand(self, targets: np.ndarray) -> None:
        """Finger position targets (rad, the 14 joints of ``HAND_JOINTS``), clipped to the
        joint ranges. The body keeps its WBC torques. Only on a ``hands=True`` model."""
        if not self.hands:
            raise ValueError("model built without hands")
        t = np.asarray(targets, float)
        if t.shape != (N_HAND,) or not np.isfinite(t).all():
            raise ValueError(f"finger targets must be {N_HAND} finite values")
        self.hand_targets[:] = np.clip(t, self.hand_range[:, 0], self.hand_range[:, 1])

    def hand_positions(self) -> np.ndarray:
        """Finger joint encoders (rad)."""
        return self.data.qpos[self.hand_qadr].copy()

    # -- stepping ----------------------------------------------------------------------
    def step(self, controller) -> None:
        """One 50 Hz policy tick and DECIMATION physics steps of PD torque."""
        controller.tick(self.proprioception())
        for _ in range(DECIMATION):
            self.data.ctrl[self.body_actuators] = controller.torques(self.proprioception())
            self.data.ctrl[self.finger_actuators] = self.hand_targets
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
