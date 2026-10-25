"""The walking G1 in Isaac Sim 6.2 (PhysX + RTX), as a drop-in for ``G1MobileEnv``.

Same embodiment and controller contract as the validated MuJoCo path, in a different
simulator and in scenes DREAM did not author:
- robot: NVIDIA's g1_gear_wbc.xml converted to USD unmodified (experiments/prepare_g1_usd.py);
- control: NVIDIA's pretrained Decoupled WBC actors through ``WBCLocomotion`` unchanged, with
  their PD torques computed here at the 200 Hz physics rate and clipped at the MJCF actuator
  force limits, as MuJoCo does. PhysX joint drives are zeroed so they add nothing;
- sensing: pelvis IMU orientation and body-frame gyro, joint encoders, a head RGB-D camera
  at the MuJoCo mount, with the same depth/RGB noise model and 3 m range; ODOMETRY IS
  ORACLE-ASSISTED exactly as in ``G1MobileEnv`` (true increments, noise, gyro bias).

Simulator differences are recorded by ``parity_report`` rather than hidden: PhysX contacts
are rigid with patch friction (MuJoCo: soft contacts, solref 0.02, torsional friction 0.005);
MJCF friction loss (0.1 N m) has no exact PhysX counterpart and is omitted; friction pairs
combine by max, as MuJoCo does, through the robot's own material (the scene is untouched).

Import this module only after ``SimulationApp`` has started. ``true_pose`` and the contact
log are for the evaluator only. The pure helpers at the top run without Isaac (tests).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path

import numpy as np

from dream.robot.g1_wbc import DECIMATION, DEFAULT, DT_PHYSICS, N_BODY, Proprioception
from dream.sim.camera import MAX_RANGE, CameraSpec
from dream.types import CameraCalibration, RGBDFrame

ROOT = Path(__file__).resolve().parents[2]
MJCF = ROOT / "third_party/gr00t_wbc_g1/g1_gear_wbc.xml"
STAND_HEIGHT = 0.79  # m above the floor at reset, as dream/sim/g1_mobile.py
JOINT_ARMATURE = 0.01  # kg m^2, every joint of g1_gear_wbc.xml
JOINT_DAMPING = 0.001  # N m s/rad, every joint of g1_gear_wbc.xml (passive, unclipped)
ROBOT_FRICTION = 1.0  # g1_gear_wbc.xml geom friction (sliding)
ROBOT_PREFIX = "/World/G1"


def mjcf_actuators(path: Path = MJCF) -> tuple[list[str], np.ndarray]:
    """Actuated joint names in actuator (= policy) order, and their force limits (N m)."""
    root = ET.parse(path).getroot()
    limits = {}
    for joint in root.iter("joint"):
        if joint.get("actuatorfrcrange"):
            lo, hi = (float(v) for v in joint.get("actuatorfrcrange").split())
            limits[joint.get("name")] = max(-lo, hi)
    names = [m.get("joint") for m in root.find("actuator")]
    if len(names) != N_BODY or any(n not in limits for n in names):
        raise ValueError("MJCF actuator contract differs from the 29-DoF G1")
    return names, np.array([limits[n] for n in names])


def camera_axes(spec: CameraSpec) -> np.ndarray:
    """Camera-to-mount rotation (columns x right, y up, z backward), MuJoCo and USD alike."""
    x, y = np.reshape(spec.xyaxes(), (2, 3))
    return np.column_stack([x, y, np.cross(x, y)])


def apertures(spec: CameraSpec, focal: float = 10.0) -> tuple[float, float]:
    """(horizontal, vertical) USD apertures giving ``spec.fovy_deg`` with square pixels."""
    vertical = 2 * focal * np.tan(np.radians(spec.fovy_deg) / 2)
    return vertical * spec.width / spec.height, vertical


def sensor_depth(distance: np.ndarray, spec: CameraSpec, rng: np.random.Generator) -> np.ndarray:
    """The MuJoCo driver's depth model on a renderer's distance-to-image-plane image."""
    depth = np.asarray(distance, np.float32).copy()
    depth[~np.isfinite(depth) | (depth <= 0) | (depth > MAX_RANGE)] = np.nan
    a, b = spec.depth_noise
    sigma = a + b * np.nan_to_num(depth) ** 2
    noisy = depth + rng.normal(size=depth.shape).astype(np.float32) * sigma
    return (np.round(noisy * 1000) / 1000).astype(np.float32)


def sensor_rgb(rgb: np.ndarray, spec: CameraSpec, rng: np.random.Generator) -> np.ndarray:
    rgb = np.asarray(rgb)[..., :3]
    if spec.rgb_noise <= 0:
        return rgb.astype(np.uint8).copy()
    n = rng.normal(scale=spec.rgb_noise, size=rgb.shape)
    return np.clip(rgb.astype(np.float32) + n, 0, 255).astype(np.uint8)


def self_mask(ids: np.ndarray, id_to_path: dict, prefix: str = ROBOT_PREFIX) -> np.ndarray:
    """Pixels whose nearest surface is the robot's own geometry (the robot self-filter)."""
    own = [int(k) for k, v in id_to_path.items() if str(v).startswith(prefix + "/")]
    return np.isin(np.asarray(ids), own)


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_of(q) -> float:
    w, x, y, z = q
    return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass(frozen=True)
class ContactEvent:
    """Evaluator record: a robot body touched something other than the floor."""

    time_s: float
    robot_body: str
    other: str
    impulse: float  # N s, summed over the step's contact points


class IsaacG1Env:
    """Synchronous physics owner with the ``G1MobileEnv`` sensing/stepping interface."""

    def __init__(
        self,
        world,
        robot_usd: str,
        rng: np.random.Generator,
        noise,
        head_camera: CameraSpec,
        floor_paths: tuple[str, ...],
        checkpoint=lambda stage: None,
        self_collisions: bool = True,
    ) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade

        self.world, self.rng, self.noise, self.head = world, rng, noise, head_camera
        self.floor_paths = floor_paths
        self.checkpoint = checkpoint
        self.stage = world.stage
        add_reference_to_stage(robot_usd, ROBOT_PREFIX)
        checkpoint("robot_referenced")
        names, self.force_limit = mjcf_actuators()
        material = UsdShade.Material.Define(self.stage, ROBOT_PREFIX + "/DreamContactMaterial")
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(ROBOT_FRICTION)
        mat.CreateDynamicFrictionAttr(ROBOT_FRICTION)
        mat.CreateRestitutionAttr(0.0)
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_mat.CreateFrictionCombineModeAttr("max")  # MuJoCo combines friction by max
        roots, joints, bodies, colliders, torso = [], {}, [], 0, None
        robot = self.stage.GetPrimAtPath(ROBOT_PREFIX)
        # Instance proxies included: a referenced asset may be composed as instances.
        for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not path.startswith(ROBOT_PREFIX + "/"):
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(path)
                art = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                art.CreateEnabledSelfCollisionsAttr(self_collisions)
                # Isaac Lab's G1 articulation settings (the policy family's training sim).
                art.CreateSolverPositionIterationCountAttr(8)
                art.CreateSolverVelocityIterationCountAttr(4)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joints[prim.GetName()] = prim
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(0.0)
                drive.CreateMaxForceAttr(1e6)  # limits are applied to the computed torque
                joint = PhysxSchema.PhysxJointAPI.Apply(prim)
                joint.CreateArmatureAttr(JOINT_ARMATURE)
                joint.CreateMaxJointVelocityAttr(1e5)  # deg/s: MuJoCo has no velocity cap
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(path)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                if prim.GetName() == "torso_link":
                    torso = path
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders += 1
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.weakerThanDescendants, "physics"
                )
        if len(roots) != 1 or set(names) - set(joints) or torso is None:
            children = [str(c.GetPath()) for c in robot.GetChildren()] if robot else []
            raise ValueError(
                "converted G1 USD does not match the MJCF contract: "
                f"robot prim valid={bool(robot and robot.IsValid())}, children={children[:8]}, "
                f"roots={roots}, joints={len(joints)}, "
                f"missing={sorted(set(names) - set(joints))[:5]}"
                f", bodies={len(bodies)}, torso={torso}, colliders={colliders}"
            )
        self.root_path, self.torso_path, self.bodies = roots[0], torso, bodies
        self.contract = {"articulation_root": roots[0], "revolute_joints": len(joints),
                         "rigid_bodies": len(bodies), "colliders": colliders,
                         "self_collisions": self_collisions}  # fmt: skip
        self.camera_path = self.torso_path + "/" + head_camera.name
        self._define_camera(Sdf)
        checkpoint("robot_contract_checked")
        self.art = world.scene.add(SingleArticulation(self.root_path, name="dream_g1"))
        world.reset()
        checkpoint("world_reset")
        dofs = list(self.art.dof_names)
        if sorted(dofs) != sorted(names):
            raise ValueError(f"articulation DOFs differ from the MJCF actuators: {dofs}")
        self.index = np.array([dofs.index(n) for n in names])  # MJCF order -> Isaac DOF
        self.dof_names = dofs
        self.art.get_articulation_controller().set_gains(
            kps=np.zeros(len(dofs)), kds=np.zeros(len(dofs))
        )
        self._annotators = None
        self.physics_steps = 0
        self.contacts: list[ContactEvent] = []
        self.contact_steps = 0
        self._sim_time = 0.0

    # -- scene initialisation (not agent actions) -----------------------------------------
    def _define_camera(self, Sdf) -> None:
        from pxr import Gf, UsdGeom

        cam = UsdGeom.Camera.Define(self.stage, self.camera_path)
        h_ap, v_ap = apertures(self.head)
        cam.CreateFocalLengthAttr(10.0)
        cam.CreateHorizontalApertureAttr(h_ap)
        cam.CreateVerticalApertureAttr(v_ap)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
        r = camera_axes(self.head)
        m = Gf.Matrix4d(1.0)
        for i in range(3):  # USD row-vector convention: row i is the image of axis i
            m.SetRow(i, Gf.Vec4d(*r[:, i], 0.0))
        m.SetRow(3, Gf.Vec4d(*self.head.position, 1.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m)
        self._cam_local = (r, np.array(self.head.position, float))

    def reset(self, x: float, y: float, yaw: float, floor_z: float = 0.0) -> None:
        q = np.zeros(len(self.dof_names))
        q[self.index[: len(DEFAULT)]] = DEFAULT
        self.art.set_world_pose(
            position=np.array([x, y, floor_z + STAND_HEIGHT]),
            orientation=np.array([cos(yaw / 2), 0, 0, sin(yaw / 2)]),
        )
        self.art.set_joint_positions(q)
        self.art.set_joint_velocities(np.zeros_like(q))
        self.art.set_linear_velocity(np.zeros(3))
        self.art.set_angular_velocity(np.zeros(3))
        self._odom = np.zeros(3)
        self._last_true = self.true_pose()
        self._bias = float(self.rng.normal(scale=self.noise.gyro_bias_std))

    # -- robot sensing -----------------------------------------------------------------------
    def _root(self) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.art.get_world_pose()
        return np.asarray(p, float), np.asarray(q, float)

    def proprioception(self) -> Proprioception:
        _, quat = self._root()
        r = quat_matrix(quat)
        gyro = r.T @ np.asarray(self.art.get_angular_velocity(), float)  # body frame, as MuJoCo
        q = np.asarray(self.art.get_joint_positions(), float)[self.index]
        dq = np.asarray(self.art.get_joint_velocities(), float)[self.index]
        return Proprioception(quat, gyro, q, dq)

    def odometry(self) -> np.ndarray:
        """(x, y, yaw) in the odometry frame. ORACLE-ASSISTED, as ``G1MobileEnv``."""
        return self._odom.copy()

    def _torso_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Torso link world pose from the articulation's link state (forward kinematics)."""
        view = self.art._articulation_view._physics_view
        names = list(self.art._articulation_view.body_names)
        t = np.asarray(view.get_link_transforms())[0, names.index("torso_link")]
        x, y, z, w = t[3:7]  # PhysX tensors are xyzw
        return t[:3].astype(float), quat_matrix([w, x, y, z])

    def _annotate(self):
        if self._annotators is None:
            import omni.replicator.core as rep

            rp = rep.create.render_product(self.camera_path, (self.head.width, self.head.height))
            names = ("rgb", "distance_to_image_plane", "instance_id_segmentation_fast")
            self._annotators = {}
            for name in names:
                a = rep.AnnotatorRegistry.get_annotator(name)
                a.attach([rp])
                self._annotators[name] = a
            self._render_product = rp
        return self._annotators

    def capture_head(self, warmup_renders: int = 2) -> RGBDFrame:
        """Render the head RGB-D; physics must not advance while rendering (checked)."""
        annotators = self._annotate()
        before = self._root()[0]
        steps = self.physics_steps
        for _ in range(warmup_renders + 1):
            self.world.render()
        if self.physics_steps != steps or np.abs(self._root()[0] - before).max() > 1e-9:
            raise RuntimeError("rendering advanced physics; camera capture is not passive")
        rgb = np.asarray(annotators["rgb"].get_data())
        dist = np.asarray(annotators["distance_to_image_plane"].get_data())
        seg = annotators["instance_id_segmentation_fast"].get_data()
        ids, info = np.asarray(seg["data"]), seg["info"].get("idToLabels", {})
        shape = (self.head.height, self.head.width)
        if rgb.shape[:2] != shape or dist.shape != shape or ids.shape != shape:
            raise RuntimeError(f"camera returned {rgb.shape}/{dist.shape}/{ids.shape}")
        calibration = self._odom_calibration()
        return RGBDFrame(
            self.time,
            sensor_rgb(rgb, self.head, self.rng),
            sensor_depth(dist, self.head, self.rng),
            self_mask(ids, info),
            calibration,
        )

    def _odom_calibration(self) -> CameraCalibration:
        pelvis_p, quat = self._root()
        rp = quat_matrix(quat)
        tp, tr = self._torso_pose()
        r_cam, p_cam = self._cam_local
        world_r, world_p = tr @ r_cam, tp + tr @ p_cam
        cam_in_pelvis_r = rp.T @ world_r
        cam_in_pelvis_p = rp.T @ (world_p - pelvis_p)
        yaw = yaw_of(quat)
        cy, sy = cos(yaw), sin(yaw)
        tilt = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]]) @ rp  # IMU roll/pitch only
        x, y, oyaw = self._odom
        c, s = cos(oyaw), sin(oyaw)
        r_est = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ tilt
        p_est = np.array([x, y, pelvis_p[2]])  # height from leg kinematics
        return CameraCalibration(
            width=self.head.width,
            height=self.head.height,
            fovy_deg=self.head.fovy_deg,
            position=p_est + r_est @ cam_in_pelvis_p,
            rotation=r_est @ cam_in_pelvis_r,
        )

    # -- stepping -----------------------------------------------------------------------------
    def step(self, controller) -> None:
        """One 50 Hz policy tick and DECIMATION physics steps of clipped PD torque."""
        controller.tick(self.proprioception())
        for _ in range(DECIMATION):
            prop = self.proprioception()
            tau = np.clip(controller.torques(prop), -self.force_limit, self.force_limit)
            tau = tau - JOINT_DAMPING * np.asarray(prop.dq)
            efforts = np.zeros(len(self.dof_names))
            efforts[self.index] = tau
            self.art.set_joint_efforts(efforts)
            self.world.step(render=False)
            self.physics_steps += 1
            self._sim_time += DT_PHYSICS
            self._read_contacts()
        from dream.sim.g1_mobile import G1MobileEnv

        G1MobileEnv._integrate_odometry(self)  # the identical oracle-assisted model

    def _read_contacts(self) -> None:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        headers, data = get_physx_simulation_interface().get_contact_report()
        touched = False
        for h in headers:
            a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            ca = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
            cb = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            ra, rb = a.startswith(ROBOT_PREFIX + "/"), b.startswith(ROBOT_PREFIX + "/")
            if ra == rb:
                continue
            robot, other = (a, cb) if ra else (b, ca)
            if any(other == f or other.startswith(f + "/") for f in self.floor_paths):
                continue
            start, n = h.contact_data_offset, h.num_contact_data
            impulse = float(sum(np.linalg.norm(data[i].impulse) for i in range(start, start + n)))
            self.contacts.append(ContactEvent(self.time, robot.rsplit("/", 1)[-1], other, impulse))
            touched = True
        self.contact_steps += touched

    # -- evaluation only ----------------------------------------------------------------------
    def true_pose(self) -> np.ndarray:
        p, q = self._root()
        return np.array([p[0], p[1], yaw_of(q)])

    def true_height(self) -> float:
        return float(self._root()[0][2])

    @property
    def time(self) -> float:
        return round(self._sim_time, 9)

    def parity_report(self) -> dict:
        return {
            **self.contract,
            "physics_dt_s": DT_PHYSICS,
            "policy_decimation": DECIMATION,
            "joint_armature_kg_m2": JOINT_ARMATURE,
            "joint_damping_n_m_s": JOINT_DAMPING,
            "torque_limits": "MJCF actuatorfrcrange, applied to the PD torque",
            "robot_friction": {"sliding": ROBOT_FRICTION, "combine": "max (as MuJoCo)"},
            "not_reproduced": [
                "MJCF frictionloss 0.1 N m per joint (no exact PhysX equivalent)",
                "MuJoCo soft contact (solref 0.02, solimp) vs PhysX rigid contact",
                "MuJoCo torsional friction 0.005 (PhysX patch friction instead)",
            ],
            "solver": "PhysX TGS, articulation 8 position / 4 velocity iterations",
        }
