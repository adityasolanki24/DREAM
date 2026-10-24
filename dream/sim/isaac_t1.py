"""Experimental Isaac Sim 5.1 T1 adapter; import only after SimulationApp starts.

Own implementation of the audited Isaac Lab v2 PPO deployment contract. APM reference
provenance is in navigation_sources.json. No APM import, world-pose planning, state writes
during execution, learned navigation, or claim of walking validation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from math import acos, cos, sin
from pathlib import Path

import numpy as np

from dream.navigation.control import NavigationState
from dream.robot.t1 import T1Policy


def joint_armature(name: str) -> float:
    """Match pinned lab/t1.py: head inertia 0.005, other joints 0.01 kg m²."""
    return 0.005 if name in ("AAHead_yaw", "Head_pitch") else 0.01


def rotation_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 0.01:
        raise ValueError("invalid robot orientation")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class IsaacT1:
    """50 Hz PPO / 200 Hz physics, RGB-D sensing and ideal-proprioceptive odometry.

    Body velocity/gyro/orientation are ideal robot proprioception in this initial adapter,
    not demonstrated VIO. Translation integrates velocity; true world translation stays
    available only to the independent runner's scoring. Depth veto is reactive, not routing.
    """

    def __init__(
        self,
        world,
        asset_url: str,
        policy_dir: Path,
        hashes: dict,
        checkpoint: Callable[[str], None] = lambda _: None,
        diagnostic_dir: Path | None = None,
    ):
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import PhysxSchema, UsdPhysics

        self.checkpoint = checkpoint
        self.world = world
        self.cfg = json.loads((policy_dir / "env_cfg.json").read_text())
        add_reference_to_stage(asset_url, "/World/T1")
        checkpoint("asset_referenced")
        roots, heads = [], []
        for prim in world.stage.Traverse():
            if not (
                str(prim.GetPath()) == "/World/T1" or str(prim.GetPath()).startswith("/World/T1/")
            ):
                continue
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(str(prim.GetPath()))
                api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateEnabledSelfCollisionsAttr(False)
                api.CreateSolverPositionIterationCountAttr(8)
                api.CreateSolverVelocityIterationCountAttr(4)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(1.0)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(
                    joint_armature(prim.GetName())
                )
            if prim.GetName() == "H2":
                heads.append(str(prim.GetPath()))
        if len(roots) != 1 or len(heads) != 1:
            raise ValueError("T1 asset articulation/head contract mismatch")
        self.head_path = heads[0]
        checkpoint("asset_contract_checked")
        self.art = world.scene.add(SingleArticulation(roots[0], name="dream_t1"))
        checkpoint("articulation_reset_started")
        world.reset()
        checkpoint("articulation_reset_completed")
        names = list(self.art.dof_names)
        self.actor = T1Policy.load(policy_dir, hashes, names)
        checkpoint("policy_loaded")
        # Match the audited PPO training actuator gains, rather than the stock USD drives.
        gains = {
            ".*_Hip_.*": (100, 3),
            ".*_Knee_Pitch": (150, 4),
            "Waist": (100, 3),
            ".*_Ankle_.*": (40, 2),
            ".*_Shoulder_.*": (40, 2),
            ".*_Elbow_.*": (40, 2),
            "AAHead_yaw": (20, 1),
            "Head_pitch": (20, 1),
        }
        kp, kd = [], []
        for name in names:
            matches = [v for pattern, v in gains.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"T1 joint gains unspecified: {name}")
            p, d = matches[0]
            kp.append(p)
            kd.append(d)
        self.art.get_articulation_controller().set_gains(kps=np.array(kp), kds=np.array(kd))
        # SingleArticulation exposes structured DOF properties; get_dof_limits is
        # the plural Articulation API, not a method of this single-prim wrapper.
        self.limits = np.column_stack(
            (self.art.dof_properties["lower"], self.art.dof_properties["upper"])
        )
        if diagnostic_dir is not None:
            (diagnostic_dir / "articulation.json").write_text(
                json.dumps(
                    {
                        "root": roots[0],
                        "head": heads[0],
                        "joint_names": names,
                        "policy_indices": self.actor.indices.tolist(),
                        "kp": kp,
                        "kd": kd,
                        "limits_rad": self.limits.tolist(),
                        "asset_url": asset_url,
                    },
                    indent=2,
                )
                + "\n"
            )
        checkpoint("joint_contract_checked")
        initial = np.zeros(23)
        initial[self.actor.indices] = self.actor.default
        # Scene initialization only; subsequent changes use articulation actions and physics.
        self.art.set_joint_positions(initial)
        self.art.set_joint_velocities(np.zeros(23))
        self.art.set_world_pose(position=np.array([0, 0, 0.70]), orientation=np.array([1, 0, 0, 0]))
        self.odom = np.zeros(3)
        self.time = 0.0
        self.command = np.zeros(3)
        self.stopped = False

    def initialize_camera(self) -> None:
        """Initialize head RGB-D only after standing physics is observable."""
        from isaacsim.sensors.camera import Camera

        self.checkpoint("camera_initialization_started")
        self.camera = Camera(prim_path=self.head_path + "/dream_eye", resolution=(400, 400))
        self.camera.initialize()
        self.camera.set_local_pose(
            translation=np.array([0.10, 0, 0.16]),
            orientation=np.array([1, 0, 0, 0]),
            camera_axes="world",
        )
        self.camera.set_horizontal_aperture(20.955)
        self.camera.set_focal_length(20.955 / (2 * np.tan(np.deg2rad(120) / 2)))
        self.camera.add_distance_to_image_plane_to_frame()
        self.checkpoint("camera_initialized")

    def sensors(self) -> dict:
        # Only orientation is read here; the true robot translation is withheld.
        _, quat = self.art.get_world_pose()
        rotation = rotation_wxyz(quat)
        linear = rotation.T @ np.asarray(self.art.get_linear_velocity())
        gyro = rotation.T @ np.asarray(self.art.get_angular_velocity())
        return {
            "q": np.asarray(self.art.get_joint_positions()),
            "dq": np.asarray(self.art.get_joint_velocities()),
            "linear": linear,
            "gyro": gyro,
            "gravity": rotation.T @ np.array([0, 0, -1]),
            "tilt": acos(float(np.clip(rotation[2, 2], -1, 1))),
        }

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(3):  # warm the attached annotators without writing any physical state
            self.world.render()
        rgb = self.camera.get_rgba()
        depth = self.camera.get_depth()
        if (
            rgb is None
            or depth is None
            or rgb.shape != (400, 400, 4)
            or rgb.dtype != np.uint8
            or depth.shape != (400, 400)
            or not np.any(np.isfinite(depth) & (depth > 0))
        ):
            raise RuntimeError("RGB-D frame unavailable or invalid")
        return rgb[..., :3].copy(), depth.copy()

    def navigation_state(self, depth: np.ndarray, captured_time_s: float) -> NavigationState:
        sensor = self.sensors()
        # Central horizontal sector; invalid pixels do not count as clear space.
        h, w = depth.shape
        sector = depth[h // 3 : h // 2, w // 3 : 2 * w // 3]
        valid = sector[np.isfinite(sector) & (sector > 0)]
        clearance = float(valid.min()) if valid.size == sector.size else float("nan")
        return NavigationState(
            self.time, *self.odom, sensor["tilt"], clearance, depth_time_s=captured_time_s
        )

    def advance(self, forward_m_s: float, yaw_rad_s: float) -> dict:
        from isaacsim.core.utils.types import ArticulationAction

        sensor = self.sensors()
        if sensor["tilt"] > np.deg2rad(25):
            self.stopped = True
        desired = np.array([forward_m_s, 0, yaw_rad_s])
        if not np.isfinite(desired).all() or np.any(np.abs(desired) > [0.2, 0, 0.4]):
            self.stopped = True
            raise ValueError("unsafe locomotion command")
        if self.stopped:
            desired[:] = 0
        # Prototype command slew, fixed before physical evaluation: 0.5 m/s² and 1 rad/s².
        self.command += np.clip(desired - self.command, [-0.01, 0, -0.02], [0.01, 0, 0.02])
        target = self.actor.target(
            sensor["q"],
            sensor["dq"],
            sensor["linear"],
            sensor["gyro"],
            sensor["gravity"],
            self.command,
            self.limits,
        )
        self.art.apply_action(ArticulationAction(joint_positions=target))
        before = self.world.current_time_step_index
        for _ in range(4):
            self.world.step(render=False)
        if self.world.current_time_step_index - before != 4:
            raise RuntimeError("physics did not advance four steps")
        yaw = self.odom[2]
        vx, vy = sensor["linear"][:2]
        self.odom += 0.02 * np.array(
            [
                cos(yaw) * vx - sin(yaw) * vy,
                sin(yaw) * vx + cos(yaw) * vy,
                sensor["gyro"][2],
            ]
        )
        self.time += 0.02
        return {
            "time_s": self.time,
            "proprioception_time_s": self.time - 0.02,
            "odom": self.odom.tolist(),
            "tilt_rad": sensor["tilt"],
            "stopped": self.stopped,
            "command": self.command.tolist(),
            "joint_position_rad": sensor["q"].tolist(),
            "joint_velocity_rad_s": sensor["dq"].tolist(),
            "joint_target_rad": target.tolist(),
            "raw_ppo_action": self.actor.last.tolist(),
            "target_limit_excess_rad": self.actor.target_limit_excess.tolist(),
        }
