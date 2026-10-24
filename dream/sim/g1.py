"""Real Unitree G1 MJCF, with a fixed pelvis for isolated hand commissioning.

The vendored XML and meshes remain byte-identical upstream. MjSpec removes the floating
joint and adds a floor, lighting and optional physical object before compile. This support
is an environment simplification; no balance, locomotion or Isaac parity is implied.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from dream.robot.g1 import G1Controller
from dream.sim.camera import ROBOT_GEOM_GROUP, CameraSpec, add_camera

ASSET = (
    Path(__file__).resolve().parents[2] / "third_party/unitree_g1/g1_29dof_with_hand_rev_1_0.xml"
)


def build_g1(
    object_position: tuple[float, float, float] | None = None,
    radius: float = 0.025,
    mass: float = 0.05,
    *,
    constrained: bool = False,
    camera: CameraSpec | None = None,
    contact_timeconst: float | None = None,
    target_position: tuple[float, float, float] | None = None,
) -> mujoco.MjModel:
    """Optional free sphere and pedestal are scene setup, never runtime state writes."""
    if radius <= 0 or mass <= 0 or not np.isfinite([radius, mass]).all():
        raise ValueError("positive finite radius and mass required")
    spec = mujoco.MjSpec.from_file(str(ASSET))
    spec.delete(spec.joint("floating_base_joint"))
    spec.option.timestep = 0.001  # 1 kHz commissioning control, no upstream inertia edits
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # The same PD law as the controller, with velocity feedback integrated implicitly.
    # Explicit external damping is unstable for the upstream fingers' tiny inertias.
    for actuator in spec.actuators:
        hand = "_hand_" in actuator.name
        kp, kd = (2.0, 0.06) if hand else (80.0, 4.0)
        actuator.gainprm[0] = kp
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.biasprm[:3] = [0, -kp, -kd]
        actuator.forcelimited = True
        actuator.forcerange = spec.joint(actuator.name).actfrcrange
    spec.worldbody.add_light(pos=[1, -1, 3], dir=[-0.3, 0.2, -1])
    if camera is not None:
        # All robot geometry is used to render its own sensor self-filter. Group
        # assignment changes rendering only; upstream collision geometry is retained.
        for geom in spec.geoms:
            if geom.name != "floor":
                geom.group = ROBOT_GEOM_GROUP
        add_camera(spec.worldbody, camera)
    if object_position is not None:
        pos = np.asarray(object_position, float)
        if pos.shape != (3,) or not np.isfinite(pos).all() or pos[2] <= radius:
            raise ValueError("invalid object position")
        # A narrow pedestal lets fingers approach from the side; the sphere is free.
        height = pos[2] - radius
        spec.worldbody.add_geom(
            name="pedestal",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius * 0.6, height / 2, 0],
            pos=[pos[0], pos[1], height / 2],
            rgba=[0.3, 0.3, 0.3, 1],
        )
        body = spec.worldbody.add_body(name="test_object", pos=pos)
        if not constrained:
            body.add_freejoint(name="test_object_free")
        body.add_geom(
            name="test_object_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            mass=mass,
            rgba=[0.85, 0.15, 0.1, 1],
        )
    if target_position is not None:
        p = np.asarray(target_position, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("invalid target platform")
        spec.worldbody.add_geom(
            name="target_platform",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=p - [0, 0, 0.0125],
            size=[0.065, 0.065, 0.0125],
            rgba=[0.1, 0.2, 0.85, 1],
        )
    if contact_timeconst is not None:
        # Numerical contact-resolution experiment, NOT hardware calibration. The
        # documented positive-solref stability floor is 2 dt; use >=5 dt here.
        # https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters
        if not np.isfinite(contact_timeconst) or contact_timeconst < 5 * spec.option.timestep:
            raise ValueError("contact time constant must be at least five physics steps")
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.solref = [contact_timeconst, 1]
    return spec.compile()


class G1Env:
    """Synchronous physics owner; resetting is permitted only as episode initialization."""

    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        self.model = model if model is not None else build_g1()
        self.data = mujoco.MjData(self.model)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Interior finger home; preserve all body home coordinates from the asset.
        for side, sign in (("left", 1), ("right", -1)):
            for finger in ("thumb_2", "middle_0", "middle_1", "index_0", "index_1"):
                joint = self.model.joint(f"{side}_hand_{finger}_joint")
                self.data.qpos[joint.qposadr[0]] = sign * (0.08 if finger == "thumb_2" else -0.08)
        mujoco.mj_forward(self.model, self.data)
        self.robot = G1Controller(self.model, self.data)

    def step(self) -> None:
        self.robot.tick()
        if self.robot.stop_reason:
            return
        mujoco.mj_step(self.model, self.data)
        # Detect the final substep as well, without issuing another motor command.
        self.robot.check_safety()
        if self.robot.stop_reason:
            self.robot.tick()
