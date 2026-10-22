"""Procedural single-joint mechanisms and their MJCF scenes.

``MechanismSpec`` is ground truth. Only the simulator (to build the scene) and the evaluation
code (to score the agent) may read it.

Every mechanism family looks the same: a panel with a handle on a fixed cabinet frame. The handle
position is sampled independently of the joint, so appearance tells the agent nothing about the
kinematics, as in the report's "visually identical, kinematically different" benchmark.

Simplifications for this stage, all deliberate:
- The gripper is a ball-joint ``connect`` constraint that switches on when the end effector
  reaches the handle. There are no finger contacts.
- Panels have ``gravcomp=1``, so they behave like counterbalanced parts and a bottom-hinged flap
  does not fall open on its own.
- The mechanism has no collision geometry. Joint limits implement the stops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dream.types import GANTRY_KP, GANTRY_KV

JOINT_TYPES = ("fixed", "prismatic", "revolute")

# family -> joint type. Families decide where axes go. The agent never sees the family name.
FAMILIES = {
    "fixed": "fixed",
    "drawer": "prismatic",  # slides out toward the robot (-y)
    "slider": "prismatic",  # slides sideways (+-x), like a sliding cupboard door
    "left_hinge": "revolute",
    "right_hinge": "revolute",
    "bottom_hinge": "revolute",  # flap, like an oven door
    "top_hinge": "revolute",
}

PANEL_THICKNESS = 0.02
HANDLE_STANDOFF = 0.04  # m in front of the panel face (-y)
PANEL_CENTER = np.array([0.0, 0.0, 0.6])
EE_HOME = np.array([0.0, -0.35, 0.6])


@dataclass(frozen=True)
class MechanismSpec:
    """Ground-truth mechanism. Privileged: never give this to the agent."""

    family: str
    joint_type: str
    panel_half_extents: np.ndarray  # (3,)
    handle_offset: np.ndarray  # (3,) handle site relative to the panel centre
    axis: np.ndarray  # (3,) unit, world frame (zeros for fixed); +q opens toward the robot
    origin: np.ndarray  # (3,) world point on the joint axis (revolute only)
    upper_limit: float  # m or rad; the lower limit is 0 (closed)
    friction: float  # N, equivalent Coulomb resistance at the handle
    damping: float  # N s/m or N m s/rad

    @property
    def handle_pos(self) -> np.ndarray:
        return PANEL_CENTER + self.handle_offset


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def _tilt(axis: np.ndarray, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb ``axis`` by a random rotation of at most ``max_deg`` degrees."""
    if max_deg <= 0:
        return _unit(axis)
    perp = np.cross(axis, rng.normal(size=3))
    perp /= np.linalg.norm(perp)
    ang = np.deg2rad(rng.uniform(0, max_deg))
    return _unit(np.cos(ang) * axis + np.sin(ang) * perp)


def sample_mechanism(
    family: str,
    rng: np.random.Generator,
    *,
    max_tilt_deg: float = 0.0,
    hinge_offset_max: float = 0.0,
    friction_range: tuple[float, float] = (0.5, 3.0),
) -> MechanismSpec:
    """Sample a ground-truth mechanism from ``family``.

    ``max_tilt_deg`` and ``hinge_offset_max`` widen the distribution beyond axis-aligned joints
    at panel edges. They are used to build held-out generalisation sets.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")
    joint_type = FAMILIES[family]
    half = np.array([rng.uniform(0.15, 0.25), PANEL_THICKNESS / 2, rng.uniform(0.12, 0.22)])
    # The handle is anywhere on the front face, independent of the joint.
    handle_offset = np.array(
        [
            rng.uniform(-0.7, 0.7) * half[0],
            -half[1] - HANDLE_STANDOFF,
            rng.uniform(-0.7, 0.7) * half[2],
        ]
    )
    friction = float(rng.uniform(*friction_range))
    axis = np.zeros(3)
    origin = np.zeros(3)
    upper = 0.0
    damping = 0.0
    if family == "drawer":
        axis = _tilt(np.array([0.0, -1.0, 0.0]), max_tilt_deg, rng)
        upper = float(rng.uniform(0.15, 0.3))
        damping = 2.0
    elif family == "slider":
        axis = _tilt(np.array([rng.choice([-1.0, 1.0]), 0.0, 0.0]), max_tilt_deg, rng)
        upper = float(rng.uniform(0.15, 0.3))
        damping = 2.0
    elif joint_type == "revolute":
        edge_axis, edge_point = {
            "left_hinge": (np.array([0.0, 0.0, 1.0]), np.array([-half[0], -half[1], 0.0])),
            "right_hinge": (np.array([0.0, 0.0, 1.0]), np.array([half[0], -half[1], 0.0])),
            "bottom_hinge": (np.array([1.0, 0.0, 0.0]), np.array([0.0, -half[1], -half[2]])),
            "top_hinge": (np.array([1.0, 0.0, 0.0]), np.array([0.0, -half[1], half[2]])),
        }[family]
        # The hinge line can sit slightly in from (or beyond) the panel edge.
        inward = -np.sign(edge_point) * (np.abs(edge_point) > 0) * np.array([1, 0, 1])
        edge_point = edge_point + inward * rng.uniform(-hinge_offset_max, hinge_offset_max)
        axis = _tilt(edge_axis, max_tilt_deg, rng)
        origin = PANEL_CENTER + edge_point
        # Point +q so the far side of the panel swings toward the robot (-y).
        lever = PANEL_CENTER - origin
        if np.cross(axis, lever)[1] > 0:
            axis = -axis
        upper = float(rng.uniform(1.0, 1.6))
        damping = 0.05
    return MechanismSpec(
        family=family,
        joint_type=joint_type,
        panel_half_extents=half,
        handle_offset=handle_offset,
        axis=axis,
        origin=origin,
        upper_limit=upper,
        friction=friction,
        damping=damping,
    )


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def handle_lever_arm(spec: MechanismSpec) -> float:
    """Distance from the handle to the hinge line (revolute only)."""
    r = spec.handle_pos - spec.origin
    return float(np.linalg.norm(r - np.dot(r, spec.axis) * spec.axis))


def build_mjcf(spec: MechanismSpec) -> str:
    """Return the MJCF scene: cabinet, mechanism and a 3-axis compliant gantry end effector."""
    half = spec.panel_half_extents
    # A stiff friction time constant limits creep under sub-friction loads (MuJoCo friction
    # loss is a soft constraint; the default 0.02 s creeps several mm per second).
    common = f'limited="true" range="0 {spec.upper_limit:.6g}" solreffriction="0.004 1"'
    if spec.joint_type == "fixed":
        joint_xml = ""
    elif spec.joint_type == "prismatic":
        joint_xml = (
            f'<joint name="mech" type="slide" axis="{_fmt(spec.axis)}" {common} '
            f'frictionloss="{spec.friction:.6g}" damping="{spec.damping:.6g}"/>'
        )
    else:
        # MuJoCo hinge friction is a torque. Convert the handle-equivalent force.
        torque = spec.friction * handle_lever_arm(spec)
        joint_xml = (
            f'<joint name="mech" type="hinge" pos="{_fmt(spec.origin - PANEL_CENTER)}" '
            f'axis="{_fmt(spec.axis)}" {common} '
            f'frictionloss="{torque:.6g}" damping="{spec.damping:.6g}"/>'
        )
    frame_t = 0.02
    frame = "\n".join(
        f'<geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba=".45 .35 .25 1"/>'
        for p, s in [
            (PANEL_CENTER + [0, 0.1, 0], [half[0] + 2 * frame_t, 0.08, half[2] + 2 * frame_t]),
            (PANEL_CENTER + [-half[0] - frame_t, 0, 0], [frame_t, 0.03, half[2] + 2 * frame_t]),
            (PANEL_CENTER + [half[0] + frame_t, 0, 0], [frame_t, 0.03, half[2] + 2 * frame_t]),
        ]
    )
    actuators = "\n".join(
        f'<position name="act_{a}" joint="ee_{a}" kp="{GANTRY_KP:g}" kv="{GANTRY_KV:g}" '
        'forcelimited="true" forcerange="-20 20"/>'
        for a in "xyz"
    )
    h = spec.handle_offset
    return f"""
<mujoco model="dream_{spec.family}">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -1.5 2" dir="0 1 -1"/>
    <geom type="plane" size="1.5 1.5 0.05" rgba=".9 .9 .9 1"/>
    <camera name="front" pos="0.9 -1.2 1.1" xyaxes="0.8 0.6 0 -0.25 0.33 0.91"/>
    {frame}
    <body name="panel" pos="{_fmt(PANEL_CENTER)}" gravcomp="1">
      {joint_xml}
      <geom type="box" size="{_fmt(half)}" mass="1" rgba=".75 .6 .4 1"/>
      <geom type="box" pos="{h[0]:.6g} {(h[1] - half[1]) / 2:.6g} {h[2]:.6g}"
            size="0.01 {(-h[1] - half[1]) / 2:.6g} 0.01" mass="0.02" rgba=".2 .2 .2 1"/>
      <site name="handle" pos="{_fmt(h)}" size="0.012" rgba=".9 .1 .1 1"/>
    </body>
    <body name="ee_base" pos="{_fmt(EE_HOME)}">
      <body name="ee" gravcomp="1">
        <joint name="ee_x" type="slide" axis="1 0 0" range="-0.8 0.8" damping="2"/>
        <joint name="ee_y" type="slide" axis="0 1 0" range="-0.8 0.8" damping="2"/>
        <joint name="ee_z" type="slide" axis="0 0 1" range="-0.8 0.8" damping="2"/>
        <geom type="sphere" size="0.02" mass="0.3" rgba=".1 .4 .9 1"/>
        <site name="ee_tip" size="0.005"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect name="grasp" site1="ee_tip" site2="handle" active="false" solref="0.004 1"/>
  </equality>
  <actuator>
    {actuators}
  </actuator>
  <sensor>
    <framepos name="handle_pos" objtype="site" objname="handle"/>
    <framequat name="handle_quat" objtype="site" objname="handle"/>
    <framepos name="ee_pos" objtype="site" objname="ee_tip"/>
    <actuatorfrc name="frc_x" actuator="act_x"/>
    <actuatorfrc name="frc_y" actuator="act_y"/>
    <actuatorfrc name="frc_z" actuator="act_z"/>
  </sensor>
</mujoco>
"""
