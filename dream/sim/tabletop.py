"""Tabletop manipulation scenes: the Menagerie Franka Panda, free cubes and a walled tray.

``TabletopSpec`` is ground truth: exact object poses and the tray's geometry. Like
``MechanismSpec``, it may be used to build scenes and to score results, never by decision-making
code. The robot is the unmodified Menagerie ``panda.xml`` (third_party/, Apache-2.0). This
module only adds a TCP site, the table, the objects and the tray around it through ``MjSpec``.

Objects are free bodies with real collision geometry. Nothing in the scene attaches objects to
the gripper, so grasps succeed or fail through finger contact and friction alone.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from dream.robot.panda import ROBOT_BODIES
from dream.sim.compartment import CompartmentSpec, add_compartment, sample_compartment

PANDA_XML = (
    pathlib.Path(__file__).resolve().parents[2]
    / "third_party"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)
GRIP_STIFFNESS_SCALE = 10.0
TCP_OFFSET = 0.1034  # m from the hand frame to the centre of the fingertip pads
COLORS = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "blue": (0.15, 0.3, 0.85, 1.0),
    "green": (0.15, 0.65, 0.25, 1.0),
    "yellow": (0.9, 0.75, 0.1, 1.0),
}
WORKSPACE_XY = ((0.35, 0.7), (-0.3, 0.3))  # region where objects and the tray are placed
TRAY_FLOOR_HALF_THICKNESS = 0.002  # m
TRAY_FLOOR_TOP = 2 * TRAY_FLOOR_HALF_THICKNESS  # m


@dataclass(frozen=True)
class ObjectSpec:
    name: str  # also the colour name, e.g. "red"
    half_size: float  # m, cube half edge
    xy: tuple[float, float]  # m, initial position on the table
    yaw: float  # rad
    mass: float = 0.05  # kg


@dataclass(frozen=True)
class TraySpec:
    xy: tuple[float, float]  # m, centre
    inner_half: float = 0.09  # m, half width of the square floor (fits four 4 cm cubes)
    wall_height: float = 0.03  # m
    wall_thickness: float = 0.005  # m


@dataclass(frozen=True)
class TabletopSpec:
    """Ground truth for one scene. Privileged: never give this to decision-making code."""

    objects: tuple[ObjectSpec, ...]
    tray: TraySpec | None
    compartment: CompartmentSpec | None = None
    seed: int | None = field(default=None, compare=False)


def sample_tabletop(
    rng: np.random.Generator,
    colors: tuple[str, ...] = ("red", "blue", "green"),
    half_size: float = 0.02,
    min_gap: float = 0.09,
) -> TabletopSpec:
    """Random non-overlapping cube placements and tray position inside ``WORKSPACE_XY``."""
    (x0, x1), (y0, y1) = WORKSPACE_XY
    tray = TraySpec(
        xy=(float(rng.uniform(x0 + 0.05, x1 - 0.05)), float(rng.uniform(y0 + 0.05, y1 - 0.05)))
    )
    tray_clearance = tray.inner_half + tray.wall_thickness + half_size + 0.03
    placed: list[ObjectSpec] = []
    for color in colors:
        for _ in range(1000):
            xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            if np.max(np.abs(xy - tray.xy)) < tray_clearance:
                continue
            if all(np.linalg.norm(xy - o.xy) >= min_gap for o in placed):
                break
        else:
            raise RuntimeError("could not place objects without overlap")
        placed.append(
            ObjectSpec(
                color, half_size, (float(xy[0]), float(xy[1])), float(rng.uniform(-np.pi, np.pi))
            )
        )
    return TabletopSpec(objects=tuple(placed), tray=tray)


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def build_model(spec: TabletopSpec) -> mujoco.MjModel:
    s = mujoco.MjSpec.from_file(str(PANDA_XML))
    s.option.timestep = 0.002
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # better frictional grasps than pyramidal
    s.option.noslip_iterations = 3  # suppresses slow slip of grasped objects
    s.visual.global_.offwidth = 640
    s.visual.global_.offheight = 480

    for name in ROBOT_BODIES:  # Franka controllers compensate gravity internally; so do we
        s.body(name).gravcomp = 1.0
    # Menagerie's gripper is a soft position servo (kp 100 N/m on the finger tendon): it squeezes
    # ~2 N on a 4 cm cube, too little to pull a handle against joint friction. A real Franka Hand
    # grips with up to 70 N. Stiffen the servo x10 here (~20 N on a cube); the vendored file is
    # unchanged.
    grip = s.actuator("actuator8")
    grip.gainprm[0] *= GRIP_STIFFNESS_SCALE
    grip.biasprm[1] *= GRIP_STIFFNESS_SCALE
    grip.biasprm[2] *= GRIP_STIFFNESS_SCALE
    s.body("hand").add_site(
        name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005] * 3, rgba=[1, 0, 0, 0.5]
    )
    world = s.worldbody
    world.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], diffuse=[0.8] * 3)
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.2, 1.2, 0.05],
        rgba=[0.85, 0.83, 0.8, 1],
    )
    world.add_camera(name="overview", pos=[1.35, 0.0, 0.85], xyaxes=[0, 1, 0, -0.5, 0, 0.87])

    for obj in spec.objects:
        b = world.add_body(name=obj.name, pos=[*obj.xy, obj.half_size], quat=_yaw_quat(obj.yaw))
        b.add_freejoint(name=f"{obj.name}_free")
        b.add_geom(
            name=f"{obj.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[obj.half_size] * 3,
            mass=obj.mass,
            rgba=COLORS[obj.name],
            friction=[1.5, 0.005, 0.0001],
            condim=4,
        )
        b.add_site(name=f"{obj.name}_site", size=[0.003] * 3)

    if spec.compartment is not None:
        add_compartment(s, spec.compartment)
    if spec.tray is not None:
        _add_tray(world, spec.tray)
    return s.compile()


def _add_tray(world, t: TraySpec) -> None:
    tray = world.add_body(name="tray", pos=[*t.xy, 0])
    floor_half = t.inner_half + t.wall_thickness
    tray.add_geom(
        name="tray_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[floor_half, floor_half, TRAY_FLOOR_HALF_THICKNESS],
        pos=[0, 0, TRAY_FLOOR_HALF_THICKNESS],
        rgba=[0.5, 0.5, 0.55, 1],
    )
    h = t.wall_height / 2
    for i, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        size = [t.wall_thickness, floor_half, h] if dx else [floor_half, t.wall_thickness, h]
        pos = [dx * (t.inner_half + t.wall_thickness), dy * (t.inner_half + t.wall_thickness), h]
        tray.add_geom(
            name=f"tray_wall{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            rgba=[0.45, 0.45, 0.5, 1],
        )


def sample_compartment_scene(
    kind: str,
    rng: np.random.Generator,
    colors: tuple[str, ...] = ("red", "blue"),
    half_size: float = 0.02,
    looks_like: str | None = None,
) -> TabletopSpec:
    """A compartment with a hidden mechanism plus cubes placed clear of it, all in reach."""
    comp = sample_compartment(kind, rng, looks_like=looks_like)
    centre = np.array(comp.xy)
    # Keep cubes off the box, its handle, the drawer's pull-out path and the lid's swing.
    keep_out = comp.outer_half[0] + 0.12
    placed: list[ObjectSpec] = []
    for color in colors:
        for _ in range(2000):
            xy = np.array([rng.uniform(0.3, 0.6), rng.uniform(-0.35, 0.35)])
            if np.linalg.norm(xy - centre) < keep_out + 0.05:
                continue
            if all(np.linalg.norm(xy - o.xy) >= 0.09 for o in placed):
                break
        else:
            raise RuntimeError("could not place objects clear of the compartment")
        placed.append(
            ObjectSpec(
                color, half_size, (float(xy[0]), float(xy[1])), float(rng.uniform(-np.pi, np.pi))
            )
        )
    return TabletopSpec(objects=tuple(placed), tray=None, compartment=comp)
