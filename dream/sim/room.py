"""Procedural indoor rooms for the mobile G1: walls, furniture, and a cabinet whose mechanism
is hidden behind a closed door.

``RoomSpec`` is ground truth, like ``CompartmentSpec``: for building the scene and scoring
only, never for decision-making code. Every cabinet looks alike from outside (a closed box
with a handle on its front); what moves is hidden: a door hinged on its left or right edge,
a drawer that slides out, or nothing (fixed). Distractor crates have a cabinet's size and a
colour from the same palette but no handle, so neither size nor colour identifies the
cabinet; a handle on a box does.

Rooms are sampled from a seed. ``shifted=True`` draws from a wider distribution (larger
rooms, more clutter, more distractors) for changed-layout evaluation. ``with_cabinet=False``
is the negative control: the same room, with a handle-less crate of the cabinet's size and
colour where the cabinet would be, so the correct outcome is "not found".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin

import mujoco
import numpy as np

WALL_THICKNESS = 0.1  # m
WALL_HEIGHT = 2.2  # m
PANEL = 0.02  # m, cabinet panel thickness
MECHANISMS = ("hinge_left", "hinge_right", "drawer", "fixed")
# Saturated colours the detector names distinctly (dream/perception/detect.color_name).
PALETTE = {
    "red": (0.75, 0.15, 0.12),
    "orange": (0.85, 0.45, 0.1),
    "yellow": (0.85, 0.75, 0.15),
    "green": (0.2, 0.6, 0.25),
    "blue": (0.15, 0.3, 0.75),
    "purple": (0.5, 0.25, 0.65),
}
HANDLE_RGBA = (0.15, 0.15, 0.17, 1.0)


@dataclass(frozen=True)
class FurnitureSpec:
    kind: str  # "table" | "crate" | "shelf"
    xy: tuple[float, float]
    yaw: float
    half: tuple[float, float, float]  # m
    color: str


@dataclass(frozen=True)
class CabinetSpec:
    """Ground truth. Local frame: origin at the base centre, +x out of the front."""

    xy: tuple[float, float]
    yaw: float
    half: tuple[float, float, float]  # m (depth, width, height) halves
    mechanism: str  # one of MECHANISMS, or "none": a crate, no door and no handle
    color: str
    friction: float  # N m (hinge) or N (drawer), joint frictionloss
    handle_height_frac: float = 0.75  # of the door height


@dataclass(frozen=True)
class RoomSpec:
    width: float  # m, along world x
    depth: float  # m, along world y
    cabinet: CabinetSpec
    furniture: tuple[FurnitureSpec, ...] = field(default_factory=tuple)
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # robot x, y, yaw


def _footprint_ok(xy, radius, placed, room_w, room_d, margin=0.15) -> bool:
    x, y = xy
    if abs(x) > room_w / 2 - radius - margin or abs(y) > room_d / 2 - radius - margin:
        return False
    return all(np.hypot(x - px, y - py) > radius + pr + 0.25 for (px, py, pr) in placed)


def sample_room(
    rng: np.random.Generator, shifted: bool = False, with_cabinet: bool = True
) -> RoomSpec:
    w = float(rng.uniform(6.5, 8.0) if shifted else rng.uniform(4.5, 6.0))
    d = float(rng.uniform(5.5, 7.0) if shifted else rng.uniform(4.0, 5.5))
    placed: list[tuple[float, float, float]] = []
    colors = list(PALETTE)
    # Cabinet against a random wall, front facing into the room.
    half = (
        float(rng.uniform(0.22, 0.3)),
        float(rng.uniform(0.25, 0.4)),
        float(rng.uniform(0.4, 0.55)),
    )
    wall = int(rng.integers(4))
    for _ in range(200):
        along = float(rng.uniform(-0.35, 0.35))
        if wall in (0, 1):  # +x / -x walls
            sign = 1 if wall == 0 else -1
            xy = (sign * (w / 2 - half[0] - 0.05), along * d)
            yaw = pi if sign == 1 else 0.0
        else:
            sign = 1 if wall == 2 else -1
            xy = (along * w, sign * (d / 2 - half[0] - 0.05))
            yaw = -pi / 2 if sign == 1 else pi / 2
        r = float(np.hypot(half[0], half[1]))
        if _footprint_ok(xy, r, placed, w, d, margin=-r):
            break
    placed.append((xy[0], xy[1], float(np.hypot(half[0], half[1])) + 0.8))  # keep its front clear
    mechanism = str(rng.choice(MECHANISMS))  # drawn either way: same room for the control
    cabinet = CabinetSpec(
        xy=(float(xy[0]), float(xy[1])),
        yaw=float(yaw),
        half=half,
        mechanism=mechanism if with_cabinet else "none",
        color=str(rng.choice(colors)),
        friction=float(rng.uniform(0.3, 1.5)),
    )
    furniture = []
    n_distract = int(rng.integers(1, 3) if shifted else rng.integers(0, 2))
    n_other = int(rng.integers(3, 6) if shifted else rng.integers(1, 4))
    kinds = ["crate"] * n_distract + list(rng.choice(["table", "shelf", "crate"], n_other))
    for k, kind in enumerate(kinds):
        if kind == "table":
            fh = (rng.uniform(0.3, 0.5), rng.uniform(0.4, 0.7), rng.uniform(0.35, 0.4))
        elif kind == "shelf":
            fh = (rng.uniform(0.15, 0.2), rng.uniform(0.35, 0.6), rng.uniform(0.6, 0.8))
        else:  # crates: the first n_distract are cabinet-sized
            fh = half if k < n_distract else tuple(rng.uniform(0.15, 0.3, 3))
        r = float(np.hypot(fh[0], fh[1]))
        for _ in range(300):
            fxy = (float(rng.uniform(-w / 2, w / 2)), float(rng.uniform(-d / 2, d / 2)))
            if _footprint_ok(fxy, r, placed, w, d):
                placed.append((fxy[0], fxy[1], r))
                furniture.append(
                    FurnitureSpec(
                        kind=str(kind),
                        xy=fxy,
                        yaw=float(rng.uniform(-pi, pi)),
                        half=tuple(float(v) for v in fh),
                        color=str(rng.choice(colors)),
                    )
                )
                break
    for _ in range(500):
        sxy = (float(rng.uniform(-w / 2, w / 2)), float(rng.uniform(-d / 2, d / 2)))
        if _footprint_ok(sxy, 0.45, placed, w, d, margin=0.3):
            break
    else:
        raise RuntimeError("no free start position")
    return RoomSpec(w, d, cabinet, tuple(furniture), (sxy[0], sxy[1], float(rng.uniform(-pi, pi))))


def _quat(yaw: float) -> list[float]:
    return [cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)]


def _rgba(color: str) -> list[float]:
    return [*PALETTE[color], 1.0]


def add_room(spec: mujoco.MjSpec, room: RoomSpec) -> None:
    """Add walls, furniture and the cabinet to an ``MjSpec`` world (scene construction)."""
    world = spec.worldbody
    wall_rgba = [0.82, 0.82, 0.8, 1.0]
    hw, hd, t = room.width / 2, room.depth / 2, WALL_THICKNESS / 2
    for name, pos, size in (
        ("wall_px", (hw + t, 0, WALL_HEIGHT / 2), (t, hd + 2 * t, WALL_HEIGHT / 2)),
        ("wall_nx", (-hw - t, 0, WALL_HEIGHT / 2), (t, hd + 2 * t, WALL_HEIGHT / 2)),
        ("wall_py", (0, hd + t, WALL_HEIGHT / 2), (hw, t, WALL_HEIGHT / 2)),
        ("wall_ny", (0, -hd - t, WALL_HEIGHT / 2), (hw, t, WALL_HEIGHT / 2)),
    ):
        world.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=wall_rgba
        )
    for i, f in enumerate(room.furniture):
        body = world.add_body(name=f"furniture_{i}", pos=[*f.xy, 0], quat=_quat(f.yaw))
        if f.kind == "table":  # top plus four legs: the space underneath is open
            hx, hy, hz = f.half
            top = 0.02
            body.add_geom(
                name=f"furniture_{i}_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0, 0, 2 * hz - top], size=[hx, hy, top], rgba=_rgba(f.color),
            )  # fmt: skip
            for j, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
                body.add_geom(
                    name=f"furniture_{i}_leg{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[sx * (hx - 0.03), sy * (hy - 0.03), hz - top],
                    size=[0.025, 0.025, hz - top], rgba=_rgba(f.color),
                )  # fmt: skip
        else:
            body.add_geom(
                name=f"furniture_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0, 0, f.half[2]], size=list(f.half), rgba=_rgba(f.color),
            )  # fmt: skip
    _add_cabinet(world, room.cabinet)


def _add_cabinet(world, c: CabinetSpec) -> None:
    hx, hy, hz = c.half
    rgba = _rgba(c.color)
    body = world.add_body(name="cabinet", pos=[*c.xy, 0], quat=_quat(c.yaw))

    def panel(name, pos, size, parent=body):
        parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba)

    p = PANEL / 2
    panel("cabinet_back", [-hx + p, 0, hz], [p, hy, hz])
    panel("cabinet_left", [0, hy - p, hz], [hx, p, hz])
    panel("cabinet_right", [0, -hy + p, hz], [hx, p, hz])
    panel("cabinet_top", [0, 0, 2 * hz - p], [hx, hy, p])
    panel("cabinet_bottom", [0, 0, p], [hx, hy, p])
    if c.mechanism == "none":  # negative control: a closed crate, no handle
        body.add_geom(
            name="cabinet_crate", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, hz],
            size=[hx, hy, hz], rgba=rgba,
        )  # fmt: skip
        return
    door_h = hz - PANEL  # half height of the opening
    handle_z = PANEL + 2 * door_h * c.handle_height_frac
    if c.mechanism in ("hinge_left", "hinge_right"):
        side = 1 if c.mechanism == "hinge_left" else -1  # +y is the cabinet's left
        door = body.add_body(name="cabinet_door", pos=[hx - p, side * (hy - p), hz])
        door.add_joint(
            name="cabinet_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, side],
            range=[0, 1.9], frictionloss=c.friction, damping=0.5,
        )  # fmt: skip
        panel("cabinet_door_panel", [0, -side * (hy - p), 0], [p, hy - p, door_h], door)
        handle_local = [p + 0.03, -side * (2 * hy - 0.08), handle_z - hz]
    elif c.mechanism == "drawer":
        door = body.add_body(name="cabinet_door", pos=[hx - p, 0, hz])
        door.add_joint(
            name="cabinet_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
            range=[0, 1.6 * hx], frictionloss=c.friction * 5, damping=2.0,
        )  # fmt: skip
        panel("cabinet_door_panel", [0, 0, 0], [p, hy - p, door_h], door)
        # The drawer box behind the front panel.
        panel(
            "cabinet_drawer_floor",
            [-hx + 2 * PANEL, 0, -door_h + p],
            [hx - 2 * PANEL, hy - 2 * PANEL, p],
            door,
        )
        handle_local = [p + 0.03, 0.0, handle_z - hz]
    else:  # fixed: the front is part of the carcass
        door = body
        panel("cabinet_door_panel", [hx - p, 0, hz], [p, hy - p, door_h])
        handle_local = [hx + 0.03, 0.0, handle_z]
    # A horizontal bar handle on two posts; grey, so it contrasts with every palette colour.
    door.add_geom(
        name="cabinet_handle", type=mujoco.mjtGeom.mjGEOM_BOX, pos=handle_local,
        size=[0.012, 0.06, 0.012], rgba=list(HANDLE_RGBA),
    )  # fmt: skip
    for k, dy in enumerate((-0.05, 0.05)):
        post = [handle_local[0] - 0.018, handle_local[1] + dy, handle_local[2]]
        door.add_geom(
            name=f"cabinet_handle_post{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=post,
            size=[0.012, 0.008, 0.008], rgba=list(HANDLE_RGBA),
        )  # fmt: skip
    door.add_site(
        name="cabinet_handle_site", pos=handle_local, size=[0.01] * 3, rgba=[1, 0.5, 0, 0]
    )


def cabinet_front(c: CabinetSpec) -> tuple[np.ndarray, np.ndarray]:
    """World centre of the cabinet's front face (at mid height) and its outward normal."""
    n = np.array([cos(c.yaw), sin(c.yaw)])
    centre = np.array(c.xy) + n * c.half[0]
    return np.array([*centre, c.half[2]]), n
