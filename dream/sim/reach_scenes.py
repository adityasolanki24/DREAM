"""Scenes for the unfamiliar-height reach test: a fixture with a small coloured handle.

``ReachScene`` is GROUND TRUTH, like ``RoomSpec``: it builds the world and is read by the
independent evaluator (dream/eval/reach.py); decision code never sees it.

World: an empty floor (the floor of the robot MJCF), the robot at the origin facing +x (so the
odometry frame and the world frame coincide at the start), and a fixture, a solid box in a
neutral grey, whose front face is ``distance`` ahead and ``lateral`` to the left of the start,
turned to face the start with a yaw jitter. A small, saturated-colour bar handle protrudes from
the middle of the front face at the requested height. Optional distractors: handle-less crates
in other palette colours, and a second fixture whose handle has another colour.

Fingers stay open (``build_mobile_g1(hands=True)`` default targets): the variant that passed the
locomotion regression (tests/test_g1_hands.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, pi, sin

import mujoco
import numpy as np

from dream.sim.room import PALETTE

HANDLE_HALF = (0.012, 0.045, 0.012)  # m: protrusion / width / thickness halves of the bar
MIN_FIXTURE_ABOVE = 0.12  # m the fixture top is kept above the handle centre
CLEAR_RADIUS = 1.0  # m around the target face kept free of distractors
START_CLEAR = 0.8  # m around the start kept free of distractors
HANDLE_COLORS = tuple(PALETTE)  # the detector's palette: red orange yellow green blue purple


@dataclass(frozen=True)
class SolidSpec:
    """A box: ``center`` (x, y, z), ``half`` extents in its local frame, ``yaw``."""

    name: str
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class ReachScene:
    seed: int
    handle_height: float  # m, handle centre above the floor
    handle_color: str
    handle_center: tuple[float, float, float]  # world; also the odometry frame at the start
    face_normal: tuple[float, float]  # outward unit normal of the fixture front face
    fixture: SolidSpec
    handle: SolidSpec
    distractors: tuple[SolidSpec, ...] = field(default_factory=tuple)
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # robot x, y, yaw
    params: dict = field(default_factory=dict)  # the drawn parameters, for the record


def _grey(v: float) -> tuple[float, float, float, float]:
    return (v, v, v, 1.0)


def sample_reach_scene(
    seed: int,
    height: float,
    distance: float,
    lateral: float,
    yaw_jitter: float,
    n_crates: int = 0,
    second_fixture: bool = False,
    handle_color: str | None = None,
) -> ReachScene:
    """Scene from explicit conditions (the config draws them); the seed draws the rest:
    handle colour (unless given), fixture size and shade, distractor placement."""
    rng = np.random.default_rng([seed, 0])
    colors = list(HANDLE_COLORS)
    color = handle_color or str(rng.choice(colors))
    # Fixture: depth, width and a height above the handle.
    hd, hw = float(rng.uniform(0.15, 0.3)), float(rng.uniform(0.25, 0.5))
    lo = max(0.55, height + MIN_FIXTURE_ABOVE)
    hh = float(rng.uniform(lo, lo + 0.5)) / 2
    face = np.array([distance, lateral])
    facing = atan2(-lateral, -distance) + yaw_jitter  # outward normal toward the start
    n = np.array([cos(facing), sin(facing)])
    centre = face - n * hd
    fixture = SolidSpec(
        "fixture_body", (float(centre[0]), float(centre[1]), hh), (hd, hw, hh), facing,
        _grey(float(rng.uniform(0.45, 0.85))),
    )  # fmt: skip
    hh_, hw_, ht_ = HANDLE_HALF
    hxy = face + n * hh_
    hcentre = (float(hxy[0]), float(hxy[1]), float(height))
    handle = SolidSpec(
        "reach_handle", hcentre, (hh_, hw_, ht_), facing, (*PALETTE[color], 1.0)
    )  # fmt: skip
    others = [c for c in colors if c != color]
    placed: list[tuple[float, float, float]] = [(float(face[0]), float(face[1]), CLEAR_RADIUS)]
    placed.append((0.0, 0.0, START_CLEAR))
    distractors: list[SolidSpec] = []

    def place(r: float) -> tuple[float, float] | None:
        for _ in range(200):
            p = (float(rng.uniform(0.3, distance + 0.8)), float(rng.uniform(-1.6, 1.6)))
            if all(np.hypot(p[0] - x, p[1] - y) > r + pr for x, y, pr in placed):
                placed.append((p[0], p[1], r))
                return p
        return None

    if second_fixture:
        half = (0.2, 0.3, 0.45)
        p = place(0.45)
        if p is not None:
            yaw = atan2(-p[1], -p[0]) + float(rng.uniform(-0.3, 0.3))
            nn = np.array([cos(yaw), sin(yaw)])
            dc = str(rng.choice(others))
            distractors.append(
                SolidSpec(
                    "distractor_fixture",
                    (p[0], p[1], half[2]),
                    half,
                    yaw,
                    _grey(float(rng.uniform(0.45, 0.85))),
                )  # fmt: skip
            )
            hp = np.array(p) + nn * (half[0] + hh_)
            distractors.append(
                SolidSpec(
                    "distractor_handle",
                    (float(hp[0]), float(hp[1]), 0.5),
                    HANDLE_HALF,
                    yaw,
                    (*PALETTE[dc], 1.0),
                )  # fmt: skip
            )
    for i in range(n_crates):
        half = tuple(float(v) for v in rng.uniform(0.12, 0.25, 3))
        p = place(float(np.hypot(half[0], half[1])) + 0.1)
        if p is None:
            continue
        distractors.append(
            SolidSpec(
                f"distractor_crate_{i}",
                (p[0], p[1], half[2]),
                half,
                float(rng.uniform(-pi, pi)),
                (*PALETTE[str(rng.choice(others))], 1.0),
            )  # fmt: skip
        )
    return ReachScene(
        seed, float(height), color, hcentre, (float(n[0]), float(n[1])), fixture, handle,
        tuple(distractors),
        params={"distance": distance, "lateral": lateral, "yaw_jitter": yaw_jitter,
                "fixture_half": [hd, hw, hh], "n_crates": n_crates,
                "second_fixture": second_fixture},
    )  # fmt: skip


def _quat(yaw: float) -> list[float]:
    return [cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)]


def add_reach_scene(spec: mujoco.MjSpec, scene: ReachScene) -> None:
    """Scene construction: static box geoms on the world body (the handle is fixed to the
    fixture; nothing moves, the task is to touch it)."""
    for s in (scene.fixture, scene.handle, *scene.distractors):
        spec.worldbody.add_geom(
            name=s.name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(s.center), size=list(s.half),
            quat=_quat(s.yaw), rgba=list(s.rgba),
        )  # fmt: skip
