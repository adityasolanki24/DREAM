"""Procedural multi-room homes for semantic object search: GROUND TRUTH, for scene building and
scoring only, never for decision-making code (AGENTS.md invariant 1).

A home is an outer rectangle split into a grid of rooms by 0.1 m interior walls with doorways
(a random spanning tree over room adjacency plus each other adjacency with probability 0.3).
Rooms have a type; objects come from a furnishing table (``FURNISHING_NOMINAL``,
``FURNISHING_SHIFTED``) that is the BENCHMARK WORLD's distribution, written by the benchmark
author. It is independent of anything the agent knows: the agent's priors come from an external
knowledge resource, never from this table.

Conventions: world frame, home centred on the origin, SI units. An object's local frame has its
origin at the base centre on the floor and +x out of its front (the side a person stands at).
``half`` is the half extent (depth, width, height) of its bounding box. Every geom of an object
is named ``sem:{instance_id}:{part}``; walls are ``wall:{name}``; the floor is not added here.

``sample_home`` builds a home to a scenario recipe (A to L, below), verifies the recipe's
invariant on the finished geometry (including pixel counts from ray casting through the same
head-camera model the robot uses) and resamples until it holds, or raises. Scenarios ``I`` and
``K`` draw a variant: ``I_moved``, ``I_removed``, ``I_valid``; ``K_studio``, ``K_mismatch``.

Scenario recipes (T is the target's typical room type, see ``TARGETS``):
  A  target visible from the start: start in its room, 1.5 to 3.0 m away, facing it.
  B  target in T; start in another room; not visible at the start.
  C  T is partially observable: an occluder (partition wall, tall storage_shelf or tall
     kitchen_cabinet) hides the target from 0.5 m inside every doorway of T; start outside T.
  D  target in an atypical room (``TargetInfo.atypical``); a T room exists without it.
  E  a T room exists; the target is nowhere.
  F  T has the target plus a second room with confusers (``TargetInfo.confusers``); the
     target is in T (p 0.8) or the confuser room.
  G  target in T behind an occluder: visible from some in-room viewpoints, hidden from others.
  H  no T room and no target.
  I  stale memory: ``memory_prior`` gives the target's remembered place (age 1 to 7 days).
     I_moved: it is now in another room. I_removed: it is nowhere. I_valid: the memory is right.
  J  no T room; the target is in one of ``TargetInfo.pool``, with 2+ such rooms present.
  K  conflicting room evidence: K_studio is a room mixing a bed and a kitchenette (no T room);
     K_mismatch puts two mismatched objects in the T room. The target is in the mixed room
     (p 0.5) or another room.
  L  B physically, but the request is a manipulation request, which this system cannot do.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import cos, pi, sin

import mujoco
import numpy as np

WALL_THICKNESS = 0.1  # m
WALL_HEIGHT = 2.2  # m
DOOR_WIDTH = (1.0, 1.3)  # m
MIN_ROOM = 2.8  # m, each side of a room
ROBOT_RADIUS = 0.35  # m, the planner's inflated robot radius
CELL = 0.05  # m, raster resolution for reachability
STAND_RANGE = 1.5  # m: every object needs a reachable standing cell this close with line of sight
DOOR_CLEAR = 0.9  # m: nothing stands this close to a doorway
EXTENT_LIMIT = 11.0  # m: the agent's 24 m occupancy grid is centred on the start
MAX_ATTEMPTS = 80

CLASSES = (
    "refrigerator", "countertop", "sink", "stove", "kitchen_cabinet", "dining_table", "chair",
    "microwave", "bed", "wardrobe", "nightstand", "desk", "office_chair", "bookshelf", "sofa",
    "coffee_table", "tv_stand", "armchair", "toilet", "washbasin", "shower", "bathtub",
    "workbench", "tool_rack", "storage_shelf", "washing_machine", "dryer", "chest_freezer",
    "filing_cabinet",
)  # fmt: skip
ROOM_TYPES = (
    "kitchen", "bedroom", "bathroom", "living_room", "office", "garage", "laundry", "storage",
)  # fmt: skip


GENERATORS = ("nominal", "shifted", "xl")
# The ``xl`` generator (a new environment for the semantic-knowledge comparison): a 4 x 3 grid
# of 12 rooms of about 3.5 m, nominal furnishing and object scales, doorway logic as above.
# Rooms are as small as the nominal and shifted ones; the home is larger and the travel longer.
XL_GRID = (4, 3)  # columns, rows
XL_WIDTH = (13.0, 14.5)  # m
XL_DEPTH = (10.0, 11.5)  # m
XL_TYPES = (  # kitchen, living room, 3 bedrooms, 2 bathrooms, office, garage, laundry, storage
    "kitchen", "living_room", "bedroom", "bedroom", "bedroom", "bathroom", "bathroom", "office",
    "garage", "laundry", "storage",
)  # fmt: skip


class HomeGenerationError(RuntimeError):
    """No home satisfying the scenario was found within the attempt budget."""


# -- ground-truth records --------------------------------------------------------------------
@dataclass(frozen=True)
class Part:
    """One MuJoCo primitive of an object, in the object's local frame."""

    name: str
    shape: str  # "box" (size = half extents), "cylinder" (radius, half height), "sphere" (radius)
    pos: tuple[float, float, float]
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees


@dataclass(frozen=True)
class ObjectSpec:
    instance_id: str
    cls: str
    room_id: int
    xy: tuple[float, float]
    yaw: float  # rad; the front faces (cos yaw, sin yaw)
    half: tuple[float, float, float]  # m (depth, width, height) halves of the bounding box
    parts: tuple[Part, ...]
    z0: float = 0.0  # m, height of the base above the floor (a microwave on a counter)
    front_clear: float = 0.5  # m of floor in front that is kept free


@dataclass(frozen=True)
class RoomBox:
    """Axis-aligned room. ``lo``/``hi`` run to the centreline of interior walls and to the
    inner face of exterior walls."""

    id: int
    type: str
    lo: tuple[float, float]
    hi: tuple[float, float]


@dataclass(frozen=True)
class Doorway:
    a: int  # room ids
    b: int
    centre: tuple[float, float]
    width: float  # m
    axis: str  # "x" or "y": the direction one walks through it


@dataclass(frozen=True)
class WallSeg:
    name: str
    centre: tuple[float, float]
    half: tuple[float, float]  # half extents along x, y; the height is WALL_HEIGHT


@dataclass(frozen=True)
class HomeSpec:
    width: float  # m, along world x
    depth: float  # m, along world y
    rooms: tuple[RoomBox, ...]
    doorways: tuple[Doorway, ...]
    objects: tuple[ObjectSpec, ...]
    start: tuple[float, float, float]  # robot x, y, yaw
    scenario: str
    target_class: str | None
    request: str
    memory_prior: tuple[tuple[str, tuple[float, float, float], float], ...] = ()  # cls, xyz, age s
    generator: str = "nominal"
    seed: int = 0
    walls: tuple[WallSeg, ...] = ()


# -- small geometry --------------------------------------------------------------------------
@dataclass(frozen=True)
class Rect:
    cx: float
    cy: float
    hx: float
    hy: float
    yaw: float = 0.0

    def corners(self) -> np.ndarray:
        c, s = cos(self.yaw), sin(self.yaw)
        local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * [self.hx, self.hy]
        return local @ np.array([[c, s], [-s, c]]) + [self.cx, self.cy]

    def local(self, pts: np.ndarray) -> np.ndarray:
        c, s = cos(self.yaw), sin(self.yaw)
        d = np.asarray(pts, float) - [self.cx, self.cy]
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        c, s = cos(self.yaw), sin(self.yaw)
        dx, dy = pts[..., 0] - self.cx, pts[..., 1] - self.cy
        return (np.abs(c * dx + s * dy) <= self.hx) & (np.abs(-s * dx + c * dy) <= self.hy)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Euclidean distance from points to the rectangle (0 inside)."""
        u = np.abs(self.local(pts)) - [self.hx, self.hy]
        return np.hypot(np.maximum(u[..., 0], 0), np.maximum(u[..., 1], 0))


def _overlap(a: Rect, b: Rect, gap: float = 0.0) -> bool:
    """Separating-axis test for two oriented rectangles, each grown by gap / 2 (pure Python:
    the generator calls this hundreds of thousands of times)."""
    g = gap / 2
    ahx, ahy, bhx, bhy = a.hx + g, a.hy + g, b.hx + g, b.hy + g
    dx, dy = b.cx - a.cx, b.cy - a.cy
    ca, sa, cb, sb = cos(a.yaw), sin(a.yaw), cos(b.yaw), sin(b.yaw)
    for ux, uy in ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb)):
        ra = ahx * abs(ux * ca + uy * sa) + ahy * abs(-ux * sa + uy * ca)
        rb = bhx * abs(ux * cb + uy * sb) + bhy * abs(-ux * sb + uy * cb)
        if abs(dx * ux + dy * uy) > ra + rb:
            return False
    return True


def footprint(obj: ObjectSpec) -> Rect:
    return Rect(obj.xy[0], obj.xy[1], obj.half[0], obj.half[1], obj.yaw)


def footprints(home: HomeSpec) -> dict[str, np.ndarray]:
    """Instance id -> (4, 2) world polygon of its footprint."""
    return {o.instance_id: footprint(o).corners() for o in home.objects}


def instances_of(home: HomeSpec, cls: str) -> tuple[ObjectSpec, ...]:
    return tuple(o for o in home.objects if o.cls == cls)


def room_at(home: HomeSpec, xy) -> RoomBox | None:
    """The room containing a world point (wall centrelines are the boundaries)."""
    x, y = float(xy[0]), float(xy[1])
    for r in home.rooms:
        if r.lo[0] <= x <= r.hi[0] and r.lo[1] <= y <= r.hi[1]:
            return r
    return None


def room_interior(home: HomeSpec, room: RoomBox) -> tuple[np.ndarray, np.ndarray]:
    """Free-floor bounds of a room: its box minus half the thickness of interior walls."""
    lo, hi = np.array(room.lo), np.array(room.hi)
    ext = np.array([home.width, home.depth]) / 2
    t = WALL_THICKNESS / 2
    return lo + np.where(np.abs(lo + ext) < 1e-9, 0, t), hi - np.where(
        np.abs(hi - ext) < 1e-9, 0, t
    )


def door_keepouts(home: HomeSpec) -> list[Rect]:
    """The clear zone in front of every doorway, on both sides."""
    along = DOOR_CLEAR + WALL_THICKNESS / 2
    out = []
    for d in home.doorways:
        hx, hy = (along, d.width / 2) if d.axis == "x" else (d.width / 2, along)
        out.append(Rect(d.centre[0], d.centre[1], hx, hy))
    return out


# -- object dimensions and furnishing --------------------------------------------------------
# Full (depth, width, height) in metres, from typical product dimensions (rounded). Written by
# the benchmark author, not measured.
DIMS: dict[str, tuple[float, float, float]] = {
    "refrigerator": (0.7, 0.7, 1.8), "countertop": (0.6, 1.2, 0.9), "sink": (0.6, 1.2, 0.9),
    "stove": (0.6, 0.6, 0.9), "kitchen_cabinet": (0.6, 0.8, 0.9), "dining_table": (0.9, 1.5, 0.75),
    "chair": (0.45, 0.45, 0.9), "microwave": (0.35, 0.5, 0.3), "bed": (2.0, 1.4, 0.55),
    "wardrobe": (0.6, 1.0, 1.9), "nightstand": (0.4, 0.45, 0.55), "desk": (0.65, 1.4, 0.75),
    "office_chair": (0.5, 0.5, 0.95), "bookshelf": (0.3, 0.9, 1.8), "sofa": (0.9, 2.0, 0.85),
    "coffee_table": (0.6, 1.0, 0.45), "tv_stand": (0.4, 1.4, 1.0), "armchair": (0.85, 0.85, 0.85),
    "toilet": (0.7, 0.4, 0.75), "washbasin": (0.5, 0.6, 0.85), "shower": (0.9, 0.9, 2.0),
    "bathtub": (0.75, 1.7, 0.55), "workbench": (0.7, 1.8, 0.9), "tool_rack": (0.15, 1.2, 1.8),
    "storage_shelf": (0.4, 1.0, 1.9), "washing_machine": (0.6, 0.6, 0.85),
    "dryer": (0.6, 0.6, 0.85), "chest_freezer": (0.6, 1.0, 0.85),
    "filing_cabinet": (0.62, 0.46, 1.3),
}  # fmt: skip
SCALE_NOMINAL = (0.94, 1.06)  # per-axis scale range on DIMS
SCALE_SHIFTED = (0.80, 1.25)  # the shifted generator's wider range, mean shifted up
FRONT_CLEAR = {"bed": 0.6, "sofa": 0.6, "armchair": 0.5, "shower": 0.5, "tool_rack": 0.6}
LOOSE = {"coffee_table", "chair", "office_chair", "nightstand", "dining_table", "microwave"}
KITCHEN_RUN = {"countertop", "sink", "stove", "kitchen_cabinet"}

# type -> ((class, probability, min count, max count), ...)
FURNISHING_NOMINAL: dict[str, tuple[tuple[str, float, int, int], ...]] = {
    "kitchen": (("countertop", 0.95, 1, 2), ("sink", 0.9, 1, 1), ("stove", 0.8, 1, 1),
                ("kitchen_cabinet", 0.6, 1, 2), ("dining_table", 0.35, 1, 1),
                ("microwave", 0.3, 1, 1)),
    "bedroom": (("bed", 1.0, 1, 1), ("wardrobe", 0.8, 1, 1), ("nightstand", 0.6, 1, 2),
                ("desk", 0.3, 1, 1)),
    "bathroom": (("toilet", 1.0, 1, 1), ("washbasin", 0.95, 1, 1), ("shower", 0.55, 1, 1),
                 ("bathtub", 0.4, 1, 1)),
    "living_room": (("sofa", 0.95, 1, 1), ("coffee_table", 0.7, 1, 1), ("tv_stand", 0.8, 1, 1),
                    ("armchair", 0.4, 1, 2), ("bookshelf", 0.3, 1, 1)),
    "office": (("desk", 1.0, 1, 1), ("bookshelf", 0.7, 1, 1), ("filing_cabinet", 0.5, 1, 2)),
    "garage": (("workbench", 0.9, 1, 1), ("tool_rack", 0.7, 1, 1), ("storage_shelf", 0.6, 1, 2),
               ("chest_freezer", 0.2, 1, 1)),
    "laundry": (("washing_machine", 0.95, 1, 1), ("dryer", 0.7, 1, 1), ("washbasin", 0.4, 1, 1),
                ("storage_shelf", 0.3, 1, 1), ("chest_freezer", 0.1, 1, 1)),
    "storage": (("storage_shelf", 1.0, 1, 3), ("filing_cabinet", 0.3, 1, 1),
                ("chest_freezer", 0.12, 1, 1), ("wardrobe", 0.15, 1, 1)),
    "studio": (("bed", 1.0, 1, 1), ("nightstand", 0.6, 1, 1), ("sink", 0.9, 1, 1),
               ("countertop", 0.9, 1, 1), ("stove", 0.7, 1, 1), ("wardrobe", 0.4, 1, 1)),
}  # fmt: skip
# Shifted: different probabilities, larger counts, and (in the generator) extra clutter.
FURNISHING_SHIFTED: dict[str, tuple[tuple[str, float, int, int], ...]] = {
    "kitchen": (("countertop", 0.8, 1, 3), ("sink", 0.7, 1, 1), ("stove", 0.6, 1, 1),
                ("kitchen_cabinet", 0.8, 1, 3), ("dining_table", 0.55, 1, 1),
                ("microwave", 0.5, 1, 1)),
    "bedroom": (("bed", 1.0, 1, 1), ("wardrobe", 0.6, 1, 2), ("nightstand", 0.7, 1, 2),
                ("desk", 0.55, 1, 1), ("bookshelf", 0.35, 1, 1)),
    "bathroom": (("toilet", 1.0, 1, 1), ("washbasin", 0.85, 1, 2), ("shower", 0.45, 1, 1),
                 ("bathtub", 0.55, 1, 1), ("storage_shelf", 0.2, 1, 1)),
    "living_room": (("sofa", 0.85, 1, 2), ("coffee_table", 0.6, 1, 1), ("tv_stand", 0.6, 1, 1),
                    ("armchair", 0.6, 1, 2), ("bookshelf", 0.5, 1, 2)),
    "office": (("desk", 1.0, 1, 2), ("bookshelf", 0.6, 1, 2), ("filing_cabinet", 0.7, 1, 2),
               ("storage_shelf", 0.3, 1, 1)),
    "garage": (("workbench", 0.8, 1, 2), ("tool_rack", 0.5, 1, 2), ("storage_shelf", 0.8, 1, 2),
               ("chest_freezer", 0.35, 1, 1), ("washing_machine", 0.15, 1, 1)),
    "laundry": (("washing_machine", 0.85, 1, 1), ("dryer", 0.6, 1, 1), ("washbasin", 0.55, 1, 1),
                ("storage_shelf", 0.5, 1, 1), ("chest_freezer", 0.25, 1, 1)),
    "storage": (("storage_shelf", 1.0, 2, 4), ("filing_cabinet", 0.5, 1, 2),
                ("chest_freezer", 0.25, 1, 1), ("wardrobe", 0.3, 1, 2)),
    "studio": FURNISHING_NOMINAL["studio"],
}  # fmt: skip
CLUTTER_POOL = (
    "bookshelf", "storage_shelf", "filing_cabinet", "armchair", "coffee_table", "nightstand",
    "kitchen_cabinet", "wardrobe", "chest_freezer", "chair",
)  # fmt: skip
TYPE_WEIGHTS = {  # how often a filler room has each type
    "bedroom": 3.0, "bathroom": 2.0, "living_room": 2.0, "office": 1.5, "garage": 1.0,
    "laundry": 1.0, "storage": 1.0, "kitchen": 1.5,
}  # fmt: skip


@dataclass(frozen=True)
class TargetInfo:
    typical: str  # room type where the target is usually found
    atypical: tuple[str, ...]  # scenario D: unusual rooms for it
    pool: tuple[str, ...]  # scenario J: the plausible rooms when no typical room exists
    confusers: tuple[str, ...]  # scenario F: look-alikes placed in a second room
    mismatch: tuple[str, str]  # scenario K: out-of-place objects in the typical room


TARGETS: dict[str, TargetInfo] = {
    "refrigerator": TargetInfo("kitchen", ("garage", "office", "living_room", "laundry"),
                               ("garage", "living_room", "storage", "office"),
                               ("chest_freezer", "washbasin", "storage_shelf"),
                               ("washbasin", "toilet")),
    "bed": TargetInfo("bedroom", ("living_room", "office", "storage"),
                      ("living_room", "office", "storage", "garage"),
                      ("sofa", "armchair", "nightstand"), ("sink", "countertop")),
    "toilet": TargetInfo("bathroom", ("laundry", "garage", "storage"),
                         ("laundry", "garage", "storage", "office"),
                         ("washbasin", "shower", "storage_shelf"), ("stove", "countertop")),
    "stove": TargetInfo("kitchen", ("garage", "laundry", "storage"),
                        ("garage", "laundry", "storage", "office"),
                        ("countertop", "kitchen_cabinet", "sink"), ("washbasin", "toilet")),
    "washing_machine": TargetInfo("laundry", ("kitchen", "bathroom", "garage"),
                                  ("kitchen", "bathroom", "garage", "storage"),
                                  ("dryer", "chest_freezer", "storage_shelf"), ("bed", "wardrobe")),
    "workbench": TargetInfo("garage", ("storage", "laundry", "office"),
                            ("storage", "laundry", "office", "living_room"),
                            ("desk", "dining_table", "tool_rack"), ("bed", "toilet")),
    "bathtub": TargetInfo("bathroom", ("laundry", "bedroom", "storage"),
                          ("laundry", "bedroom", "storage", "garage"),
                          ("shower", "washbasin", "chest_freezer"), ("stove", "sink")),
    "sofa": TargetInfo("living_room", ("office", "bedroom", "garage"),
                       ("office", "bedroom", "garage", "storage"),
                       ("bed", "armchair", "tv_stand"), ("toilet", "washbasin")),
}  # fmt: skip
SCENARIOS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")
MANIPULATION_REQUESTS = ("Pick up the {t}", "Open the {t}", "Bring me the kettle")


# -- object parts ----------------------------------------------------------------------------
WHITE = (0.92, 0.92, 0.9, 1.0)
CREAM = (0.88, 0.84, 0.74, 1.0)
SILVER = (0.8, 0.82, 0.85, 1.0)
STEEL = (0.6, 0.62, 0.66, 1.0)
DARK = (0.12, 0.12, 0.14, 1.0)
WOOD = (0.55, 0.38, 0.22, 1.0)
LIGHT_WOOD = (0.72, 0.56, 0.36, 1.0)
GLASS = (0.6, 0.8, 0.9, 0.35)
FABRIC = (0.35, 0.4, 0.55, 1.0)
CARD = (0.65, 0.5, 0.33, 1.0)


def _b(name, x, y, z, sx, sy, sz, rgba, euler=(0.0, 0.0, 0.0)) -> Part:
    """Box centred at (x, y, z) with half extents (sx, sy, sz)."""
    return Part(name, "box", (x, y, z), (sx, sy, sz), rgba, euler)


def _c(name, x, y, z, r, hh, rgba, euler=(0.0, 0.0, 0.0)) -> Part:
    return Part(name, "cylinder", (x, y, z), (r, hh), rgba, euler)


def _legs(prefix, hx, hy, h, r, rgba) -> list[Part]:
    return [
        _b(f"{prefix}{i}", sx * (hx - r), sy * (hy - r), h / 2, r, r, h / 2, rgba)
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    ]


def _shelving(hx, hy, hz, rgba, n, items) -> list[Part]:
    """Open shelving unit: back panel, two sides, n boards, and boxes on some boards."""
    p = [_b("back", -hx + 0.01, 0, hz, 0.01, hy, hz, rgba)]
    p += [_b(f"side{i}", 0, s * (hy - 0.01), hz, hx, 0.01, hz, rgba) for i, s in enumerate((1, -1))]
    for k in range(n):
        z = 0.05 + (2 * hz - 0.1) * k / (n - 1)
        p.append(_b(f"board{k}", 0, 0, z, hx, hy - 0.02, 0.012, rgba))
        if k < n - 1 and k in items:
            p.append(
                _b(f"box{k}", 0.02, hy * 0.3 * (k % 3 - 1), z + 0.1, hx * 0.6, hy * 0.2, 0.1, CARD)
            )
    return p


def _parts(cls: str, hx: float, hy: float, hz: float) -> list[Part]:  # noqa: C901
    H = 2 * hz
    if cls == "refrigerator":
        return [
            _b("body", 0, 0, hz, hx, hy, hz, SILVER),
            _b("seam", hx + 0.002, 0, 0.68 * H, 0.003, hy * 0.98, 0.006, DARK),
            _b("handle_top", hx + 0.03, hy * 0.7, 0.8 * H, 0.012, 0.012, 0.14, STEEL),
            _b("handle_bottom", hx + 0.03, hy * 0.7, 0.5 * H, 0.012, 0.012, 0.25, STEEL),
        ]
    if cls == "wardrobe":
        return [
            _b("body", 0, 0, hz, hx, hy, hz, WOOD),
            _b("seam", hx + 0.002, 0, hz, 0.003, 0.004, hz * 0.97, DARK),
            _b("knob_l", hx + 0.02, 0.06, hz, 0.02, 0.02, 0.02, STEEL),
            _b("knob_r", hx + 0.02, -0.06, hz, 0.02, 0.02, 0.02, STEEL),
            _b("plinth", hx + 0.01, 0, 0.03, 0.01, hy * 0.98, 0.03, DARK),
        ]
    if cls == "chest_freezer":
        return [
            _b("body", 0, 0, hz - 0.03, hx, hy, hz - 0.03, WHITE),
            _b("lid", 0, 0, H - 0.03, hx + 0.01, hy + 0.01, 0.03, SILVER),
            _b("handle", hx + 0.03, 0, H - 0.05, 0.012, 0.12, 0.012, STEEL),
        ]
    if cls == "storage_shelf":
        return _shelving(hx, hy, hz, STEEL, 5, (0, 1, 3))
    if cls == "bookshelf":
        books = [
            _b(f"books{k}", 0.02, 0.0, 0.05 + (H - 0.1) * k / 4 + 0.13, hx * 0.7, hy * 0.7, 0.12,
               (0.5 + 0.1 * k, 0.25, 0.3, 1.0))
            for k in (0, 1, 2)
        ]  # fmt: skip
        return _shelving(hx, hy, hz, WOOD, 5, ()) + books
    if cls == "countertop":
        return [
            _b("base", 0, 0, hz - 0.02, hx - 0.01, hy, hz - 0.02, CREAM),
            _b("top", 0.01, 0, H - 0.02, hx, hy, 0.02, (0.25, 0.25, 0.27, 1.0)),
            _b("seam", hx - 0.005 + 0.006, 0, hz - 0.1, 0.002, 0.004, hz * 0.8, DARK),
            _b("knob", hx + 0.02, 0.1, H - 0.15, 0.015, 0.015, 0.015, STEEL),
        ]
    if cls == "sink":
        bx, by = hx * 0.6, hy * 0.45
        return [
            _b("base", 0, 0, hz - 0.02, hx - 0.01, hy, hz - 0.02, CREAM),
            _b(
                "top_back",
                -hx + (hx - bx) / 2 + 0.005,
                0,
                H - 0.02,
                (hx - bx) / 2 + 0.005,
                hy,
                0.02,
                STEEL,
            ),
            _b("top_front", hx - (hx - bx) / 2, 0, H - 0.02, (hx - bx) / 2, hy, 0.02, STEEL),
            _b("top_left", 0, hy - (hy - by) / 2, H - 0.02, bx, (hy - by) / 2, 0.02, STEEL),
            _b("top_right", 0, -hy + (hy - by) / 2, H - 0.02, bx, (hy - by) / 2, 0.02, STEEL),
            _b("basin", 0, 0, H - 0.17, bx, by, 0.01, DARK),  # recessed 15 cm below the top
            _c("faucet_post", -hx + 0.06, 0, H + 0.1, 0.015, 0.1, STEEL),
            _b("faucet_spout", -hx + 0.12, 0, H + 0.19, 0.06, 0.012, 0.012, STEEL),
        ]
    if cls == "stove":
        burners = [
            _c(f"burner{i}", sx * 0.13, sy * 0.13, H + 0.012, 0.075, 0.006, DARK)
            for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
        ]
        return [
            _b("body", 0, 0, hz - 0.01, hx, hy, hz - 0.01, WHITE),
            _b("top", 0, 0, H - 0.01, hx, hy, 0.01, DARK),
            _b("oven_window", hx + 0.003, 0, hz * 0.8, 0.003, hy * 0.6, hz * 0.35, DARK),
            _b("oven_handle", hx + 0.04, 0, hz * 1.3, 0.012, hy * 0.7, 0.012, STEEL),
            *burners,
        ]
    if cls == "kitchen_cabinet":
        return [
            _b("body", 0, 0, hz, hx, hy, hz, CREAM),
            _b("seam", hx + 0.002, 0, hz, 0.003, 0.004, hz * 0.95, DARK),
            _b("knob_l", hx + 0.02, 0.05, min(H * 0.7, H - 0.12), 0.015, 0.015, 0.015, STEEL),
            _b("knob_r", hx + 0.02, -0.05, min(H * 0.7, H - 0.12), 0.015, 0.015, 0.015, STEEL),
        ]
    if cls == "microwave":
        return [
            _b("body", 0, 0, hz, hx, hy, hz, SILVER),
            _b("window", hx + 0.003, -hy * 0.2, hz, 0.003, hy * 0.55, hz * 0.65, DARK),
            _b("panel", hx + 0.003, hy * 0.75, hz, 0.003, hy * 0.15, hz * 0.7, STEEL),
        ]
    if cls in ("dining_table", "coffee_table", "desk", "workbench"):
        top = 0.04 if cls != "workbench" else 0.05
        color = {"dining_table": LIGHT_WOOD, "coffee_table": WOOD, "desk": LIGHT_WOOD}.get(
            cls, STEEL
        )
        parts = [_b("top", 0, 0, H - top / 2, hx, hy, top / 2, color)]
        parts += _legs("leg", hx, hy, H - top, 0.03, STEEL if cls == "workbench" else color)
        if cls == "desk":
            parts.append(_b("drawers", 0, hy * 0.7, H * 0.65, hx * 0.9, hy * 0.28, H * 0.2, color))
        if cls == "workbench":
            parts.append(_b("lower_shelf", 0, 0, 0.25, hx - 0.03, hy - 0.03, 0.015, STEEL))
            parts.append(_b("vice", hx * 0.5, hy * 0.75, H + 0.07, 0.07, 0.05, 0.07, DARK))
        return parts
    if cls in ("chair", "office_chair"):
        seat_h = 0.45 if cls == "chair" else 0.5
        color = WOOD if cls == "chair" else DARK
        parts = [_b("seat", 0, 0, seat_h, hx, hy, 0.03, color)]
        parts.append(_b("back", -hx + 0.03, 0, (seat_h + H) / 2, 0.03, hy, (H - seat_h) / 2, color))
        if cls == "chair":
            return parts + _legs("leg", hx, hy, seat_h, 0.025, color)
        return parts + [
            _c("pole", 0, 0, seat_h / 2, 0.025, seat_h / 2, STEEL),
            _c("base", 0, 0, 0.03, hx * 0.95, 0.03, DARK),
        ]
    if cls == "bed":
        return [
            _b("frame", 0, 0, 0.15, hx, hy, 0.15, WOOD),
            _b("mattress", 0.0, 0, 0.35, hx - 0.03, hy - 0.03, 0.1, WHITE),
            _b("blanket", 0.35, 0, 0.46, hx * 0.55, hy - 0.02, 0.02, FABRIC),
            _b("headboard", -hx + 0.03, 0, 0.55, 0.03, hy, 0.55, WOOD),
            _b("pillow_l", -hx + 0.3, hy * 0.5, 0.5, 0.2, 0.25, 0.06, WHITE),
            _b("pillow_r", -hx + 0.3, -hy * 0.5, 0.5, 0.2, 0.25, 0.06, WHITE),
        ]
    if cls == "nightstand":
        return [
            _b("body", 0, 0, hz, hx, hy, hz, LIGHT_WOOD),
            _b("drawer", hx + 0.002, 0, hz * 1.3, 0.003, hy * 0.85, hz * 0.3, WOOD),
            _b("knob", hx + 0.02, 0, hz * 1.3, 0.015, 0.015, 0.015, STEEL),
        ]
    if cls in ("sofa", "armchair"):
        arm = 0.12
        return [
            _b("base", 0, 0, 0.15, hx, hy, 0.15, FABRIC),
            _b("seat", 0.08, 0, 0.35, hx - 0.1, hy - arm, 0.08, (0.4, 0.45, 0.62, 1.0)),
            _b("back", -hx + 0.12, 0, hz * 1.1, 0.12, hy, hz * 0.55, FABRIC),
            _b("arm_l", 0, hy - arm / 2, 0.35, hx, arm / 2, 0.2, FABRIC),
            _b("arm_r", 0, -hy + arm / 2, 0.35, hx, arm / 2, 0.2, FABRIC),
        ]
    if cls == "tv_stand":
        return [
            _b("cabinet", 0, 0, 0.25, hx, hy, 0.25, DARK),
            _b("screen", -0.02, 0, 0.5 + 0.3, 0.025, hy * 0.75, 0.27, (0.03, 0.03, 0.05, 1.0)),
            _b("stand", -0.02, 0, 0.52, 0.1, 0.12, 0.02, DARK),
        ]
    if cls == "toilet":
        w = hy
        return [
            _b("tank", -hx + 0.1, 0, 0.55, 0.1, w * 0.9, 0.2, WHITE),
            _b("lid", -hx + 0.1, 0, 0.77, 0.11, w * 0.92, 0.015, WHITE),
            _c("bowl", 0.0, 0, 0.2, w * 0.95, 0.2, WHITE),
            _b("bowl_front", 0.12, 0, 0.2, hx * 0.4, w * 0.8, 0.2, WHITE),
            _b("seat", 0.1, 0, 0.41, hx * 0.55, w * 0.95, 0.015, (0.85, 0.85, 0.8, 1.0)),
            _b("button", -hx + 0.1, 0, 0.78, 0.03, 0.03, 0.01, STEEL),
        ]
    if cls == "washbasin":
        return [
            _c("pedestal", -hx + 0.15, 0, 0.35, 0.08, 0.35, WHITE),
            _b("basin", -hx + 0.25, 0, 0.78, hx * 0.5, hy, 0.06, WHITE),
            _b("backsplash", -hx + 0.01, 0, 0.88, 0.01, hy, 0.1, WHITE),
            _c("faucet_post", -hx + 0.06, 0, 0.9, 0.012, 0.06, STEEL),
            _b("faucet_spout", -hx + 0.1, 0, 0.96, 0.05, 0.01, 0.01, STEEL),
        ]
    if cls == "shower":
        return [
            _b("tray", 0, 0, 0.04, hx, hy, 0.04, WHITE),
            _b("wall_back", -hx + 0.01, 0, hz, 0.01, hy, hz, GLASS),
            _b("wall_left", 0, hy - 0.01, hz, hx, 0.01, hz, GLASS),
            _b("wall_right", 0, -hy + 0.01, hz, hx, 0.01, hz, GLASS),
            _c("head", -hx + 0.15, 0, H - 0.3, 0.1, 0.01, STEEL, (0.0, 0.0, 0.0)),
            _b("pipe", -hx + 0.04, 0, hz, 0.012, 0.012, hz * 0.9, STEEL),
        ]
    if cls == "bathtub":
        return [
            _b("floor", 0, 0, 0.08, hx, hy, 0.08, WHITE),
            _b("wall_back", -hx + 0.04, 0, hz, 0.04, hy, hz, WHITE),
            _b("wall_front", hx - 0.04, 0, hz, 0.04, hy, hz, WHITE),
            _b("wall_left", 0, hy - 0.04, hz, hx, 0.04, hz, WHITE),
            _b("wall_right", 0, -hy + 0.04, hz, hx, 0.04, hz, WHITE),
            _c("faucet_post", -hx + 0.06, hy - 0.15, H + 0.06, 0.015, 0.06, STEEL),
            _b("faucet_spout", -hx + 0.12, hy - 0.15, H + 0.12, 0.06, 0.012, 0.012, STEEL),
        ]
    if cls == "tool_rack":
        tools = [
            _b(f"tool{i}", hx + 0.03, -hy * 0.7 + i * hy * 0.35, hz * (1.2 + 0.15 * (i % 2)),
               0.03, 0.02, 0.12, (0.7, 0.2 + 0.1 * i, 0.15, 1.0))
            for i in range(5)
        ]  # fmt: skip
        return [_b("board", 0, 0, hz, hx, hy, hz * 0.9, LIGHT_WOOD), *tools]
    if cls in ("washing_machine", "dryer"):
        color = WHITE if cls == "washing_machine" else (0.75, 0.78, 0.82, 1.0)
        return [
            _b("body", 0, 0, hz, hx, hy, hz, color),
            _c("door", hx + 0.01, 0, hz * 0.85, hy * 0.62, 0.015, DARK, (0.0, 90.0, 0.0)),
            _c("window", hx + 0.03, 0, hz * 0.85, hy * 0.45, 0.01, GLASS, (0.0, 90.0, 0.0)),
            _b("panel", hx + 0.003, 0, H - 0.07, 0.003, hy * 0.9, 0.05, STEEL),
        ]
    if cls == "filing_cabinet":
        drawers = [
            _b(f"drawer{k}", hx + 0.003, 0, H * (0.15 + 0.23 * k), 0.003, hy * 0.92, H * 0.1, STEEL)
            for k in range(4)
        ]  # fmt: skip
        return [_b("body", 0, 0, hz, hx, hy, hz, (0.55, 0.57, 0.6, 1.0)), *drawers]
    raise KeyError(cls)


def make_object(
    rng: np.random.Generator,
    cls: str,
    instance_id: str,
    shifted: bool = False,
    tall: bool = False,
) -> tuple[tuple[float, float, float], tuple[Part, ...]]:
    """Sample an object's dimensions and build its parts. ``tall`` makes a kitchen_cabinet a
    2.0 m pantry unit (used as an occluder)."""
    lo, hi = SCALE_SHIFTED if shifted else SCALE_NOMINAL
    d = np.array(DIMS[cls]) * rng.uniform(lo, hi, 3)
    if cls == "kitchen_cabinet" and tall:
        d[2] = rng.uniform(1.9, 2.1)
        d[1] = min(d[1], 0.8)
    if cls == "storage_shelf":
        d[2] = min(d[2], 2.1)
    half = (float(d[0] / 2), float(d[1] / 2), float(d[2] / 2))
    return half, tuple(_parts(cls, *half))


# -- scene construction ----------------------------------------------------------------------
def _quat(yaw: float) -> list[float]:
    return [cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)]


_SHAPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
}


def add_home(spec: mujoco.MjSpec, home: HomeSpec) -> None:
    """Add the walls and all objects to an ``MjSpec`` world (scene construction). The floor is
    the caller's (the G1 model already has one)."""
    world = spec.worldbody
    for w in home.walls:
        world.add_geom(
            name=f"wall:{w.name}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[w.centre[0], w.centre[1], WALL_HEIGHT / 2],
            size=[w.half[0], w.half[1], WALL_HEIGHT / 2], rgba=[0.82, 0.82, 0.8, 1.0],
        )  # fmt: skip
    for o in home.objects:
        body = world.add_body(name=f"obj:{o.instance_id}", pos=[*o.xy, o.z0], quat=_quat(o.yaw))
        for p in o.parts:
            size = list(p.size) + [0.0] * (3 - len(p.size))
            body.add_geom(
                name=f"sem:{o.instance_id}:{p.name}", type=_SHAPES[p.shape], pos=list(p.pos),
                size=size, rgba=list(p.rgba), euler=list(p.euler),
            )  # fmt: skip


def semantic_index(model: mujoco.MjModel, home: HomeSpec) -> dict[int, tuple[str, str]]:
    """Geom id -> (instance_id, class) for every object geom. Sim side only (sensor, scorer)."""
    classes = {o.instance_id: o.cls for o in home.objects}
    out = {}
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith("sem:"):
            iid = name.split(":")[1]
            out[g] = (iid, classes[iid])
    return out


# -- navigability and visibility on the ground-truth geometry ---------------------------------
def _cell_centres(home: HomeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x0, y0, centres (ny, nx, 2)) of a raster covering the outer walls."""
    x0, y0 = -home.width / 2 - 0.2, -home.depth / 2 - 0.2
    nx, ny = int((home.width + 0.4) / CELL) + 1, int((home.depth + 0.4) / CELL) + 1
    xs = x0 + (np.arange(nx) + 0.5) * CELL
    ys = y0 + (np.arange(ny) + 0.5) * CELL
    return np.array(x0), np.array(y0), np.stack(np.meshgrid(xs, ys), -1)


def _rects(home: HomeSpec) -> list[Rect]:
    """Every obstacle footprint a robot can bump: walls and floor-standing objects."""
    rects = [Rect(w.centre[0], w.centre[1], w.half[0], w.half[1]) for w in home.walls]
    return rects + [footprint(o) for o in home.objects if o.z0 == 0.0]


def _blockers(home: HomeSpec) -> list[tuple[str | None, Rect]]:
    """Obstacles that block the view of a camera 1.2 m up: walls and objects >= 0.9 m tall."""
    out: list[tuple[str | None, Rect]] = [
        (None, Rect(w.centre[0], w.centre[1], w.half[0], w.half[1])) for w in home.walls
    ]
    return out + [
        (o.instance_id, footprint(o)) for o in home.objects if o.z0 == 0 and o.half[2] >= 0.45
    ]


def free_mask(home: HomeSpec, radius: float = ROBOT_RADIUS) -> np.ndarray:
    """(ny, nx) bool: robot centres at least ``radius`` from every wall and object footprint,
    inside the outer walls. Ground-truth raster at CELL resolution."""
    x0, y0, centres = _cell_centres(home)
    free = np.ones(centres.shape[:2], bool)
    for r in _rects(home):
        c = r.corners()
        j0, i0 = np.floor((c.min(0) - radius - [x0, y0]) / CELL).astype(int)
        j1, i1 = np.ceil((c.max(0) + radius - [x0, y0]) / CELL).astype(int) + 1
        i0, j0 = max(i0, 0), max(j0, 0)
        window = centres[i0:i1, j0:j1]
        free[i0:i1, j0:j1] &= ~(r.distance(window) <= radius)
    return free


def flood(free: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """4-connected reachable cells from ``seed`` (row, col) through ``free``."""
    reach = np.zeros_like(free)
    if not free[seed]:
        return reach
    reach[seed] = True
    while True:
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= free
        if (grown == reach).all():
            return reach
        reach = grown


def cell_of(home: HomeSpec, xy) -> tuple[int, int]:
    x0, y0, _ = _cell_centres(home)
    return int((xy[1] - y0) / CELL), int((xy[0] - x0) / CELL)


def line_of_sight(
    cells: np.ndarray, obj: ObjectSpec, blockers: list[tuple[str | None, Rect]]
) -> np.ndarray:
    """Per cell (n, 2) xy: is ``obj`` visible in plan view, i.e. does the straight line to its
    centre or front-face centre cross no wall or tall object first?"""
    reach = obj.half[0] + 0.03
    front = np.array([obj.xy[0] + reach * cos(obj.yaw), obj.xy[1] + reach * sin(obj.yaw)])
    ok = np.zeros(len(cells), bool)
    t = np.linspace(0, 1, 40)[:-1, None, None]  # stop before the target point itself
    for target in (np.array(obj.xy), front):
        pts = cells[None] + (target - cells)[None] * t
        blocked = np.zeros(len(cells), bool)
        lo, hi = pts.reshape(-1, 2).min(0), pts.reshape(-1, 2).max(0)
        for iid, b in blockers:
            r = b.hx + b.hy  # skip blockers clear of the swept region
            if (
                iid != obj.instance_id
                and lo[0] - r <= b.cx <= hi[0] + r
                and lo[1] - r <= b.cy <= hi[1] + r
            ):
                blocked |= b.inside(pts).any(0)
        ok |= ~blocked
    return ok


def problems(home: HomeSpec) -> list[str]:
    """Why a home breaks the reachability contract (empty: it satisfies it). Rasterises the
    ground-truth obstacles at CELL, inflates by ROBOT_RADIUS, floods from the start, and checks
    that every room is reached and every object has a reachable standing cell within
    STAND_RANGE with line of sight."""
    out = []
    free = free_mask(home)
    _, _, centres = _cell_centres(home)
    sx, sy, _ = home.start
    seed = cell_of(home, (sx, sy))
    if not free[seed]:
        return ["start is not in free space"]
    reach = flood(free, seed)
    for r in home.rooms:
        lo, hi = room_interior(home, r)
        inside = (centres[..., 0] > lo[0] + ROBOT_RADIUS) & (centres[..., 0] < hi[0] - ROBOT_RADIUS)
        inside &= (centres[..., 1] > lo[1] + ROBOT_RADIUS) & (
            centres[..., 1] < hi[1] - ROBOT_RADIUS
        )
        if not (reach & inside).any():
            out.append(f"room {r.id} ({r.type}) unreachable")
    for d in home.doorways:
        i, j = cell_of(home, d.centre)
        if not reach[i, j]:
            out.append(f"doorway {d.a}-{d.b} not passable")
    blockers = _blockers(home)
    pts = centres[reach]
    for o in home.objects:
        if o.z0 > 0:
            continue
        fp = footprint(o)
        near = pts[fp.distance(pts) <= STAND_RANGE]
        if len(near) == 0:
            out.append(f"{o.instance_id}: no reachable cell within {STAND_RANGE} m")
            continue
        order = np.argsort(fp.distance(near))[:: max(1, len(near) // 300)]
        if not line_of_sight(near[order], o, blockers).any():
            out.append(f"{o.instance_id}: no standing cell with line of sight")
    return out


# -- ray-cast visibility (the head camera's view without rendering) ----------------------------
class GroundTruthView:
    """Pixel counts of instances from a pose, by casting one ray per pixel of the head camera
    through the compiled scene. Equivalent to counting segmentation pixels with finite depth;
    used by the generator to verify scenario invariants and by tests. Ground truth."""

    def __init__(self, home: HomeSpec) -> None:
        spec = mujoco.MjSpec()
        add_home(spec, home)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.index = semantic_index(self.model, home)

    def geom_ids(self, instance_ids) -> set[int]:
        ids = set(instance_ids)
        return {g for g, (iid, _) in self.index.items() if iid in ids}

    def pixels(self, instance_ids, x: float, y: float, yaw: float, stride: int = 2) -> int:
        """Pixels (at the camera's resolution) with finite depth that show these instances."""
        from dream.sim.camera import MAX_RANGE
        from dream.sim.kinematic_base import HEAD_CAMERA, head_camera_pose

        spec = HEAD_CAMERA
        pos, rot = head_camera_pose(x, y, yaw)
        f = 0.5 * spec.height / np.tan(np.deg2rad(spec.fovy_deg) / 2)
        us = (np.arange(0, spec.width, stride) + 0.5 - spec.width / 2) / f
        vs = -(np.arange(0, spec.height, stride) + 0.5 - spec.height / 2) / f
        uu, vv = np.meshgrid(us, vs)
        cam = np.stack([uu, vv, -np.ones_like(uu)], -1).reshape(-1, 3)
        norm = np.linalg.norm(cam, axis=1)
        vec = (cam / norm[:, None]) @ rot.T
        n = len(vec)
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            pos,
            vec.ravel(),
            None,
            1,
            -1,
            geomid,
            dist,
            None,
            n,
            MAX_RANGE * 2,
        )
        depth = dist / norm
        hit = (
            np.isin(geomid, list(self.geom_ids(instance_ids))) & (depth <= MAX_RANGE) & (dist >= 0)
        )
        return int(hit.sum()) * stride * stride


# -- scenario invariants (also the generator's final gate) -------------------------------------
def _room_of(home: HomeSpec, obj: ObjectSpec) -> RoomBox:
    return home.rooms[obj.room_id]


def door_poses(home: HomeSpec, room: RoomBox) -> list[tuple[float, float, float]]:
    """Poses 0.5 m inside ``room`` from each of its doorways, facing into the room."""
    out = []
    for d in home.doorways:
        if room.id not in (d.a, d.b):
            continue
        other = home.rooms[d.b if room.id == d.a else d.a]
        rc = (np.array(room.lo) + room.hi) / 2
        oc = (np.array(other.lo) + other.hi) / 2
        k = 0 if d.axis == "x" else 1
        inward = np.zeros(2)
        inward[k] = np.sign(rc[k] - oc[k])
        p = np.array(d.centre) + 0.5 * inward
        out.append((float(p[0]), float(p[1]), float(np.arctan2(inward[1], inward[0]))))
    return out


def viewpoints(
    home: HomeSpec, obj: ObjectSpec, n: int, seed: int
) -> list[tuple[float, float, float]]:
    """``n`` standing poses in the object's room, 1.0 to 2.8 m from it, facing it, in free space
    (deterministic given the seed)."""
    rng = np.random.default_rng(seed)
    room = _room_of(home, obj)
    lo, hi = room_interior(home, room)
    rects = _rects(home)
    out: list[tuple[float, float, float]] = []
    for _ in range(40 * n):
        p = rng.uniform(lo + ROBOT_RADIUS, hi - ROBOT_RADIUS)
        d = np.array(obj.xy) - p
        if not 1.0 <= np.linalg.norm(d) <= 2.8 or any(r.distance(p) <= ROBOT_RADIUS for r in rects):
            continue
        out.append(
            (float(p[0]), float(p[1]), float(np.arctan2(d[1], d[0]) + rng.uniform(-0.1, 0.1)))
        )
        if len(out) == n:
            break
    return out


def g_visibility(home: HomeSpec, view: GroundTruthView, obj: ObjectSpec) -> tuple[int, int, int]:
    """(poses tried, poses seeing >= 150 px, poses seeing none) among in-room viewpoints."""
    poses = viewpoints(home, obj, 30, home.seed)
    px = [view.pixels([obj.instance_id], *p, stride=2) for p in poses]
    return len(poses), sum(v >= 150 for v in px), sum(v == 0 for v in px)


def verify_scenario(home: HomeSpec) -> list[str]:
    """Violations of the home's scenario recipe (empty: it holds). Independent of how the home
    was generated; uses ground truth only."""
    sc, tc = home.scenario, home.target_class
    out = problems(home)
    ex = EXTENT_LIMIT
    sx, sy, _ = home.start
    if (home.width / 2 + WALL_THICKNESS + abs(sx) > ex) or (
        home.depth / 2 + WALL_THICKNESS + abs(sy) > ex
    ):
        out.append("home exceeds +-11 m of the start")
    if tc not in TARGETS:
        return [*out, f"unknown target {tc}"]
    info = TARGETS[tc]
    targets = instances_of(home, tc)
    types = [r.type for r in home.rooms]
    typical = [r for r in home.rooms if r.type == info.typical]
    view = GroundTruthView(home)
    here = room_at(home, (sx, sy))
    ids = [o.instance_id for o in targets]
    px0 = view.pixels(ids, sx, sy, home.start[2], stride=1) if ids else 0
    letter = sc[0]
    if letter in "ABCGL" or sc == "I_valid":
        if len(targets) != 1 or not typical or _room_of(home, targets[0]).type != info.typical:
            out.append("target must be one instance in its typical room")
    if letter == "A":
        if px0 < 400:
            out.append(f"target shows {px0} px at the start (< 400)")
        if here is None or targets and here.id != targets[0].room_id:
            out.append("start must be in the target's room")
    if letter in "BL" and (px0 != 0 or (here and targets and here.id == targets[0].room_id)):
        out.append("target must be out of the start room and unseen at the start")
    if letter == "C" and targets:
        room = _room_of(home, targets[0])
        if here is not None and here.id == room.id:
            out.append("start must be outside the target room")
        poses = door_poses(home, room)
        if not poses:
            out.append("target room has no doorway")
        for p in poses:
            if view.pixels(ids, *p, stride=1) != 0:
                out.append("target visible from a doorway")
    if letter == "G" and targets:
        n, seen, hidden = g_visibility(home, view, targets[0])
        if n < 10 or seen < 3 or hidden < 3:
            out.append(f"G visibility not mixed: {seen} seen, {hidden} hidden of {n}")
    if letter == "D":
        if len(targets) != 1 or types[targets[0].room_id] not in info.atypical or not typical:
            out.append("target must be in an atypical room with a typical room present")
    if letter == "E" and (targets or not typical):
        out.append("a typical room must exist and the target must not")
    if letter == "F":
        extra = {c for c in info.confusers if instances_of(home, c)}
        if len(targets) != 1 or len(extra) < len(info.confusers):
            out.append("F needs the target and a room of all confusers")
    if letter == "H" and (targets or typical):
        out.append("no typical room and no target allowed")
    if letter == "J":
        pool = [r for r in home.rooms if r.type in info.pool]
        if (
            len(pool) < 2
            or typical
            or len(targets) != 1
            or types[targets[0].room_id] not in info.pool
        ):
            out.append("J needs 2+ pool rooms, no typical room, target in a pool room")
    if letter == "I":
        mem = home.memory_prior
        if len(mem) != 1 or mem[0][0] != tc or not 86400 <= mem[0][2] <= 7 * 86400:
            out.append("I needs one memory entry, 1 to 7 days old")
        elif (
            sc == "I_valid"
            and targets
            and np.hypot(*(np.array(targets[0].xy) - mem[0][1][:2])) > 0.05
        ):
            out.append("valid memory must match the target position")
        elif sc == "I_moved" and (
            len(targets) != 1 or room_at(home, mem[0][1][:2]).id == targets[0].room_id
        ):
            out.append("moved: the target must be in another room than the memory")
        elif sc == "I_removed" and targets:
            out.append("removed: no target may exist")
    if letter == "K" and len(targets) > 1:
        out.append("K allows at most one target")
    if letter == "L" and not home.request.startswith(("Pick up", "Open", "Bring")):
        out.append("L request must be a manipulation request")
    return out


# -- generator -------------------------------------------------------------------------------
class _Fail(Exception):
    """This attempt cannot satisfy the recipe; resample."""


FREE_STYLE = {"dining_table", "coffee_table"}
DEFAULT_FRONT = 0.5


def replace_walls(home: HomeSpec, without: WallSeg) -> HomeSpec:
    """The home with one wall segment removed (to test what a partition hides)."""
    return replace(home, walls=tuple(w for w in home.walls if w.name != without.name))


def _split(rng: np.random.Generator, length: float, n: int, alpha: float) -> np.ndarray:
    """Edges of n segments of at least MIN_ROOM covering ``length``, centred on 0."""
    extra = length - n * MIN_ROOM
    sizes = MIN_ROOM + extra * rng.dirichlet([alpha] * n)
    return -length / 2 + np.concatenate([[0.0], np.cumsum(sizes)])


class _Gen:
    def __init__(self, rng, scenario: str, target: str, generator: str, seed: int) -> None:
        if target not in TARGETS:
            raise ValueError(f"unknown target {target}")
        if generator not in GENERATORS:
            raise ValueError(f"unknown generator {generator}")
        if scenario[:1] not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        self.rng, self.target, self.generator, self.seed = rng, target, generator, seed
        self.shifted = generator == "shifted"
        self.xl = generator == "xl"  # nominal furnishing and scales in a larger home
        self.info = TARGETS[target]
        self.furn = FURNISHING_SHIFTED if self.shifted else FURNISHING_NOMINAL
        if scenario == "I":
            scenario = str(rng.choice(["I_moved", "I_removed", "I_valid"]))
        if scenario == "K":
            scenario = str(rng.choice(["K_studio", "K_mismatch"]))
        self.scenario, self.letter = scenario, scenario[0]

    # -- layout ---------------------------------------------------------------------
    def layout(self) -> None:
        rng = self.rng
        if self.shifted:
            cols, rows = [(3, 2), (2, 3), (3, 3)][int(rng.integers(3))]
            w, d, alpha = rng.uniform(9.0, 12.0), rng.uniform(7.5, 10.0), 3.0
        elif self.xl:
            cols, rows = XL_GRID
            w, d, alpha = rng.uniform(*XL_WIDTH), rng.uniform(*XL_DEPTH), 6.0
        else:
            cols, rows = [(2, 2), (2, 2), (3, 2), (2, 3)][int(rng.integers(4))]
            w, d, alpha = rng.uniform(7.0, 10.0), rng.uniform(6.0, 9.0), 6.0
        w, d = max(w, cols * MIN_ROOM + 0.2), max(d, rows * MIN_ROOM + 0.2)
        self.w, self.d = float(w), float(d)
        xe, ye = _split(rng, w, cols, alpha), _split(rng, d, rows, alpha)
        self.rooms = [
            [r * cols + c, (float(xe[c]), float(ye[r])), (float(xe[c + 1]), float(ye[r + 1]))]
            for r in range(rows)
            for c in range(cols)
        ]
        horiz = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
        vert = [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
        edges = horiz + vert
        parent = list(range(len(self.rooms)))

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        chosen = []
        for k in rng.permutation(len(edges)):
            a, b = edges[k]
            if find(a) != find(b):
                parent[find(a)] = find(b)
                chosen.append(edges[k])
            elif rng.random() < 0.3:
                chosen.append(edges[k])
        self.doorways: list[Doorway] = []
        self.walls: list[WallSeg] = []
        t, hw, hd = WALL_THICKNESS / 2, w / 2, d / 2
        for name, c, h in (
            ("px", (hw + t, 0), (t, hd + 2 * t)), ("nx", (-hw - t, 0), (t, hd + 2 * t)),
            ("py", (0, hd + t), (hw, t)), ("ny", (0, -hd - t), (hw, t)),
        ):  # fmt: skip
            self.walls.append(WallSeg(f"outer_{name}", c, h))
        for a, b in edges:  # every adjacency has a wall; chosen ones get a door
            ra, rb = self.rooms[a], self.rooms[b]
            vertical = (a, b) in horiz  # wall at constant x, running along y
            k = 1 if vertical else 0  # coordinate running along the wall
            pos = ra[2][1 - k]  # wall centreline
            s0, s1 = max(ra[1][k], rb[1][k]), min(ra[2][k], rb[2][k])
            cuts = [s0 - t]
            if (a, b) in chosen:
                width = float(rng.uniform(*DOOR_WIDTH))
                c = float(rng.uniform(s0 + 0.45 + width / 2, s1 - 0.45 - width / 2))
                centre = (pos, c) if vertical else (c, pos)
                self.doorways.append(Doorway(a, b, centre, width, "x" if vertical else "y"))
                cuts += [c - width / 2, c + width / 2]
            cuts.append(s1 + t)
            for j in range(0, len(cuts), 2):
                lo, hi = cuts[j], cuts[j + 1]
                mid, half = (lo + hi) / 2, (hi - lo) / 2
                self.walls.append(
                    WallSeg(
                        f"in_{a}_{b}_{j // 2}",
                        (pos, mid) if vertical else (mid, pos),
                        (t, half) if vertical else (half, t),
                    )
                )
        self.partitions = 0

    # -- room types -----------------------------------------------------------------
    def assign_types(self) -> None:
        rng, info, sc = self.rng, self.info, self.scenario
        n = len(self.rooms)
        T = info.typical
        required: list[str] = []
        forbidden = {"studio", T} if sc != "K_studio" else {"studio"}
        if sc == "K_studio":
            forbidden.add(T)  # a studio home has no ordinary typical room
            required.append("studio")
            forbidden.discard("studio")
        elif self.letter not in "HJ":
            forbidden.discard(T)
            required.append(T)
        if self.letter in "DF":
            required.append(str(rng.choice(info.atypical)))
        if self.letter == "J":
            required += [
                str(x) for x in rng.choice(info.pool, int(rng.integers(2, 4)), replace=False)
            ]
        if sc == "I_moved":
            required.append(str(rng.choice([t for t in ROOM_TYPES if t != T])))
        types = list(required)
        if self.xl:  # every room type the generator knows, with spare bedrooms and bathrooms
            need = Counter(XL_TYPES)
            for t in required:
                if need[t] > 0:
                    need[t] -= 1
            rest = [t for t in need.elements() if t not in forbidden]
            types += [str(t) for t in rng.permutation(rest)[: max(n - len(types), 0)]]
        allowed = [t for t in ROOM_TYPES if t not in forbidden and t != "studio"]
        while len(types) < n:
            pool = [t for t in allowed if self.shifted or self.xl or t not in types] or allowed
            w = np.array([TYPE_WEIGHTS[t] for t in pool])
            types.append(str(rng.choice(pool, p=w / w.sum())))
        if len(types) > n:
            raise _Fail
        for r, t in zip(self.rooms, rng.permutation(types), strict=True):
            r.append(str(t))

    # -- placement ------------------------------------------------------------------
    def home(self, start=(0.0, 0.0, 0.0), memory=(), request="") -> HomeSpec:
        rooms = tuple(RoomBox(i, t, lo, hi) for i, lo, hi, t in self.rooms)
        return HomeSpec(
            self.w, self.d, rooms, tuple(self.doorways), tuple(self.objects), start,
            self.scenario, self.target, request, tuple(memory), self.generator, self.seed,
            tuple(self.walls),
        )  # fmt: skip

    @staticmethod
    def _front(o: ObjectSpec) -> Rect:
        d = o.half[0] + o.front_clear / 2
        c, s = cos(o.yaw), sin(o.yaw)
        return Rect(o.xy[0] + d * c, o.xy[1] + d * s, o.front_clear / 2, o.half[1], o.yaw)

    def _fits(self, room: RoomBox, fp: Rect, front: Rect | None, cls: str) -> bool:
        lo, hi = self._interior[room.id]
        c = fp.corners()
        if (c < lo - 1e-6).any() or (c > hi + 1e-6).any():
            return False
        for k in self._keep:
            if _overlap(fp, k) or (front is not None and _overlap(front, k)):
                return False
        if any(_overlap(fp, r, 0.05) for r in self.reserved):
            return False
        for w in self.walls:
            if w.name.startswith("partition") and _overlap(fp, Rect(*w.centre, *w.half), 0.05):
                return False
        loose = cls in LOOSE
        for o in self.objects:
            if o.z0 > 0:
                continue
            ofp = footprint(o)
            if _overlap(fp, ofp, 0.05):
                return False
            if loose or o.cls in LOOSE:
                continue
            if o.front_clear > 0 and _overlap(fp, self._front(o)):
                return False
            if front is not None and _overlap(front, ofp):
                return False
        return True

    def _next_id(self, cls: str) -> str:
        self.counts[cls] = self.counts.get(cls, -1) + 1
        return f"{cls}_{self.counts[cls]}"

    def _poses(self, room: RoomBox, cls: str, half, style: str):
        """Candidate (xy, yaw) for one object; the caller tests each."""
        rng = self.rng
        lo, hi = self._interior[room.id]
        hx, hy = half[0], half[1]
        while True:
            if style == "free":
                p = rng.uniform(lo + 0.6 + hy * 0.5, hi - 0.6 - hy * 0.5)
                yield (float(p[0]), float(p[1])), float(rng.integers(4)) * pi / 2
                continue
            prev = [o for o in self.objects if o.room_id == room.id and o.cls in KITCHEN_RUN]
            if cls in KITCHEN_RUN and prev and rng.random() < 0.8:
                o = prev[int(rng.integers(len(prev)))]
                side = 1 if rng.random() < 0.5 else -1
                lx, ly = -o.half[0] + hx, side * (o.half[1] + hy + 0.01)
                c, s = cos(o.yaw), sin(o.yaw)
                yield (o.xy[0] + c * lx - s * ly, o.xy[1] + s * lx + c * ly), o.yaw
                continue
            wall, t = int(rng.integers(4)), float(rng.uniform(0, 1))
            if wall < 2:  # +x / -x walls
                x = (hi[0] - hx - 0.01) if wall == 0 else (lo[0] + hx + 0.01)
                y = lo[1] + hy + 0.02 + t * max(hi[1] - lo[1] - 2 * hy - 0.04, 0)
                yield (float(x), float(y)), (pi if wall == 0 else 0.0)
            else:
                y = (hi[1] - hx - 0.01) if wall == 2 else (lo[1] + hx + 0.01)
                x = lo[0] + hy + 0.02 + t * max(hi[0] - lo[0] - 2 * hy - 0.04, 0)
                yield (float(x), float(y)), (-pi / 2 if wall == 2 else pi / 2)

    def put(
        self, cls: str, room: RoomBox, style: str | None = None, tall: bool = False,
        pose=None, tries: int = 60, z0: float = 0.0,
    ) -> ObjectSpec | None:  # fmt: skip
        """Place one object in a room: at ``pose`` if given, else the first valid draw of
        ``tries`` (None if none fits)."""
        half, parts = make_object(self.rng, cls, "tmp", self.shifted, tall)
        style = style or ("free" if cls in FREE_STYLE else "wall")
        fc = 0.0 if cls in LOOSE or z0 > 0 else FRONT_CLEAR.get(cls, DEFAULT_FRONT)
        if pose is not None:
            cands = [pose]
        else:
            gen = self._poses(room, cls, half, style)
            cands = (next(gen) for _ in range(tries))
        for xy, yaw in cands:
            xy = (float(xy[0]), float(xy[1]))
            spec = ObjectSpec("tmp", cls, room.id, xy, float(yaw), half, parts, z0, fc)
            front = self._front(spec) if fc > 0 else None
            if z0 > 0 or self._fits(room, footprint(spec), front, cls):
                spec = ObjectSpec(
                    self._next_id(cls), cls, room.id, xy, float(yaw), half, parts, z0, fc
                )
                self.objects.append(spec)
                return spec
        return None

    def put_beside(self, cls: str, anchor: ObjectSpec, kind: str) -> ObjectSpec | None:
        """Companions placed relative to an anchor (nightstand, chair, office chair); the same
        validity tests apply as for everything else."""
        rng = self.rng
        room = self._rooms[anchor.room_id]
        half, _ = make_object(rng, cls, "tmp", self.shifted)
        c, s = cos(anchor.yaw), sin(anchor.yaw)
        ax, ay = anchor.half[0], anchor.half[1]
        for _ in range(12):
            side = 1 if rng.random() < 0.5 else -1
            if kind == "beside_bed":
                lx, ly, dyaw = -ax + half[0], side * (ay + half[1] + 0.02), 0.0
            elif kind == "desk_chair":
                lx, ly, dyaw = ax + half[0] + 0.02, float(rng.uniform(-0.3, 0.3)), pi
            elif rng.random() < 0.7:  # around a table: face it from a side ...
                lx, ly = float(rng.uniform(-ax, ax)), side * (ay + half[0] + 0.05)
                dyaw = -side * pi / 2
            else:  # ... or from an end
                lx, ly = side * (ax + half[0] + 0.05), float(rng.uniform(-ay, ay))
                dyaw = pi if side == 1 else 0.0
            xy = (anchor.xy[0] + c * lx - s * ly, anchor.xy[1] + s * lx + c * ly)
            got = self.put(cls, room, pose=(xy, anchor.yaw + dyaw))
            if got:
                return got
        return None

    # -- furnishing -----------------------------------------------------------------
    def furnish(self, room: RoomBox, exclude: set[str]) -> None:
        rng = self.rng
        placed: dict[str, list[ObjectSpec]] = {}
        for cls, p, nmin, nmax in self.furn[room.type]:
            if cls in exclude or rng.random() >= p:
                continue
            for _ in range(int(rng.integers(nmin, nmax + 1))):
                if cls == "microwave":
                    base = [
                        o for o in self.objects if o.room_id == room.id and o.cls == "countertop"
                    ]
                    if base:
                        self.put(
                            "microwave",
                            room,
                            pose=(base[0].xy, base[0].yaw),
                            z0=2 * base[0].half[2],
                        )
                    continue
                o = self.put(cls, room)
                if o:
                    placed.setdefault(cls, []).append(o)
        if room.type == "bathroom" and not (placed.get("shower") or placed.get("bathtub")):
            for cls in rng.permutation(["shower", "bathtub"]):
                if cls not in exclude and self.put(str(cls), room):
                    break
        for t in placed.get("dining_table", []):
            for _ in range(int(rng.integers(2, 5))):
                self.put_beside("chair", t, "around")
        for d in placed.get("desk", []):
            if rng.random() < 0.9:
                self.put_beside("office_chair", d, "desk_chair")
        for b in placed.get("bed", []):
            if "nightstand" not in exclude and not placed.get("nightstand") and rng.random() < 0.5:
                self.put_beside("nightstand", b, "beside_bed")
        if self.shifted:
            for _ in range(int(rng.integers(1, 4))):
                self.put(str(rng.choice([c for c in CLUTTER_POOL if c not in exclude])), room)

    # -- recipes --------------------------------------------------------------------
    def rooms_of(self, *types: str) -> list[RoomBox]:
        return [r for r in self._rooms if r.type in types]

    def pick(self, rooms: list[RoomBox]) -> RoomBox:
        if not rooms:
            raise _Fail
        return rooms[int(self.rng.integers(len(rooms)))]

    def place_target(self, room: RoomBox) -> ObjectSpec:
        o = self.put(self.target, room, tries=120)
        if o is None:
            raise _Fail
        return o

    def drop(self, thing) -> None:
        if isinstance(thing, WallSeg):
            self.walls.remove(thing)
        elif thing is not None:
            self.objects.remove(thing)

    def occluder(self, room: RoomBox):
        """A tall storage_shelf, a tall kitchen_cabinet or a partition wall stub in the room;
        returns the ObjectSpec or WallSeg added, or None."""
        rng = self.rng
        kind = int(rng.integers(3))
        if kind == 0:
            return self.put("storage_shelf", room, tries=30)
        if kind == 1:
            return self.put("kitchen_cabinet", room, tall=True, tries=30)
        lo, hi = self._interior[room.id]
        wall, length, t = (
            int(rng.integers(4)),
            float(rng.uniform(1.2, 2.2)),
            float(rng.uniform(0.15, 0.85)),
        )
        h, name = WALL_THICKNESS / 2, f"partition_{self.partitions}"
        if wall < 2:  # stub from a +x / -x wall, running along x
            x = hi[0] - length / 2 if wall == 0 else lo[0] + length / 2
            seg = WallSeg(name, (float(x), float(lo[1] + t * (hi[1] - lo[1]))), (length / 2, h))
        else:
            y = hi[1] - length / 2 if wall == 2 else lo[1] + length / 2
            seg = WallSeg(name, (float(lo[0] + t * (hi[0] - lo[0])), float(y)), (h, length / 2))
        rect = Rect(*seg.centre, *seg.half)
        if any(_overlap(rect, k) for k in self._keep):
            return None
        if any(_overlap(rect, footprint(o), 0.3) for o in self.objects if o.z0 == 0):
            return None
        self.partitions += 1
        self.walls.append(seg)
        return seg

    def _plan_view(self, home: HomeSpec, target: ObjectSpec, poses, without=None) -> np.ndarray:
        """Cheap plan-view test per pose: target centre inside the camera's horizontal cone and
        range and the line of sight clear. A filter only; rays decide."""
        xy = np.array([p[:2] for p in poses])
        v = np.array(target.xy) - xy
        ang = np.abs(
            (np.arctan2(v[:, 1], v[:, 0]) - np.array([p[2] for p in poses]) + pi) % (2 * pi) - pi
        )
        near = (np.hypot(v[:, 0], v[:, 1]) <= 2.8) & (ang <= np.radians(32))
        blockers = [b for b in _blockers(home) if without is None or b[0] != without]
        return near & line_of_sight(xy, target, blockers)

    def hidden_target(self, room: RoomBox) -> ObjectSpec:
        """Scenarios C and G: place the target and an occluder so the recipe's visibility
        condition holds on the geometry placed so far (re-verified on the finished home)."""
        for _ in range(400):
            target = self.place_target(room)
            occ = self.occluder(room)
            if occ is None:
                self.drop(target)
                continue
            ids = [target.instance_id]
            home = self.home()
            oid = occ.instance_id if isinstance(occ, ObjectSpec) else None
            if self.letter == "C":
                poses = door_poses(home, room)
                if poses and not self._plan_view(home, target, poses).any():
                    # the occluder must be what hides it: a doorway would see it without
                    bare = self.home()
                    bare = replace_walls(bare, occ) if isinstance(occ, WallSeg) else bare
                    if self._plan_view(bare, target, poses, without=oid).any():
                        view = GroundTruthView(home)
                        if all(view.pixels(ids, *p, stride=1) == 0 for p in poses):
                            return target
            else:
                poses = viewpoints(home, target, 30, home.seed)
                if len(poses) >= 10:
                    vis = self._plan_view(home, target, poses)
                    if vis.sum() >= 4 and (~vis).sum() >= 4:
                        n, seen, hid = g_visibility(home, GroundTruthView(home), target)
                        if n >= 10 and seen >= 3 and hid >= 3:
                            return target
            self.drop(occ)
            self.drop(target)
        raise _Fail("no occluder placement found")

    def build_objects(self) -> tuple[tuple, str]:
        """Run the recipe, then furnish; returns (memory_prior, request)."""
        rng, info, sc, tgt = self.rng, self.info, self.scenario, self.target
        typical = self.rooms_of(info.typical)
        exclude = {tgt}
        memory: tuple = ()
        name = tgt.replace("_", " ")
        request = f"Find the {name}"
        letter = self.letter
        if letter in "ABL":
            self.place_target(typical[0])
        elif letter in "CG":
            self.hidden_target(typical[0])
        elif letter == "D":
            self.place_target(self.pick(self.rooms_of(*info.atypical)))
        elif letter == "J":
            self.place_target(self.pick(self.rooms_of(*info.pool)))
        elif letter == "F":
            room = self.pick(self.rooms_of(*info.atypical))
            for c in info.confusers:
                if self.put(c, room, tries=120) is None:
                    raise _Fail
            self.place_target(typical[0] if rng.random() < 0.8 else room)
            exclude |= set(info.confusers)
        elif letter == "I":
            o = self.place_target(typical[0])
            age = float(rng.uniform(86400, 7 * 86400))
            memory = ((tgt, (o.xy[0], o.xy[1], o.half[2]), age),)
            if sc != "I_valid":
                self.drop(o)
                self.reserved.append(footprint(o))
                if sc == "I_moved":
                    self.place_target(self.pick([r for r in self._rooms if r.id != typical[0].id]))
        elif sc in ("K_studio", "K_mismatch"):
            mixed = self.rooms_of("studio")[0] if sc == "K_studio" else typical[0]
            if sc == "K_mismatch":
                for c in info.mismatch:
                    if self.put(c, mixed, tries=80) is None:
                        raise _Fail
            where = (
                mixed
                if rng.random() < 0.5
                else self.pick([r for r in self._rooms if r.id != mixed.id])
            )
            self.place_target(where)
        if letter == "L":
            request = str(rng.choice([t.format(t=name) for t in MANIPULATION_REQUESTS]))
        for r in self._rooms:
            self.furnish(r, exclude)
        return memory, request

    # -- start pose -----------------------------------------------------------------
    def choose_start(self, home: HomeSpec, memory, request) -> HomeSpec:
        rng = self.rng
        free = free_mask(home)
        _, _, centres = _cell_centres(home)
        mid = (np.array(self._rooms[0].lo) + self._rooms[0].hi) / 2
        dist = np.where(free, np.hypot(*(centres - mid).transpose(2, 0, 1)), np.inf)
        seed = np.unravel_index(np.argmin(dist), dist.shape)
        if not np.isfinite(dist[seed]):
            raise _Fail("no free cell")
        reach = flood(free, seed)
        limit = EXTENT_LIMIT - WALL_THICKNESS - np.array([home.width, home.depth]) / 2
        ok = reach & (np.abs(centres[..., 0]) <= limit[0]) & (np.abs(centres[..., 1]) <= limit[1])
        view = GroundTruthView(home)
        targets = instances_of(home, self.target)
        ids = [o.instance_id for o in targets]

        def cells_in(rooms, lo_d=0.0, hi_d=np.inf, centre=None):
            pts = centres[ok]
            keep = np.zeros(len(pts), bool)
            for r in rooms:
                keep |= ((pts >= r.lo) & (pts <= r.hi)).all(1)
            if centre is not None:
                d = np.hypot(*(pts - centre).T)
                keep &= (d >= lo_d) & (d <= hi_d)
            pts = pts[keep]
            return pts[rng.permutation(len(pts))]

        def made(p, yaw):
            return self.home((float(p[0]), float(p[1]), float(yaw)), memory, request)

        if self.letter == "A":
            t = targets[0]
            pts = cells_in([self._rooms[t.room_id]], 1.5, 3.0, np.array(t.xy))
            if len(pts) == 0:
                raise _Fail("no start cell for A")
            pts = pts[line_of_sight(pts, t, _blockers(home))]
            for p in pts[:40]:
                v = np.array(t.xy) - p
                yaw = float(np.arctan2(v[1], v[0]) + rng.uniform(-0.15, 0.15))
                if view.pixels(ids, p[0], p[1], yaw, stride=1) >= 450:
                    return made(p, yaw)
            raise _Fail("target not visible from any start")
        if self.letter in "BCL":
            t = targets[0]
            pts = cells_in([r for r in self._rooms if r.id != t.room_id])
            for p in pts[:30]:
                yaw = float(rng.uniform(-pi, pi))
                if view.pixels(ids, p[0], p[1], yaw, stride=1) == 0:
                    return made(p, yaw)
            raise _Fail("no start cell out of sight")
        pts = cells_in(self._rooms)
        if len(pts) == 0:
            raise _Fail("no start cell")
        return made(pts[0], rng.uniform(-pi, pi))

    # -- driver ---------------------------------------------------------------------
    def attempt(self) -> HomeSpec:
        self.layout()
        self.assign_types()
        self._rooms = [RoomBox(i, t, lo, hi) for i, lo, hi, t in self.rooms]
        self.objects: list[ObjectSpec] = []
        base = self.home()
        self._keep = door_keepouts(base)
        self._interior = [room_interior(base, r) for r in self._rooms]
        self.reserved: list[Rect] = []
        self.counts: dict[str, int] = {}
        memory, request = self.build_objects()
        home = self.choose_start(self.home(), memory, request)
        bad = verify_scenario(home)
        if bad:
            raise _Fail(bad)
        return home


def sample_home(
    rng: np.random.Generator,
    scenario: str,
    target: str = "refrigerator",
    generator: str = "nominal",
    seed: int | None = None,
) -> HomeSpec:
    """Sample a home satisfying a scenario recipe (see the module docstring). ``seed`` fixes the
    home exactly; otherwise one is drawn from ``rng`` and recorded in ``HomeSpec.seed``. Raises
    ``HomeGenerationError`` after ``MAX_ATTEMPTS`` failed attempts."""
    seed = int(rng.integers(2**31)) if seed is None else int(seed)
    gen = _Gen(np.random.default_rng(seed), scenario, target, generator, seed)
    reasons: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        try:
            return gen.attempt()
        except _Fail as e:
            reasons.append(str(e.args[0]) if e.args else "placement")
    raise HomeGenerationError(
        f"scenario {scenario} target {target} ({generator}) failed {MAX_ATTEMPTS} times; "
        f"last reasons: {reasons[-3:]}"
    )
