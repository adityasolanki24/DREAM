"""Spatial-semantic memory for a mobile robot: an occupancy map plus furniture candidates.

Built only from the robot's own RGB-D frames, whose calibration is in its odometry frame.
- The occupancy map (dream/perception/occupancy.py) says where the robot can walk.
- Furniture candidates come from DREAM's classical detector (dream/perception/detect.py):
  box-like detections 0.3-1.6 m tall in a saturated colour, merged across views by colour and
  proximity. Treating neutral (grey/white) detections as room structure is a TASK-SPECIFIC
  HEURISTIC for these rooms, whose walls are untextured and neutral; open-vocabulary
  perception should replace it.
- A handle is a small detection the detector attributes to a box's side (``kind ==
  "handle"``). A candidate with a handle is a box that probably opens; how it opens is not
  observable here and is left unknown.
- Each candidate remembers from which of SIDES directions around its centre it has been
  seen from close range (CLOSE_VIEW), because a handle is only resolvable there. No shape is
  assumed: a box fitted to partial views misplaced faces by 15-30 cm (development rooms
  0-1), so viewpoints are bearings around the centre, not faces of a fitted rectangle.
This is per-episode state, not persistent memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dream.perception.detect import Detection, detect
from dream.perception.occupancy import OccupancyGrid
from dream.types import RGBDFrame

FLOOR_PLANE = (np.array([0.0, 0.0, 1.0]), 0.0)  # odometry frame: z = 0 is the floor stood on
# The head camera's depth noise model, sigma = a + b z^2 (m): its sensor specification
# (dream/sim/camera.py CameraSpec.depth_noise), known to the robot like a datasheet value.
DEPTH_NOISE = (0.001, 0.0015)
NEUTRAL = ("grey", "white", "black")
MERGE = 0.6  # m: same-coloured box detections closer than this are one candidate
# m: a handle 0.12 x 0.024 m spans ~130 pixels of the 320x240 head camera at 1 m and ~1 pixel
# row at 2.5 m (measured on development room seed 1), so faces count as checked within this.
CLOSE_VIEW = 1.4
SIDES = 8  # bearings around a candidate, 45 degrees apart
HANDLE_GAP = 0.12  # m: a handle protrudes at most this far beyond the object's visible extent
MAX_MISSES = 3


@dataclass
class Candidate:
    id: int
    color: str
    position: np.ndarray  # (x, y, z) visible-box centre, odometry frame
    yaw: float  # rad, orientation of the fitted rectangle
    extent: np.ndarray  # (a, b, height) m, the largest seen
    views: int = 1
    weight: float = 0.0  # sum of view weights behind ``position``
    misses: int = 0  # close views in which it should have been detected and was not
    handle: np.ndarray | None = None  # handle position (odometry frame), if any was seen
    handle_views: int = 0
    sides_close: set = field(default_factory=set)  # bearing indices seen within CLOSE_VIEW

    @property
    def radius(self) -> float:
        return float(max(self.extent[:2])) / 2

    def side_of(self, xy) -> int:
        """Bearing index (0..SIDES-1) of ``xy`` as seen from the candidate's centre."""
        d = np.asarray(xy)[:2] - self.position[:2]
        return int(np.round(np.arctan2(d[1], d[0]) / (2 * np.pi / SIDES))) % SIDES

    def side_direction(self, k: int) -> np.ndarray:
        a = 2 * np.pi * k / SIDES
        return np.array([np.cos(a), np.sin(a)])

    def handle_side(self) -> int | None:
        return None if self.handle is None else self.side_of(self.handle)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "color": self.color,
            "position": np.round(self.position, 3).tolist(),
            "extent": np.round(self.extent, 3).tolist(),
            "views": self.views,
            "handle": None if self.handle is None else np.round(self.handle, 3).tolist(),
            "handle_views": self.handle_views,
            "handle_side": self.handle_side(),
            "sides_seen_close": sorted(self.sides_close),
        }


class SpatialMemory:
    def __init__(self, rng: np.random.Generator) -> None:
        self.grid = OccupancyGrid()
        self.candidates: list[Candidate] = []
        self.rng = rng
        self.frames = 0
        self._next = 1

    def update(self, frame: RGBDFrame) -> list[Detection]:
        self.grid.integrate(frame)
        dets, _ = detect(frame, self.rng, FLOOR_PLANE, DEPTH_NOISE)
        cam = frame.calibration.position
        boxes = [
            d
            for d in dets
            if d.kind in ("box", "container")
            and d.color not in NEUTRAL
            and 0.3 <= d.extent[2] <= 1.6
            and max(d.extent[:2]) <= 2.0
        ]
        seen = {self._add_box(d, cam) for d in boxes}
        self._count_misses(frame, seen)
        for h in (d for d in dets if d.kind == "handle"):
            self._add_handle(h)
        self.frames += 1
        return dets

    def _count_misses(self, frame: RGBDFrame, seen: set[int]) -> None:
        """A candidate that should be plainly in view up close but is not detected is evidence
        it is not there. After MAX_MISSES such views it is dropped (development room seed 1:
        two stray far detections of a table became a candidate the robot kept walking to)."""
        cal = frame.calibration
        axis = -cal.rotation[:, 2]  # optical axis (MuJoCo cameras look along -z)
        heading = np.arctan2(axis[1], axis[0])
        half_fov = np.radians(cal.fovy_deg * cal.width / cal.height) / 2
        keep = []
        for c in self.candidates:
            rel = c.position[:2] - cal.position[:2]
            dist = float(np.linalg.norm(rel))
            off = abs((np.arctan2(rel[1], rel[0]) - heading + np.pi) % (2 * np.pi) - np.pi)
            if c.id not in seen and 0.5 < dist <= CLOSE_VIEW + c.radius and off < 0.6 * half_fov:
                c.misses += 1
            if c.misses < MAX_MISSES or c.handle is not None:
                keep.append(c)
        self.candidates = keep

    def _add_box(self, d: Detection, cam: np.ndarray) -> int:
        near = [
            c
            for c in self.candidates
            if c.color == d.color and np.linalg.norm(c.position[:2] - d.position[:2]) < MERGE
        ]
        # Close views locate an object better (depth noise grows with range squared).
        weight = 1.0 / max(float(np.linalg.norm(cam[:2] - d.position[:2])), 0.5) ** 2
        if near:
            c = near[0]
            w = weight / (c.weight + weight)
            c.position = (1 - w) * c.position + w * d.position
            c.weight += weight
            c.extent = np.maximum(c.extent, d.extent)
            c.views += 1
        else:
            c = Candidate(self._next, d.color, d.position.copy(), d.yaw, d.extent.copy())
            c.weight = weight
            self._next += 1
            self.candidates.append(c)
        dist = float(np.linalg.norm(cam[:2] - c.position[:2]))
        if dist <= CLOSE_VIEW + c.radius:
            k = c.side_of(cam)  # a face is legible up to ~45 degrees off its normal
            c.sides_close.update({(k - 1) % SIDES, k, (k + 1) % SIDES})
        return c.id

    def _add_handle(self, h: Detection) -> None:
        """Attribute a fully visible handle to the candidate it sits on (within HANDLE_GAP).

        A handle is recognised by being small, so a fragment cut by the image border or
        hidden behind something (``occluded``) is not one: its size is unknown. Development
        room seed 1 produced phantom handles from wall fragments cut by the top border.
        """
        if h.occluded:
            return
        best, best_d = None, np.inf
        for c in self.candidates:
            if not 0.1 <= h.position[2] <= c.extent[2] + 0.03:
                continue
            gap = float(np.linalg.norm(h.position[:2] - c.position[:2])) - c.radius
            if gap <= HANDLE_GAP and gap < best_d:
                best, best_d = c, gap
        if best is None:
            return
        c = best
        w = 1.0 / (c.handle_views + 1)
        c.handle = h.position.copy() if c.handle is None else (1 - w) * c.handle + w * h.position
        c.handle_views += 1
