"""Relocalise a new session's odometry frame against a stored occupancy map.

Each session's odometry frame starts at wherever the robot happens to stand, so a remembered
position (stored in an earlier session's frame) means nothing until the two frames are
aligned. This aligns them from the robot's own sensing only: the occupied cells of the local
occupancy grid (built from the head RGB-D camera after a look-around) are matched to the
occupied cells of the stored grid. Scan-to-map matching, in two stages:

1. Exhaustive coarse search over heading (COARSE_YAW_DEG steps) and translation
   (COARSE_STEP_M) of a chamfer cost: the mean distance from each transformed local obstacle
   cell to the nearest stored obstacle cell, truncated at TRUNCATE_M so unmapped clutter does
   not dominate.
2. Local refinement of the best few coarse hypotheses (FINE_YAW_DEG, FINE_STEP_M).

The answer is refused, not guessed, when the evidence is weak: too few inlier cells
(MIN_INLIER_FRACTION within INLIER_M) or a second, distinct hypothesis (more than
DISTINCT_M or DISTINCT_DEG away) whose cost is within AMBIGUITY_RATIO of the best; a
rectangular room with little furniture is the typical ambiguous case. Simulator poses are not
an input; evaluations compare the result with them afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from dream.perception.occupancy import CELL, OccupancyGrid

TRUNCATE_M = 0.5
COARSE_YAW_DEG, COARSE_STEP_M = 4.0, 0.2
FINE_YAW_DEG, FINE_STEP_M = 0.5, 0.05
MAX_POINTS = 400
INLIER_M = 0.1
MIN_INLIER_FRACTION = 0.5
AMBIGUITY_RATIO = 1.15
DISTINCT_M, DISTINCT_DEG = 0.5, 20.0


@dataclass(frozen=True)
class Alignment:
    """Pose of the new session's odometry frame in the stored map frame (x, y, yaw)."""

    accepted: bool
    x: float
    y: float
    yaw: float
    cost_m: float  # mean truncated distance of local obstacle cells to stored ones
    inlier_fraction: float
    ambiguity: float  # cost of the best distinct rival / best cost (>1; near 1 is ambiguous)
    reason: str

    def to_map(self, xy) -> np.ndarray:
        """Session (odometry) coordinates -> stored map coordinates."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        p = np.asarray(xy, float)
        return np.stack([c * p[..., 0] - s * p[..., 1] + self.x,
                         s * p[..., 0] + c * p[..., 1] + self.y], -1)  # fmt: skip

    def to_session(self, xy) -> np.ndarray:
        """Stored map coordinates -> session (odometry) coordinates."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        p = np.asarray(xy, float) - np.array([self.x, self.y])
        return np.stack([c * p[..., 0] + s * p[..., 1], -s * p[..., 0] + c * p[..., 1]], -1)


class AlignedMemory:
    """A ``HomeMemory`` seen from the current session's odometry frame through an accepted
    ``Alignment``: beliefs come out in session coordinates, absences and new sightings go
    in converted back to the stored map frame. Everything else is passed through."""

    def __init__(self, memory, alignment: Alignment) -> None:
        if not alignment.accepted:
            raise ValueError("only an accepted alignment may translate memory")
        self.memory, self.alignment = memory, alignment

    def where_is(self, label: str, map_id: str, now: float | None = None):
        out = []
        for b in self.memory.where_is(label, map_id, now=now):
            p = np.array(b.position, float)
            p[:2] = self.alignment.to_session(p[:2])
            out.append(replace(b, position=p))
        return out

    def record_absence(self, center, radius: float, map_id: str, **kw):
        return self.memory.record_absence(self.alignment.to_map(center), radius, map_id, **kw)

    def observe(self, label: str, position, map_id: str, **kw):
        p = np.array(position, float)
        p[:2] = self.alignment.to_map(p[:2])
        return self.memory.observe(label, p, map_id, **kw)

    def __getattr__(self, name):
        return getattr(self.memory, name)


def distance_field(grid: OccupancyGrid, truncate: float = TRUNCATE_M) -> np.ndarray:
    """Distance (m) from each cell to the nearest occupied cell, truncated (8-neighbour
    dilation: exact to within one cell, enough for a cost)."""
    occ = grid.occupied()
    dist = np.full(occ.shape, truncate, np.float32)
    dist[occ] = 0.0
    reached = occ.copy()
    for k in range(1, int(math.ceil(truncate / CELL)) + 1):
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        new = grown & ~reached
        dist[new] = min(k * CELL, truncate)
        reached = grown
    return dist


def _points(grid: OccupancyGrid, rng: np.random.Generator) -> np.ndarray:
    i, j = np.nonzero(grid.occupied())
    pts = grid.centre(i, j)
    if len(pts) > MAX_POINTS:
        pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
    return pts


def _costs(field, stored: OccupancyGrid, pts, yaws, txs, tys) -> np.ndarray:
    """Cost for every (yaw, tx, ty) combination: (len(yaws), len(txs), len(tys))."""
    out = np.empty((len(yaws), len(txs), len(tys)), np.float32)
    for a, yaw in enumerate(yaws):
        c, s = math.cos(yaw), math.sin(yaw)
        rx = c * pts[:, 0] - s * pts[:, 1]
        ry = s * pts[:, 0] + c * pts[:, 1]
        i = np.floor((rx[None, None, :] + txs[:, None, None] + stored.size / 2) / CELL)
        j = np.floor((ry[None, None, :] + tys[None, :, None] + stored.size / 2) / CELL)
        i, j = i.astype(np.int32), j.astype(np.int32)
        inside = (i >= 0) & (i < stored.n) & (j >= 0) & (j < stored.n)
        d = np.where(inside, field[np.clip(i, 0, stored.n - 1), np.clip(j, 0, stored.n - 1)],
                     TRUNCATE_M)  # fmt: skip
        out[a] = d.mean(axis=2)
    return out


def relocalise(stored: OccupancyGrid, local: OccupancyGrid, search_radius: float = 6.0,
               seed: int = 0) -> Alignment:  # fmt: skip
    """Align ``local`` (this session's grid) to ``stored`` (an earlier session's grid)."""
    rng = np.random.default_rng(seed)
    pts = _points(local, rng)
    if len(pts) < 30 or stored.occupied().sum() < 30:
        return Alignment(False, 0, 0, 0, TRUNCATE_M, 0, 1, "too few obstacle cells")
    field = distance_field(stored)
    yaws = np.deg2rad(np.arange(0, 360, COARSE_YAW_DEG))
    ts = np.arange(-search_radius, search_radius + 1e-9, COARSE_STEP_M)
    coarse = _costs(field, stored, pts, yaws, ts, ts)
    order = np.argsort(coarse, axis=None)
    hyps = []
    for flat in order[:200]:  # best distinct coarse hypotheses
        a, b, c = np.unravel_index(flat, coarse.shape)
        h = (yaws[a], ts[b], ts[c])
        if all(_distinct(h, g) for g in hyps):
            hyps.append(h)
        if len(hyps) == 4:
            break
    refined = [_refine(field, stored, pts, h) for h in hyps]
    refined.sort(key=lambda r: r[0])
    best_cost, (yaw, x, y) = refined[0]
    rivals = [c for c, h in refined[1:] if _distinct(h, (yaw, x, y))]
    ambiguity = (rivals[0] / max(best_cost, 1e-6)) if rivals else float("inf")
    c, s = math.cos(yaw), math.sin(yaw)
    mapped = np.stack([c * pts[:, 0] - s * pts[:, 1] + x, s * pts[:, 0] + c * pts[:, 1] + y], -1)
    i, j = stored.cell(mapped)
    inside = stored.inside(i, j)
    near = np.zeros(len(pts), bool)
    near[inside] = field[i[inside], j[inside]] <= INLIER_M
    inliers = float(near.mean())
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    if inliers < MIN_INLIER_FRACTION:
        return Alignment(False, x, y, yaw, best_cost, inliers, ambiguity, "too few inliers")
    if ambiguity < AMBIGUITY_RATIO:
        return Alignment(False, x, y, yaw, best_cost, inliers, ambiguity, "ambiguous")
    return Alignment(True, x, y, yaw, best_cost, inliers, ambiguity, "ok")


def _distinct(h, g) -> bool:
    dyaw = abs((h[0] - g[0] + math.pi) % (2 * math.pi) - math.pi)
    return math.hypot(h[1] - g[1], h[2] - g[2]) > DISTINCT_M or dyaw > math.radians(DISTINCT_DEG)


def _refine(field, stored, pts, h):
    yaw0, x0, y0 = h
    yaws = yaw0 + np.deg2rad(np.arange(-COARSE_YAW_DEG, COARSE_YAW_DEG + 1e-9, FINE_YAW_DEG))
    xs = x0 + np.arange(-COARSE_STEP_M, COARSE_STEP_M + 1e-9, FINE_STEP_M)
    ys = y0 + np.arange(-COARSE_STEP_M, COARSE_STEP_M + 1e-9, FINE_STEP_M)
    cost = _costs(field, stored, pts, yaws, xs, ys)
    a, b, c = np.unravel_index(np.argmin(cost), cost.shape)
    return float(cost[a, b, c]), (float(yaws[a]), float(xs[b]), float(ys[c]))
