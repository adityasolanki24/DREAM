"""Path planning and frontier selection on an occupancy map (classical, no learning).

- ``traversable``: known-free cells farther than the robot's radius from any occupied cell.
- ``distances``: Dijkstra over traversable cells from the robot (8-connected), restricted to
  the explored bounding box; it prices every candidate goal at once.
- ``frontiers``: clusters of traversable cells next to unknown space, each scored by how much
  unknown area lies within sensing range of it (an optimistic information-gain proxy that
  ignores occlusion), against the walking distance to reach it.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from dream.perception.occupancy import CELL, OccupancyGrid

ROBOT_RADIUS = 0.35  # m: G1 shoulder half-width ~0.25 m plus arm swing and tracking error
SENSE_RANGE = 2.5  # m: within the head camera's 3 m depth range, where cells are reliable
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    out = mask.copy()
    n, m = mask.shape
    r = radius_cells
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r or (di == 0 and dj == 0):
                continue
            src = mask[max(0, -di) : n - max(0, di), max(0, -dj) : m - max(0, dj)]
            out[max(0, di) : n - max(0, -di), max(0, dj) : m - max(0, -dj)] |= src
    return out


def traversable(grid: OccupancyGrid, radius: float = ROBOT_RADIUS) -> np.ndarray:
    blocked = dilate(grid.occupied(), int(np.ceil(radius / CELL)))
    return grid.free() & ~blocked


def distances(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Path length (m) from ``start`` to every reachable cell; inf elsewhere."""
    known = np.argwhere(passable)
    dist = np.full(passable.shape, np.inf)
    if not len(known):
        return dist
    lo = np.maximum(np.minimum(known.min(0), start) - 1, 0)
    hi = np.minimum(np.maximum(known.max(0), start) + 2, passable.shape)
    si, sj = start
    dist[si, sj] = 0.0
    heap = [(0.0, si, sj)]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj in _NEIGHBOURS:
            a, b = i + di, j + dj
            if not (lo[0] <= a < hi[0] and lo[1] <= b < hi[1]) or not passable[a, b]:
                continue
            nd = d + CELL * (1.4142 if di and dj else 1.0)
            if nd < dist[a, b]:
                dist[a, b] = nd
                heapq.heappush(heap, (nd, a, b))
    return dist


def path_to(dist: np.ndarray, goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Steepest descent on ``dist`` from ``goal`` back to the start (start excluded)."""
    if not np.isfinite(dist[goal]):
        return []
    path = [goal]
    i, j = goal
    while dist[i, j] > 0:
        best = min(
            ((i + di, j + dj) for di, dj in _NEIGHBOURS),
            key=lambda c: (
                dist[c] if 0 <= c[0] < dist.shape[0] and 0 <= c[1] < dist.shape[1] else np.inf
            ),
        )
        if dist[best] >= dist[i, j]:
            break
        i, j = best
        path.append(best)
    return path[::-1][1:]


@dataclass(frozen=True)
class Frontier:
    cell: tuple[int, int]  # the reachable cell of the cluster closest to its centroid
    size: int  # frontier cells in the cluster
    gain: float  # unknown area (m^2) within SENSE_RANGE of ``cell``
    cost: float  # walking distance (m)


def frontiers(grid: OccupancyGrid, passable: np.ndarray, dist: np.ndarray) -> list[Frontier]:
    unknown = grid.unknown()
    edge = np.zeros_like(unknown)
    edge[1:, :] |= unknown[:-1, :]
    edge[:-1, :] |= unknown[1:, :]
    edge[:, 1:] |= unknown[:, :-1]
    edge[:, :-1] |= unknown[:, 1:]
    cand = passable & edge & np.isfinite(dist)
    seen = np.zeros_like(cand)
    out = []
    r = int(SENSE_RANGE / CELL)
    for start in map(tuple, np.argwhere(cand)):
        if seen[start]:
            continue
        cluster, stack = [], [start]
        seen[start] = True
        while stack:
            c = stack.pop()
            cluster.append(c)
            for di, dj in _NEIGHBOURS:
                a = (c[0] + di, c[1] + dj)
                if 0 <= a[0] < cand.shape[0] and 0 <= a[1] < cand.shape[1]:
                    if cand[a] and not seen[a]:
                        seen[a] = True
                        stack.append(a)
        if len(cluster) < 4:  # speckle, not an opening into unexplored space
            continue
        pts = np.array(cluster)
        c = tuple(pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))])
        win = unknown[max(0, c[0] - r) : c[0] + r + 1, max(0, c[1] - r) : c[1] + r + 1]
        out.append(Frontier(c, len(cluster), float(win.sum()) * CELL * CELL, float(dist[c])))
    return out
