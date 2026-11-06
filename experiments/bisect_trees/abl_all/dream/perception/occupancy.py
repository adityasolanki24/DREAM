"""A 2-D occupancy map of the floor around a mobile robot, built from its RGB-D frames.

Standard log-odds occupancy mapping in the robot's odometry frame: depth pixels are back-
projected with the frame's calibration (which the mobile robot reports in its odometry frame,
never in simulator coordinates). Points between OBSTACLE_MIN and OBSTACLE_MAX above the floor
mark their cell occupied; every cell a camera ray passes over before its hit point (and a
floor hit's own cell) is evidence of free space. Unvisited cells stay unknown (log-odds 0).
Odometry drift is not corrected: the map is only as good as the odometry it is built on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dream.perception.geometry import backproject
from dream.types import RGBDFrame

CELL = 0.05  # m
OBSTACLE_MIN = 0.06  # m above the floor: lower returns are floor (1 mm + 1.5 mm/m^2 noise)
OBSTACLE_MAX = 1.8  # m: above the robot's height, nothing it can collide with
HIT, MISS, CLAMP = 0.9, -0.4, 4.0  # log-odds increments and saturation
OCCUPIED, FREE = 0.6, -0.6  # log-odds thresholds for a confident state
PIXEL_STRIDE = 3


@dataclass
class OccupancyGrid:
    size: float = 16.0  # m, square, centred on the odometry origin
    logodds: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = int(round(self.size / CELL))
        if self.logodds is None:
            self.logodds = np.zeros((n, n), np.float32)

    @property
    def n(self) -> int:
        return self.logodds.shape[0]

    def cell(self, xy) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, float)
        ij = np.floor((xy + self.size / 2) / CELL).astype(int)
        return ij[..., 0], ij[..., 1]

    def centre(self, i, j) -> np.ndarray:
        return np.stack([np.asarray(i), np.asarray(j)], -1) * CELL - self.size / 2 + CELL / 2

    def inside(self, i, j) -> np.ndarray:
        return (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)

    def integrate(self, frame: RGBDFrame) -> None:
        pts = backproject(frame)[::PIXEL_STRIDE, ::PIXEL_STRIDE].reshape(-1, 3)
        mask = frame.self_mask[::PIXEL_STRIDE, ::PIXEL_STRIDE].reshape(-1)
        pts = pts[np.isfinite(pts).all(axis=1) & ~mask]
        if not len(pts):
            return
        cam = frame.calibration.position
        obstacle = (pts[:, 2] > OBSTACLE_MIN) & (pts[:, 2] < OBSTACLE_MAX)
        floor = pts[:, 2] <= OBSTACLE_MIN
        # Free space: cells under each ray before its hit, sampled every half cell.
        d = pts[:, :2] - cam[:2]
        length = np.linalg.norm(d, axis=1)
        steps = np.arange(0.0, length.max(), CELL / 2)
        frac = steps[None, :] / np.maximum(length[:, None], 1e-6)
        before = frac < 1.0 - CELL / np.maximum(length[:, None], 1e-6)
        before |= floor[:, None] & (frac <= 1.0)
        xy = cam[None, None, :2] + frac[..., None] * d[:, None, :]
        i, j = self.cell(xy[before])
        ok = self.inside(i, j)
        free = np.zeros_like(self.logodds, bool)
        free[i[ok], j[ok]] = True
        hit = np.zeros_like(free)
        hi, hj = self.cell(pts[obstacle, :2])
        ok = self.inside(hi, hj)
        hit[hi[ok], hj[ok]] = True
        self.logodds[free & ~hit] += MISS
        self.logodds[hit] += HIT
        np.clip(self.logodds, -CLAMP, CLAMP, out=self.logodds)

    def occupied(self) -> np.ndarray:
        return self.logodds > OCCUPIED

    def free(self) -> np.ndarray:
        return self.logodds < FREE

    def unknown(self) -> np.ndarray:
        return ~self.occupied() & ~self.free()
