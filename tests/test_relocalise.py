"""Scan-to-map relocalisation between two sessions' occupancy grids."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dream.memory.relocalise import relocalise
from dream.perception.occupancy import OccupancyGrid


def room_points(furniture=True):
    """Obstacle points of a 5 x 4 m room (walls) with asymmetric furniture, world frame."""
    pts = []
    for x in np.arange(-2.5, 2.5, 0.05):
        pts += [(x, -2.0), (x, 2.0)]
    for y in np.arange(-2.0, 2.0, 0.05):
        pts += [(-2.5, y), (2.5, y)]
    if furniture:
        for x in np.arange(1.2, 2.0, 0.05):  # a cabinet in one corner
            for y in np.arange(1.2, 1.7, 0.05):
                pts.append((x, y))
        for x in np.arange(-1.5, -1.0, 0.05):  # a crate on the other side
            for y in np.arange(-0.9, -0.5, 0.05):
                pts.append((x, y))
    return np.array(pts)


def grid_seen_from(pose, pts, visible_fraction=1.0, seed=0):
    """Occupancy grid in the odometry frame of a robot standing at ``pose`` (world)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rel = pts - (x, y)
    local = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], -1)
    rng = np.random.default_rng(seed)
    local = local[rng.random(len(local)) < visible_fraction]
    g = OccupancyGrid()
    i, j = g.cell(local)
    ok = g.inside(i, j)
    g.logodds[i[ok], j[ok]] = 2.0
    return g


def test_recovers_the_offset_between_two_start_poses():
    pts = room_points()
    a, b = (0.3, -0.4, 0.2), (-0.8, 0.9, 2.4)
    al = relocalise(grid_seen_from(a, pts), grid_seen_from(b, pts, visible_fraction=0.6))
    assert al.accepted, al
    # truth: b's frame expressed in a's frame
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    tx, ty = c * dx + s * dy, -s * dx + c * dy
    tyaw = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
    assert al.x == pytest.approx(tx, abs=0.08) and al.y == pytest.approx(ty, abs=0.08)
    assert abs((al.yaw - tyaw + math.pi) % (2 * math.pi) - math.pi) < math.radians(2)
    p = np.array([0.5, -0.3])
    assert np.allclose(al.to_session(al.to_map(p)), p)


def test_an_empty_rectangular_room_is_refused_as_ambiguous():
    pts = room_points(furniture=False)
    al = relocalise(grid_seen_from((0, 0, 0), pts), grid_seen_from((0.5, 0.2, 1.0), pts))
    assert not al.accepted and al.reason == "ambiguous"


def test_a_different_room_is_not_accepted():
    pts = room_points()
    other = pts * np.array([1.4, 0.7])  # a 7 x 2.8 m room with furniture elsewhere
    al = relocalise(grid_seen_from((0, 0, 0), pts), grid_seen_from((0, 0, 0), other))
    assert not al.accepted
