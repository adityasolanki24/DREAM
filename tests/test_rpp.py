"""Contracts of the Regulated Pure Pursuit navigator (controller-level, on an ideal unicycle;
these are not humanoid physics: the G1 comparison is experiments/g1_navigation_eval.py)."""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np

from dream.navigation.follow import PathFollower
from dream.navigation.rpp import RegulatedPurePursuit, densify, line_of_sight, shortcut, smooth


def drive(ctrl, pose, steps=400, dt=0.1, **kw):
    log = []
    pose = np.array(pose, float)
    for _ in range(steps):
        v, w, status = ctrl.update(pose, **kw)
        log.append((v, w, status, pose.copy()))
        if status == "arrived":
            break
        pose = pose + [v * cos(pose[2]) * dt, v * sin(pose[2]) * dt, w * dt]
    return log


def l_path():
    return densify([np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([2.0, 2.0])])


def test_line_of_sight_and_shortcut_remove_grid_staircase():
    free = np.ones((40, 40), bool)
    free[10:30, 18:22] = False  # a wall
    assert line_of_sight(free, (5, 5), (5, 35))
    assert not line_of_sight(free, (20, 5), (20, 35))
    stair = [(i, i) for i in range(1, 9)]  # a diagonal grid path
    assert shortcut(stair, free, (0, 0)) == [(8, 8)]


def test_smoothing_keeps_endpoints_and_free_space():
    pts = densify([np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])])
    out = smooth(pts, lambda p: p[0] <= 1.0 + 1e-9)
    np.testing.assert_allclose(out[0], pts[0])
    np.testing.assert_allclose(out[-1], pts[-1])
    assert all(p[0] <= 1.0 + 1e-9 for p in out)
    corner = np.array([1.0, 0.0])
    assert min(np.linalg.norm(np.array(out) - corner, axis=1)) > 0.02  # the corner is rounded


def test_l_turn_is_one_continuous_motion_without_turning_in_place():
    ctrl = RegulatedPurePursuit()
    ctrl.set_path(l_path())
    log = drive(ctrl, [0, 0, 0])
    assert log[-1][2] == "arrived"
    walking = [r for r in log if r[2] == "walking"]
    assert not [r for r in log if r[2] == "turning"]
    stops = sum(1 for a, b in zip(walking, walking[1:], strict=False) if a[0] > 0.05 >= b[0])
    assert stops == 0
    old = PathFollower()
    old.set_path([p for p in l_path()[::4]])
    old_log = drive(old, [0, 0, 0])
    assert old_log[-1][2] == "arrived"


def test_commands_respect_acceleration_and_envelope_limits():
    ctrl = RegulatedPurePursuit()
    ctrl.set_path(l_path())
    log = drive(ctrl, [0, 0, 0])
    v = np.array([r[0] for r in log])
    w = np.array([r[1] for r in log])
    assert np.all(np.diff(v) <= ctrl.max_accel * ctrl.dt + 1e-9)
    assert np.all(np.diff(v) >= -ctrl.max_decel * ctrl.dt - 1e-9)
    assert np.all(np.abs(np.diff(w)) <= ctrl.max_angular_accel * ctrl.dt + 1e-9)
    assert v.max() <= 0.5 and np.abs(w).max() <= 0.6  # dream/robot/g1_wbc.MAX_COMMAND


def test_turns_in_place_only_when_the_path_is_behind():
    ctrl = RegulatedPurePursuit()
    ctrl.set_path(densify([np.array([0.0, 0.0]), np.array([-2.0, 0.0])]))
    log = drive(ctrl, [0, 0, 0])
    statuses = [r[2] for r in log]
    assert statuses[0] == "turning" and "walking" in statuses and statuses[-1] == "arrived"


def test_final_heading_aligned_within_tolerance():
    ctrl = RegulatedPurePursuit()
    ctrl.set_path(densify([np.array([0.0, 0.0]), np.array([1.5, 0.0])]), final_heading=pi / 2)
    log = drive(ctrl, [0, 0, 0])
    assert log[-1][2] == "arrived"
    assert abs(log[-1][3][2] - pi / 2) < ctrl.heading_tolerance + 0.1


def wall_clearance(p):
    return float(max(0.0, 1.0 - p[0]))  # a wall at x = 1


def test_projected_collision_stops_before_contact():
    ctrl = RegulatedPurePursuit()
    ctrl.set_path(densify([np.array([0.0, 0.0]), np.array([3.0, 0.0])]))
    log = drive(ctrl, [0, 0, 0], clearance=wall_clearance, min_clearance=0.32)
    assert log[-1][2] != "arrived"
    assert [r for r in log if r[2] == "blocked"]
    assert max(r[3][0] for r in log) < 1.0 - 0.32 + 0.05


def test_starting_inside_the_clearance_band_does_not_block_moving_away():
    """Regression (dev room 1): stopped 0.2 m from furniture, every arc was 'blocked',
    including those leading away; three approaches failed and cost up to 48 s each."""
    ctrl = RegulatedPurePursuit()
    ctrl.set_path(densify([np.array([0.8, 0.0]), np.array([-1.5, 0.0])]))
    log = drive(ctrl, [0.8, 0, np.pi], clearance=wall_clearance, min_clearance=0.32)
    assert log[-1][2] == "arrived"
    assert not [r for r in log if r[2] == "blocked"]


def test_speed_regulated_near_obstacles():
    a, b = RegulatedPurePursuit(), RegulatedPurePursuit()
    for c in (a, b):
        c.set_path(densify([np.array([0.0, 0.0]), np.array([3.0, 0.0])]))
    fa = drive(a, [0, 0, 0], steps=30)
    fb = drive(b, [0, 0, 0], steps=30, clearance=lambda p: 0.3)
    assert max(r[0] for r in fb) < max(r[0] for r in fa)
