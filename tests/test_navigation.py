"""Contract tests only: no navigation-model or humanoid performance claims."""

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from dream.navigation.control import (
    LocalWaypoint,
    NavigationRequest,
    NavigationState,
    WaypointTracker,
    decode_waypoints,
)
from dream.robot.t1 import T1Policy


def state(**kwargs):
    kwargs.setdefault("depth_time_s", kwargs.get("time_s", 0))
    return replace(NavigationState(0, 0, 0, 0, 0, 2), **kwargs)


def test_navigation_scale_is_applied_once_and_targets_are_cumulative():
    raw = np.tile([0.5, 0.5, 0.5], (8, 1))
    request = NavigationRequest("Walk to the table", 0, 0)
    decoded = decode_waypoints(raw, request, normalized=True)
    assert decoded[7] == decoded[0]
    assert decoded[0].forward_m == 0.5 and decoded[0].left_m == 0.2165
    assert decoded[0].heading_rad == pytest.approx(pi / 3)
    metric = np.tile([0.5, 0.2165, pi / 3], (8, 1))
    assert decode_waypoints(metric, request, normalized=False) == decoded
    assert np.array_equal(raw, np.tile([0.5, 0.5, 0.5], (8, 1)))
    with pytest.raises(ValueError):
        decode_waypoints(np.full((8, 3), np.nan), request, normalized=False)


def test_left_and_forward_waypoints_use_observation_frame_and_measured_progress():
    tracker = WaypointTracker()
    observed = state(odom_yaw_rad=pi / 2)
    tracker.submit(LocalWaypoint(0.5, 0, 0, 0, 0), observed)
    first = tracker.update(observed)
    assert first.forward_m_s == 0.2 and abs(first.yaw_rad_s) < 1e-9
    reached = tracker.update(state(time_s=2, odom_y_m=0.5, odom_yaw_rad=pi / 2))
    assert reached.status == "waypoint reached" and reached.forward_m_s == 0
    tracker.submit(LocalWaypoint(0, 0.5, 0, 2, 0), state(time_s=2))
    turn = tracker.update(state(time_s=2))
    assert turn.forward_m_s == 0 and turn.yaw_rad_s > 0


@pytest.mark.parametrize(
    "changed",
    [
        {"clearance_m": 0.1},
        {"clearance_m": float("nan")},
        {"tilt_rad": 0.5},
        {"time_s": 9},
        {"source": "simulator_ground_truth"},
    ],
)
def test_navigation_safety_revokes_motion_authority(changed):
    tracker = WaypointTracker()
    target = LocalWaypoint(0.5, 0, 0, 0, 0)
    tracker.submit(target, state())
    result = tracker.update(state(**changed))
    assert result.forward_m_s == result.yaw_rad_s == 0
    assert tracker.target is None and tracker.authority == 1
    with pytest.raises(ValueError, match="revoked"):
        tracker.submit(target, state())


def test_stale_waypoints_and_backward_sensor_time_are_rejected():
    tracker = WaypointTracker()
    with pytest.raises(ValueError, match="stale"):
        tracker.submit(LocalWaypoint(0.5, 0, 0, 0, 0), state(time_s=3))
    tracker.submit(LocalWaypoint(0.5, 0, 0, 1, 0), state(time_s=1))
    assert tracker.update(state(time_s=0.5)).forward_m_s == 0


def test_stale_depth_stops_even_if_robot_proprioception_is_current():
    tracker = WaypointTracker()
    tracker.submit(LocalWaypoint(0.5, 0, 0, 0, 0), state())
    assert tracker.update(state(time_s=1, depth_time_s=0)).status == "invalid observation"


def test_policy_observation_order_named_joints_and_recurrent_action_contract():
    cfg = {
        "joint_names": [str(i) for i in range(23)],
        "default_joint_pos": [0.0] * 23,
        "sim_dt": 0.005,
        "decimation": 4,
        "action_scale": 0.5,
    }
    observed = []

    def actor(obs):
        observed.append(obs.copy())
        return np.arange(23) * 0.01

    adapter = T1Policy(cfg, actor, list(reversed(cfg["joint_names"])))
    limits = np.tile([-3, 3], (23, 1))
    q, dq = np.arange(23) * 0.01, np.arange(23) * 0.02
    result = adapter.target(q, dq, [0.1, 0, 0], [0, 0, 0.1], [0, 0, -1], [0.2, 0, 0], limits)
    assert np.allclose(result, np.arange(22, -1, -1) * 0.005)
    assert observed[0].shape == (81,)
    assert np.allclose(observed[0][12:35], q[::-1])
    assert np.allclose(observed[0][35:58], dq[::-1])
    adapter.target(q, dq, [0, 0, 0], [0, 0, 0], [0, 0, -1], [0, 0, 0], limits)
    assert np.allclose(observed[1][-23:], np.arange(23) * 0.01)
    with pytest.raises(ValueError, match="command"):
        adapter.target(q, dq, [0, 0, 0], [0, 0, 0], [0, 0, -1], [1, 0, 0], limits)
    adapter.policy = lambda _: np.full(23, np.nan)
    with pytest.raises(ValueError, match="action"):
        adapter.target(q, dq, [0, 0, 0], [0, 0, 0], [0, 0, -1], [0, 0, 0], limits)


def test_robot_rotation_is_scalar_first_and_preserves_forward_left_convention():
    from dream.sim.isaac_t1 import rotation_wxyz

    r = rotation_wxyz([np.cos(pi / 4), 0, 0, np.sin(pi / 4)])
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(r.T @ r, np.eye(3))
    with pytest.raises(ValueError, match="orientation"):
        rotation_wxyz([2, 0, 0, 0])


def test_t1_out_of_limit_setpoints_are_recorded():
    cfg = {
        "joint_names": [str(i) for i in range(23)],
        "default_joint_pos": [0.0] * 23,
        "sim_dt": 0.005,
        "decimation": 4,
        "action_scale": 0.5,
    }
    actor = T1Policy(
        cfg,
        lambda _: np.full(23, 2.0, dtype=np.float32),
        cfg["joint_names"],
    )
    limits = np.tile([-0.1, 0.1], (23, 1))
    q = np.zeros(23)

    target = actor.target(
        q,
        q,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, -1],
        [0, 0, 0],
        limits,
    )
    assert np.allclose(target, 1.0)
    assert np.allclose(actor.target_limit_excess, 0.9)

    # Actual physical joint positions outside limits remain invalid.
    with pytest.raises(ValueError, match="proprioception"):
        actor.target(
            np.full(23, 0.2),
            q,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [0, 0, 0],
            limits,
        )


@pytest.mark.parametrize("residual", [-0.000062669729, 0.000062669729, -0.05, 0.05])
def test_uncalibrated_realized_joint_excursions_never_reach_actor(residual):
    """Keep both the captured residual and independent large excursions fail-closed."""
    cfg = {
        "joint_names": [str(i) for i in range(23)],
        "default_joint_pos": [0.0] * 23,
        "sim_dt": 0.005,
        "decimation": 4,
        "action_scale": 0.5,
    }
    actor = T1Policy(
        cfg, lambda _: pytest.fail("invalid physical state reached actor"), cfg["joint_names"]
    )
    limits = np.tile([-1.0, 1.0], (23, 1))
    q = np.zeros(23)
    q[7] = np.sign(residual) + residual
    original = q.copy()
    with pytest.raises(ValueError, match="proprioception"):
        actor.target(q, np.zeros(23), [0, 0, 0], [0, 0, 0], [0, 0, -1], [0, 0, 0], limits)
    np.testing.assert_array_equal(q, original)
