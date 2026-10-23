"""Physical controller contracts. Synthetic targets are labelled, never model evidence."""

import numpy as np
import pytest

from dream.robot.joint import JointTargetController
from dream.sim.tabletop import sample_tabletop
from dream.sim.tabletop_env import TabletopEnv


def make_env(seed=220, clock=None):
    env = TabletopEnv(sample_tabletop(np.random.default_rng(seed)), np.random.default_rng(seed))
    options = {} if clock is None else {"clock": clock}
    env.robot = JointTargetController(env.model, env.data, **options)
    return env


@pytest.mark.parametrize("seed,sign", [(220, 1), (221, -1)])
def test_safe_recorded_targets_move_through_physics_with_bounded_commands(seed, sign):
    env = make_env(seed)
    initial = env.robot_state().joint_pos.copy()
    goal = initial.copy()
    goal[6] += sign * 0.03
    receipt = env.robot.submit(goal, 0.07, 1 / 15)
    previous_v = env.robot_state().joint_vel.copy()
    peak_v, peak_a = 0.0, 0.0
    for _ in range(int((receipt["duration_s"] + 0.3) / env.dt)):
        env.step()
        assert np.max(np.abs(env.robot.command_velocity)) <= 0.5 + 1e-9
        assert np.max(np.abs(env.robot.command_acceleration)) <= 1 + 1e-9
        v = env.robot_state().joint_vel
        peak_v = max(peak_v, np.max(np.abs(v)))
        peak_a = max(peak_a, np.max(np.abs(v - previous_v)) / env.dt)
        previous_v = v.copy()
    assert not env.robot_state().protective_stop, env.robot_state().stop_reason
    assert np.max(np.abs(env.robot_state().joint_pos - goal)) < 0.01
    assert np.linalg.norm(env.robot_state().joint_pos - initial) > 0.02
    assert peak_v < 1 and peak_a < 4
    assert receipt["time_dilation"] > 1


def test_invalid_replacement_cancels_motion_and_has_no_cartesian_fallback():
    env = make_env()
    q = env.robot_state().joint_pos
    env.robot.submit(q + [0, 0, 0, 0, 0, 0, 0.02], 0.08, 1 / 15)
    bad = q.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="joint"):
        env.robot.submit(bad, 0.08, 1 / 15)
    assert not env.robot.moving
    with pytest.raises(RuntimeError, match="fallback"):
        env.robot.set_tcp_target([0.5, 0, 0.2])
    with pytest.raises(ValueError, match="workspace"):
        env.robot.submit(np.array([0, 1.7, 0, -1.57, 0, 1.57, 0]), 0.08, 1 / 15)


def test_watchdog_latches_and_hold_interrupts_a_target():
    now = [0.0]
    env = make_env(clock=lambda: now[0])
    q = env.robot_state().joint_pos
    env.robot.submit(q + [0, 0, 0, 0, 0, 0, 0.03], 0.08, 1 / 15)
    now[0] = 6
    env.step()
    assert env.robot_state().protective_stop and "watchdog" in env.robot_state().stop_reason
    assert not env.robot.moving
    env = make_env()
    env.robot.submit(q + [0, 0, 0, 0, 0, 0, 0.03], 0.08, 1 / 15)
    env.step(40)
    env.robot.hold()
    held = env.robot_state().joint_pos.copy()
    env.step(100)
    assert not env.robot.moving
    assert np.max(np.abs(env.robot_state().joint_pos - held)) < 0.01


def test_execution_cancel_during_motion_holds_and_reports_partial_displacement():
    import threading

    from dream.eval.joint import execute_target

    env = make_env()
    cancel = threading.Event()
    env.step_hook = lambda *args: cancel.set() if env.time > 0.6 else None
    q = env.robot_state().joint_pos.copy()
    target = q + [0, 0, 0, 0, 0, 0, 0.1]
    result = execute_target(env, target, 0.08, 1 / 15, cancel)
    assert result["executed"] and not result["target_reached"]
    assert result["reason"] == "cancelled" and not env.robot.moving
    assert abs(env.robot_state().joint_pos[6] - q[6]) < 0.05


def test_measured_limits_are_enforced_while_holding():
    env = make_env()
    # Test setup injects a measured robot velocity before the controller acts.
    env.data.qvel[env.robot.dadr[0]] = 1.1
    env.step()
    assert env.robot.protective_stop and "velocity" in env.robot.stop_reason


def test_joint_hold_remains_safe_while_real_physics_continues_during_inference():
    import time

    from dream.session import LiveSession

    session = LiveSession(seed=222, perception="vision", rate=1)
    session.env.robot = JointTargetController(session.env.model, session.env.data)
    session.env.robot.hold()
    before = session.env.robot_state()
    session.start()
    try:
        time.sleep(0.2)  # stand-in latency only; this test does not claim model inference
        after = session.env.robot_state()
        assert after.time - before.time > 0.1
        assert np.max(np.abs(after.joint_pos - before.joint_pos)) < 0.002
        assert not after.protective_stop
    finally:
        session.close()
