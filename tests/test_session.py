"""Live session: background physics, goal submission, pause/resume/cancel, logs."""

import time

import numpy as np
import pytest

from dream.eval.manipulation import object_in_tray
from dream.session import LiveSession


@pytest.fixture
def session(tmp_path):
    # rate=4: realistic for an interactive session. In max-speed mode (rate=0) the physics thread
    # starves the agent thread. One unexplained stall was seen in 20 such runs (README limitations).
    s = LiveSession(seed=11, rate=4, log_dir=tmp_path)
    s.start()
    yield s
    s.close()


def test_physics_keeps_running_while_idle_and_rejects_natural_language(session):
    t0 = session.env.time
    time.sleep(0.3)
    assert session.env.time > t0  # the control loop runs without any goal
    assert "not a structured goal" in session.submit("please put the red cube in the tray")
    assert not session.busy


def test_cancel_interrupts_and_robot_holds_still(session):
    assert session.submit("in red tray") == ""
    deadline = time.time() + 20
    while session.status.current_action is None and time.time() < deadline:
        time.sleep(0.01)
    session.cancel()
    assert session.status.state == "cancelled" and not session.busy
    q = session.env.robot_state().joint_pos
    time.sleep(0.5)
    assert np.max(np.abs(session.env.robot_state().joint_pos - q)) < 0.02


@pytest.mark.slow
def test_pause_resume_completes_goal_and_writes_logs(session, tmp_path):
    session.submit("in red tray")
    time.sleep(1.0)
    session.pause()
    assert session.status.state == "paused"
    assert session.resume() == ""
    assert session.wait(300)
    assert session.status.state == "done"
    time.sleep(0.3)
    assert object_in_tray(session.env, "red")
    session.close()
    assert (tmp_path / "episodes.jsonl").exists() and (tmp_path / "trajectory.json").exists()


@pytest.mark.slow
def test_compartment_session_pauses_mid_investigation_holds_and_resumes(tmp_path):
    from dream.eval.manipulation import object_in_compartment

    s = LiveSession(seed=3, rate=4, log_dir=tmp_path, scene="drawer")
    s.start()
    try:
        assert s.submit("in red compartment") == ""
        deadline = time.time() + 300
        while (s.status.discovery is None) and time.time() < deadline:
            time.sleep(0.05)
        assert s.status.discovery is not None  # probe 1 reported with probabilities
        assert set(s.status.discovery["type_probs"]) == {"fixed", "prismatic", "revolute"}
        s.pause()
        q = s.env.robot_state().joint_pos
        time.sleep(0.5)
        assert np.max(np.abs(s.env.robot_state().joint_pos - q)) < 0.02  # safe hold
        assert s.resume() == ""
        assert s.wait(600)
        assert s.status.state == "done"
        time.sleep(0.3)
        assert object_in_compartment(s.env, "red")
    finally:
        s.close()
    log = (tmp_path / "trajectory.json").read_text()
    assert '"scene": "drawer"' in log and '"discovery"' in log
