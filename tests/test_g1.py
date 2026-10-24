"""Real CPU physics contracts for the supported G1; no network or policy weights."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from dream.sim.g1 import G1Env


def test_both_hands_move_within_ranges_and_upstream_effort_limits():
    env = G1Env()
    start = env.robot.state().joint_pos.copy()
    for side in ("left", "right"):
        env.robot.set_hand(side, 1)
    for _ in range(1600):
        env.step()
        assert not env.robot.state().protective_stop, env.robot.stop_reason
        assert np.all(env.data.qfrc_actuator[env.robot.dadr] <= env.robot.effort[:, 1] + 1e-9)
        assert np.all(env.data.qfrc_actuator[env.robot.dadr] >= env.robot.effort[:, 0] - 1e-9)
    for side in ("left", "right"):
        ids = [i for i, n in enumerate(env.robot.names) if n.startswith(side + "_hand_")]
        assert np.count_nonzero(abs(env.robot.state().joint_pos[ids] - start[ids]) > 0.3) >= 5


@pytest.mark.slow
def test_runner_accepts_default_relative_config(tmp_path):
    from experiments.g1_dexterous import run

    cfg = Path("experiments/configs/g1_dex3_dev.json")
    # Run the real unrendered pipeline, retaining every development case.
    results = run(cfg, tmp_path / "run")
    assert set(results) == {"nominal", "smaller", "shifted"}
    assert all(r["physics_executed"] for r in results.values())
    assert json.loads((tmp_path / "run/metadata.json").read_text())["source_sha256"]


def test_names_map_to_correct_joints_even_with_different_right_finger_order():
    env = G1Env()
    robot = env.robot
    for i, name in enumerate(robot.names):
        assert env.model.joint(name).id == robot.joints[i]
        assert env.model.joint(name).qposadr[0] == robot.qadr[i]
    assert tuple(robot.qadr) != tuple(range(43))


def test_targets_reject_atomically_and_state_does_not_alias_physics():
    env = G1Env()
    robot = env.robot
    before = robot.target.copy()
    with pytest.raises(ValueError):
        robot.command_joints({robot.names[0]: 0.5, robot.names[1]: float("nan")})
    np.testing.assert_array_equal(before, robot.target)
    for side, closure in (("wrong", 0.5), ("left", 1.1), ("right", float("nan"))):
        with pytest.raises(ValueError):
            robot.set_hand(side, closure)
    observed = robot.state()
    observed.joint_pos[:] = 999
    assert np.max(abs(robot.state().joint_pos)) < 1


def test_reset_reproduces_actual_dynamics_and_latched_stop():
    env = G1Env()
    outcomes = []
    for _ in range(2):
        env.reset()
        env.robot.set_hand("right", 0.5)
        for _ in range(100):
            env.step()
        outcomes.append(env.data.qpos.copy())
    np.testing.assert_array_equal(*outcomes)
    env.data.qvel[env.robot.dadr[0]] = 21
    env.step()
    assert env.robot.state().protective_stop
    # A protective hold must not command all position servos back to zero.
    np.testing.assert_allclose(
        env.data.ctrl - env.data.qfrc_bias[env.robot.dadr] / env.robot.kp,
        env.robot.state().joint_pos,
        atol=1e-15,  # float64 cancellation in the equivalent gravity-offset reference
    )
    with pytest.raises(ValueError):
        env.robot.set_hand("left", 1)


def test_adapter_rejects_incompatible_motor_convention():
    from dream.robot.g1 import G1Controller

    env = G1Env()
    env.model.actuator_biasprm[0, 2] = 0  # raw motor / different damping contract
    with pytest.raises(ValueError, match="servo"):
        G1Controller(env.model, env.data)


def test_tools_reject_unsupported_grasp_and_honor_mid_execution_cancel():
    from dream.tools.dexterous import DexterousToolbox

    env = G1Env()
    cancel = threading.Event()

    def step():
        env.step()
        if env.data.time >= 0.01:
            cancel.set()

    tools = DexterousToolbox(env.robot, step, cancel)
    assert tools.call("grasp_object", {"object": "sphere"}).status == "rejected"
    assert tools.call("close_hand", {"side": "left", "extra": 1}).status == "rejected"
    result = tools.call("close_hand", {"side": "left"})
    assert result.status == "stopped" and result.sim_time < 0.02
    assert tools.call("get_robot_state").ok


def test_existing_vlm_boundary_can_dispatch_g1_tools_without_claiming_grasp():
    from dream.runtime.vlm import VLMAgent
    from dream.tools.dexterous import DexterousToolbox

    class Backend:
        name = "scripted plumbing test double; not model inference"

        def __init__(self):
            self.outputs = iter(
                [
                    '<tool_call>{"name":"close_hand","arguments":{"side":"left"}}</tool_call>',
                    '<tool_call>{"name":"finish","arguments":'
                    '{"success":false,"reason":"grasp unverified"}}</tool_call>',
                ]
            )

        def generate(self, messages, tools):
            assert "Unitree G1" in messages[0]["content"][0]["text"]
            assert "close_hand" in [t["function"]["name"] for t in tools]
            assert "grasp" not in [t["function"]["name"] for t in tools]
            return next(self.outputs)

    env = G1Env()
    episode = VLMAgent(DexterousToolbox(env.robot, env.step), Backend()).run("Close left hand")
    assert not episode.success_claimed
    result = episode.steps[0].result
    assert result.ok and result.data["max_motion_rad"] > 0.3
    assert not result.data["object_grasp_verified"]


def test_grasp_scoring_requires_unsupported_lift_and_two_distinct_fingers():
    from dream.eval.g1 import score_episode

    gates = json.loads(Path("experiments/configs/g1_dex3_dev.json").read_text())["gates"]
    base = {
        "phase": "open",
        "time_s": 0,
        "object_position_m": [0, 0, 0.9],
        "contacts": [],
        "stop_reason": "",
    }
    thumb = {"body": "left_hand_thumb_2_link", "normal_force_n": 1}
    index = {"body": "left_hand_index_1_link", "normal_force_n": 1}
    hold = {
        **base,
        "phase": "hold",
        "time_s": 1,
        "object_position_m": [0, 0, 0.95],
        "contacts": [thumb, index],
    }
    assert score_episode([base, hold], gates)["grasp_success"]
    for contacts in (
        [thumb],
        [thumb, thumb],
        [thumb, index, {"body": "world", "normal_force_n": 1}],
    ):
        assert not score_episode([base, {**hold, "contacts": contacts}], gates)["grasp_success"]
    assert not score_episode([base, {**hold, "object_position_m": [0, 0, 0.9]}], gates)[
        "grasp_success"
    ]
