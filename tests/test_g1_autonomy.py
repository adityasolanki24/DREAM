"""Reasoning pauses and recovery must preserve measured contact, not erase it."""

from types import SimpleNamespace

import numpy as np

from dream.perception.sphere import SphereEstimate
from dream.sim.g1 import G1Env
from dream.tools.object_exploration import ExplorationToolbox


def test_reasoning_hold_stops_arm_motion_without_resetting_finger_setpoints():
    env = G1Env()
    robot = env.robot
    robot.set_hand("left", 0.5)
    robot.move_arm("left", np.array([0, 0, 0, 0.1, 0, 0, 0]))
    for _ in range(100):
        env.step()
    hand = np.array(["_hand_" in name for name in robot.names])
    target, reference = robot.target.copy(), robot.reference.copy()
    q = robot.state().joint_pos
    robot.hold_for_reasoning()
    np.testing.assert_array_equal(robot.target[hand], target[hand])
    np.testing.assert_array_equal(robot.reference[hand], reference[hand])
    np.testing.assert_array_equal(robot.target[~hand], q[~hand])
    assert robot.cartesian_target is None


def test_inspection_does_not_move_or_open_when_a_fresh_object_is_visible():
    obj = SphereEstimate(0, np.array([0.37, 0.11, 0.92]), np.eye(3) * 4e-6, 0.025, 0, 100)
    robot = SimpleNamespace(grasp_pose=lambda side: (obj.position, np.array([1, 0, 0, 0])))
    toolbox = ExplorationToolbox(robot, lambda: None, SimpleNamespace(observe=lambda: obj))
    toolbox._open = lambda: (_ for _ in ()).throw(AssertionError("visible object opened"))
    toolbox._move = lambda point: (_ for _ in ()).throw(AssertionError("visible object moved"))
    assert toolbox._inspect_object("red").ok


def test_tracking_loss_during_inspection_preserves_grasp_and_reacquires():
    obj = SphereEstimate(0, np.array([0.37, 0.11, 0.92]), np.eye(3) * 4e-6, 0.025, 0, 100)
    observations = iter([None, obj])
    robot = SimpleNamespace(
        grasp_pose=lambda side: (obj.position, np.array([1, 0, 0, 0])),
        finger_forces=lambda side: {"thumb": 1.0, "middle": 1.0},
    )
    toolbox = ExplorationToolbox(
        robot, lambda: None, SimpleNamespace(observe=lambda: next(observations))
    )
    toolbox.holding = "red"
    toolbox.closure = 0.4
    moves = []
    toolbox._move = lambda point: moves.append(point)
    toolbox._open = lambda: (_ for _ in ()).throw(AssertionError("lost held object released"))
    assert toolbox._inspect_object("red").ok
    assert toolbox.holding == "red" and toolbox.closure == 0.4
    np.testing.assert_allclose(moves, [obj.position + [0, 0, 0.035]])


def test_observed_slip_clears_stale_grasp_and_placement_state():
    obj = SphereEstimate(0, np.array([0.37, 0.11, 0.92]), np.eye(3) * 4e-6, 0.025, 0, 100)
    robot = SimpleNamespace(
        grasp_pose=lambda side: (obj.position + [0, 0, 0.05], np.array([1, 0, 0, 0])),
        finger_forces=lambda side: {"thumb": 0.0},
    )
    perception = SimpleNamespace(observe=lambda: obj, target=None)
    toolbox = ExplorationToolbox(robot, lambda: None, perception)
    toolbox.holding, toolbox.at_target = "red", True
    scene = toolbox._observe_scene().data
    assert scene["holding"] is None and not scene["at_target"]
    assert scene["grasp_observation"] == "object no longer tracks hand"
