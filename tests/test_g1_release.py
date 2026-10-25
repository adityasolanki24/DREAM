"""Release height must follow measured object geometry and pose uncertainty."""

from types import SimpleNamespace

import numpy as np
import pytest

from dream.tools.object_exploration import ActionStopped, ExplorationToolbox


def test_release_lowers_by_measured_radius_and_three_sigma_before_unloading():
    events = []
    robot = SimpleNamespace(
        grasp_pose=lambda side: (np.array([0.4, 0.26, 0.95]), np.array([1, 0, 0, 0])),
        finger_forces=lambda side: {"thumb": 0.0, "index": 0.0, "middle": 0.0},
        hold_for_reasoning=lambda: events.append("hold_preserving_fingers"),
        set_hand=lambda *args: events.append("unload"),
    )
    toolbox = ExplorationToolbox(robot, lambda: None, SimpleNamespace(target=(np.zeros(3), 0.065)))
    toolbox.holding = "red"
    toolbox._object = lambda: SimpleNamespace(radius=0.021, covariance=np.eye(3) * 0.002**2)
    targets = []
    toolbox._move = lambda target, **kwargs: targets.append(target.copy())
    toolbox._advance = lambda *args: None
    toolbox._open = lambda: events.append("full_open")
    toolbox._release_object()
    np.testing.assert_allclose(targets[0], [0, 0, 0.027])


@pytest.mark.parametrize("error,preserve", [(ValueError, True), (ActionStopped, False)])
def test_ordinary_skill_failure_preserves_grip_but_safety_stop_does_not(error, preserve):
    events = []
    robot = SimpleNamespace(
        grasp_pose=lambda side: (np.zeros(3), np.array([1, 0, 0, 0])),
        state=lambda: SimpleNamespace(time=0.0, protective_stop=False),
        hold=lambda: events.append("hard"),
        hold_for_reasoning=lambda: events.append("preserve"),
    )
    toolbox = ExplorationToolbox(robot, lambda: None, None)

    def failed(**kwargs):
        raise error("bounded execution failure")

    toolbox._reach_object = failed
    result = toolbox.call("reach_object", {"object": "red"})
    assert not result.ok
    assert events == (["preserve"] if preserve else ["hard"])


def test_two_release_failures_exhaust_recovery_budget():
    from dream.runtime.agent import Finish, Step, ToolCall
    from dream.runtime.exploration import ExplorationPlanner
    from dream.tools.goals import InRegion
    from dream.tools.toolbox import ToolResult

    history = [
        Step(ToolCall("release_object"), ToolResult("release_object", "failed", "timeout"), 0)
    ] * 2
    decision = ExplorationPlanner().next_action(
        InRegion("red", "target"), {"objects": []}, {}, history
    )
    assert isinstance(decision, Finish) and not decision.success


def test_lowering_uses_observed_support_instead_of_requiring_arm_convergence():
    clock = [0.0]
    robot = SimpleNamespace(
        grasp_pose=lambda side: (np.zeros(3), np.array([1, 0, 0, 0])),
        state=lambda: SimpleNamespace(time=clock[0]),
        reach=lambda *args: None,
        hold_for_reasoning=lambda: None,
    )
    centre = np.array([0.4, 0.26, 0.895])
    toolbox = ExplorationToolbox(robot, lambda: None, SimpleNamespace(target=(centre, 0.065)))
    toolbox._object = lambda: SimpleNamespace(
        position=centre + [0, 0, 0.021],
        radius=0.021,
        covariance=np.eye(3) * 0.002**2,
    )
    toolbox._advance = lambda dt: clock.__setitem__(0, clock[0] + dt)
    error = toolbox._move(centre + [0, 0, 0.027], stop_near_support=True)
    assert error > 0.1 and clock[0] < 0.5
