"""The tool outcome comes from robot sensing and safety, independently of scoring."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from dream.tools.humanoid import HumanoidToolbox
from dream.types import DexterousState


class _Robot:
    def __init__(self):
        self.observation = DexterousState(
            0.0,
            ("joint",),
            np.zeros(1),
            np.zeros(1),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            False,
        )

    def state(self):
        return self.observation

    def hold(self, reason):
        self.observation = replace(self.observation, protective_stop=True, stop_reason=reason)

    def finger_forces(self, side):
        return {"pad": 0.5 if side == "left" else 1.0}


@pytest.mark.parametrize(("observed", "status"), [(True, "succeeded"), (False, "failed")])
def test_grasp_reports_sensed_outcome_once(observed, status):
    robot = _Robot()

    def grasp():
        robot.observation = replace(robot.observation, time=4.0)
        return observed

    toolbox = HumanoidToolbox(robot, grasp, Event())
    assert toolbox.call("grasp_tabletop", {"seed": 1}).status == "rejected"
    assert robot.state().time == 0.0
    result = toolbox.call("grasp_tabletop")
    assert result.status == status
    assert result.data == {"grasp_observed": observed}
    assert result.sim_time == 4.0
    assert toolbox.call("grasp_tabletop").status == "rejected"


def test_cancelled_tool_does_not_move_robot():
    robot, cancel = _Robot(), Event()
    toolbox = HumanoidToolbox(robot, lambda: pytest.fail("grasp was called"), cancel)
    cancel.set()
    assert toolbox.call("grasp_tabletop").status == "stopped"
    state = toolbox.call("get_robot_state").data
    assert state["stop_reason"] == "cancelled"
    assert state["finger_forces_n"] == {"left": {"pad": 0.5}, "right": {"pad": 1.0}}


def test_protective_stop_overrides_positive_sensor_result():
    robot = _Robot()

    def grasp():
        robot.hold("torque limit")
        return True

    result = HumanoidToolbox(robot, grasp, Event()).call("grasp_tabletop")
    assert result.status == "stopped"
    assert result.message == "torque limit"
