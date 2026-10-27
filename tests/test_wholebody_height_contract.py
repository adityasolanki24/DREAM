"""Height command contract without simulation or pretrained weights."""

from unittest.mock import Mock

import numpy as np
import pytest

from dream.robot.g1_wbc import WBCLocomotion
from dream.robot.wholebody import (
    ControllerCapabilities,
    PostureCommand,
    WBCBackend,
    WholeBodyTarget,
    synchronized_path,
)

CAPS = ControllerCapabilities(
    "test",
    frozenset({"posture"}),
    (0.3, 0.74),
    (-0.3,) * 3,
    (0.3,) * 3,
    measured=True,
    source="test",
)
INCIDENT = 0.7400000107504702


def target(height):
    return WholeBodyTarget(np.zeros(36), (0, 0, 0), (), height, (0, 0, 0), np.zeros(14))


@pytest.mark.parametrize("height", [0.3, 0.74, 0.5])
def test_measured_height_bounds_are_inclusive(height):
    assert not CAPS.violations(height, (0, 0, 0))


@pytest.mark.parametrize("height", [0.74 + 1e-8, INCIDENT, 0.3 - 1e-8])
def test_outside_measured_height_is_rejected(height):
    assert CAPS.violations(height, (0, 0, 0))


@pytest.mark.parametrize("height", [0.74 + 1e-8, INCIDENT, 0.3 - 1e-8])
def test_backend_refuses_outside_height_before_any_actuation(height):
    plant = Mock()
    controller = Mock()
    controller.height, controller.rpy, controller.arm_target = 0.74, np.zeros(3), np.zeros(14)
    backend = WBCBackend(plant, controller, CAPS)
    backend._arm_feedforward = Mock(side_effect=AssertionError("feedforward invoked"))
    backend._run = Mock(side_effect=AssertionError("trajectory invoked"))
    report = backend.execute(target(height))
    assert not report.completed and report.refused
    backend._arm_feedforward.assert_not_called()
    backend._run.assert_not_called()
    plant.assert_not_called()
    assert not plant.mock_calls and not controller.mock_calls


@pytest.mark.parametrize("start,goal", [(0.3, 0.74), (0.74, 0.3), (0.74, 0.74), (0.5, 0.74)])
def test_interpolation_stays_within_legal_height_bounds(start, goal):
    command = lambda h: PostureCommand(h, (0, 0, 0), np.zeros(14))  # noqa: E731
    path = synchronized_path(command(start), command(goal), CAPS)
    assert path[-1].height == goal
    assert all(0.3 <= c.height <= 0.74 for c in path)


def test_wbc_setter_rejects_incident_height_without_weights():
    controller = WBCLocomotion.__new__(WBCLocomotion)
    with pytest.raises(ValueError, match="height command"):
        controller.set_posture(INCIDENT)
