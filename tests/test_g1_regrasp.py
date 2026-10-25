"""A recovered grasp must establish a new placement, not reuse an earlier one."""

from types import SimpleNamespace

import numpy as np

from dream.perception.sphere import SphereEstimate
from dream.tools.object_exploration import ExplorationToolbox


def test_verified_regrasp_invalidates_previous_placement_state():
    def observed(z):
        return SphereEstimate(0, np.array([0.37, 0.11, z]), np.eye(3) * 4e-6, 0.025, 0, 100)

    observations = iter([observed(0.92), observed(0.955)])
    robot = SimpleNamespace(
        grasp_pose=lambda side: (observed(0.955).position, np.array([1, 0, 0, 0]))
    )
    toolbox = ExplorationToolbox(
        robot, lambda: None, SimpleNamespace(observe=lambda: next(observations))
    )
    toolbox.belief = SimpleNamespace(p_movable=0.99, inadequate=False)
    toolbox.at_target, toolbox.released = True, True
    toolbox._close = lambda: True
    toolbox._move = lambda point: None
    toolbox._advance = lambda duration: None
    assert toolbox._grasp_object("red").ok
    assert toolbox.holding == "red"
    assert not toolbox.at_target and not toolbox.released
