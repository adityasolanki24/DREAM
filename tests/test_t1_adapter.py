"""Boundary tests with simulator doubles, never locomotion validation."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from dream.sim.isaac_t1 import IsaacT1


@pytest.mark.parametrize("lower,upper", [(-1.7, 2.2), (-0.3, 0.8)])
def test_single_articulation_limit_query_uses_its_public_api(lower, upper):
    """Exercise the actual init expression against the documented single-prim boundary."""
    properties = np.zeros(23, dtype=[("lower", float), ("upper", float)])
    properties["lower"], properties["upper"] = lower, upper
    body = SimpleNamespace(art=SimpleNamespace(dof_properties=properties))
    tree = ast.parse(textwrap.dedent(inspect.getsource(IsaacT1.__init__)))
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "limits" for t in node.targets)
    )
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), "init_limits", "exec"),
        {"self": body, "np": np},
    )
    assert body.limits.shape == (23, 2)
    np.testing.assert_equal(body.limits[:, 0], lower)
    np.testing.assert_equal(body.limits[:, 1], upper)


def test_joint_armature_matches_pinned_training_head_exception():
    from dream.sim import isaac_t1

    assert isaac_t1.joint_armature("AAHead_yaw") == 0.005
    assert isaac_t1.joint_armature("Head_pitch") == 0.005
    assert isaac_t1.joint_armature("Left_Ankle_Pitch") == 0.01


def test_rgbd_requires_aligned_nonempty_arrays_and_depth_returns():
    body = object.__new__(IsaacT1)
    body.world = SimpleNamespace(render=lambda: None)
    body.camera = SimpleNamespace(
        get_rgba=lambda: np.zeros((400, 400, 4), np.uint8),
        get_depth=lambda: np.ones((400, 400), np.float32),
    )
    rgb, depth = body.capture()
    assert rgb.shape == (400, 400, 3) and depth.shape == (400, 400)
    for value in (None, np.empty((0, 0)), np.full((400, 400), np.inf)):
        body.camera.get_depth = lambda value=value: value
        with pytest.raises(RuntimeError, match="RGB-D"):
            body.capture()
