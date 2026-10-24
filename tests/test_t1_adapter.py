"""Boundary tests with simulator doubles, never locomotion validation."""

import ast
import inspect
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from dream.sim.isaac_t1 import IsaacT1


def test_physics_step_reports_and_stops_on_new_tilt(monkeypatch):
    """A fall during decimation must not be reported as the previous upright sample."""
    body = object.__new__(IsaacT1)
    world = SimpleNamespace(current_time_step_index=0)
    world.step = lambda **kw: setattr(
        world, "current_time_step_index", world.current_time_step_index + 1
    )
    body.world = world
    body.art = SimpleNamespace(apply_action=lambda action: None)
    body.actor = SimpleNamespace(
        target=lambda *args: np.zeros(23), last=np.zeros(23), target_limit_excess=np.zeros(23)
    )
    body.command, body.odom = np.zeros(3), np.zeros(3)
    body.time, body.stopped = 0.0, False
    body.limits = np.tile([-2, 2], (23, 1))

    def sensors():
        return {
            "q": np.zeros(23),
            "dq": np.zeros(23),
            "linear": np.zeros(3),
            "gyro": np.zeros(3),
            "gravity": np.array([0, 0, -1]),
            "tilt": 0.6 if world.current_time_step_index else 0.0,
            "orientation_wxyz": np.array([1, 0, 0, 0]),
        }

    body.sensors = sensors
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.core.utils.types",
        SimpleNamespace(ArticulationAction=lambda **kw: kw),
    )
    row = body.advance(0, 0)
    assert row["tilt_rad"] == 0.6 and row["stopped"]
    assert row["physics_step_count"] == 4 and row["state_time_s"] == 0.02
    assert row["proprioception_time_s"] == 0.0


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
