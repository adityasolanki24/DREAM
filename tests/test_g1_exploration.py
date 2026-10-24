"""Contracts use actual RGB-D and physics where stated; no learned inference is implied."""

import json
from pathlib import Path

import numpy as np
import pytest

from dream.perception.sphere import detect_sphere, detect_target
from dream.sim.camera import CameraSpec, RGBDCamera
from dream.sim.g1 import G1Env, build_g1


def test_probe_measures_resistance_instead_of_requiring_successful_target_tracking():
    """An obstructed exploration command is evidence, not an ordinary reach failure."""
    from dream.perception.sphere import SphereEstimate
    from dream.tools.object_exploration import ExplorationToolbox

    class Robot:
        def grasp_pose(self, side):
            return np.array([0.37, 0.11, 0.92]), np.array([1.0, 0, 0, 0])

        def finger_forces(self, side):
            return {"thumb": 1.0, "middle": 1.0, "index": 0.0}

        def reach(self, *args):
            pass

    toolbox = ExplorationToolbox(Robot(), lambda: None, None)
    estimate = SphereEstimate(
        0, np.array([0.37, 0.11, 0.92]), np.eye(3) * 0.002**2, 0.025, 0.001, 100
    )
    toolbox._object = lambda: estimate
    toolbox._close = lambda: True
    toolbox._advance = lambda duration: None
    toolbox._move = lambda *args: (_ for _ in ()).throw(ValueError("obstructed reach"))
    result = toolbox._probe_object("red")
    assert result.ok
    assert result.data["evidence"]["measured_displacement_m"] == 0
    assert toolbox.belief.p_movable < 0.02


def test_development_scene_starts_without_finger_fixture_interpenetration():
    cfg = json.loads(Path("experiments/configs/g1_explore_dev.json").read_text())
    scene = cfg["cases"][0]
    env = G1Env(
        build_g1(
            tuple(scene["position_m"]),
            camera=CameraSpec(**cfg["camera"]),
            contact_timeconst=cfg["contact_timeconst_s"],
            target_position=tuple(cfg["target_position_m"]),
        )
    )
    assert sum(env.robot.finger_forces("left").values()) == 0


@pytest.mark.slow
def test_blue_target_detection_rejects_blue_floor_and_sphere_pose_comes_from_pixels():
    camera_spec = CameraSpec(
        position=(0.85, -0.45, 1.35), target=(0.36, 0.1, 0.91), width=320, height=240
    )
    pos = np.array([0.37, 0.11, 0.92])
    target = np.array([0.4, 0.26, 0.895])
    env = G1Env(build_g1(tuple(pos), camera=camera_spec, target_position=tuple(target)))
    camera = RGBDCamera(env.model, camera_spec, np.random.default_rng(35))
    try:
        frame = camera.capture(env.data)
        sphere = detect_sphere(frame)
        fixture = detect_target(frame)
        assert sphere is not None and np.linalg.norm(sphere.position - pos) < 0.006
        assert fixture is not None and np.linalg.norm(fixture[0] - target) < 0.008
    finally:
        camera.close()


@pytest.mark.slow
@pytest.mark.parametrize(
    "position,radius", [((0.35, 0.09, 0.915), 0.02), ((0.39, 0.125, 0.93), 0.028)]
)
def test_actual_arm_reaches_two_independent_sensor_derived_locations(position, radius):
    camera_spec = CameraSpec(
        position=(0.85, -0.45, 1.35), target=(0.36, 0.1, 0.91), width=320, height=240
    )
    env = G1Env(build_g1(position, radius, camera=camera_spec, contact_timeconst=0.005))
    camera = RGBDCamera(env.model, camera_spec, np.random.default_rng(52))
    try:
        estimate = detect_sphere(camera.capture(env.data))
        assert estimate is not None
        initial = np.linalg.norm(env.robot.grasp_pose("left")[0] - estimate.position)
        env.robot.reach("left", estimate.position, env.robot.grasp_pose("left")[1])
        for _ in range(2500):
            env.step()
            assert not env.robot.state().protective_stop
        final = np.linalg.norm(env.robot.grasp_pose("left")[0] - estimate.position)
        assert final < 0.006 and final < initial / 3
        np.testing.assert_array_less(env.robot.target, env.robot.limits[:, 1] + 1e-12)
    finally:
        camera.close()


def test_belief_eig_baseline_contact_validity_and_inadequacy():
    from dream.agent.movability import MovabilityBelief

    belief = MovabilityBelief()
    short = belief.select(0.005, "fixed_short")
    adaptive = belief.select(0.005, "eig")
    assert adaptive["candidates"] == short["candidates"]
    assert adaptive["distance_m"] == 0.025 and short["distance_m"] == 0.012
    belief.update(0, 0.025, 0.005, False)
    assert belief.p_movable == 0.5  # a miss is not evidence of immobility
    belief.update(0.024, 0.025, 0.005, True)
    assert belief.p_movable > 0.95
    fixed = MovabilityBelief()
    fixed.update(0, 0.025, 0.005, True)
    assert fixed.p_movable < 0.02
    unknown = MovabilityBelief()
    unknown.update(-0.03, 0.025, 0.005, True)
    assert unknown.inadequate and unknown.p_movable == 0.5


def test_failed_grasp_replans_inspection_instead_of_blind_retry():
    from dream.runtime.agent import Step, ToolCall
    from dream.runtime.exploration import ExplorationPlanner
    from dream.tools.goals import InRegion
    from dream.tools.toolbox import ToolResult

    scene = {
        "objects": [{"name": "red"}],
        "belief": {"model_inadequate": False, "p_movable": 0.99},
        "holding": None,
        "reached": True,
    }
    history = [
        Step(
            ToolCall("grasp_object", {"object": "red"}),
            ToolResult("grasp_object", "failed", "lift unverified"),
            0,
        )
    ]
    action = ExplorationPlanner().next_action(InRegion("red", "target"), scene, {}, history)
    assert action.name == "inspect_object"


def test_established_contact_does_not_chase_the_nominal_tool_frame():
    from dream.runtime.exploration import ExplorationPlanner
    from dream.tools.goals import InRegion

    scene = {
        "objects": [{"name": "red"}],
        "belief": {"model_inadequate": False, "p_movable": 0.99, "probes": 1},
        "holding": None,
        "reached": False,
        "hand_closed": True,
        "in_contact": True,
    }
    action = ExplorationPlanner().next_action(InRegion("red", "target"), scene, {}, [])
    assert action.name == "grasp_object"


def test_release_unloads_contact_and_withdraws_before_full_finger_sweep():
    from types import SimpleNamespace

    from dream.perception.sphere import SphereEstimate
    from dream.tools.object_exploration import ExplorationToolbox

    events = []

    class Robot:
        position = np.array([0.4, 0.26, 0.96])
        closure = 0.5

        def grasp_pose(self, side):
            return self.position.copy(), np.array([1.0, 0, 0, 0])

        def set_hand(self, side, closure):
            self.closure = closure
            events.append(("hand", closure))

        def finger_forces(self, side):
            return {"thumb": float(self.closure > 0.3), "index": float(self.closure > 0.3)}

    robot = Robot()
    tools = ExplorationToolbox(
        robot, lambda: None, SimpleNamespace(target=(np.array([0.4, 0.26, 0.895]), 0.065))
    )
    tools.holding, tools.closure = "red", 0.5
    tools._object = lambda: SphereEstimate(
        0, robot.position, np.eye(3) * 0.002**2, 0.025, 0.001, 100
    )
    tools._advance = lambda duration: None

    def move(point, **kwargs):
        robot.position = point.copy()
        events.append(("move", point.copy()))
        return 0

    tools._move = move
    tools._release_object()
    first_full_open = next(
        i for i, (kind, value) in enumerate(events) if kind == "hand" and value == 0
    )
    assert any(
        kind == "move" and value[2] > 0.995 and abs(value[0] - 0.4) < 0.005
        for kind, value in events[:first_full_open]
    )


def test_cartesian_targets_are_bounded_and_no_object_pose_enters_ik():
    env = G1Env()
    before = env.robot.target.copy()
    with pytest.raises(ValueError):
        env.robot.reach("left", np.array([0.9, 0.1, 0.9]), np.array([1.0, 0, 0, 0]))
    np.testing.assert_array_equal(before, env.robot.target)


def test_vlm_exploration_schemas_and_finish_cannot_certify_an_unseen_object():
    from types import SimpleNamespace

    from dream.runtime.vlm import VLMAgent
    from dream.tools.goals import InRegion
    from dream.tools.object_exploration import ExplorationToolbox

    class Backend:
        name = "scripted boundary test, NOT genuine model inference"

        def __init__(self):
            self.outputs = iter(
                [
                    '<tool_call>{"name":"finish","arguments":{"success":true,"reason":"done",'
                    '"goal":{"type":"in","object":"red","target":"target"}}}</tool_call>',
                    '<tool_call>{"name":"finish","arguments":{"success":false,'
                    '"reason":"object not observed"}}</tool_call>',
                ]
            )

        def generate(self, messages, tools):
            names = [t["function"]["name"] for t in tools]
            assert "reach_object" in names and "probe_object" in names
            assert "close_hand" not in names and "move_arm" not in names
            return next(self.outputs)

    env = G1Env()
    perception = SimpleNamespace(observe=lambda: None, target=None)
    tools = ExplorationToolbox(env.robot, env.step, perception)
    episode = VLMAgent(tools, Backend()).run(
        "Investigate and transfer red", expected_goal=InRegion("red", "target")
    )
    assert not episode.success_claimed and "not verified" in episode.steps[0].error
