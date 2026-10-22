"""Contracts for the model-driven agent (dream/runtime/vlm.py).

``ScriptedBackend`` replays fixed outputs. It is a test double, not a model: these tests show
that parsing, validation, error feedback, success verification and the real toolbox and
physics behave as specified. They are not evidence of any model's ability.
"""

import json

import numpy as np
import pytest

from dream.runtime.vlm import (
    MAX_IMAGES_IN_CONTEXT,
    VLMAgent,
    _bounded_images,
    parse_tool_call,
    tool_schemas,
)
from dream.sim.tabletop import sample_tabletop
from dream.sim.tabletop_env import TabletopEnv
from dream.tools.toolbox import TOOL_SPECS, Toolbox


class ScriptedBackend:
    """Replays outputs in order and records what it was shown (a test double, not a model)."""

    name = "scripted-test-double"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen = []

    def generate(self, messages, tools):
        self.seen.append((messages, tools))
        return self.outputs.pop(0)


def call(name, **args):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"


def _agent(outputs, seed=3, **kw):
    env = TabletopEnv(sample_tabletop(np.random.default_rng(seed)), np.random.default_rng(seed))
    backend = ScriptedBackend(outputs)
    agent = VLMAgent(
        Toolbox(env.embodiment(), env), backend, camera=lambda: env.capture_frame().rgb, **kw
    )
    return env, agent, backend


def test_every_tool_is_offered_as_a_json_schema_function():
    schemas = tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == [t["name"] for t in TOOL_SPECS] + ["finish"]
    for s in schemas:
        json.dumps(s)  # serialisable
        p = s["function"]["parameters"]
        assert p["type"] == "object" and set(p["required"]) <= set(p["properties"])


def test_tool_calls_are_parsed_as_json_only():
    ok, err = parse_tool_call(
        'think...\n<tool_call>\n{"name": "grasp", "arguments": {"object": "red"}}\n</tool_call>'
    )
    assert ok == {"name": "grasp", "arguments": {"object": "red"}} and err == ""
    assert parse_tool_call(
        '<tool_call>{"name": "grasp", "arguments": "{\\"object\\": \\"red\\"}"}</tool_call>'
    )[0]["arguments"] == {"object": "red"}
    for bad in (
        "grasp red",
        "<tool_call>__import__('os').system('ls')</tool_call>",
        '<tool_call>{"arguments": {}}</tool_call>',
        '<tool_call>{"name": "x", "arguments": [1]}</tool_call>',
    ):
        call_, err = parse_tool_call(bad)
        assert call_ is None and err


def test_invalid_outputs_get_bounded_feedback_and_the_run_stops():
    _, agent, backend = _agent(
        ["no call here", call("teleport", object="red"), "{oops"], max_invalid=2
    )
    ep = agent.run("Put the red cube in the tray.")
    assert not ep.success_claimed and "invalid model" in ep.reason
    # one unparseable output, then an unknown tool (rejected, not run): two in a row, stop
    assert ep.invalid_outputs == 1 and ep.rejected_calls == 1 and len(ep.steps) == 2
    feedback = backend.seen[1][0][-1]["content"][-1]["text"]
    assert feedback.startswith("Error:")


def test_a_success_claim_is_rejected_unless_the_world_verifies_it():
    env, agent, _ = _agent(
        [
            call(
                "finish",
                success=True,
                reason="done",
                goal={"type": "in", "object": "red", "target": "tray"},
            ),
            call("finish", success=False, reason="I cannot verify it"),
        ]
    )
    ep = agent.run("Put the red cube in the tray.")
    assert not ep.success_claimed and ep.reason.startswith("model declined")
    assert "success rejected" in ep.steps[0].error


def test_the_model_sees_the_camera_image_and_the_tool_results():
    env, agent, backend = _agent(
        [call("get_robot_state"), call("finish", success=False, reason="test")]
    )
    agent.run("Describe the robot state, then stop.")
    first, _ = backend.seen[0]
    images = [p["image"] for m in first for p in m["content"] if p["type"] == "image"]
    assert len(images) == 1 and images[0].shape == (360, 480, 3) and images[0].dtype == np.uint8
    second, tools = backend.seen[1]
    assert "get_robot_state result" in second[-1]["content"][-1]["text"]
    assert {t["function"]["name"] for t in tools} >= {"grasp", "place", "investigate_physics"}


def test_old_images_are_dropped_from_the_context():
    img = np.zeros((2, 2, 3), np.uint8)
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": str(i)}],
        }
        for i in range(5)
    ]
    kept = sum(p["type"] == "image" for m in _bounded_images(msgs) for p in m["content"])
    assert kept == MAX_IMAGES_IN_CONTEXT


@pytest.mark.slow
def test_scripted_calls_run_through_real_physics_and_success_is_verified():
    from dream.eval.manipulation import object_in_tray

    env, agent, _ = _agent(
        [
            call("observe_scene"),
            call("grasp", object="red"),
            call("place", object="red", target="tray"),
            call("verify_goal", goal={"type": "in", "object": "red", "target": "tray"}),
            call(
                "finish",
                success=True,
                reason="red is in the tray",
                goal={"type": "in", "object": "red", "target": "tray"},
            ),
        ]
    )
    ep = agent.run("Put the red cube in the tray.")
    env.step(250)
    assert ep.success_claimed and object_in_tray(env, "red")
    assert ep.tool_calls == ["observe_scene", "grasp", "place", "verify_goal", "finish"]
