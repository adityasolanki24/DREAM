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
    MODEL_EXCLUDED_TOOLS,
    VLMAgent,
    _bounded_images,
    parse_tool_call,
    tool_schemas,
)
from dream.sim.tabletop import sample_tabletop
from dream.sim.tabletop_env import TabletopEnv
from dream.tools.toolbox import TOOL_SPECS, Toolbox


def test_multiple_calls_are_rejected_without_executing_a_prefix():
    assert parse_tool_call(call("grasp", object="red") + call("retract"))[0] is None


def test_model_coordinates_and_benchmark_opening_strategy_are_not_dispatched(monkeypatch):
    _, agent, _ = _agent(
        [
            call("move_end_effector", position=[0.5, 0.0, 0.1]),
            call("open_articulated", object="compartment", strategy="fixed_pull"),
            call("finish", success=False, reason="unsupported"),
        ]
    )
    original = agent.tools.call
    dispatched = []

    def tools(name, args=None):
        dispatched.append(name)
        return original(name, args)

    monkeypatch.setattr(agent.tools, "call", tools)
    ep = agent.run("Put red in the container")
    assert ep.rejected_calls == 2
    assert "move_end_effector" not in dispatched and "open_articulated" not in dispatched
    assert "move_end_effector" not in [s["function"]["name"] for s in tool_schemas()]


def test_a_success_claim_cannot_substitute_an_easier_goal(monkeypatch):
    from dream.tools.goals import parse_goal
    from dream.tools.toolbox import ToolResult

    _, agent, _ = _agent(
        [
            call(
                "finish",
                success=True,
                reason="done",
                goal={"type": "in", "object": "blue", "target": "tray"},
            ),
            call("finish", success=False, reason="wrong object"),
        ]
    )
    original = agent.tools.call
    verified = []

    def tools(name, args=None):
        if name == "verify_goal":
            verified.append(args)
            return ToolResult(name, "succeeded", data={"satisfied": True})
        return original(name, args)

    monkeypatch.setattr(agent.tools, "call", tools)
    ep = agent.run("Put red in the tray", expected_goal=parse_goal("in red tray"))
    assert not ep.success_claimed and "different" in ep.steps[0].error
    assert not verified  # substituted objective never reaches the verifier


def test_cancellation_during_inference_prevents_dispatch(monkeypatch):
    _, agent, backend = _agent([call("grasp", object="red")])
    generate = backend.generate

    def cancelled(messages, tools):
        agent.tools.cancel.set()
        return generate(messages, tools)

    monkeypatch.setattr(backend, "generate", cancelled)
    ep = agent.run("Put red in the tray")
    assert not ep.success_claimed and "cancel" in ep.reason
    assert not ep.tool_calls


def test_inference_holds_current_pose_instead_of_continuing_an_old_target(monkeypatch):
    env, agent, backend = _agent([])
    start = env.robot_state()
    env.embodiment().move_tcp(start.tcp_pos + [0.05, 0.0, 0.0], start.tcp_quat)

    def inspect_hold(*args):
        assert np.linalg.norm(env.robot.target_pos - env.robot_state().tcp_pos) < 0.001
        return call("finish", success=False, reason="test")

    monkeypatch.setattr(backend, "generate", inspect_hold)
    agent.run("Test holding during cloud inference")


def test_hosted_backend_does_not_drop_additional_tool_calls(monkeypatch):
    import io
    import urllib.request

    from dream.runtime.qwen import OpenAICompatibleBackend

    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "grasp", "arguments": '{"object":"red"}'}},
                        {"function": {"name": "retract", "arguments": "{}"}},
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(response).encode())
    )
    text = OpenAICompatibleBackend("http://unused").generate([], [])
    assert text.count("<tool_call>") == 2
    assert parse_tool_call(text)[0] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": []}]},
        {"choices": [{"message": {"tool_calls": [{}]}}]},
    ],
)
def test_malformed_hosted_responses_become_bounded_inference_errors(monkeypatch, payload):
    import io
    import urllib.request

    from dream.runtime.qwen import OpenAICompatibleBackend

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode())
    )
    with pytest.raises(ValueError):
        OpenAICompatibleBackend("http://unused").generate([], [])


def test_local_backend_passes_images_tools_and_cancellation_to_official_interfaces(monkeypatch):
    import contextlib
    import sys
    import threading
    from types import SimpleNamespace

    from dream.runtime.qwen import QwenTransformersBackend

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list),
    )
    backend = QwenTransformersBackend.__new__(QwenTransformersBackend)
    backend.cancel = threading.Event()
    backend.timeout_s = 60.0
    backend.max_new_tokens = 10
    backend._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    seen = {}

    class Inputs(dict):
        def to(self, device, dtype):
            seen["device"], seen["dtype"] = device, dtype
            return self

    def template(messages, **kwargs):
        seen["messages"], seen["kwargs"] = messages, kwargs
        return Inputs(input_ids=np.array([[1, 2]]))

    def generate(**kwargs):
        assert not kwargs["stopping_criteria"][0](None, None)
        backend.cancel.set()
        assert kwargs["stopping_criteria"][0](None, None)
        return np.array([[1, 2, 3]])

    backend.processor = SimpleNamespace(
        apply_chat_template=template, decode=lambda *a, **k: "<tool_call>{}</tool_call><|im_end|>"
    )
    backend.model = SimpleNamespace(device="cpu", dtype="bf16", generate=generate)
    text = backend.generate(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": np.zeros((2, 2, 3), dtype=np.uint8)},
                    {"type": "text", "text": "task"},
                ],
            }
        ],
        tool_schemas(),
    )
    assert text == "<tool_call>{}</tool_call>"
    assert seen["messages"][0]["content"][0]["image"].size == (2, 2)
    assert seen["kwargs"]["tools"] == tool_schemas()
    assert seen["kwargs"]["tokenize"] and seen["kwargs"]["add_generation_prompt"]
    assert (seen["device"], seen["dtype"]) == ("cpu", "bf16")
    backend.cancel.clear()
    backend.timeout_s = -1.0
    with pytest.raises(TimeoutError):
        backend.generate([], [])


def test_inference_failure_stops_without_a_physical_tool_call(monkeypatch):
    _, agent, backend = _agent([])

    def timeout(*args):
        raise TimeoutError("test backend timeout")

    monkeypatch.setattr(backend, "generate", timeout)
    ep = agent.run("Put red in the tray")
    assert not ep.tool_calls and not ep.success_claimed and "TimeoutError" in ep.reason


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
    assert names == [t["name"] for t in TOOL_SPECS if t["name"] not in MODEL_EXCLUDED_TOOLS] + [
        "finish"
    ]
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
    from dream.tools.goals import parse_goal

    ep = agent.run("Put the red cube in the tray.", expected_goal=parse_goal("in red tray"))
    env.step(250)
    assert ep.success_claimed and object_in_tray(env, "red")
    assert ep.tool_calls == ["observe_scene", "grasp", "place", "verify_goal", "finish"]
