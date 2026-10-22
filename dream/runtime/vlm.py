"""Model-driven task agent: a vision-language model chooses DREAM's tools.

The model receives the user's instruction in natural language, the current camera image, the
structured observations returned by tools (metric positions come from perception and
proprioception, never from the model's reading of pixels), and the result of every action. Each
turn it emits one tool call in the Qwen/Hermes format:

    <tool_call>{"name": "grasp", "arguments": {"object": "blue"}}</tool_call>

Safety and validity do not depend on the model:
- outputs are parsed as JSON only, never executed as code;
- every call goes through ``Toolbox.call``, which validates the schema, checks physical
  preconditions and runs skills on the safety-limited controller;
- invalid outputs get a short error message back, and a run stops after ``max_invalid``
  consecutive invalid turns;
- the model ends a task with ``finish``. A claim of success is accepted only if the structured
  goal the model declares is verified by ``verify_goal`` on fresh observations, and nothing is
  left in the gripper. A rejected claim is reported back to the model.

The model sits behind ``VLMBackend``. Backends for Qwen3-VL (local transformers checkpoint, or
an OpenAI-compatible server such as vLLM) are in ``dream/runtime/qwen.py``. Tests use a
scripted test double, which is not a model and shows only that the plumbing works.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dream.tools.goals import Goal, parse_goal
from dream.tools.toolbox import TOOL_SPECS, Toolbox, ToolResult

FINISH_SPEC = {
    "name": "finish",
    "description": (
        "End the task. success=true only when the goal is achieved; give the goal as a "
        'structured goal so it can be verified: {"type": "in" | "on", "object": ..., '
        '"target": ...}. Use success=false with a reason when the task is impossible, unsafe, '
        "unsupported by the robot's tools, or keeps failing."
    ),
    "parameters": {
        "success": {"type": "boolean", "required": True},
        "reason": {"type": "string", "required": True},
        "goal": {"type": "goal", "required": False},
    },
}

SYSTEM_PROMPT = """You control a robot arm (a Franka Panda with a two-finger gripper) in a \
simulated tabletop workspace, through tools. You see the scene through a camera image and \
through the structured results of tools.

Rules:
- Call exactly one tool per turn, as <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
- Positions and sizes come from tool results (observe_scene, get_robot_state); do not estimate \
coordinates from the image.
- Objects are named by the tools (for example by colour); use those names in arguments.
- After acting, check the result. If an action failed, observe again and decide what to do \
next; do not repeat the same failing call blindly.
- If a container is closed or its mechanism is unknown, investigate_physics finds out how it \
moves; open_articulated opens it using what was learned.
- When the goal is achieved, call verify_goal and then finish with success=true and the goal. \
If the task cannot be done with these tools or this robot, call finish with success=false and \
explain why."""

# Images are large in the model's context: keep only the most recent ones.
MAX_IMAGES_IN_CONTEXT = 2
MAX_RESULT_CHARS = 3000


class VLMBackend(Protocol):
    """A vision-language model. ``messages`` are chat messages whose ``content`` is a list of
    parts: {"type": "text", "text": str} or {"type": "image", "image": HxWx3 uint8 array}.
    ``tools`` are JSON-schema function definitions. Returns the raw generated text."""

    name: str

    def generate(self, messages: list[dict], tools: list[dict]) -> str: ...


def tool_schemas() -> list[dict]:
    """The toolbox's typed tools (and ``finish``) as JSON-schema function definitions."""
    kinds = {
        "string": {"type": "string"},
        "number": {"type": "number"},
        "boolean": {"type": "boolean"},
        "vector3": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "goal": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["in", "on"]},
                "object": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["type", "object", "target"],
        },
    }
    out = []
    for spec in [*TOOL_SPECS, FINISH_SPEC]:
        props = {k: dict(kinds[p["type"]]) for k, p in spec["parameters"].items()}
        required = [k for k, p in spec["parameters"].items() if p.get("required")]
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }
        )
    return out


_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> tuple[dict | None, str]:
    """The first tool call in a model output, or (None, error message). JSON only."""
    found = _CALL.findall(text)
    if not found:
        return None, "no <tool_call> found; call exactly one tool per turn"
    try:
        call = json.loads(found[0])
    except json.JSONDecodeError as e:
        return None, f"tool call is not valid JSON ({e.msg})"
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None, "a tool call needs a string 'name'"
    args = call.get("arguments", {})
    if isinstance(args, str):  # some servers return arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None, "'arguments' is not valid JSON"
    if not isinstance(args, dict):
        return None, "'arguments' must be an object"
    return {"name": call["name"], "arguments": args}, ""


@dataclass
class VLMStep:
    output: str  # raw model output
    call: dict | None  # parsed call, None if invalid
    result: ToolResult | None
    error: str  # parse or finish-rejection error, "" if none
    model_s: float  # inference latency
    tool_s: float


@dataclass
class VLMEpisode:
    instruction: str
    backend: str
    success_claimed: bool  # accepted (verified) success
    reason: str
    declared_goal: Goal | None
    steps: list[VLMStep] = field(default_factory=list)

    @property
    def invalid_outputs(self) -> int:
        return sum(1 for s in self.steps if s.call is None)

    @property
    def rejected_calls(self) -> int:
        return sum(1 for s in self.steps if s.result is not None and s.result.status == "rejected")

    @property
    def tool_calls(self) -> list[str]:
        return [s.call["name"] for s in self.steps if s.call is not None]

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "backend": self.backend,
            "success_claimed": self.success_claimed,
            "reason": self.reason,
            "declared_goal": repr(self.declared_goal),
            "invalid_outputs": self.invalid_outputs,
            "rejected_calls": self.rejected_calls,
            "steps": [
                {
                    "output": s.output[:500],
                    "call": s.call,
                    "status": None if s.result is None else s.result.status,
                    "message": s.error or ("" if s.result is None else s.result.message),
                    "model_s": round(s.model_s, 3),
                    "tool_s": round(s.tool_s, 3),
                }
                for s in self.steps
            ],
        }


class VLMAgent:
    def __init__(
        self,
        toolbox: Toolbox,
        backend: VLMBackend,
        camera: Callable[[], np.ndarray] | None = None,
        max_steps: int = 20,
        max_invalid: int = 3,
    ) -> None:
        """``camera()`` returns the current RGB image (HxWx3 uint8) sent with each turn."""
        self.tools = toolbox
        self.backend = backend
        self.camera = camera
        self.max_steps = max_steps
        self.max_invalid = max_invalid
        self.schemas = tool_schemas()

    def run(self, instruction: str, on_event: Callable[[str, dict], None] | None = None):
        emit = on_event or (lambda kind, data: None)
        emit("goal", {"instruction": instruction})
        scene = self.tools.call("observe_scene")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            self._user(f"Instruction: {instruction}\n\nobserve_scene result:\n{_brief(scene)}"),
        ]
        ep = VLMEpisode(instruction, self.backend.name, False, "", None)
        invalid_run = 0
        for _ in range(self.max_steps):
            if self.tools.cancel.is_set():
                return self._end(ep, False, "cancelled by user", emit)
            if self.tools.robot.state().protective_stop:
                return self._end(
                    ep, False, "robot in protective stop (operator reset needed)", emit
                )
            t0 = time.perf_counter()
            output = self.backend.generate(_bounded_images(messages), self.schemas)
            model_s = time.perf_counter() - t0
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output}]})
            call, error = parse_tool_call(output)
            result, tool_s = None, 0.0
            if call is not None and call["name"] == "finish":
                done, error = self._finish(call["arguments"], ep)
                if done:
                    ep.steps.append(VLMStep(output, call, None, "", model_s, 0.0))
                    return self._end(ep, ep.success_claimed, ep.reason, emit)
            elif call is not None:
                emit("action", {"tool": call["name"], "args": call["arguments"]})
                t1 = time.perf_counter()
                result = self.tools.call(call["name"], call["arguments"])
                tool_s = time.perf_counter() - t1
                emit(
                    "result",
                    {"tool": call["name"], "status": result.status, "message": result.message},
                )
            ep.steps.append(VLMStep(output, call, result, error, model_s, tool_s))
            if error:
                invalid_run += 1
                emit("invalid", {"error": error})
                if invalid_run >= self.max_invalid:
                    return self._end(
                        ep, False, f"{invalid_run} invalid model outputs in a row: {error}", emit
                    )
                messages.append(self._user(f"Error: {error}. Try again with one valid call."))
                continue
            # A call the toolbox rejected (unknown tool, bad arguments, unmet precondition) did
            # nothing; it counts toward the same consecutive-invalid budget.
            invalid_run = invalid_run + 1 if result.status == "rejected" else 0
            if result.status == "stopped":
                return self._end(ep, False, result.message, emit)
            if invalid_run >= self.max_invalid:
                return self._end(
                    ep, False, f"{invalid_run} invalid model calls in a row: {result.message}", emit
                )
            messages.append(self._user(f"{call['name']} result:\n{_brief(result)}"))
        return self._end(ep, False, f"step budget ({self.max_steps}) exhausted", emit)

    def _finish(self, args: dict, ep: VLMEpisode) -> tuple[bool, str]:
        """(accepted, error). A success claim must verify on fresh observations."""
        success, reason = args.get("success"), args.get("reason", "")
        if not isinstance(success, bool) or not isinstance(reason, str):
            return False, "finish needs a boolean 'success' and a string 'reason'"
        if not success:
            ep.success_claimed, ep.reason = False, f"model declined: {reason}"
            return True, ""
        try:
            goal = parse_goal(args.get("goal"))
        except (ValueError, AttributeError, TypeError) as e:
            return False, f"finish(success=true) needs a valid structured goal ({e})"
        ep.declared_goal = goal
        v = self.tools.call("verify_goal", {"goal": goal})
        holding = self.tools.held_object()
        if v.ok and v.data.get("satisfied") and holding is None:
            ep.success_claimed, ep.reason = True, f"verified: {reason}"
            return True, ""
        why = "still holding " + repr(holding) if holding else v.message or "not satisfied"
        return False, f"success rejected: the goal is not verified ({why})"

    def _user(self, text: str) -> dict:
        content: list[dict] = [{"type": "text", "text": text}]
        if self.camera is not None:
            content.insert(0, {"type": "image", "image": self.camera()})
        return {"role": "user", "content": content}

    @staticmethod
    def _end(ep: VLMEpisode, success: bool, reason: str, emit) -> VLMEpisode:
        ep.success_claimed, ep.reason = success, reason
        emit("finish", {"success": success, "reason": reason})
        return ep


def _brief(result: ToolResult) -> str:
    """A tool result as compact JSON for the model, truncated to a bounded size."""
    text = json.dumps(
        {"status": result.status, "message": result.message, "data": result.data},
        default=lambda o: o.tolist() if isinstance(o, np.ndarray) else repr(o),
    )
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + " ...(truncated)"


def _bounded_images(messages: list[dict]) -> list[dict]:
    """Keep only the last MAX_IMAGES_IN_CONTEXT images; older ones become a text marker."""
    seen, out = 0, []
    for m in reversed(messages):
        parts = []
        for p in m["content"]:
            if p["type"] == "image":
                seen += 1
                if seen > MAX_IMAGES_IN_CONTEXT:
                    p = {"type": "text", "text": "[earlier camera image omitted]"}
            parts.append(p)
        out.append({**m, "content": parts})
    return out[::-1]
