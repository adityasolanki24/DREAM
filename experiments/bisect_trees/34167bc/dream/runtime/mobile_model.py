"""Model tool selection for mobile TaskAgent, with caller-bound structured goals.

This is a Planner, not an execution loop: TaskAgent observes, verifies, executes and
checks safety. Model calls are bounded to observed candidates and frontiers.
"""

from __future__ import annotations

import json

from dream.runtime.agent import Finish, ToolCall
from dream.runtime.vlm import parse_tool_call
from dream.tools.goals import GoToPlace, Inspect


def schema(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


SCHEMAS = [
    schema("look_around", "Observe the surroundings."),
    schema(
        "approach",
        "Visit a currently observed viewpoint.",
        {"candidate": {"type": "integer"}, "side": {"type": "integer"}},
        ["candidate", "side"],
    ),
    schema(
        "inspect",
        "Inspect an observed candidate.",
        {"candidate": {"type": "integer"}},
        ["candidate"],
    ),
    schema(
        "explore_frontier",
        "Visit a frontier by its index in the current observation.",
        {"index": {"type": "integer"}},
        ["index"],
    ),
    schema(
        "finish",
        "Finish only if the observation verifies the bound goal, or decline.",
        {"success": {"type": "boolean"}, "reason": {"type": "string"}},
        ["success", "reason"],
    ),
]

GOAL_SCHEMAS = [
    schema("inspect_cabinet", "Find the cabinet and stand where it can be examined."),
    schema(
        "go_to_place",
        "Go to a named place already present in home memory.",
        {"place": {"type": "string"}},
        ["place"],
    ),
    schema(
        "unsupported",
        "The request cannot be represented by the available mobile skills.",
        {"reason": {"type": "string"}},
        ["reason"],
    ),
]


def interpret_mobile_goal(backend, utterance, places, scene, camera):
    """Ask the selected VLM to map a spoken request to a supported typed mobile goal."""
    message = {
        "request": utterance,
        "available_places": sorted(places),
        "observed_scene": scene,
        "capabilities": ["inspect a cabinet", "navigate to a remembered place"],
    }
    parts = [{"type": "text", "text": json.dumps(message, default=str)}]
    frame = camera()
    if frame is not None:
        parts.append({"type": "image", "image": frame})
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Interpret the user's goal from the request and current robot observation. "
                        "Select one typed goal only when it preserves the user's intent and is "
                        "supported by the listed mobile skills. Never invent a place or "
                        "capability; "
                        "otherwise call unsupported."
                    ),
                }
            ],
        },
        {"role": "user", "content": parts},
    ]
    call, error = parse_tool_call(backend.generate(messages, GOAL_SCHEMAS))
    if call is None:
        raise ValueError(f"goal interpretation failed: {error}")
    name, args = call["name"], call["arguments"]
    if name == "inspect_cabinet" and not args:
        return Inspect("cabinet")
    if name == "go_to_place" and set(args) == {"place"}:
        place = args["place"]
        if isinstance(place, str) and place in places:
            return GoToPlace(place)
        raise ValueError("model selected a place that is not in home memory")
    if name == "unsupported" and set(args) == {"reason"} and isinstance(args["reason"], str):
        raise ValueError(f"unsupported goal: {args['reason']}")
    raise ValueError(f"invalid goal interpretation: {name}")


class MobileModelPlanner:
    """A single model decision per TaskAgent step; invalid decisions fail closed."""

    def __init__(self, backend, instruction, camera):
        self.backend, self.instruction, self.camera = backend, instruction, camera

    def next_action(self, goal, scene, verdict, history):
        prior = [
            {
                "tool": s.call.name,
                "args": s.call.args,
                "status": s.result.status,
                "message": s.result.message,
            }
            for s in history
        ]
        message = {
            "instruction": self.instruction,
            "bound_goal": repr(goal),
            "observation": scene,
            "verdict": verdict,
            "history": prior,
        }
        parts = [{"type": "text", "text": json.dumps(message, default=str)}]
        frame = self.camera()
        if frame is not None:
            parts.append({"type": "image", "image": frame})
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Select exactly one available tool. Use only observed "
                        "candidates, viewpoints and frontiers. Finish successfully only "
                        "if the verdict is satisfied. "
                        "If the goal is unsupported or unsafe, finish with success false.",
                    }
                ],
            },
            {"role": "user", "content": parts},
        ]
        call, error = parse_tool_call(self.backend.generate(messages, SCHEMAS))
        if call is None:
            return Finish(False, f"invalid model decision: {error}")
        name, args = call["name"], call["arguments"]
        if name == "finish" and set(args) == {"success", "reason"}:
            if type(args["success"]) is not bool or not isinstance(args["reason"], str):
                return Finish(False, "invalid model finish")
            if args["success"] and not verdict.get("satisfied", False):
                return Finish(False, "model success rejected: bound goal not verified")
            return Finish(args["success"], args["reason"])
        if name == "look_around" and not args:
            return ToolCall(name)
        if name == "inspect" and set(args) == {"candidate"}:
            if type(args["candidate"]) is int and any(
                c["id"] == args["candidate"] for c in scene.get("candidates", [])
            ):
                return ToolCall(name, args)
        if name == "approach" and set(args) == {"candidate", "side"}:
            if all(type(args[k]) is int for k in args) and any(
                v["candidate"] == args["candidate"] and v["side"] == args["side"]
                for v in scene.get("viewpoints", [])
            ):
                return ToolCall(name, args)
        if name == "explore_frontier" and set(args) == {"index"}:
            index, frontiers = args["index"], scene.get("frontiers", [])
            if type(index) is int and 0 <= index < len(frontiers):
                x, y = frontiers[index]["xy"]
                return ToolCall("go_to", {"x": x, "y": y})
        return Finish(False, f"invalid or unobserved model tool: {name}")
