"""Model tool selection for mobile TaskAgent, with caller-bound structured goals.

This is a Planner, not an execution loop: TaskAgent observes, verifies, executes and
checks safety. Model calls are bounded to observed candidates and frontiers.
"""

from __future__ import annotations

import json
import re

from dream.runtime.agent import Finish, ToolCall
from dream.runtime.vlm import parse_tool_call
from dream.tools.goals import GoToObserved, GoToPlace, Inspect, UnsupportedGoal


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
    schema(
        "inspect_instance",
        "Inspect exactly this observed candidate track ID; do not substitute another instance.",
        {"candidate": {"type": "integer"}},
        ["candidate"],
    ),
    schema(
        "go_to_frontier",
        "Navigate to the index of an observed reachable frontier.",
        {"index": {"type": "integer"}},
        ["index"],
    ),
    schema(
        "go_to_viewpoint",
        "Navigate to a validated observed viewpoint.",
        {"candidate": {"type": "integer"}, "side": {"type": "integer"}},
        ["candidate", "side"],
    ),
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
    # Stable references supplied by the user are constraints, not a model-chosen replacement.
    words = {
        name: i
        for i, name in enumerate(
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        )
    }
    match = re.search(
        r"\bcandidate\s*[:#]?\s*(\d+|" + "|".join(words) + r")\b", utterance.casefold()
    )
    requested = (
        None if match is None else (int(match[1]) if match[1].isdigit() else words[match[1]])
    )
    message = {
        "request": utterance,
        "required_candidate_instance": requested,
        "available_places": sorted(places),
        "observed_scene": scene,
        "capabilities": [
            "inspect a cabinet",
            "inspect an observed candidate instance",
            "navigate to a remembered place or observed reachable frontier/viewpoint",
        ],
        "unsupported_capabilities": ["grasp", "open", "carry", "manipulate"],
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
        return UnsupportedGoal(f"Could not ground the request in available tools: {error}")
    name, args = call["name"], call["arguments"]
    if name == "inspect_instance" and set(args) == {"candidate"}:
        cid = args["candidate"]
        if requested is not None and cid != requested:
            return UnsupportedGoal("Model did not preserve the requested candidate instance.")
        if type(cid) is int and any(c["id"] == cid for c in scene.get("candidates", [])):
            return Inspect("cabinet", candidate=cid)
        return UnsupportedGoal("Requested candidate is not currently observed; look around first.")
    if name == "go_to_frontier" and set(args) == {"index"}:
        index = args["index"]
        if type(index) is int and 0 <= index < len(scene.get("frontiers", [])):
            xy = scene["frontiers"][index]["xy"]
            return GoToObserved(float(xy[0]), float(xy[1]), f"frontier:{index}")
        return UnsupportedGoal("Requested frontier is not observed and reachable.")
    if name == "go_to_viewpoint" and set(args) == {"candidate", "side"}:
        for v in scene.get("viewpoints", []):
            if v["candidate"] == args["candidate"] and v["side"] == args["side"]:
                return GoToObserved(
                    *map(float, v["xy"]), f"candidate:{v['candidate']}:side:{v['side']}"
                )
        return UnsupportedGoal("Requested viewpoint has no currently validated path.")
    if name == "inspect_cabinet" and not args:
        if requested is not None:
            if not any(c["id"] == requested for c in scene.get("candidates", [])):
                return UnsupportedGoal(f"Candidate {requested} is not currently observed.")
            return Inspect("cabinet", candidate=requested)
        return Inspect("cabinet")
    if name == "go_to_place" and set(args) == {"place"}:
        place = args["place"]
        if isinstance(place, str) and place in places:
            return GoToPlace(place)
        return UnsupportedGoal(
            "Requested place is not in home memory; use an observed reachable target."
        )
    if name == "unsupported" and set(args) == {"reason"} and isinstance(args["reason"], str):
        return UnsupportedGoal(
            f"Unsupported request: {args['reason']}. Available: inspect observed candidates, "
            "navigate to reachable observed targets or remembered places."
        )
    return UnsupportedGoal(f"Unsupported goal interpretation: {name}")


def available_actions(goal, scene, verdict, history):
    """Bounded admissible operations, grounded in the current observation, never a chosen plan."""

    def attempts(name, args):
        return sum(s.call.name == name and s.call.args == args for s in history)

    actions = []
    if isinstance(goal, GoToObserved) or isinstance(goal, GoToPlace):
        xy = verdict.get("place_xy")
        if xy:
            args = {"x": xy[0], "y": xy[1]}
            if attempts("go_to", args) < 2:
                actions.append(ToolCall("go_to", args))
        return actions
    cid = getattr(goal, "candidate", None)
    if not any(s.call.name == "look_around" for s in history):
        actions.append(ToolCall("look_around"))
    for v in scene.get("viewpoints", []):
        if cid is not None and v["candidate"] != cid:
            continue
        args = {"candidate": v["candidate"], "side": v["side"]}
        if attempts("approach", args) < 2:
            actions.append(ToolCall("approach", args))
    for c in scene.get("candidates", []):
        if cid is not None and c["id"] != cid:
            continue
        pose = scene.get("inspection_poses", {}).get(str(c["id"]))
        args = {"candidate": c["id"]}
        # Legacy observation fixtures lack pose fields; live scenes always include them.
        if (pose is None or pose["valid"]) and attempts("inspect", args) < 2:
            actions.append(ToolCall("inspect", args))
    for _index, f in enumerate(scene.get("frontiers", [])):
        args = {"x": f["xy"][0], "y": f["xy"][1]}
        if attempts("go_to", args) < 2:
            actions.append(ToolCall("go_to", args))
    return actions


class MobileModelPlanner:
    """Qwen chooses every action; rejected decisions get bounded, structured repair feedback."""

    def __init__(self, backend, instruction, camera):
        self.backend, self.instruction, self.camera = backend, instruction, camera

    def next_action(self, goal, scene, verdict, history):
        if isinstance(goal, UnsupportedGoal):
            return Finish(False, goal.reason)
        alternatives = available_actions(goal, scene, verdict, history)
        prior = [
            {
                "tool": s.call.name,
                "args": s.call.args,
                "status": s.result.status,
                "message": s.result.message,
                "measured_outcome": s.result.data,
            }
            for s in history
        ]
        message = {
            "instruction": self.instruction,
            "bound_goal": repr(goal),
            "observation": scene,
            "verdict": verdict,
            "history": prior,
            "admissible_actions": [{"tool": a.name, "args": a.args} for a in alternatives],
            "remaining_repair_attempts": 3,
        }
        parts = [{"type": "text", "text": json.dumps(message, default=str)}]
        frame = self.camera()
        if frame is not None:
            parts.append({"type": "image", "image": frame})
        instruction = (
            "Choose exactly one admissible typed tool using the observations. Tool "
            "execution success is not goal completion. A partial inspection requires a "
            "valid reachable inspection viewpoint, then another inspection. Instance- "
            "bound goals must never substitute a different candidate. If verification "
            "fails, consider the admissible alternatives. Finish true only if "
            "verdict.satisfied; finish false when no safe admissible alternatives "
            "remain. No raw motor commands. Repeated actions are bounded."
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    }
                ],
            },
            {"role": "user", "content": parts},
        ]
        schemas = SCHEMAS + [
            schema(
                "go_to",
                "Only a position listed in admissible_actions.",
                {"x": {"type": "number"}, "y": {"type": "number"}},
                ["x", "y"],
            )
        ]
        last_error = ""
        for repair in range(3):
            output = self.backend.generate(messages, schemas)
            call, error = parse_tool_call(output)
            action = None
            if call is not None:
                name, args = call["name"], call["arguments"]
                if name == "finish" and set(args) == {"success", "reason"}:
                    if type(args["success"]) is bool and isinstance(args["reason"], str):
                        if args["success"] and verdict.get("satisfied"):
                            return Finish(True, args["reason"])
                        if not args["success"] and not alternatives:
                            return Finish(False, args["reason"])
                    error = "finish rejected: goal not verified or admissible alternatives remain"
                else:
                    if name == "explore_frontier" and set(args) == {"index"}:
                        i = args["index"]
                        if type(i) is int and 0 <= i < len(scene.get("frontiers", [])):
                            xy = scene["frontiers"][i]["xy"]
                            action = ToolCall("go_to", {"x": xy[0], "y": xy[1]})
                    else:
                        action = ToolCall(name, args)
                    if action in alternatives:
                        return action
                    error = "invalid, exhausted, unobserved, wrong-instance or unsafe tool target"
            last_error = error or "invalid model decision"
            messages.extend(
                [
                    {"role": "assistant", "content": [{"type": "text", "text": output}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "recovery": {
                                            "reason": last_error,
                                            "goal_verification": verdict,
                                            "admissible_alternatives": message[
                                                "admissible_actions"
                                            ],
                                            "remaining_repair_attempts": 2 - repair,
                                        }
                                    }
                                ),
                            }
                        ],
                    },
                ]
            )
        return Finish(False, f"model repair budget exhausted: {last_error}")
