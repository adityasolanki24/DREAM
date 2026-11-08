"""Model tool selection for mobile TaskAgent, with caller-bound structured goals.

This is a Planner, not an execution loop: TaskAgent observes, verifies, executes and
checks safety. Model calls are bounded to observed candidates and frontiers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping

from dream.runtime.agent import Finish, ToolCall
from dream.runtime.vlm import parse_tool_call
from dream.tools.goals import (
    EXPLORATION_REACHED,
    GoToObserved,
    GoToPlace,
    Inspect,
    UnsupportedGoal,
)
from dream.tools.intent import (
    OUTCOMES,
    UNKNOWN,
    GoalSpecification,
    check_anchoring,
    check_binding,
)
from dream.tools.mobile import frontier_already_reached


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


def _reached(frontier, scene) -> bool:
    pose = scene.get("pose")
    return frontier_already_reached(
        frontier["xy"], frontier.get("cost_m"), None if pose is None else pose[:2]
    )


def interpret_mobile_goal(backend, utterance, places, scene, camera, spec=None, attempts=3):
    """Ask the selected VLM to map a spoken request to a supported typed mobile goal.

    With ``spec`` (the immutable original request, dream/tools/intent.py) every proposed goal
    must pass ``check_binding``: one that would complete something else is rejected and the
    model is told why, up to ``attempts`` proposals, then the request is refused. Without
    ``spec`` this is the 7 Nov 2026 behaviour, kept unchanged as the comparison arm."""
    if spec is None:
        return _interpret_once(backend, utterance, places, scene, camera)
    feedback: list[dict] = []
    for _ in range(attempts):
        goal = _interpret_once(backend, utterance, places, scene, camera, spec, feedback)
        ok, why = check_binding(spec, goal)
        if ok:
            return goal
        feedback.append({"rejected_goal": repr(goal), "reason": why})
    return UnsupportedGoal(
        f"no available goal completes '{spec.original_instruction}' without changing it: "
        + "; ".join(f["reason"] for f in feedback)
    )


def _interpret_once(backend, utterance, places, scene, camera, spec=None, feedback=()):
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
    if spec is not None:
        message["original_request"] = {
            "target": spec.target_class,
            "required_outcome": spec.required_outcome,
            "rule": "the goal must establish exactly this outcome on exactly this target; "
            "if no listed goal does, call unsupported",
        }
        if feedback:
            message["rejected_proposals"] = list(feedback)
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
            f = scene["frontiers"][index]
            if _reached(f, scene):
                return UnsupportedGoal(
                    "Requested frontier is where the robot already stands (already reached); "
                    "it is not a navigation target."
                )
            return GoToObserved(float(f["xy"][0]), float(f["xy"][1]), f"frontier:{index}")
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
        if _reached(f, scene):
            continue
        args = {"x": f["xy"][0], "y": f["xy"][1]}
        if attempts("go_to", args) < 2:
            actions.append(ToolCall("go_to", args))
    return actions


class MobileModelPlanner:
    """Qwen chooses every action; rejected decisions get bounded, structured repair feedback."""

    def __init__(self, backend, instruction, camera):
        self.backend, self.instruction, self.camera = backend, instruction, camera
        # One record per decision that needed repair; read by evaluation, never by decisions.
        self.repairs: list[dict] = []

    def next_action(self, goal, scene, verdict, history):
        if isinstance(goal, UnsupportedGoal):
            return Finish(False, goal.reason)
        if (
            isinstance(goal, GoToObserved)
            and goal.is_exploration_point
            and (verdict.get("reached") or verdict.get("satisfied"))
        ):
            # Typed outcome, decided without the model: nothing here observes the target.
            return Finish(
                False,
                "reached an exploration point; the requested target was not observed, "
                "so the task is not complete",
                EXPLORATION_REACHED,
            )
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
        rejected: list[dict] = []

        def settled(result, action):
            if rejected:
                self.repairs.append({"result": result, "rejected": rejected, "then": repr(action)})
            return action

        for repair in range(3):
            output = self.backend.generate(messages, schemas)
            call, error = parse_tool_call(output)
            action = None
            if call is not None:
                name, args = call["name"], call["arguments"]
                if name == "finish" and set(args) == {"success", "reason"}:
                    if type(args["success"]) is bool and isinstance(args["reason"], str):
                        if args["success"] and verdict.get("satisfied"):
                            return settled(
                                "accepted_alternative",
                                Finish(
                                    True,
                                    f"verified: {verdict.get('reason', 'goal satisfied')}",
                                    model_reason=args["reason"],
                                ),
                            )
                        if not args["success"] and not alternatives:
                            return settled(
                                "accepted_alternative",
                                Finish(
                                    False,
                                    "not completed: "
                                    f"{verdict.get('reason', 'goal not verified')}; "
                                    "no admissible alternatives remain",
                                    model_reason=args["reason"],
                                ),
                            )
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
                        return settled("accepted_alternative", action)
                    error = "invalid, exhausted, unobserved, wrong-instance or unsafe tool target"
            last_error = error or "invalid model decision"
            rejected.append({"model_output": output[:300], "reason": last_error})
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
        return settled("exhausted", Finish(False, f"model repair budget exhausted: {last_error}"))


SPECIFY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "specify_request",
        "description": "Restate the user's request in the robot's vocabulary. Use only words "
        "the user said (or a listed synonym); never replace the target with something easier.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "a vocabulary class or place, or ''"},
                "target_kind": {"type": "string", "enum": ["object", "room", "place", "none"]},
                "outcome": {"type": "string", "enum": [*OUTCOMES, UNKNOWN]},
                "also_required": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["target", "target_kind", "outcome"],
        },
    },
}


def propose_grounding(
    backend,
    text: str,
    objects: Mapping[str, Iterable[str]],
    rooms: Mapping[str, Iterable[str]] = (),
    places: Iterable[str] = (),
    attempts: int = 2,
) -> tuple[GoalSpecification | None, list[dict]]:
    """Ask a language model to ground what the grammar could not, then validate it against
    the request's own words. Returns (spec or None, the rejected proposals)."""
    places = list(places)
    message = {
        "request": text,
        "object_vocabulary": {k: list(v) for k, v in objects.items()},
        "room_vocabulary": sorted(dict(rooms)),
        "remembered_places": sorted(places),
        "outcomes": list(OUTCOMES),
    }
    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "Restate the request with specify_request. The target must be what the user "
            "named, in the robot's vocabulary; the outcome must be the action the user asked "
            "for. If the user named something outside the vocabulary or an action not listed, "
            "say so with target '' or outcome 'unknown'. Never choose a different target or "
            "an easier outcome.")}]},
        {"role": "user", "content": [{"type": "text", "text": json.dumps(message)}]},
    ]  # fmt: skip
    rejected = []
    for _ in range(attempts):
        output = backend.generate(messages, [SPECIFY_SCHEMA])
        call, error = parse_tool_call(output)
        problems = [error] if call is None else []
        spec = None
        if call is not None:
            a = call["arguments"]
            target = a.get("target") or None
            kind = a.get("target_kind")
            spec = GoalSpecification(
                original_instruction=text,
                target_class=target if isinstance(target, str) else None,
                target_kind=None if kind in (None, "none") or target is None else kind,
                target_instance=None,
                required_outcome=a.get("outcome") if isinstance(a.get("outcome"), str) else "",
                also_required=tuple(x for x in a.get("also_required") or () if isinstance(x, str)),
                grounding="model",
            )
            problems = check_anchoring(spec, objects, rooms, places)
        if not problems:
            return spec, rejected
        rejected.append({"model_output": str(output)[:300], "problems": problems})
        messages += [
            {"role": "assistant", "content": [{"type": "text", "text": str(output)}]},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(
                {"rejected": problems, "note": "use only what the request states"})}]},
        ]  # fmt: skip
    return None, rejected
