"""Deterministic exploration planner for TaskAgent; no alternate agent framework or NLP.

Actions depend on current RGB-D, measured bounded-probe belief and previous failures.
One red sphere/blue target is a declared task family. No seed, mass or scene type is read.
"""

from dream.runtime.agent import Finish, Step, ToolCall
from dream.tools.goals import Goal


class ExplorationPlanner:
    def next_action(self, goal: Goal, scene: dict, verdict: dict, history: list[Step]):
        if verdict.get("satisfied"):
            return Finish(True, "released object verified in target by RGB-D")
        if any(
            sum(s.call.name == n and not s.result.ok for s in history) >= 2
            for n in ("reach_object", "grasp_object", "move_object", "inspect_object")
        ):
            return Finish(False, "bounded recovery budget exhausted")
        if not scene["objects"]:
            return ToolCall("inspect_object", {"object": "red"})
        belief = scene["belief"]
        if belief["model_inadequate"]:
            return Finish(False, "observed motion is outside the bounded-lift model")
        if belief["p_movable"] <= 0.02:
            if scene["hand_closed"]:
                return ToolCall("release_object")
            return Finish(False, "no resolved movement under bounded probes; decline transfer")
        if history and not history[-1].result.ok and history[-1].call.name != "inspect_object":
            return ToolCall("inspect_object", {"object": "red"})
        if scene["holding"]:
            return (
                ToolCall("release_object")
                if scene["at_target"]
                else ToolCall("move_object", {"object": "red", "target": "target"})
            )
        if belief["p_movable"] >= 0.95 and scene.get("in_contact") and scene.get("hand_closed"):
            return ToolCall("grasp_object", {"object": "red"})
        if scene.get("in_contact") and scene.get("hand_closed") and belief["probes"] < 2:
            return ToolCall("probe_object", {"object": "red"})
        if not scene["reached"]:
            return ToolCall("reach_object", {"object": "red"})
        if belief["p_movable"] < 0.95:
            if belief["probes"] >= 2:
                return Finish(False, "movability remains uncertain after two bounded probes")
            return ToolCall("probe_object", {"object": "red"})
        return ToolCall("grasp_object", {"object": "red"})
