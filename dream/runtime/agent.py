"""Task-level agent: observe -> plan one action -> execute a tool -> observe -> verify -> repeat.

The planner is behind the ``Planner`` protocol. ``RuleBasedPlanner`` is a deterministic planner
over structured goals. It is not a language model and has no language understanding. It
reduces every supported goal to one shared procedure, "get object X to target T", using the
same tools for every goal type. A pretrained (vision-)language model can later implement
``Planner`` and emit the same tool calls without touching the motor layer.

Recovery comes from the loop, not from special cases. After every action the agent
re-observes and re-plans, so a missed grasp is retried and a dropped object is picked up again.
A placement that verification rejects is redone. Attempt budgets bound all of this, and the
agent stops and reports when they run out.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dream.tools.goals import Goal, InRegion, OnTop, resting_on
from dream.tools.toolbox import Toolbox, ToolResult
from dream.types import ObjectObservation


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Finish:
    success: bool
    reason: str


Action = ToolCall | Finish


@dataclass
class Step:
    call: ToolCall
    result: ToolResult
    wall_time: float


class Planner(Protocol):
    def next_action(
        self, goal: Goal, scene: dict, verdict: dict, history: list[Step]
    ) -> Action: ...


@dataclass
class RuleBasedPlanner:
    max_attempts: int = 3  # per action kind, before giving up

    def next_action(self, goal: Goal, scene: dict, verdict: dict, history: list[Step]) -> Action:
        names = {o["name"] for o in scene["objects"]}
        target = goal.region if isinstance(goal, InRegion) else goal.support
        if goal.object not in names:
            return Finish(False, f"'{goal.object}' is not visible")
        if isinstance(goal, InRegion) and target not in scene["fixtures"]:
            return Finish(False, f"unknown region '{target}'")
        if isinstance(goal, OnTop) and target not in names:
            return Finish(False, f"'{target}' is not visible")
        holding = scene["holding"]
        if verdict["satisfied"] and holding is None:
            return Finish(True, "goal verified by observation")
        failures = _failures(history)
        if holding == goal.object:
            if failures["place"] >= self.max_attempts:
                return Finish(False, "placement keeps failing")
            return ToolCall("place", {"object": goal.object, "target": target})
        if holding is not None:  # holding something else (e.g. a cleared obstacle): set it aside
            return ToolCall("place", {"object": holding, "target": "table"})
        # Anything resting on the object, or on the support we stack onto, must be cleared first.
        observed = [_obs(o) for o in scene["objects"]]
        blockers = resting_on(goal.object, observed)
        if isinstance(goal, OnTop):
            blockers += [b for b in resting_on(goal.support, observed) if b != goal.object]
        if blockers:
            if failures["grasp"] >= self.max_attempts:
                return Finish(False, f"could not clear {blockers}")
            return ToolCall("grasp", {"object": blockers[0]})
        if failures["grasp"] >= self.max_attempts:
            return Finish(
                False, f"could not grasp '{goal.object}' after {failures['grasp']} attempts"
            )
        if _placements(history) >= self.max_attempts:
            return Finish(False, "goal still not verified after repeated placements")
        return ToolCall("grasp", {"object": goal.object})


def _obs(o: dict) -> ObjectObservation:
    return ObjectObservation(o["name"], o["color"], np.asarray(o["position"]), o["yaw"], o["size"])


def _failures(history: list[Step]) -> dict[str, int]:
    out = {"grasp": 0, "place": 0}
    for s in history:
        if s.call.name in out and not s.result.ok:
            out[s.call.name] += 1
    return out


def _placements(history: list[Step]) -> int:
    return sum(1 for s in history if s.call.name == "place" and s.result.ok)


@dataclass
class Episode:
    goal: Goal
    success_claimed: bool
    reason: str
    steps: list[Step]

    @property
    def tool_calls(self) -> int:
        return len(self.steps)

    @property
    def failed_actions(self) -> int:
        return sum(1 for s in self.steps if not s.result.ok)

    @property
    def recovered(self) -> bool:
        """At least one action failed, and the agent still claimed success afterwards."""
        return self.failed_actions > 0 and self.success_claimed

    def to_dict(self) -> dict:
        return {
            "goal": repr(self.goal),
            "success_claimed": self.success_claimed,
            "reason": self.reason,
            "tool_calls": self.tool_calls,
            "failed_actions": self.failed_actions,
            "steps": [
                {
                    "tool": s.call.name,
                    "args": s.call.args,
                    "status": s.result.status,
                    "message": s.result.message,
                    "sim_time": round(s.result.sim_time, 3),
                    "wall_time": round(s.wall_time, 3),
                }
                for s in self.steps
            ],
        }


class TaskAgent:
    def __init__(self, toolbox: Toolbox, planner: Planner | None = None, max_steps: int = 12):
        self.tools = toolbox
        self.planner = planner or RuleBasedPlanner()
        self.max_steps = max_steps

    def run(self, goal: Goal, on_event: Callable[[str, dict], None] | None = None) -> Episode:
        emit = on_event or (lambda kind, data: None)
        steps: list[Step] = []
        emit("goal", {"goal": repr(goal)})
        for _ in range(self.max_steps):
            if self.tools.cancel.is_set():
                return self._finish(goal, False, "cancelled by user", steps, emit)
            observed = [
                self.tools.call("observe_scene"),
                self.tools.call("verify_goal", {"goal": goal}),
                self.tools.call("get_robot_state"),
            ]
            bad = next((r for r in observed if not r.ok), None)
            if bad is not None:
                return self._finish(goal, False, f"{bad.tool} failed: {bad.message}", steps, emit)
            scene, verdict, state = (r.data for r in observed)
            if state["protective_stop"]:
                return self._finish(
                    goal, False, "robot in protective stop (operator reset needed)", steps, emit
                )
            action = self.planner.next_action(goal, scene, verdict, steps)
            if isinstance(action, Finish):
                return self._finish(goal, action.success, action.reason, steps, emit)
            emit("action", {"tool": action.name, "args": action.args})
            t0 = time.perf_counter()
            result = self.tools.call(action.name, action.args)
            steps.append(Step(action, result, time.perf_counter() - t0))
            emit(
                "result", {"tool": action.name, "status": result.status, "message": result.message}
            )
            if result.status == "stopped":
                return self._finish(goal, False, result.message, steps, emit)
        return self._finish(goal, False, f"step budget ({self.max_steps}) exhausted", steps, emit)

    @staticmethod
    def _finish(goal, success, reason, steps, emit) -> Episode:
        emit("finish", {"success": success, "reason": reason})
        return Episode(goal, success, reason, steps)
