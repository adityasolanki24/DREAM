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

Containers: "put X in Y" has the precondition "Y is accessible", meaning perception sees an
open region named Y big enough for X. If Y is an articulated object that is not accessible,
the planner first establishes how it moves (``investigate_physics``) and then opens it along
the learned constraint (``open_articulated``). It never chooses a motion itself: the opening
motion comes from the investigation's model, and when the investigation does not support
opening (identified as fixed, mechanism not identified, model inadequate) the planner stops and
says why instead of forcing it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dream.tools.goals import Goal, InRegion, OnTop, resting_on
from dream.tools.toolbox import WALL_CLEARANCE, Toolbox, ToolResult
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


OPENING_MARGIN = 0.005  # m beyond the minimum accessible half width when opening a container


@dataclass
class RuleBasedPlanner:
    max_attempts: int = 3  # per action kind, before giving up
    exploration: str = "eig"  # probe selection for investigate_physics: "eig" | "random"
    opening: str = "learned"  # "learned" | "fixed_pull" (baseline: no discovery)
    probe_budget: int = 12

    def next_action(self, goal: Goal, scene: dict, verdict: dict, history: list[Step]) -> Action:
        names = {o["name"] for o in scene["objects"]}
        target = goal.region if isinstance(goal, InRegion) else goal.support
        if goal.object not in names:
            return Finish(False, f"'{goal.object}' is not visible")
        holding = scene["holding"]
        if verdict["satisfied"] and holding is None and scene.get("holding_handle") is None:
            return Finish(True, "goal verified by observation")
        if isinstance(goal, InRegion) and not _accessible(target, goal.object, scene):
            if target not in {a["name"] for a in scene.get("articulated", [])}:
                if target in scene["fixtures"]:
                    return Finish(False, f"'{target}' is too small for '{goal.object}'")
                return Finish(False, f"unknown region '{target}'")
            if holding is not None:  # free the gripper before operating the container
                if _failures(history)["place"] >= self.max_attempts:
                    return Finish(False, f"could not set '{holding}' aside")
                return ToolCall("place", {"object": holding, "target": "table"})
            size = next(o["size"] for o in scene["objects"] if o["name"] == goal.object)
            need = round(size / 2 + WALL_CLEARANCE + OPENING_MARGIN, 4)
            return self._make_accessible(target, need, scene, history)
        if scene.get("holding_handle") is not None:  # accessible (e.g. probing opened it enough)
            return ToolCall("release")
        if isinstance(goal, OnTop) and target not in names:
            return Finish(False, f"'{target}' is not visible")
        if verdict["satisfied"] and holding is None:
            return Finish(True, "goal verified by observation")
        failures = _failures(history)
        if holding == goal.object:
            if failures["place"] >= self.max_attempts:
                return Finish(False, "placement keeps failing")
            return ToolCall("place", {"object": goal.object, "target": target})
        if holding is not None:  # holding something else (e.g. a cleared obstacle): set it aside
            if failures["place"] >= self.max_attempts:
                return Finish(False, f"could not set '{holding}' aside")
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

    def _make_accessible(
        self, target: str, need: float, scene: dict, history: list[Step]
    ) -> Action:
        opens = sum(1 for s in history if s.call.name == "open_articulated")
        if self.opening == "fixed_pull":
            if opens >= self.max_attempts:
                return Finish(False, f"'{target}' still not accessible after {opens} pulls")
            return ToolCall("open_articulated", {"object": target, "strategy": "fixed_pull"})
        inv = scene["investigations"].get(target)
        tries = sum(1 for s in history if s.call.name == "investigate_physics")
        if inv is None or inv["status"] in ("unsafe", "unobservable", "failed"):
            if tries >= self.max_attempts:
                why = inv["reason"] if inv else "no investigation"
                return Finish(False, f"could not investigate '{target}': {why}")
            return ToolCall(
                "investigate_physics",
                {"object": target, "budget": self.probe_budget, "strategy": self.exploration},
            )
        if not inv["opening_supported"]:
            return Finish(False, _unsupported(target, inv))
        if opens >= self.max_attempts:
            return Finish(False, f"'{target}' still not accessible after {opens} openings")
        if opens and _since(history, "open_articulated", "retract"):
            # Opened, but not seen open: the arm may hide the opening from the cameras. Look
            # again with the arm out of the way before pulling more.
            return ToolCall("retract")
        return ToolCall("open_articulated", {"object": target, "min_opening": need})


def _since(history: list[Step], event: str, absent: str) -> bool:
    """True if ``absent`` has not been called since the last ``event``."""
    for s in reversed(history):
        if s.call.name == absent:
            return False
        if s.call.name == event:
            return True
    return False


def _accessible(region: str, obj: str, scene: dict) -> bool:
    """The region is visible and wide enough to set the object down clear of its walls."""
    if region not in scene["fixtures"]:
        return False
    size = next(o["size"] for o in scene["objects"] if o["name"] == obj)
    return scene["fixtures"][region]["half_width"] >= size / 2 + WALL_CLEARANCE


def _unsupported(target: str, inv: dict) -> str:
    n = inv["interactions"]
    if inv["status"] == "identified":  # identified, and the class does not open
        return (
            f"'{target}' does not open: identified as {inv['mechanism']} "
            f"(P={inv['confidence']:.2f}) after {n} probes; not forcing it"
        )
    if inv["status"] == "model_inadequate":
        return (
            f"'{target}' moves in a way none of my mechanism models explain "
            f"(chi2/dof {inv['chi2_per_dof']:.2f} after {n} probes); not attempting to open it"
        )
    return (
        f"mechanism of '{target}' not identified within {n} probes (best guess "
        f"{inv['mechanism']}, P={inv['confidence']:.2f}); not attempting to open it"
    )


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
    suspended: bool = False  # stopped at an action boundary for a higher-priority task

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

    def run(
        self,
        goal: Goal,
        on_event: Callable[[str, dict], None] | None = None,
        history: list[Step] | None = None,
        preempt=None,
    ) -> Episode:
        """Run to completion, failure or the step budget.

        ``history`` resumes a suspended episode: its steps count toward the budget and the
        planner sees them, but the scene is observed afresh before the next action, so a
        changed world is revalidated rather than assumed. ``preempt`` (an Event) suspends
        at the next action boundary, after the running tool has brought the robot to a safe
        state, returning ``suspended=True`` with the steps so far as the checkpoint."""
        emit = on_event or (lambda kind, data: None)
        steps: list[Step] = list(history or [])
        emit("goal", {"goal": repr(goal), "resumed_steps": len(steps)})
        for _ in range(self.max_steps - len(steps)):
            hold = getattr(self.tools, "hold_for_reasoning", None)
            if hold is not None:
                hold()  # Same deterministic pause contract as the VLM on this embodiment.
            if self.tools.cancel.is_set():
                return self._finish(goal, False, "cancelled by user", steps, emit)
            if preempt is not None and preempt.is_set():
                emit("suspended", {"steps": len(steps)})
                return Episode(goal, False, "suspended for a higher-priority task", steps, True)
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
            if result.status == "interrupted":  # the tool stopped safely for a preemption
                continue
        return self._finish(goal, False, f"step budget ({self.max_steps}) exhausted", steps, emit)

    @staticmethod
    def _finish(goal, success, reason, steps, emit) -> Episode:
        emit("finish", {"success": success, "reason": reason})
        return Episode(goal, success, reason, steps)
