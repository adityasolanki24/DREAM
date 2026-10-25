"""Deterministic planner for "inspect cabinet" on a mobile robot (TaskAgent's Planner protocol).

Classical, not learned, no language. Each step it looks at what is currently known and picks
one typed tool call:
1. Look around once at the start.
2. If a box with a handle has been seen: walk to a viewpoint in front of the handle,
   then inspect it. The goal is verified by observation (dream/tools/mobile.py).
3. Otherwise, if some candidate object has a side not yet seen up close, go and look from
   the cheapest such side: a handle is only resolvable at close range.
4. Otherwise explore: walk to a frontier of the map, chosen by expected new area minus
   walking cost ("gain"), or simply the nearest one ("nearest", the classical baseline).
5. If nothing is left, stop and say what was and was not found.
Failed actions are counted per target; a target that keeps failing is dropped. No room
layout, object identity, mechanism or seed is known to this code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from dream.runtime.agent import Finish, Step, ToolCall
from dream.tools.goals import Goal
from dream.tools.mobile import FRONTIER_COST_WEIGHT

MAX_FAILURES = 2  # approaches per (candidate, side); failed go_to per frontier area
DROP_RADIUS = 0.5  # m: frontiers this close to a failed go_to target are skipped


@dataclass
class RoomSearchPlanner:
    frontier_choice: str = "gain"  # "gain" | "nearest"

    def next_action(self, goal: Goal, scene: dict, verdict: dict, history: list[Step]):
        if verdict.get("satisfied"):
            c = verdict.get("candidate")
            return Finish(True, f"found and inspected candidate {c}: a box with a handle")
        if not any(s.call.name == "look_around" for s in history):
            return ToolCall("look_around")
        # Every approach counts, not only failed ones: a viewpoint reached twice without
        # changing what is known will not change it a third time.
        failed = Counter(
            (int(s.call.args["candidate"]), int(s.call.args["side"]))
            for s in history
            if s.call.name == "approach"
        )
        cands = scene["candidates"]
        with_handle = [c for c in cands if c["handle"] is not None]
        if with_handle:
            c = max(with_handle, key=lambda c: c["handle_views"])
            last = history[-1]
            if (
                last.call.name == "approach"
                and last.result.ok
                and int(last.call.args["candidate"]) == c["id"]
            ):
                return ToolCall("inspect", {"candidate": c["id"]})
            options = [
                v
                for v in scene["viewpoints"]
                if v["candidate"] == c["id"] and failed[(c["id"], v["side"])] < MAX_FAILURES
            ]
            best = [v for v in options if v["side"] == c["handle_side"]] or options
            if best:
                v = min(best, key=lambda v: v["cost_m"])
                return ToolCall("approach", {"candidate": v["candidate"], "side": v["side"]})
            if str(c["id"]) not in scene["inspected"]:
                return ToolCall("inspect", {"candidate": c["id"]})
            # No reachable viewpoint yet: more of the room may make one reachable.
            found_unreachable = c["id"]
        else:
            found_unreachable = None
        vps = [v for v in scene["viewpoints"] if failed[(v["candidate"], v["side"])] < MAX_FAILURES]
        if found_unreachable is not None:  # the cabinet is found: only explore toward it
            vps = []
        if vps:
            v = min(vps, key=lambda v: v["cost_m"])
            return ToolCall("approach", {"candidate": v["candidate"], "side": v["side"]})
        dead = [
            np.array([s.call.args["x"], s.call.args["y"]])
            for s in history
            if s.call.name == "go_to" and not s.result.ok
        ]
        fr = [
            f
            for f in scene["frontiers"]
            if sum(np.linalg.norm(np.array(f["xy"]) - d) < DROP_RADIUS for d in dead) < MAX_FAILURES
        ]
        if fr:
            if self.frontier_choice == "nearest":
                f = min(fr, key=lambda f: f["cost_m"])
            else:
                f = max(fr, key=lambda f: f["gain_m2"] - FRONTIER_COST_WEIGHT * f["cost_m"])
            return ToolCall("go_to", {"x": f["xy"][0], "y": f["xy"][1]})
        if found_unreachable is not None:
            return Finish(
                False, f"candidate {found_unreachable} has a handle, but no reachable viewpoint"
            )
        n = len(cands)
        return Finish(
            False,
            f"explored the reachable space; {n} candidate object(s) checked, none with a handle",
        )


@dataclass
class PlacePlanner:
    """Walk to a known place (``GoToPlace``): one go_to, retried once, then report failure."""

    def next_action(self, goal: Goal, scene: dict, verdict: dict, history: list[Step]):
        if verdict.get("satisfied"):
            return Finish(True, f"at {goal.place}")
        if verdict.get("unknown_place"):
            return Finish(False, f"{goal.place} is not a place I know")
        x, y = verdict["place_xy"]
        tries = sum(1 for s in history if s.call.name == "go_to" and
                    (s.call.args["x"], s.call.args["y"]) == (x, y))  # fmt: skip
        if tries >= 2:
            return Finish(False, f"could not reach {goal.place}")
        return ToolCall("go_to", {"x": x, "y": y})
