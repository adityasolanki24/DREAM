"""Robot-time budget wrapper for the room-search planner, with a frontier-decision recorder."""

from __future__ import annotations

from dream.runtime.agent import Finish
from dream.runtime.search import RoomSearchPlanner


class Budgeted:
    """Stop the run when the robot-time budget is spent (an episode limit, not a strategy).

    Also an evaluation-side recorder of frontier decisions: whenever the planner picks a
    frontier, both policies' choices in that same state are computed by calling fresh
    stateless planners on the same inputs. This never changes what is returned.
    """

    def __init__(self, planner, base, seconds: float, method: str | None = None) -> None:
        self.planner, self.base, self.seconds = planner, base, seconds
        self.method = method or getattr(planner, "frontier_choice", None)
        self.decisions: list[dict] = []
        self.frontier_steps: list[list] = []

    def next_action(self, goal, scene, verdict, history):
        if not verdict.get("satisfied") and self.base.time > self.seconds:
            return Finish(False, f"time budget ({self.seconds:.0f} s) spent")
        action = self.planner.next_action(goal, scene, verdict, history)
        if getattr(action, "name", None) == "go_to":
            try:
                self._record_frontier(action, goal, scene, verdict, history)
            except Exception:  # never let the recorder affect the run
                pass
        return action

    def _record_frontier(self, action, goal, scene, verdict, history) -> None:
        frontiers = scene["frontiers"]
        picks = {}
        for m in ("gain", "nearest"):
            a = RoomSearchPlanner(m).next_action(goal, scene, verdict, history)
            picks[m] = [a.args["x"], a.args["y"]] if getattr(a, "name", None) == "go_to" else None
        chosen = [action.args["x"], action.args["y"]]
        f = next((f for f in frontiers if list(f["xy"]) == chosen), None)
        self.decisions.append({
            "time_s": round(float(self.base.time), 2), "step": len(history),
            "chosen_xy": chosen, "cost_m": f and f["cost_m"], "gain_m2": f and f["gain_m2"],
            "method": self.method, "gain_policy_xy": picks["gain"],
            "nearest_policy_xy": picks["nearest"],
            "diverges": picks["gain"] is not None and picks["gain"] != picks["nearest"],
        })  # fmt: skip
        self.frontier_steps.append([
            round(float(self.base.time), 2),
            [[*fr["xy"], fr["gain_m2"], fr["cost_m"]] for fr in frontiers],
        ])  # fmt: skip
