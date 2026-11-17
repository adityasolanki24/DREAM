"""Budgeted: the robot-time budget wrapper and its evaluation-side frontier-decision recorder.
The recorder only calls stateless planners on the same inputs and never changes the action."""

from types import SimpleNamespace

from dream.runtime.agent import Finish, Step, ToolCall
from dream.runtime.budget import Budgeted
from dream.runtime.search import RoomSearchPlanner
from dream.tools.goals import Inspect
from dream.tools.toolbox import ToolResult


def _step(name, ok=True, **args):
    return Step(ToolCall(name, args), ToolResult(name, "succeeded" if ok else "failed"), 0.0)


def test_frontier_log_records_both_policies_and_divergence_without_changing_the_action():
    scene = {
        "candidates": [],
        "viewpoints": [],
        "inspected": {},
        "frontiers": [
            {"xy": [5, 5], "gain_m2": 9.0, "cost_m": 4.0},
            {"xy": [1, 1], "gain_m2": 1.0, "cost_m": 1.0},
        ],
    }
    hist = [_step("look_around")]
    base = SimpleNamespace(time=12.0)
    for method, expected in (("gain", [5, 5]), ("nearest", [1, 1])):
        planned = RoomSearchPlanner(method).next_action(Inspect("cabinet"), scene, {}, hist)
        b = Budgeted(RoomSearchPlanner(method), base, 240.0, method)
        a = b.next_action(Inspect("cabinet"), scene, {}, hist)
        assert a == planned == ToolCall("go_to", {"x": expected[0], "y": expected[1]})
        (dec,) = b.decisions
        assert dec["chosen_xy"] == expected and dec["method"] == method
        assert dec["gain_policy_xy"] == [5, 5] and dec["nearest_policy_xy"] == [1, 1]
        assert dec["diverges"] is True
        assert (dec["cost_m"], dec["gain_m2"]) == ((4.0, 9.0) if method == "gain" else (1.0, 1.0))
        assert b.frontier_steps == [[12.0, [[5, 5, 9.0, 4.0], [1, 1, 1.0, 1.0]]]]


def test_frontier_log_agreement_and_budget_stop():
    scene = {
        "candidates": [],
        "viewpoints": [],
        "inspected": {},
        "frontiers": [{"xy": [2, 2], "gain_m2": 5.0, "cost_m": 1.0}],
    }
    b = Budgeted(RoomSearchPlanner("gain"), SimpleNamespace(time=1.0), 240.0)
    b.next_action(Inspect("cabinet"), scene, {}, [_step("look_around")])
    assert b.decisions[0]["diverges"] is False and b.method == "gain"
    late = Budgeted(RoomSearchPlanner("gain"), SimpleNamespace(time=300.0), 240.0)
    a = late.next_action(Inspect("cabinet"), scene, {}, [_step("look_around")])
    assert isinstance(a, Finish) and late.decisions == []
