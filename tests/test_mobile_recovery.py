"""Saved failure and independent contracts; model doubles are only local tests."""

import json
from pathlib import Path

import numpy as np

from dream.runtime.agent import Finish, Step, ToolCall
from dream.runtime.mobile_model import MobileModelPlanner
from dream.tools.goals import Inspect
from dream.tools.toolbox import ToolResult
from experiments.audit_mobile_inspection import load_saved

SAVED = (
    Path(__file__).resolve().parents[1] / "results/mobile_inspection_semantics/saved_failure.json"
)


def saved():
    data, tb, pose, history = load_saved(SAVED)
    return data, tb, pose, history


class Sequence:
    def __init__(self, calls):
        self.calls, self.inputs = iter(calls), []

    def generate(self, messages, tools):
        self.inputs.append(messages)
        name, args = next(self.calls)
        return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"


def test_wrong_side_observation_is_not_completed_inspection(monkeypatch):
    _, tb, _, _ = saved()
    tb.base.walk = lambda *a: True
    monkeypatch.setattr(
        tb,
        "_look",
        lambda: setattr(tb._candidate(2), "handle_views", tb._candidate(2).handle_views + 1),
    )
    result = tb.call("inspect", {"candidate": 2})
    assert not result.ok
    assert result.data["handle_observed"]
    assert not result.data["inspection_complete"]
    assert not result.data["handle_confirmed_close"]


def test_instance_goal_never_verifies_most_seen_other_candidate():
    _, tb, pose, _ = saved()
    c = tb._candidate(9)
    pose[:] = [-1.21, -1.485, 0.01]
    tb.inspections[9] = {"inspection_complete": True, "handle_confirmed_close": True}
    assert tb.at_inspection_pose(c)[0]
    verdict = tb.call("verify_goal", {"goal": Inspect("cabinet", candidate=9)}).data
    assert verdict["satisfied"] and verdict["candidate"] == 9
    assert not tb.call("verify_goal", {"goal": Inspect("cabinet", candidate=2)}).data["satisfied"]


def test_saved_candidate_sampling_finds_valid_visible_reachable_poses():
    _, tb, _, _ = saved()
    ok, _, dist = tb._passable()
    for cid in (2,):
        c = tb._candidate(cid)
        choices = [v for v in tb.viewpoints(ok, dist) if v.candidate == cid]
        assert choices, f"no valid reachable viewpoint for {cid}"
        for v in choices:
            assert tb.inspection_pose(c, np.array([*v.xy, v.heading]))[0]
            cell = tuple(int(x) for x in tb.memory.grid.cell(v.xy))
            assert ok[cell] and np.isfinite(dist[cell])
            assert tb.visible_handle(c, v.xy)


def test_unreachable_handle_side_not_advertised():
    _, tb, _, _ = saved()
    blocked = np.zeros_like(tb.memory.grid.logodds, bool)
    assert tb.viewpoints(blocked, np.full(blocked.shape, np.inf)) == []


def test_saved_premature_finish_gets_feedback_and_model_selects_alternative():
    data, tb, _, history = saved()
    tb._look = lambda: None
    scene = tb.call("observe_scene").data
    v = next(v for v in scene["viewpoints"] if v["candidate"] == 2)
    backend = Sequence(
        [
            ("finish", {"success": False, "reason": "wrong side"}),
            ("approach", {"candidate": 2, "side": v["side"]}),
        ]
    )
    planner = MobileModelPlanner(backend, "inspect cabinet", lambda: None)
    action = planner.next_action(Inspect("cabinet"), scene, data["decision"]["verdict"], history)
    assert action == ToolCall("approach", {"candidate": 2, "side": v["side"]})
    assert "recovery" in str(backend.inputs[-1])


def test_invalid_call_and_misleading_success_cannot_end_the_task():
    backend = Sequence(
        [
            ("inspect", {"candidate": 999}),
            ("finish", {"success": True, "reason": "tool succeeded"}),
            ("approach", {"candidate": 1, "side": 0}),
        ]
    )
    planner = MobileModelPlanner(backend, "inspect candidate 1", lambda: None)
    scene = {
        "candidates": [{"id": 1}],
        "inspection_poses": {"1": {"valid": False}},
        "viewpoints": [{"candidate": 1, "side": 0}],
        "frontiers": [],
    }
    h = [
        Step(
            ToolCall("inspect", {"candidate": 1}),
            ToolResult("inspect", "succeeded", data={"at_inspection_pose": False}),
            0,
        )
    ]
    assert (
        planner.next_action(Inspect("cabinet", candidate=1), scene, {"satisfied": False}, h).name
        == "approach"
    )


def test_repeated_unproductive_action_exhausts_bounded_search():
    scene = {
        "candidates": [{"id": 1}],
        "inspection_poses": {"1": {"valid": False}},
        "viewpoints": [{"candidate": 1, "side": 0}],
        "frontiers": [],
    }
    h = [Step(ToolCall("look_around"), ToolResult("look_around", "succeeded"), 0)] + [
        Step(ToolCall("approach", {"candidate": 1, "side": 0}), ToolResult("approach", "failed"), 0)
    ] * 2
    backend = Sequence([("finish", {"success": False, "reason": "no alternatives"})])
    action = MobileModelPlanner(backend, "inspect candidate 1", lambda: None).next_action(
        Inspect("cabinet", candidate=1), scene, {"satisfied": False}, h
    )
    assert isinstance(action, Finish) and not action.success


def test_invalid_model_repair_is_bounded():
    backend = Sequence([("invented_tool", {})] * 3)
    action = MobileModelPlanner(backend, "inspect", lambda: None).next_action(
        Inspect("cabinet"), {"candidates": [], "viewpoints": [], "frontiers": []}, {}, []
    )
    assert isinstance(action, Finish) and not action.success
    assert len(backend.inputs) <= 3


def test_candidate_nine_path_is_not_enough_when_handle_ray_is_occluded():
    _, tb, _, _ = saved()
    ok, _, dist = tb._passable()
    assert not [v for v in tb.viewpoints(ok, dist) if v.candidate == 9]
    assert not tb.visible_handle(tb._candidate(9), np.array([-1.21, -1.485]))


def test_saved_candidate_one_and_unsupported_request_are_structured():
    from dream.runtime.mobile_model import interpret_mobile_goal
    from dream.tools.goals import GoToObserved, UnsupportedGoal

    records = json.loads((SAVED.parent / "subsequent_requests.json").read_text())["records"]
    original = records[0]
    message = json.loads(next(m["text"] for m in original["messages"] if m["role"] == "user"))
    backend = Sequence([("inspect_cabinet", {})])
    goal = interpret_mobile_goal(
        backend, message["request"], {"entrance": [0, 0]}, message["observed_scene"], lambda: None
    )
    assert goal.candidate == 1
    backend = Sequence([("go_to_frontier", {"index": 0})])
    goal = interpret_mobile_goal(
        backend, "go to that observed frontier", {}, message["observed_scene"], lambda: None
    )
    assert isinstance(goal, GoToObserved)
    backend = Sequence([("unsupported", {"reason": "no grasp or pen tool"})])
    goal = interpret_mobile_goal(
        backend, "grab the black pen", {}, message["observed_scene"], lambda: None
    )
    assert isinstance(goal, UnsupportedGoal) and "Available" in goal.reason
