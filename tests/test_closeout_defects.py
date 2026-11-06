"""Regression tests for the 1 Nov 2026 genuine-Qwen session defects (R3-a, R3-b).

R3-a: a request bound to a frontier already under the robot, verified at 0.09 m, was reported
as task success. R3-b: a viewpoint at 1.37 m of a 1.4 m inspection limit was chosen although
measured arrival errors are 0.10-0.15 m. Model doubles are local tests only.
"""

import json
import threading
from types import SimpleNamespace

import numpy as np

from dream.perception.spatial import Candidate, SpatialMemory
from dream.runtime.agent import Finish, TaskAgent
from dream.runtime.mobile_model import MobileModelPlanner, available_actions, interpret_mobile_goal
from dream.tools.goals import GoToObserved, UnsupportedGoal
from dream.tools.mobile import (
    ARRIVAL_MARGIN,
    INSPECT_RANGE,
    PLACE_TOLERANCE,
    MobileToolbox,
)
from dream.tools.toolbox import ToolResult

EXPLORATION = "exploration_point_reached_target_unverified"


class Model:
    def __init__(self, *calls):
        self.calls = list(calls)
        self.n = 0

    def generate(self, messages, tools):
        name, args = self.calls[min(self.n, len(self.calls) - 1)]
        self.n += 1
        return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"


def scene_with(frontiers, pose=(-0.078, -0.002, 0.003)):
    return {"pose": list(pose), "candidates": [], "viewpoints": [], "frontiers": frontiers}


# ----- D1: a frontier already reached is not a navigation target ---------------------------
def test_frontier_under_the_robot_is_not_bound_as_a_goal():
    # The saved 1 Nov observation: cost 0.07 m, 0.09 m from the robot.
    scene = scene_with([{"xy": [-0.025, -0.075], "gain_m2": 23.76, "cost_m": 0.07}])
    goal = interpret_mobile_goal(
        Model(("go_to_frontier", {"index": 0})), "walk to the fridge", {}, scene, lambda: None
    )
    assert isinstance(goal, UnsupportedGoal) and "already reached" in goal.reason


def test_distant_frontier_is_still_bound():
    scene = scene_with([{"xy": [-1.6, 2.1], "gain_m2": 16.0, "cost_m": 2.65}])
    goal = interpret_mobile_goal(
        Model(("go_to_frontier", {"index": 0})), "go there", {}, scene, lambda: None
    )
    assert isinstance(goal, GoToObserved) and goal.reference == "frontier:0"


def test_reached_frontier_is_not_an_admissible_action():
    scene = scene_with(
        [
            {"xy": [-0.025, -0.075], "gain_m2": 1.0, "cost_m": 0.07},
            {"xy": [2.0, 2.0], "gain_m2": 1.0, "cost_m": 3.0},
        ]
    )
    goal = SimpleNamespace(candidate=None)
    xs = [a.args["x"] for a in available_actions(goal, scene, {}, []) if a.name == "go_to"]
    assert xs == [2.0]


def test_observation_withholds_reached_frontiers():
    memory = SpatialMemory(np.random.default_rng(0))
    memory.grid.logodds[:] = -2.0
    base = SimpleNamespace(odometry=lambda: np.zeros(3), stop_reason=None, time=0.0)
    tb = MobileToolbox(base, memory, navigator="rpp")
    tb._look = lambda: None
    data = tb.call("observe_scene").data
    assert all(
        f["cost_m"] > PLACE_TOLERANCE and np.hypot(*f["xy"]) > PLACE_TOLERANCE
        for f in data["frontiers"]
    )
    assert "frontiers_note" in data or data["frontiers"]


def test_observation_states_when_no_frontier_remains():
    memory = SpatialMemory(np.random.default_rng(0))
    memory.grid.logodds[:] = -2.0  # everything known: no frontier at all
    base = SimpleNamespace(odometry=lambda: np.zeros(3), stop_reason=None, time=0.0)
    tb = MobileToolbox(base, memory, navigator="rpp")
    tb._look = lambda: None
    data = tb.call("observe_scene").data
    assert data["frontiers"] == [] and "no unexplored frontier" in data["frontiers_note"]


# ----- D2: exploration is not task completion ---------------------------------------------
class FakeTools:
    """Observation doubles for the saved 1 Nov verdict: reached at 0.09 m, sim time 0."""

    def __init__(self, verdict):
        self.verdict, self.cancel = verdict, threading.Event()

    def call(self, name, args=None):
        data = {
            "observe_scene": scene_with([]),
            "verify_goal": self.verdict,
            "get_robot_state": {"protective_stop": False},
        }.get(name, {})
        return ToolResult(name, "succeeded", "", data)


def test_exploration_goal_cannot_succeed_even_if_verifier_and_model_say_so():
    goal = GoToObserved(-0.025, -0.075, "frontier:0")
    saved_verdict = {"satisfied": True, "reason": "0.09 m away", "reference": "frontier:0"}
    model = Model(("finish", {"success": True, "reason": "The refrigerator has been inspected."}))
    planner = MobileModelPlanner(model, "walk to the refrigerator", lambda: None)
    episode = TaskAgent(FakeTools(saved_verdict), planner).run(goal)
    assert not episode.success_claimed
    assert episode.completion == EXPLORATION
    assert "refrigerator" not in episode.reason  # user-facing text is not the model's
    assert model.n == 0  # no model call is needed (or allowed) to reach the typed outcome


def test_agent_guard_overrides_any_planner_success_for_exploration_goals():
    class Liar:
        def next_action(self, goal, scene, verdict, history):
            return Finish(True, "arrived at the refrigerator")

    goal = GoToObserved(1.0, 1.0, "frontier:3")
    episode = TaskAgent(FakeTools({"satisfied": True, "reason": "0.0 m away"}), Liar()).run(goal)
    assert not episode.success_claimed and episode.completion == EXPLORATION
    assert episode.model_reason == "arrived at the refrigerator"
    assert episode.to_dict()["completion"] == EXPLORATION


def test_non_frontier_go_to_observed_is_unchanged():
    class Done:
        def next_action(self, goal, scene, verdict, history):
            return Finish(True, "at target")

    goal = GoToObserved(1.0, 1.0, "candidate:2:side:3")
    episode = TaskAgent(FakeTools({"satisfied": True, "reason": "0.0 m away"}), Done()).run(goal)
    assert episode.success_claimed and episode.completion is None


def test_verifier_reports_frontier_target_as_reached_but_unverified():
    memory = SpatialMemory(np.random.default_rng(0))
    pose = np.array([0.0, 0.0, 0.0])
    base = SimpleNamespace(odometry=lambda: pose.copy(), stop_reason=None, time=0.0)
    tb = MobileToolbox(base, memory, navigator="rpp")
    v = tb.call("verify_goal", {"goal": GoToObserved(0.1, 0.0, "frontier:0")}).data
    assert not v["satisfied"] and v["reached"] and v["completion"] == EXPLORATION
    v = tb.call("verify_goal", {"goal": GoToObserved(0.1, 0.0, "candidate:1:side:2")}).data
    assert v["satisfied"] and "completion" not in v


def test_finish_text_comes_from_the_typed_verdict_not_the_model():
    planner = MobileModelPlanner(
        Model(("finish", {"success": True, "reason": "ZZZ free text"})), "x", lambda: None
    )
    verdict = {"satisfied": True, "reason": "handle observed from a valid inspection pose"}
    action = planner.next_action(object(), {}, verdict, [])
    assert action.success and "ZZZ" not in action.reason
    assert action.model_reason == "ZZZ free text"


# ----- D4: leave an arrival margin when choosing a viewpoint ------------------------------
def saved_candidate_one():
    # 1 Nov session 150528, decision 0009: the viewpoint (1, 5) was 1.37 m from the handle.
    return Candidate(
        1,
        "orange",
        np.array([1.143, 0.204, 0.523]),
        0.0,
        np.array([1.308, 1.97, 1.104]),
        views=14,
        handle=np.array([1.094, -0.679, 0.366]),
        handle_views=2,
        sides_close={3, 4, 5},
    )


def open_room_toolbox(pose):
    memory = SpatialMemory(np.random.default_rng(0))
    memory.grid.logodds[:] = -2.0
    memory.candidates.append(saved_candidate_one())
    base = SimpleNamespace(odometry=lambda: np.array(pose, float), stop_reason=None, time=0.0)
    return MobileToolbox(base, memory, navigator="rpp")


def test_selected_viewpoint_leaves_an_arrival_margin():
    assert ARRIVAL_MARGIN == 0.15
    tb = open_room_toolbox([0.08, 0.192, -0.507])
    ok, _, dist = tb._passable()
    c = tb._candidate(1)
    chosen_any = False
    for k in range(8):
        chosen = tb.side_point(c, k, ok, dist)
        if chosen is not None:
            chosen_any = True
            d = float(np.linalg.norm(chosen[0] - c.handle[:2]))
            assert INSPECT_RANGE[0] <= d <= INSPECT_RANGE[1] - ARRIVAL_MARGIN + 1e-9, (k, d)
    assert chosen_any


def test_old_boundary_viewpoint_is_not_selected_but_predicate_is_unchanged():
    tb = open_room_toolbox([0.08, 0.192, -0.507])
    c = tb._candidate(1)
    old = np.array([-0.275, -0.625])  # 1.37 m from the handle: the 1 Nov choice
    assert (
        INSPECT_RANGE[1] - ARRIVAL_MARGIN < np.linalg.norm(old - c.handle[:2]) <= INSPECT_RANGE[1]
    )
    # Only a thin band at the inspection limit is reachable, as on the saved map.
    grid = tb.memory.grid
    cells = np.argwhere(np.ones_like(grid.logodds, bool))
    ring = np.zeros(grid.logodds.shape, bool)
    d = np.linalg.norm(grid.centre(cells[:, 0], cells[:, 1]) - c.handle[:2], axis=1)
    ring[cells[:, 0], cells[:, 1]] = (d > 1.30) & (d <= 1.40)
    dist = np.where(ring, 1.0, np.inf)
    assert tb.side_point(c, 5, ring, dist) is None
    # The inspection predicate is not weakened or tightened: a pose within 1.4 m still counts.
    facing = np.arctan2(*(c.handle[:2] - old)[::-1])
    assert tb.inspection_pose(c, np.array([*old, facing]))[0]
