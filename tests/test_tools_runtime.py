"""Contracts for structured goals, the typed toolbox and the task runtime."""

import numpy as np
import pytest

from dream.eval.manipulation import object_in_tray, object_on_top
from dream.runtime.agent import Finish, RuleBasedPlanner, Step, TaskAgent, ToolCall
from dream.sim.tabletop import sample_tabletop
from dream.sim.tabletop_env import TabletopEnv
from dream.tools.goals import InRegion, OnTop, parse_goal, resting_on, verify
from dream.tools.toolbox import TOOL_SPECS, Toolbox, ToolResult
from dream.types import ObjectObservation

TRAY = {"tray": (np.array([0.5, 0.0, 0.0]), 0.07)}


def _obs(name, x, y, z=0.02, size=0.04):
    return ObjectObservation(name, name, np.array([x, y, z]), 0.0, size)


# ----- goals ----------------------------------------------------------------------------


def test_parse_structured_goals_and_reject_everything_else():
    assert parse_goal("in red tray") == InRegion("red", "tray")
    assert parse_goal({"type": "on", "object": "red", "target": "blue"}) == OnTop("red", "blue")
    for bad in ("put the red cube in the tray", "on red red", "under red blue", "in red"):
        with pytest.raises(ValueError):
            parse_goal(bad)


def test_verify_in_region_and_on_top_from_observations():
    objs = [_obs("red", 0.5, 0.02, 0.024), _obs("blue", 0.3, 0.2), _obs("green", 0.3, 0.2, 0.06)]
    assert verify(InRegion("red", "tray"), objs, TRAY).satisfied
    assert not verify(InRegion("blue", "tray"), objs, TRAY).satisfied
    assert verify(OnTop("green", "blue"), objs, TRAY).satisfied
    assert not verify(OnTop("blue", "green"), objs, TRAY).satisfied
    assert not verify(InRegion("purple", "tray"), objs, TRAY).satisfied
    held = [_obs("red", 0.5, 0.0, 0.18)]  # above the tray but still in the hand
    assert not verify(InRegion("red", "tray"), held, TRAY).satisfied


def test_resting_on_detects_stacks_only():
    objs = [_obs("blue", 0.3, 0.2), _obs("green", 0.3, 0.21, 0.06), _obs("red", 0.4, 0.2)]
    assert resting_on("blue", objs) == ["green"]
    assert resting_on("red", objs) == [] and resting_on("green", objs) == []


# ----- planner (no simulator) ------------------------------------------------------------------


def _scene(objs, holding=None):
    return {
        "objects": [
            {
                "name": o.name,
                "color": o.color,
                "position": o.position.tolist(),
                "yaw": 0.0,
                "size": o.size,
            }
            for o in objs
        ],
        "fixtures": {"tray": {"centre": [0.5, 0.0, 0.004], "half_width": 0.09}},
        "holding": holding,
    }


def _container_scene(objs, fixtures=None, investigation=None, holding=None):
    scene = _scene(objs, holding)
    scene["fixtures"] = fixtures or {}
    scene["articulated"] = [{"name": "box"}]
    scene["investigations"] = {} if investigation is None else {"box": investigation}
    return scene


def _inv(status, mechanism, supported, confidence=0.99):
    return {
        "status": status,
        "reason": "",
        "mechanism": mechanism,
        "confidence": confidence,
        "chi2_per_dof": 0.1,
        "interactions": 4,
        "opening_supported": supported,
    }


def test_planner_investigates_then_opens_a_closed_container_before_grasping():
    p = RuleBasedPlanner(probe_budget=9)
    objs = [_obs("red", 0.4, 0.1)]
    goal, unsat = InRegion("red", "box"), {"satisfied": False}
    assert p.next_action(goal, _container_scene(objs), unsat, []) == ToolCall(
        "investigate_physics", {"object": "box", "budget": 9, "strategy": "eig"}
    )
    learned = _container_scene(objs, investigation=_inv("identified", "prismatic", True))
    assert p.next_action(goal, learned, unsat, []) == ToolCall(
        "open_articulated",
        {"object": "box", "min_opening": 0.05},  # 2 + 2.5 + 0.5 cm
    )
    too_small = {"box": {"centre": [0.6, 0, 0.008], "half_width": 0.03}}
    ajar = _container_scene(objs, too_small, _inv("identified", "prismatic", True))
    assert p.next_action(goal, ajar, unsat, []).name == "open_articulated"  # open further
    wide = {"box": {"centre": [0.6, 0, 0.008], "half_width": 0.06}}
    opened = _container_scene(objs, wide, _inv("identified", "prismatic", True))
    assert p.next_action(goal, opened, unsat, []) == ToolCall("grasp", {"object": "red"})
    opened = [
        Step(
            ToolCall("open_articulated", {"object": "box"}),
            ToolResult("open_articulated", "succeeded"),
            0.0,
        )
    ]
    not_seen = _container_scene(objs, investigation=_inv("identified", "prismatic", True))
    assert p.next_action(goal, not_seen, unsat, opened) == ToolCall("retract")  # look again
    looked = opened + [Step(ToolCall("retract"), ToolResult("retract", "succeeded"), 0.0)]
    assert p.next_action(goal, not_seen, unsat, looked).name == "open_articulated"
    held = _container_scene(objs, investigation=_inv("identified", "prismatic", True))
    held["holding"] = "red"  # holding the cube while the box is shut: set it down first
    assert p.next_action(goal, held, unsat, []) == ToolCall(
        "place", {"object": "red", "target": "table"}
    )


@pytest.mark.parametrize(
    "inv,phrase",
    [
        (_inv("identified", "fixed", False), "not forcing"),
        (_inv("model_inadequate", "unknown", False), "none of my mechanism models"),
        (_inv("insufficient_information", "revolute", False, 0.7), "not identified"),
    ],
)
def test_planner_declines_to_open_without_a_supported_model(inv, phrase):
    p = RuleBasedPlanner()
    scene = _container_scene([_obs("red", 0.4, 0.1)], investigation=inv)
    act = p.next_action(InRegion("red", "box"), scene, {"satisfied": False}, [])
    assert isinstance(act, Finish) and not act.success and phrase in act.reason


def test_fixed_pull_baseline_never_investigates_and_gives_up_after_its_attempts():
    p = RuleBasedPlanner(opening="fixed_pull", max_attempts=2)
    scene = _container_scene([_obs("red", 0.4, 0.1)])
    goal, unsat = InRegion("red", "box"), {"satisfied": False}
    call = ToolCall("open_articulated", {"object": "box", "strategy": "fixed_pull"})
    assert p.next_action(goal, scene, unsat, []) == call
    pulls = [Step(call, ToolResult("open_articulated", "succeeded"), 0.0)] * 2
    act = p.next_action(goal, scene, unsat, pulls)
    assert isinstance(act, Finish) and not act.success


def test_planner_reduces_both_goal_types_to_grasp_then_place():
    p = RuleBasedPlanner()
    objs = [_obs("red", 0.4, 0.1), _obs("blue", 0.4, -0.1)]
    unsat = {"satisfied": False}
    assert p.next_action(InRegion("red", "tray"), _scene(objs), unsat, []) == ToolCall(
        "grasp", {"object": "red"}
    )
    assert p.next_action(OnTop("red", "blue"), _scene(objs, "red"), unsat, []) == ToolCall(
        "place", {"object": "red", "target": "blue"}
    )
    assert p.next_action(InRegion("red", "tray"), _scene(objs), {"satisfied": True}, []) == Finish(
        True, "goal verified by observation"
    )


def test_planner_clears_obstructions_and_sets_aside_wrong_objects():
    p = RuleBasedPlanner()
    stacked = [_obs("blue", 0.4, 0.1), _obs("green", 0.4, 0.1, 0.06), _obs("red", 0.4, -0.1)]
    unsat = {"satisfied": False}
    assert p.next_action(InRegion("blue", "tray"), _scene(stacked), unsat, []) == ToolCall(
        "grasp", {"object": "green"}
    )
    assert p.next_action(OnTop("red", "blue"), _scene(stacked), unsat, []) == ToolCall(
        "grasp", {"object": "green"}
    )
    assert p.next_action(InRegion("blue", "tray"), _scene(stacked, "green"), unsat, []) == ToolCall(
        "place", {"object": "green", "target": "table"}
    )


def test_planner_gives_up_after_attempt_budget_and_on_missing_objects():
    p = RuleBasedPlanner(max_attempts=2)
    objs = [_obs("red", 0.4, 0.1)]
    fails = [
        Step(ToolCall("grasp", {"object": "red"}), ToolResult("grasp", "failed", "missed"), 0.0)
    ] * 2
    act = p.next_action(InRegion("red", "tray"), _scene(objs), {"satisfied": False}, fails)
    assert isinstance(act, Finish) and not act.success
    act = p.next_action(OnTop("red", "purple"), _scene(objs), {"satisfied": False}, [])
    assert isinstance(act, Finish) and not act.success and "purple" in act.reason


# ----- toolbox and runtime in simulation ---------------------------------------------------------


def _env(seed):
    return TabletopEnv(
        sample_tabletop(np.random.default_rng(seed)), np.random.default_rng(seed + 1)
    )


def test_every_tool_declares_a_schema_and_bad_calls_are_rejected():
    env = _env(5)
    tb = Toolbox(env.embodiment(), env)
    expected = (
        "observe_scene get_robot_state move_end_effector grasp release place verify_goal "
        "investigate_physics open_articulated retract"
    )
    assert {t["name"] for t in TOOL_SPECS} == set(expected.split())
    # A tray scene has no articulated object: both tools refuse before touching anything.
    assert tb.call("investigate_physics", {"object": "drawer"}).status == "rejected"
    assert tb.call("open_articulated", {"object": "drawer"}).status == "rejected"
    assert tb.call("teleport", {"object": "red"}).status == "rejected"
    assert tb.call("grasp", {}).status == "rejected"
    assert tb.call("grasp", {"object": "red", "force": 100}).status == "rejected"
    assert tb.call("move_end_effector", {"position": [0.5, 0.0]}).status == "rejected"
    assert tb.call("move_end_effector", {"position": [0.5, 0.0, -1.0]}).status == "rejected"
    assert tb.call("place", {"object": "red", "target": "tray"}).status == "rejected"  # not holding
    assert tb.call("verify_goal", {"goal": "please tidy up"}).status == "rejected"


def test_cancel_stops_a_tool_and_leaves_the_robot_holding_still():
    env = _env(5)
    tb = Toolbox(env.embodiment(), env)
    calls = {"n": 0}
    original = env.wait_until

    def wait_then_cancel(pred, timeout):
        calls["n"] += 1
        if calls["n"] == 3:
            tb.cancel.set()
        return original(pred, timeout)

    env.wait_until = wait_then_cancel
    r = tb.call("grasp", {"object": "red"})
    assert r.status == "stopped"
    q = env.robot_state().joint_pos
    env.step(500)
    assert np.max(np.abs(env.robot_state().joint_pos - q)) < 0.02


@pytest.mark.slow
def test_same_tools_solve_tray_and_stack_goals_in_one_persistent_scene():
    env = _env(7)
    agent = TaskAgent(Toolbox(env.embodiment(), env))
    ep = agent.run(parse_goal("in red tray"))
    env.step(250)
    assert ep.success_claimed and object_in_tray(env, "red")
    ep = agent.run(parse_goal("on blue green"))
    env.step(250)
    assert ep.success_claimed and object_on_top(env, "blue", "green")
    ep = agent.run(parse_goal("in green tray"))  # green is under blue: must clear it first
    env.step(250)
    assert ep.success_claimed and object_in_tray(env, "green")
    assert [s.call.args.get("object") for s in ep.steps][:2] == ["blue", "blue"]


def test_read_only_tools_still_answer_while_cancel_is_set():
    """Regression: a pause between agent steps made get_robot_state return no data, and the agent
    thread crashed with a KeyError."""
    env = _env(5)
    tb = Toolbox(env.embodiment(), env)
    tb.cancel.set()
    for name in ("observe_scene", "get_robot_state"):
        assert tb.call(name).ok
    assert tb.call("verify_goal", {"goal": "in red tray"}).ok
    assert tb.call("grasp", {"object": "red"}).status == "stopped"
    ep = TaskAgent(tb).run(parse_goal("in red tray"))
    assert not ep.success_claimed and "cancel" in ep.reason


def test_an_opening_whose_release_fails_is_reported_as_failed():
    """Regression (held-out drawer 5005): a stop latched while letting go was hidden."""
    from dream.skills.manipulation import SkillResult
    from dream.tools.discovery import with_release

    opened = SkillResult(True, "", details={"q_reached": 0.15})
    stuck = SkillResult(False, "protective stop: hand external force 40 N > 40 N")
    r = with_release(opened, stuck)
    assert not r.ok and "releasing the handle failed" in r.reason
    assert r.details["q_reached"] == 0.15
    assert with_release(opened, SkillResult(True)).ok
    failed = SkillResult(False, "handle left the learned constraint")
    assert with_release(failed, stuck).reason == failed.reason  # the first failure is the cause


def test_a_re_grasp_uses_where_the_hand_let_go_unless_the_camera_shows_the_handle_moved():
    from dataclasses import replace

    from dream.tools.discovery import ReleasedHandle, best_handle_estimate
    from dream.types import ArticulatedObservation

    obs = ArticulatedObservation(
        "box",
        np.array([0.5, 0.0, 0.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.ones(3),
        position_std=0.025,
        handle_id=7,
        last_seen=1.0,
    )
    released = ReleasedHandle(
        "box", 7, np.array([0.5, 0.0, 0.3]), obs.handle_pos.copy(), 0.025, 2.0
    )
    est = best_handle_estimate(obs, released, 3.0)
    assert np.allclose(est.handle_pos, released.position)
    assert est.position_std >= released.position_std  # historical estimate, not current contact
    moved = replace(obs, handle_pos=np.array([0.5, 0.0, 0.24]), position_std=0.007, last_seen=3.0)
    assert best_handle_estimate(moved, released, 3.0) is moved
    for unusable in (replace(obs, name="other"), replace(obs, handle_id=8)):
        assert best_handle_estimate(unusable, released, 3.0) is unusable
    assert best_handle_estimate(obs, None, 3.0) is obs
    assert best_handle_estimate(obs, released, 100.0) is obs


@pytest.mark.parametrize("lost,release_ok", [(False, True), (True, True), (False, False)])
def test_only_a_confirmed_grasp_and_successful_release_leave_a_regrasp_estimate(
    monkeypatch, lost, release_ok
):
    from types import SimpleNamespace

    from dream.skills.manipulation import SkillResult
    from dream.tools import discovery as d
    from dream.types import ArticulatedObservation

    state = SimpleNamespace(
        time=2.0, gripper_command=0.0, gripper_width=0.025, tcp_pos=np.array([0.5, 0.0, 0.3])
    )
    robot = SimpleNamespace(state=lambda: state)
    obs = ArticulatedObservation(
        "box",
        state.tcp_pos,
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.ones(3),
        handle_id=7,
        position_std=0.025,
    )
    inv = SimpleNamespace(
        belief=SimpleNamespace(map_type=lambda: d.REVOLUTE, h0=np.zeros(3)),
        opening_supported=True,
        handle_source="grasp",
        released_handle=None,
    )
    monkeypatch.setattr(d, "MAX_SEGMENTS", 1)
    monkeypatch.setattr(d, "_ensure_grasped", lambda *a: SkillResult(True))
    monkeypatch.setattr(d, "_handle_now", lambda *a: state.tcp_pos)
    monkeypatch.setattr(d, "plan_segment", lambda *a: ([state.tcp_pos], 0.0, 0.25))
    monkeypatch.setattr(d, "constraint_deviation", lambda *a: (0.25, 0.0))
    monkeypatch.setattr(d.art, "find_articulated", lambda *a: obs)
    monkeypatch.setattr(d.art, "release_handle", lambda *a: SkillResult(release_ok))

    def pull(*args):
        if lost:
            state.gripper_width = 0.0
        return SkillResult(True)

    monkeypatch.setattr(d.art, "pull_along", pull)
    d.open_learned(robot, SimpleNamespace(fixtures=lambda: {}), "box", inv)
    assert (inv.released_handle is not None) == (not lost and release_ok)


def test_setting_an_unwanted_object_aside_is_bounded():
    """Regression: a perceived 'held object' that could not be set aside was retried until the
    step budget ran out (15 place calls in one development run)."""
    p = RuleBasedPlanner(max_attempts=2)
    objs = [_obs("red", 0.4, 0.1), _obs("grey", 0.5, 0.0, 0.2)]
    fails = [
        Step(
            ToolCall("place", {"object": "grey", "target": "table"}),
            ToolResult("place", "timeout", "t"),
            0.0,
        )
    ] * 2
    act = p.next_action(InRegion("red", "tray"), _scene(objs, "grey"), {"satisfied": False}, fails)
    assert isinstance(act, Finish) and not act.success and "grey" in act.reason
