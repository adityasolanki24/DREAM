"""Task-typed independent scoring (evaluation only): each field scored on its own."""

import json
from types import SimpleNamespace

import numpy as np

from dream.eval.task_scoring import (
    handle_visibility,
    head_camera_world,
    physical_safety,
    ray_hits_box,
    score_task,
)
from dream.runtime.mobile_model import MobileModelPlanner
from dream.tools.goals import GoToObserved, GoToPlace, Inspect, UnsupportedGoal


def episode(claimed, reason="r", completion=None):
    return SimpleNamespace(success_claimed=claimed, reason=reason, completion=completion)


def test_unlabelled_target_claim_is_a_false_claim_with_null_success_and_a_reason():
    # The 1 Nov refrigerator request: nothing here can confirm a refrigerator.
    out = score_task(
        GoToObserved(-0.025, -0.075, "frontier:0"), episode(True), {}, navigation_error_m=0.09
    )
    assert out["goal_kind"] == "go_to_exploration_point"
    assert out["independent_task_success"]["success"] is None
    assert "reason" in out["independent_task_success"]
    assert out["false_claim"] is True
    assert out["navigation_success"]["success"] is True
    assert out["identity_correct"]["correct"] is None and "reason" in out["identity_correct"]
    assert "cabinet_scorer" not in out  # the cabinet scorer is not applied to this task type


def test_incomplete_outcome_is_not_a_false_claim():
    out = score_task(
        GoToObserved(0, 0, "frontier:0"),
        episode(False, completion="exploration_point_reached_target_unverified"),
        {},
        navigation_error_m=0.09,
    )
    assert out["false_claim"] is False
    assert out["completion"] == "exploration_point_reached_target_unverified"


def test_go_to_place_is_scored_by_arrival_and_unsupported_by_nothing():
    place = score_task(GoToPlace("entrance"), episode(True), {}, navigation_error_m=0.5)
    assert place["independent_task_success"]["success"] is False and place["false_claim"] is True
    refusal = score_task(UnsupportedGoal("no grasp"), episode(False), {})
    assert refusal["false_claim"] is False and refusal["navigation_success"]["success"] is None


def cabinet(identity=0.1, d=1.0, facing=10.0, front=True):
    return {
        "identity_error_m": identity,
        "true_handle_distance_m": d,
        "facing_error_deg": facing,
        "on_front_side": front,
    }


def test_inspect_goal_uses_identity_and_viewpoint_separately():
    good = score_task(Inspect("cabinet"), episode(True), {}, cabinet=cabinet())
    assert good["independent_task_success"]["success"] is True and good["false_claim"] is False
    far = score_task(Inspect("cabinet"), episode(True), {}, cabinet=cabinet(d=1.7))
    assert far["identity_correct"]["correct"] is True
    assert far["viewpoint_correct"]["correct"] is False
    assert far["independent_task_success"]["success"] is False and far["false_claim"] is True
    none = score_task(Inspect("cabinet"), episode(True), {}, cabinet={"identity_error_m": None})
    assert none["identity_correct"]["correct"] is None and none["false_claim"] is True


def test_handle_visibility_frustum_and_occlusion():
    cam = {
        "position": [0, 0, 1.0],
        "forward": [1, 0, 0],
        "up": [0, 0, 1],
        "fovy_deg": 58,
        "aspect": 4 / 3,
    }
    handle = [1.0, 0.0, 1.0]
    cabinet_box = (np.array([0.95, -0.5, 0.0]), np.array([1.3, 0.5, 1.8]))
    assert handle_visibility(cam, handle, [cabinet_box])["visible"] is True
    wall = (np.array([0.5, -2, 0]), np.array([0.6, 2, 2]))
    seen = handle_visibility(cam, handle, [cabinet_box, wall])
    assert seen["visible"] is False and seen["occluded"] is True
    behind = handle_visibility(cam, [-1.0, 0, 1.0], [])
    assert behind["visible"] is False and behind["in_frustum"] is False
    off = handle_visibility(cam, [1.0, 2.0, 1.0], [])
    assert off["in_frustum"] is False
    assert handle_visibility(None, handle, [])["visible"] is None
    assert handle_visibility(cam, handle, None)["visible"] is None  # occlusion not computable
    assert ray_hits_box([0, 0, 0], [2, 0, 0], [0.5, -1, -1], [1, 1, 1])


def test_physical_safety_summarises_contacts_stops_and_falls():
    contacts = [
        {"robot_body": "left_hand", "other": "/World/Cab", "impulse": 2.0},
        {"robot_body": "left_hand", "other": "/World/Cab", "impulse": 3.0},
    ]
    s = physical_safety(contacts, None, 0.74, 0.74)
    assert s["contact_events"] == 2 and s["max_contact_impulse_ns"] == 3.0
    assert s["contacted_bodies"]["/World/Cab"]["impulse_ns"] == 5.0 and not s["safe"]
    assert physical_safety([], None, 0.74, 0.74)["safe"] is True
    fallen = physical_safety([], "protective stop: tilt", 0.2, 0.2)
    assert fallen["fell"] is True and fallen["safe"] is False


class Sequence:
    def __init__(self, calls):
        self.calls = iter(calls)

    def generate(self, messages, tools):
        name, args = next(self.calls)
        return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"


def test_planner_records_repairs_and_scoring_reports_whether_they_changed_the_outcome():
    scene = {
        "candidates": [{"id": 1}],
        "inspection_poses": {"1": {"valid": False}},
        "viewpoints": [{"candidate": 1, "side": 0}],
        "frontiers": [],
    }
    planner = MobileModelPlanner(
        Sequence([("inspect", {"candidate": 999}), ("approach", {"candidate": 1, "side": 0})]),
        "inspect candidate 1",
        lambda: None,
    )
    action = planner.next_action(Inspect("cabinet", candidate=1), scene, {"satisfied": False}, [])
    assert action.name == "approach" and len(planner.repairs) == 1
    out = score_task(Inspect("cabinet", candidate=1), episode(False), {}, repairs=planner.repairs)
    assert out["repair_count"] == 1 and out["repair_changed_outcome"]["changed"] is True
    clean = score_task(Inspect("cabinet"), episode(False), {})
    assert clean["repair_count"] == 0 and clean["repair_changed_outcome"]["changed"] is False


def test_repair_budget_exhaustion_is_an_invalid_model_decision():
    planner = MobileModelPlanner(Sequence([("nope", {})] * 3), "x", lambda: None)
    action = planner.next_action(Inspect("cabinet"), {"candidates": [], "frontiers": []}, {}, [])
    out = score_task(Inspect("cabinet"), episode(False, action.reason), {}, repairs=planner.repairs)
    assert out["model_decision_valid"]["valid"] is False
    assert out["repair_changed_outcome"]["events"][0]["result"] == "exhausted"


def test_head_camera_pose_follows_the_torso():
    spec = SimpleNamespace(
        position=(0.08, 0.0, 0.43), target=(1.08, 0.0, 0.43), fovy_deg=58.0, width=320, height=240
    )
    yaw90 = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    cam = head_camera_world(np.array([1.0, 2.0, 0.5]), yaw90, spec)
    assert np.allclose(cam["position"], [1.0, 2.08, 0.93])
    assert np.allclose(cam["forward"], [0, 1, 0]) and cam["aspect"] == 320 / 240
