"""Physics regressions for the fixed-stance grasp using the cached WBC."""

import json
from pathlib import Path

import numpy as np
import pytest

from dream.robot.g1_wbc import policy_cache

pytest.importorskip("mink")
pytest.importorskip("daqp")
pytest.importorskip("onnxruntime")
from experiments.g1_grasp_bringup import HOLD_S, LIFT_M, TICK, _attempt, attempt

CONFIG = Path(__file__).resolve().parents[1] / "experiments/configs"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all(
            (policy_cache() / f"GR00T-WholeBodyControl-{mode}.onnx").exists()
            for mode in ("Balance", "Walk")
        ),
        reason="cached WBC weights required",
    ),
]


def test_tabletop_touch_does_not_trigger_a_full_lift():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_both_arms_clear.json").read_text())
    result = attempt(dict(cfg, trajectory_search=False))
    phases = result["phases"]
    assert "raise" in phases or "reach" in phases
    assert "lift" not in phases
    assert not result["grasp_verified"]
    assert not any(
        row["phase"] in {"close", "probe"} and "tabletop" in row["hand_contact_with"]
        for row in result["rows"]
    )


def test_clearance_checks_the_motion_to_a_feasible_endpoint():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_pinch.json").read_text())
    result = attempt(dict(cfg, trajectory_search=False))
    assert result["stop"]["reason"] == "hand clearance"
    assert result["phases"]["raise"]["stops"]
    assert "close" not in result["phases"]
    assert not any(
        "tabletop" in row["hand_contact_with"] or "tabletop" in row["left_hand_contact_with"]
        for row in result["rows"]
    )


@pytest.mark.parametrize("fixture", ["pinch", "lateral"])
def test_verified_free_standing_grasp_on_changed_box_positions(fixture):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(cfg)
    assert result["stop"] is None
    assert result["grasp_observed"], result["phases"]
    assert result["grasp_verified"], result["phases"]
    hold = [row for row in result["rows"] if row["phase"] == "hold"]
    assert len(hold) == round(HOLD_S / TICK)
    assert min(row["lift_m"] for row in hold) >= LIFT_M
    assert min(row["table_clearance_m"] for row in hold) >= LIFT_M
    assert all(row["hand_only_contact"] for row in hold)


def test_geometry_ranked_pinch_retains_the_farther_box():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_fresh_far.json").read_text())
    result = attempt(dict(cfg, contact_geometry_search=True))
    assert result["phases"]["grasp_geometry"]["candidates"] > 1
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


@pytest.mark.parametrize("fixture", ["shift_out", "rectangular"])
def test_rehearsal_recovers_a_failed_geometric_pinch(fixture):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(dict(cfg, contact_rehearsal_search=True))
    rank = result["phases"]["grasp_geometry"]["rank"]
    rehearsals = result["phases"]["rehearsal"]
    assert rehearsals[0]["minimum_hold_lift_m"] < LIFT_M
    assert rehearsals[rank]["minimum_hold_lift_m"] >= LIFT_M
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


@pytest.mark.parametrize(("fixture", "preview_holds"), [("pinch", False), ("shift_out", True)])
def test_whole_body_preview_filters_the_proposed_grasp(fixture, preview_holds):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(dict(cfg, whole_body_grasp_preview=True))
    preview = result["phases"]["whole_body_grasp_preview"]
    assert preview["proposed_rank"] > 0
    assert preview["attempts"][0]["verified"] is preview_holds
    expected_rank = preview["attempts"][-1]["rank"] if preview["predicted_verified"] else 0
    assert result["phases"]["grasp_geometry"]["rank"] == expected_rank
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


def test_wrap_family_holds_the_bottle_from_geometric_and_physical_selection():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_cylinder.json").read_text())
    cfg.update(
        seed=53,
        object_shape="bottle",
        object_half_extents_m=[0.033, 0.033, 0.064],
        object_mass_kg=0.10,
        object_sliding_friction=0.85,
        table_top_m=0.74,
        object_center_xy_m=[0.465, -0.175],
        whole_body_grasp_preview=True,
        grasp_depth_search=True,
    )
    result = attempt(cfg)
    assert result["phases"]["whole_body_grasp_preview"]["family"] == "proximal_wrap"
    assert result["grasp_observed"]
    assert result["grasp_verified"], result["phases"]


@pytest.mark.parametrize(("fixture", "uses_thumb_bias"), [("cylinder", False), ("shift_in", True)])
def test_expanded_contact_search_retains_the_object(fixture, uses_thumb_bias):
    cfg = json.loads((CONFIG / f"g1_grasp_bringup_{fixture}.json").read_text())
    result = attempt(dict(cfg, whole_body_grasp_preview=True, grasp_depth_search=True))
    geometry = result["phases"]["grasp_geometry"]
    assert geometry["offset_m"][0] != 0.0
    assert bool(geometry["thumb_bias_rad"]) is uses_thumb_bias
    assert result["phases"]["whole_body_grasp_preview"]["predicted_verified"]
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]


def test_observed_object_loss_stops_despite_pressure_between_fingers():
    # Reproduce the bottle prediction whose tactile pressure persisted after
    # the object slipped to the table. Candidate rank comes from its recorded
    # geometric search, with the same independent planning material and noise.
    cfg = json.loads((CONFIG / "g1_grasp_bringup_cylinder.json").read_text())
    cfg.update(
        seed=np.random.SeedSequence(0, spawn_key=(1,)),
        object_shape="bottle",
        object_half_extents_m=[0.033, 0.033, 0.064],
        object_mass_kg=0.10,
        table_top_m=0.74,
        object_center_xy_m=[0.465, -0.175],
        contact_rehearsal_search=True,
        grasp_depth_search=True,
    )
    result = _attempt(cfg, planned_rank=414)
    assert not result["grasp_verified"]
    assert result["stop"]["reason"] == "object contact lost"


@pytest.mark.parametrize("rotated", [False, True])
def test_walk_to_grasp_reaches_a_selected_stance_and_holds(rotated):
    cfg = json.loads((CONFIG / "g1_walk_grasp.json").read_text())
    heading = 0.0
    if rotated:
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        translation = np.array([-0.4, 0.25])
        for key in ("table_center_xy_m", "object_center_xy_m"):
            cfg[key] = (rotation @ cfg[key] + translation).tolist()
        start = cfg["starting_stance_xy_yaw"]
        heading = np.pi / 2
        cfg["starting_stance_xy_yaw"] = [
            *(rotation @ start[:2] + translation),
            start[2] + heading,
        ]
        cfg["seed"] = 89
    result = attempt(cfg)
    phases = result["phases"]
    assert phases["stance"]["goal"] != tuple(cfg["starting_stance_xy_yaw"])
    assert phases["stance"]["heading"] == pytest.approx(heading)
    walk = phases["walk"]
    assert walk["status"] == "arrived"
    assert walk["true_displacement_m"] >= 1.0
    assert walk["table_contact_samples"] == 0
    assert result["grasp_observed"] and result["grasp_verified"], phases
    rows = result["rows"]
    assert max(r["t"] for r in rows if r["phase"] == "walk") < min(
        r["t"] for r in rows if r["phase"] == "hold"
    )
