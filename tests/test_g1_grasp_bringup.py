"""Physics regressions for the fixed-stance grasp using the cached WBC."""

import json
from pathlib import Path

import pytest

from dream.robot.g1_wbc import policy_cache

pytest.importorskip("mink")
pytest.importorskip("daqp")
pytest.importorskip("onnxruntime")
from experiments.g1_grasp_bringup import HOLD_S, LIFT_M, TICK, attempt

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
    assert result["grasp_verified"], result["phases"]
    hold = [row for row in result["rows"] if row["phase"] == "hold"]
    assert len(hold) == round(HOLD_S / TICK)
    assert min(row["lift_m"] for row in hold) >= LIFT_M
    assert all(row["hand_only_contact"] for row in hold)


def test_geometry_ranked_pinch_retains_the_farther_box():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_fresh_far.json").read_text())
    result = attempt(dict(cfg, contact_geometry_search=True))
    assert result["phases"]["grasp_geometry"]["candidates"] > 1
    assert result["stop"] is None
    assert result["grasp_verified"], result["phases"]
