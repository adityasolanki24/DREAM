"""Regression for the fixed-stance tabletop grasp under the cached WBC."""

import json
from pathlib import Path

import pytest

from dream.robot.g1_wbc import policy_cache

pytest.importorskip("mink")
pytest.importorskip("daqp")
pytest.importorskip("onnxruntime")
from experiments.g1_grasp_bringup import attempt

CONFIG = Path(__file__).resolve().parents[1] / "experiments/configs"


@pytest.mark.slow
@pytest.mark.skipif(
    not all(
        (policy_cache() / f"GR00T-WholeBodyControl-{mode}.onnx").exists()
        for mode in ("Balance", "Walk")
    ),
    reason="cached WBC weights required",
)
def test_tabletop_touch_does_not_trigger_a_full_lift():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_both_arms_clear.json").read_text())
    result = attempt(cfg)
    phases = result["phases"]
    assert "raise" in phases or "reach" in phases
    assert "lift" not in phases
    assert not result["grasp_verified"]
    assert not any(
        row["phase"] in {"close", "probe"} and "tabletop" in row["hand_contact_with"]
        for row in result["rows"]
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not all(
        (policy_cache() / f"GR00T-WholeBodyControl-{mode}.onnx").exists()
        for mode in ("Balance", "Walk")
    ),
    reason="cached WBC weights required",
)
def test_clearance_checks_the_motion_to_a_feasible_endpoint():
    cfg = json.loads((CONFIG / "g1_grasp_bringup_pinch.json").read_text())
    result = attempt(cfg)
    assert result["stop"]["reason"] == "hand clearance"
    assert result["phases"]["raise"]["stops"]
    assert "close" not in result["phases"]
    assert not any(
        "tabletop" in row["hand_contact_with"] or "tabletop" in row["left_hand_contact_with"]
        for row in result["rows"]
    )
