"""End-to-end contracts of object search on the kinematic surrogate (slow).

The episode runner of experiments/semantic_search_eval.py drives TaskAgent with the semantic
planner and scores the ORIGINAL request against the true scene. Development seeds only."""

import json
import pathlib

import pytest

from experiments.semantic_search_eval import episode

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "experiments/configs/semantic_search_dev.json").read_text())


def run(scenario, arm="proposed", seed=100, budget=300.0):
    cfg = {**CFG, "time_budget_s": budget}
    return episode({"cfg": cfg, "arm": cfg["arms"][arm], "arm_name": arm, "seed": seed,
                    "scenario": scenario, "target": "refrigerator", "generator": "nominal",
                    "noise": "nominal", "base": "kinematic"})  # fmt: skip


@pytest.mark.slow
def test_a_visible_refrigerator_is_found_and_the_claim_is_true():
    r = run("A")
    assert r["result"] == "success" and r["claimed"] and not r["false_success_claim"]
    assert r["verdict"]["status"] == "target_found_verified"


@pytest.mark.slow
def test_an_unsupported_manipulation_request_is_refused_without_moving():
    r = run("L")
    assert r["result"] == "correct_refusal" and r["completion"] == "goal_unsupported"
    assert r["tool_calls"] == 0 and r["path_m"] == 0.0


@pytest.mark.slow
def test_an_absent_refrigerator_is_never_claimed():
    r = run("E", budget=200.0)
    assert not r["claimed"] and r["result"] == "correct_refusal"


@pytest.mark.slow
def test_room_arrival_cannot_stand_in_for_the_object():
    """Fault injection: a planner that claims success on reaching the kitchen."""
    gated = run("E", "abl_substitution_gated", budget=300.0)
    ungated = run("E", "abl_substitution_ungated", budget=300.0)
    assert not gated["claimed"] and "not accepted" in gated["reason"]
    assert ungated["claimed"] and ungated["result"] == "false_success"
