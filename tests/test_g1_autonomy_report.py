"""A missing attempt or rejected assertion cannot improve the comparison score."""

import json

from experiments.g1_autonomy_report import report


def test_missing_attempts_and_rejected_success_are_reported_separately(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cases": [{"name": "example", "seed": 1}], "model": {}}))
    run = tmp_path / "run"
    episode = run / "qwen/example"
    (episode / "inference").mkdir(parents=True)
    (episode / "summary.json").write_text(
        json.dumps({"goal_success": False, "false_success_claim": False})
    )
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "call": {"name": "finish", "arguments": {"success": True}},
                        "message": "success rejected: the goal is not verified",
                    }
                ]
            }
        )
    )
    (episode / "inference/000_output.json").write_text(
        json.dumps({"text": "test double", "latency_s": 0.1})
    )
    result = report(run, config)
    assert not result["evidence_gate_passed"]
    model, baseline = result["episodes"]
    assert model["rejected_success_assertions"] == 1 and not model["false_success_claim"]
    assert model["model_calls_attempted"] == 1
    assert not baseline["record_present"]
    assert baseline["peak_scored_hand_object_normal_force_n"] is None
    assert result["totals"]["deterministic/development"]["attempts"] == 1
    assert result["totals"]["deterministic/development"]["goal_success"] == 0
