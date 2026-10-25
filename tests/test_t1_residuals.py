"""Diagnostic logic only; no locomotion or safe nonzero tolerance is established."""

import json

import pytest

from experiments.inspect_t1_residuals import inspect


def test_inward_solver_velocity_does_not_hide_growing_sampled_penetration(tmp_path):
    (tmp_path / "articulation.json").write_text(
        json.dumps(
            {
                "joint_names": ["Left_Knee_Pitch"],
                "limits_rad": [[0, 2]],
            }
        )
    )
    rows = [
        {"time_s": t, "proprioception": {"q": [q], "dq": [dq]}, "joint_target_rad": [-0.2]}
        for t, q, dq in ((0.005, 0.001, -0.1), (0.010, -0.00001, 0.01), (0.015, -0.0001, 0.5))
    ]
    (tmp_path / "physics_substeps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    audit = inspect(tmp_path)
    assert audit["violation_samples"] == 2
    assert audit["last_sample_velocity_inward"]
    assert audit["residual_grew_on_last_sample"]
    assert audit["final_position_difference_velocity_rad_s"] == pytest.approx(-0.018)
    assert audit["runtime_change"].startswith("none")
