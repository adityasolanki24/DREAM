"""One physics episode of the reach test, end to end (slow): scene -> skill -> evaluator.

It checks the machinery and the honesty of the verdicts (the agent's claim against the
evaluator's truth), not a success rate; the rates are the experiment's job.
"""

import json
import sys
from pathlib import Path

import pytest

from dream.robot import g1_wbc as w

ROOT = Path(__file__).resolve().parents[1]
HAVE_WEIGHTS = (w.policy_cache() / "GR00T-WholeBodyControl-Walk.onnx").exists()


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_WEIGHTS, reason="WBC weights not fetched")
def test_mid_height_episode_is_judged_independently_and_does_not_collapse(tmp_path):
    sys.path.insert(0, str(ROOT))
    from experiments.g1_reach_heights import dev_specs, run_episode

    cfg = json.loads((ROOT / "experiments/configs/g1_reach_heights.json").read_text())
    spec = next(s for s in dev_specs(cfg) if s["height"] == 0.75 and s["layout"] == 0)
    r = run_episode((spec, "agent", cfg, str(tmp_path), "test", None))
    assert r["error"] is None, r["error"]
    ev, agent = r["evaluator"], r["agent"]
    assert agent["verdict"] in ("reached", "refused", "failed")
    assert not ev["collapse"]["collapsed"] and not ev["unresolved_protective_stop"]
    assert ev["reference_feasible"]  # a mid-height target in front is feasible in the envelope
    assert not ev["false_success_claim"]  # the claim is checked against the truth
    if agent["verdict"] == "reached":
        assert ev["true_distance_m"] <= 0.03
    assert agent["target"]["position"] and r["execution_trace"]["monitored_ticks"] > 0
    assert (tmp_path / "agent" / f"{spec['id']}.json").exists()  # checkpointed per episode
    assert (tmp_path / "reference" / f"{spec['id']}.json").exists()
