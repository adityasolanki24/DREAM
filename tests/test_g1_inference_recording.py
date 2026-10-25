"""Test-double decisions exercise the actual runner; these are not model evidence."""

import json
from pathlib import Path

import pytest

from experiments.g1_explore_unknown import run_case


@pytest.mark.slow
def test_g1_runner_records_observation_action_belief_and_next_decision(tmp_path):
    class Scripted:
        name = "test-double-not-qwen"

        def __init__(self):
            self.calls = iter(["reach_object", "probe_object", "inspect_object"])

        def set_cancel_event(self, cancel):
            pass

        def generate(self, messages, tools):
            call = json.dumps({"name": next(self.calls), "arguments": {"object": "red"}})
            return f"<tool_call>{call}</tool_call>"

    cfg = json.loads(Path("experiments/configs/g1_explore_dev.json").read_text())
    cfg.update(backend="qwen", max_steps=3)
    folder = tmp_path / "test_double"
    result = run_case(cfg["cases"][0], cfg, folder, backend=Scripted(), recording=False)
    assert result["physics_executed"]
    assert not result["success_claimed"]
    assert len(json.loads((folder / "belief.json").read_text())) == 1
    inputs = sorted((folder / "inference").glob("*_input.json"))
    times = [json.loads(p.read_text())["sim_time_before"] for p in inputs]
    assert len(times) == 3 and times[-1] > times[0]
    for output in (folder / "inference").glob("*_output.json"):
        record = json.loads(output.read_text())
        assert record["text"] and record["utc"] and record["latency_s"] >= 0
    assert list((folder / "frames").glob("*.npz"))
