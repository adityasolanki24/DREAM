"""Feedback wiring with explicit stubs; never evidence of pretrained model performance."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from experiments.cloud_molmo_closedloop import run_condition

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / "experiments/configs/molmo_closedloop.json").read_text())
    cfg["max_inferences"] = 2
    sensors = json.loads((ROOT / "experiments/configs/molmo_openloop.json").read_text())
    return cfg, sensors


@pytest.mark.parametrize("seed", [223, 224])
def test_feedback_reobserves_after_real_motion_and_never_runs_a_classical_fallback(
    tmp_path, seed, monkeypatch
):
    from dream.sim.camera import CAMERA_NAME, RGBDCamera

    closed = []
    original_close = RGBDCamera.close

    def close(camera):
        closed.append(camera.spec.name)
        original_close(camera)

    monkeypatch.setattr(RGBDCamera, "close", close)

    class Stub:
        """Known safe test target, not a learned output."""

        states = []

        def predict(self, request):
            self.states.append(request["state"].copy())
            raw = np.tile(request["state"], (15, 1))
            raw[:, 6] += 0.015
            raw[:, 7] = 0
            return raw

    cfg, sensors = config()
    cfg["seed"] = seed
    backend = Stub()
    result = run_condition(backend, cfg, sensors, "learned_joint", tmp_path / "test")
    assert sorted(closed) == [CAMERA_NAME, "wrist"]
    assert result["fresh_model_inference"] is False  # a test stub cannot claim real inference
    assert len(result["steps"]) == 2 and all(s["target_reached"] for s in result["steps"])
    assert len(backend.states) == 2
    assert backend.states[1][6] - backend.states[0][6] > 0.01
    assert not result["classical_fallback"] and not result["protective_stop"]
    trace = np.load(tmp_path / "test/physical_trace.npz")["trace"]
    assert trace[-1, 0] > trace[0, 0] and np.ptp(trace[:, 7]) > 0.02
    acceleration = np.diff(trace[:, 8:15], axis=0) / np.diff(trace[:, 0, None], axis=0)
    assert np.max(np.abs(acceleration)) < 4


def test_budget_expiring_in_inference_saves_output_and_does_not_dispatch(tmp_path):
    class SlowStub:
        def predict(self, request):
            time.sleep(0.1)
            raw = np.tile(request["state"], (15, 1))
            raw[:, 6] += 0.1
            return raw

    cfg, sensors = config()
    cfg["max_episode_wall_s"] = 0.05
    result = run_condition(SlowStub(), cfg, sensors, "learned_joint", tmp_path / "test")
    assert result["steps"] == [] and "no dispatch" in result["error"]
    assert (tmp_path / "test/raw_actions_000.npy").exists()
    trace = np.load(tmp_path / "test/physical_trace.npz")["trace"]
    assert np.ptp(trace[:, 7]) < 0.002
