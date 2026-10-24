"""Recording must sample real execution without advancing or changing its state."""

import json

import mujoco
import numpy as np
import pytest

from dream.sim.g1 import G1Env
from experiments.g1_recording import EpisodeRecorder, external_recording_root


def test_recorder_preserves_execution_and_samples_uniform_physics_times(tmp_path):
    env = G1Env()
    recorder = EpisodeRecorder(env, tmp_path / "recording", {"seed": 42})
    for _ in range(110):
        env.step()
        before = env.data.qpos.copy()
        recorder.sample("probe_object")
        np.testing.assert_array_equal(env.data.qpos, before)
    recorder.save()
    with np.load(recorder.root / "snapshots.npz") as data:
        np.testing.assert_allclose(data["time_s"], [0, 0.05, 0.1], atol=1e-12)
        assert data["integration_state"].shape[0] == 3
        assert data["reference_rad"].shape == (3, 43)
    assert json.loads((recorder.root / "metadata.json").read_text())["seed"] == 42


def test_recording_root_refuses_repository_and_onedrive_paths(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        external_recording_root(".")
    with pytest.raises(ValueError, match="outside"):
        external_recording_root(tmp_path / "OneDrive" / "footage")


def test_render_replay_never_steps_live_physics(tmp_path):
    env = G1Env()
    recorder = EpisodeRecorder(env, tmp_path / "recording", {})
    for _ in range(50):
        env.step()
    recorder.sample("test")
    recorder.save()
    before = env.data.qpos.copy()
    replay = recorder.replay_data(0)
    assert isinstance(replay, mujoco.MjData) and replay is not env.data
    assert replay.time == 0 and env.data.time > 0
    np.testing.assert_array_equal(env.data.qpos, before)


def test_encoder_failure_keeps_original_state_and_failed_outcome(tmp_path, monkeypatch):
    env = G1Env()
    recorder = EpisodeRecorder(env, tmp_path / "recording", {})
    for _ in range(50):
        env.step()
    recorder.sample("failed_action")
    episode = tmp_path / "evidence"
    episode.mkdir()
    (episode / "summary.json").write_text(json.dumps({"goal_success": False}))

    def failed_encoding(success):
        assert success is False
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(recorder, "render", failed_encoding)
    result = recorder.finish(episode, False)
    assert result["status"] == "render_failed"
    assert (recorder.root / "snapshots.npz").is_file()
    assert json.loads((recorder.root / "summary.json").read_text())["goal_success"] is False
    assert not list(recorder.root.rglob("*success.mp4"))
