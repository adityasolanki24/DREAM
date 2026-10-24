"""Array contracts only; no fabricated checkpoint inference or actuator execution."""

import numpy as np
import pytest

from experiments.groot_openloop import ACTION_DIMS, STATE_DIMS, load_observation, validate_actions


def test_offline_child_uses_the_cache_where_backbone_was_downloaded(tmp_path, monkeypatch):
    from experiments.cloud_groot_openloop import offline_environment

    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    env = offline_environment(tmp_path / "upstream", tmp_path / "download-cache")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "download-cache")
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"


def test_capture_calibration_is_serializable_without_losing_transform():
    import json

    from dream.types import CameraCalibration
    from experiments.capture_groot_input import calibration_record

    cal = CameraCalibration(32, 24, 45, np.zeros(3), np.eye(3))
    record = json.loads(json.dumps(calibration_record(cal)))
    assert record["position"] == [0, 0, 0] and record["rotation"] == np.eye(3).tolist()


def test_real_g1_requires_history_and_named_finite_state_streams(tmp_path):
    path = tmp_path / "measured.npz"
    arrays = {k: np.zeros((1, 1, d), np.float32) for k, d in STATE_DIMS.items()}
    np.savez(path, video=np.zeros((1, 2, 24, 32, 3), np.uint8), **arrays)
    observation = load_observation(path, "test contract")
    assert observation["language"]["annotation.human.task_description"] == [["test contract"]]
    np.savez(path, video=np.zeros((1, 1, 24, 32, 3), np.uint8), **arrays)
    with pytest.raises(ValueError, match="two"):
        load_observation(path, "test contract")


def test_real_g1_shapes_never_imply_actuator_compatibility():
    actions = {k: np.zeros((1, 40, d), np.float32) for k, d in ACTION_DIMS.items()}
    assert validate_actions(actions)["navigate_command"]["shape"] == [1, 40, 3]
    actions["left_hand"] = np.zeros((1, 40, 6), np.float32)
    with pytest.raises(ValueError, match="left_hand"):
        validate_actions(actions)


def test_groot_cloud_is_separate_and_cleanup_removes_only_its_secret(monkeypatch):
    import json

    from cloud import launch

    cfg = json.loads(launch.WORKLOAD_CONFIGS["groot-openloop"].read_text())
    plan = launch.plan("example-project", cfg, "dream-qwen-groot-test")
    assert plan["estimated_usd"] == 3.03 and plan["max_minutes"] == 45
    command = launch.create_command(plan, "a" * 40, "b" * 64)
    assert not any("dream-hf-secret" in arg for arg in command)
    calls = []
    monkeypatch.setattr(launch, "gcloud", lambda *args: calls.append(args) or "")
    assert launch.cleanup(plan, remove_results=False) == []
    assert ("secrets", "delete", plan["name"], "--project=example-project") in calls
    assert not any(args[0] == "storage" for args in calls)


def test_cloud_job_has_no_local_token_transfer_or_secret_fetch():
    from pathlib import Path

    startup = Path("cloud/vm_startup.sh").read_text()
    assert "secrets versions access" not in startup
    assert "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH" in startup
    source = Path("cloud/launch.py").read_text()
    assert '"--data-file={credential}"' not in source
    assert "/model-input" in startup and "weights.private.json" in source
