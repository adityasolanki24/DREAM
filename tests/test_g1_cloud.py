"""Inference configuration contracts; no checkpoint, cloud access or GPU required."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments import g1_explore_unknown as g1


def test_cloud_runner_help_does_not_require_inference_packages():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.cloud_g1_qwen", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_g1_gate_requires_probe_belief_feedback_not_just_reaching(tmp_path, monkeypatch):
    from experiments import cloud_g1_qwen as runner

    cfg = json.loads(runner.CONFIG.read_text())
    cfg["cases"] = [{"name": "test_double"}]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(runner, "CONFIG", path)
    rows = []
    for backend in ("qwen", "deterministic"):
        folder = tmp_path / backend / "test_double"
        folder.mkdir(parents=True)
        for name, data in (
            ("summary.json", {}),
            ("frames.json", []),
            ("belief.json", []),
            ("tools.json", [{"tool": "reach_object"}]),
        ):
            (folder / name).write_text(json.dumps(data))
        (folder / "physics.jsonl").write_text('{"time_s": 0.5}\n')
        rows.append({"case": "test_double", "backend": backend, "physics_executed": True})
    folder = tmp_path / "qwen/test_double/inference"
    folder.mkdir()
    for i in range(2):
        (folder / f"{i}_input.json").write_text(json.dumps({"sim_time_before": float(i)}))
        (folder / f"{i}_output.json").write_text(json.dumps({"text": "test double"}))
    (tmp_path / "comparison.json").write_text(json.dumps({"episodes": rows}))
    (tmp_path / "model.json").write_text(
        json.dumps({"device": "cuda:0", "revision": cfg["model"]["revision"]})
    )
    with pytest.raises(ValueError, match="belief"):
        runner.validate_evidence(tmp_path)
    (tmp_path / "qwen/test_double/belief.json").write_text(
        json.dumps([{"update_accepted": True, "contact_valid": True}])
    )
    (folder / "1_input.json").write_text(
        json.dumps(
            {
                "sim_time_before": 1.0,
                "messages": [
                    {
                        "content": [
                            {"type": "text", "text": 'probe_object result: {"posterior": 0.99}'}
                        ]
                    }
                ],
            }
        )
    )
    runner.validate_evidence(tmp_path)
    (folder / "1_output.json").write_text(json.dumps({"error": "TimeoutError"}))
    with pytest.raises(ValueError, match="belief"):
        runner.validate_evidence(tmp_path)


def test_source_metadata_works_in_committed_cloud_archive(monkeypatch):
    monkeypatch.setenv("DREAM_CODE_SHA", "a" * 40)
    monkeypatch.setattr(g1.subprocess, "check_output", lambda *a, **k: pytest.fail("no git"))
    meta = g1.source_metadata({"backend": "qwen"})
    assert meta["base_git_sha"] == "a" * 40
    assert "No local Qwen" not in meta.get("vlm_note", "")


def test_cuda_configuration_fails_before_checkpoint_loading_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        g1.load_backend({"device": "cuda:0"}, tmp_path)


def test_selected_cuda_device_reaches_existing_backend(tmp_path, monkeypatch):
    from dream.runtime import qwen

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    seen = {}

    def backend(path, **kwargs):
        seen.update(path=path, **kwargs)
        return object()

    monkeypatch.setattr(qwen, "QwenTransformersBackend", backend)
    g1.load_backend({"device": "cuda:0", "timeout_s": 75, "max_new_tokens": 256}, tmp_path)
    assert seen == {
        "path": str(tmp_path),
        "device": "cuda:0",
        "timeout_s": 75,
        "max_new_tokens": 256,
    }


def test_g1_cloud_gate_rejects_missing_inference_even_with_physics(tmp_path):
    from cloud.launch import check_pilot

    (tmp_path / "EXIT_CODE").write_text("0")
    (tmp_path / "g1").mkdir()
    (tmp_path / "g1/comparison.json").write_text(json.dumps({"episodes": []}))
    with pytest.raises(RuntimeError, match="G1"):
        check_pilot(tmp_path, "g1-qwen")


def test_g1_budget_alert_is_project_scoped_and_requires_billing_visibility(monkeypatch):
    from cloud import launch

    calls = []

    def gcloud(*args):
        calls.append(args)
        if args[:3] == ("billing", "projects", "describe"):
            return json.dumps({"billingAccountName": "billingAccounts/account"})
        if args[:2] == ("projects", "describe"):
            return json.dumps({"projectNumber": "123"})
        return json.dumps({"name": "billingAccounts/account/budgets/new"})

    monkeypatch.setattr(launch, "gcloud", gcloud)
    name = launch.create_g1_budget_alert({"project": "example-project", "name": "dream-qwen-test"})
    assert name == "billingAccounts/account/budgets/new"
    assert "--filter-projects=projects/123" in calls[-1]
    assert "--budget-amount=5AUD" in calls[-1]
    assert "--threshold-rule=percent=0.5" in calls[-1]
