"""Archive packaging regression; explicit actor stub, not policy-performance evidence."""

import json
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import isaac_t1_walk as runner


def test_archive_preflight_uses_supplied_revision_without_git(tmp_path, monkeypatch):
    (tmp_path / "env_cfg.json").write_text(json.dumps({"joint_names": [str(i) for i in range(23)]}))
    monkeypatch.setenv("DREAM_CODE_SHA", "a" * 40)
    monkeypatch.setattr(
        runner.T1Policy,
        "load",
        lambda *args: SimpleNamespace(policy=lambda obs: np.zeros(23, dtype=np.float32)),
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "test-double")
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *args, **kwargs: pytest.fail("archive called git")
    )
    result = runner.preflight(tmp_path, {"policy_hashes": {}})
    assert result["code_sha"] == "a" * 40 and result["physics_executed"] is False


def test_linux_guest_scripts_stay_lf_in_archive_with_windows_git_settings(tmp_path):
    archive = tmp_path / "source.tar"
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "archive",
            "--worktree-attributes",
            "--format=tar",
            "-o",
            str(archive),
            "HEAD",
        ],
        cwd=runner.ROOT,
        check=True,
    )
    with tarfile.open(archive) as source:
        for name in ("cloud/vm_startup.sh", "cloud/isaac_t1_job.sh"):
            contents = source.extractfile(name).read()
            assert b"\r\n" not in contents, f"Linux script archived with CRLF: {name}"


def test_application_enables_sensor_extensions_before_adapter_import(monkeypatch):
    events = []
    app = SimpleNamespace(close=lambda: events.append("close"))

    def launch(cfg):
        events.append("application")
        return app

    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=launch))
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.core.utils.extensions",
        SimpleNamespace(enable_extension=lambda name: events.append(name)),
    )
    assert runner.start_application() is app
    assert events == ["application", "isaacsim.sensors.camera", "isaacsim.storage.native"]


@pytest.mark.parametrize("fails", [False, True])
def test_episode_evidence_precedes_process_terminating_shutdown(tmp_path, monkeypatch, fails):
    """Kit shutdown may terminate Python before execute returns to its caller."""

    def close():
        raise SystemExit(0)

    monkeypatch.setattr(runner, "start_application", lambda: SimpleNamespace(close=close))

    def episode(*args):
        if fails:
            raise ValueError("diagnostic failure before first action")
        return {"gate_pass": False, "physics_executed": True}

    monkeypatch.setattr(runner, "run_episode", episode, raising=False)
    with pytest.raises(SystemExit):
        runner.execute(tmp_path, {}, "stand", tmp_path)
    assert json.loads((tmp_path / "partial_trace.json").read_text()) == []
    if fails:
        failure = json.loads((tmp_path / "failure.json").read_text())
        assert failure["type"] == "ValueError"
        assert "diagnostic failure" in failure["traceback"]
        assert failure["last_stage"] == "simulation_started"
        assert not (tmp_path / "episode.json").exists()
    else:
        assert json.loads((tmp_path / "episode.json").read_text())["gate_pass"] is False


def test_persistent_stage_logging(tmp_path):
    runner.mark_stage(tmp_path, "simulation_starting")
    runner.mark_stage(tmp_path, "world_created")

    lines = (tmp_path / "stages.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]

    assert [r["stage"] for r in records] == [
        "simulation_starting",
        "world_created",
    ]
    assert all(isinstance(r["timestamp"], float) for r in records)
