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


@pytest.mark.parametrize("completed_before_failure", [0, 3])
def test_warmup_failure_preserves_step_and_joint_diagnostics(
    tmp_path, monkeypatch, completed_before_failure
):
    class FakeWorld:
        def __init__(self, **kwargs):
            self.scene = SimpleNamespace(
                add_default_ground_plane=lambda: None,
                add=lambda obj: None,
            )

    class FakeT1:
        def __init__(self, *args, **kwargs):
            self.time = 0.0
            self.completed = 0
            self.stopped = False
            self.art = SimpleNamespace(
                dof_names=[f"joint_{i}" for i in range(23)],
                get_world_pose=lambda: (np.array([0, 0, 0.7]), np.array([1, 0, 0, 0])),
            )
            self.limits = np.tile([-2.0, 2.0], (23, 1))
            self.actor = SimpleNamespace(
                last=np.zeros(23),
                target_limit_excess=np.zeros(23),
            )

        def advance(self, forward, yaw):
            if self.completed == completed_before_failure:
                raise RuntimeError("test actuator failure")
            self.completed += 1
            self.time += 0.02
            return {"time_s": self.time}

        def sensors(self):
            return {
                "q": np.zeros(23),
                "dq": np.zeros(23),
                "tilt": 0.0,
                "gravity": np.array([0.0, 0.0, -1.0]),
            }

    monkeypatch.setitem(
        sys.modules,
        "isaacsim.core.api",
        SimpleNamespace(World=FakeWorld),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.core.api.objects",
        SimpleNamespace(FixedCuboid=lambda **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaacsim.storage.native",
        SimpleNamespace(get_assets_root_path=lambda: "/assets"),
    )
    monkeypatch.setitem(
        sys.modules,
        "dream.sim.isaac_t1",
        SimpleNamespace(IsaacT1=FakeT1),
    )

    cfg = {
        "asset_relative_url": "/robot.usd",
        "policy_hashes": {},
        "record_video": False,
    }

    with pytest.raises(RuntimeError, match="test actuator failure"):
        runner.run_episode(tmp_path, cfg, "stand", tmp_path, [])

    report = json.loads((tmp_path / "warmup_failure.json").read_text())

    assert report["failed_step"] == completed_before_failure
    assert report["completed_steps"] == completed_before_failure
    assert report["error_type"] == "RuntimeError"
    assert len(report["joint_names"]) == 23
    assert len(report["joint_limits_rad"]) == 23
    assert len(report["last_target_limit_excess_rad"]) == 23
    assert "proprioception" in report
    assert "diagnostic_error" not in report
    assert report["score_root_position_m"] == [0, 0, 0.7]
    assert report["score_root_orientation_wxyz"] == [1, 0, 0, 0]
    assert report["state_limit_excess_rad"] == [0.0] * 23
    assert "traceback" in report and "test actuator failure" in report["traceback"]
    assert len((tmp_path / "warmup.jsonl").read_text().splitlines()) == completed_before_failure
