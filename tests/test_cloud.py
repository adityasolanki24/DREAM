"""Cloud lifecycle contracts without cloud access, money or inference packages."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from cloud import launch as cloud
from dream.session import LiveSession
from experiments.cloud_qwen_pilot import RecordedBackend, embodied_episode, offline_contract


def _plan():
    return cloud.plan("example-project", json.loads(cloud.CONFIG.read_text()), "dream-qwen-test")


@pytest.mark.parametrize("transfer", [True, False])
def test_pre_vm_archive_upload_has_a_bounded_transfer_deadline(monkeypatch, transfer):
    """A slow upload must not inherit the short resource-control RPC deadline."""
    monkeypatch.setattr(cloud.shutil, "which", lambda _: "gcloud-test")
    deadlines = []

    def run(command, **kwargs):
        deadlines.append(kwargs["timeout"])
        if transfer and kwargs["timeout"] < 180:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="completed\n")

    monkeypatch.setattr(cloud.subprocess, "run", run)
    args = (
        ("storage", "cp", "source.tar.gz", "gs://test/code.tar.gz")
        if transfer
        else ("compute", "instances", "list")
    )
    assert cloud.gcloud(*args) == "completed"
    assert deadlines == [600 if transfer else 120]


def test_new_service_account_propagation_retries_only_its_missing_identity(monkeypatch):
    account = "dream-test@example-project.iam.gserviceaccount.com"
    calls, sleeps = [], []

    def grant(*args):
        calls.append(args)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(
                1, args, stderr=f"Service account {account} does not exist."
            )
        return ""

    monkeypatch.setattr(cloud, "gcloud", grant)
    monkeypatch.setattr(cloud.time, "sleep", sleeps.append)
    cloud.grant_result_access("dream-qwen-test", account)
    assert len(calls) == 3 and sleeps == [5, 10] and calls[0] == calls[1] == calls[2]

    def denied(*args):
        raise subprocess.CalledProcessError(1, args, stderr="Permission denied")

    monkeypatch.setattr(cloud, "gcloud", denied)
    with pytest.raises(subprocess.CalledProcessError):
        cloud.grant_result_access("dream-qwen-test", account)
    assert sleeps == [5, 10]


def test_pilot_is_private_isolated_and_deleted_even_if_the_job_hangs():
    p = _plan()
    command = cloud.create_command(p, "a" * 40, "b" * 64)
    assert "--no-address" in command and "--instance-termination-action=DELETE" in command
    assert "--max-run-duration=45m" in command and "--boot-disk-auto-delete" in command
    assert "--network=dream-qwen-test" in command
    assert p["estimated_usd"] < 3.0
    assert not any("nerva" in arg for arg in command)


def test_g1_runtime_builder_is_one_scene_and_budgets_retained_image_storage():
    cfg = json.loads(cloud.WORKLOAD_CONFIGS["g1-runtime-build"].read_text())
    p = cloud.plan("example-project", cfg, "dream-qwen-test")
    assert cfg["workload"] == "g1-runtime-build"
    assert [
        case["name"]
        for case in json.loads(
            (cloud.ROOT / "experiments/configs/g1_runtime_build_demo.json").read_text()
        )["cases"]
    ] == ["free_heavier"]
    assert p["estimated_usd"] == 7.16
    assert p["retained_image_monthly_estimate_usd"] == 5.0
    assert p["max_minutes"] == 30
    assert p["region"] == "asia-southeast1"
    assert p["zones"] == ["asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c"]
    command = cloud.create_command(p, "a" * 40, "b" * 64)
    assert "--instance-termination-action=DELETE" in command
    assert "--max-run-duration=30m" in command
    metadata = next(arg for arg in command if arg.startswith("--metadata="))
    assert f"dream-runtime-image-name={p['runtime_image_name']}" in metadata
    assert not any("nerva" in arg for arg in command)


def test_runtime_image_uses_real_project_metadata_and_forced_running_disk_image():
    builder = (cloud.ROOT / "cloud/build_g1_runtime_image.py").read_text()
    launcher = (cloud.ROOT / "cloud/launch.py").read_text()
    assert 'metadata("instance/attributes/dream-bucket")' in builder
    assert 'metadata("project/project-id")' in builder
    assert '"--force"' in builder
    assert "compute.images.setLabels" in launcher


def test_runtime_builder_bootstraps_the_hashed_python_environment():
    startup = (cloud.ROOT / "cloud/vm_startup.sh").read_text()
    assert "uv pip install --require-hashes -r \"/work/DREAM/cloud/$LOCK\"" in startup
    assert "uv venv --python 3.12.12 /opt/dream-runtime" in startup
    assert "source /opt/dream-runtime/bin/activate" in startup
    assert 'export MUJOCO_GL=egl' in startup
    assert '"$WORKLOAD" == g1-autonomy || "$WORKLOAD" == g1-runtime-build' in startup
    assert "apt-get install -y -q libegl1" in startup
    assert 'elif [[ "$WORKLOAD" == g1-runtime-build ]]' not in startup


def test_walking_workload_is_separate_capped_and_needs_physical_evidence(tmp_path):
    cfg = json.loads(cloud.WORKLOAD_CONFIGS["isaac-t1-walk"].read_text())
    p = cloud.plan("example-project", cfg, "dream-qwen-test")
    assert p["estimated_usd"] < 4 and p["disk_gb"] == 200
    command = cloud.create_command(p, "a" * 40, "b" * 64)
    assert "--boot-disk-size=200GB" in command and "--max-run-duration=45m" in command
    (tmp_path / "EXIT_CODE").write_text("0\n")
    with pytest.raises(RuntimeError, match="physical"):
        cloud.check_pilot(tmp_path, "isaac-t1-walk")
    for case in ("stand", "forward", "left"):
        (tmp_path / case).mkdir()
        (tmp_path / case / "episode.json").write_text(
            json.dumps(
                {
                    "physics_executed": False,
                    "real_actor_executed": True,
                    "navgpt_executed": False,
                    "gate_pass": True,
                    "rows": [{}],
                }
            )
        )
    with pytest.raises(RuntimeError, match="gate failed"):
        cloud.check_pilot(tmp_path, "isaac-t1-walk")
    # Even explicit physics flags cannot replace an independently measured settled stop.
    for case in ("stand", "forward", "left"):
        folder = tmp_path / case
        record = {
            "physics_executed": True,
            "real_actor_executed": True,
            "navgpt_executed": False,
            "gate_pass": True,
            "rows": [{"time_s": 1, "tilt_rad": 0}],
            "score_final_position_m": [0, 0, 0.7],
            "score_target_m": [0, 0, 0.7],
            "status": "waypoint reached",
            "final_speed_m_s": 0.2,
        }
        (folder / "episode.json").write_text(json.dumps(record))
        (folder / "sensor_calibration.json").write_text("{}")
        np.savez_compressed(folder / "sensor_0000.npz", rgb=np.zeros((2, 2, 3)))
    with pytest.raises(RuntimeError, match="measurements"):
        cloud.check_pilot(tmp_path, "isaac-t1-walk")
    cfg["disk_gb"] = 500
    with pytest.raises(ValueError, match="disk"):
        cloud.plan("example-project", cfg, "dream-qwen-test")


def test_stand_diagnostic_is_shorter_and_rejects_saved_failure(tmp_path):
    cfg = json.loads(cloud.WORKLOAD_CONFIGS["isaac-t1-stand"].read_text())
    plan = cloud.plan("example-project", cfg, "dream-qwen-test")
    assert plan["max_minutes"] == 25 and plan["estimated_usd"] < 3.5
    assert "--max-run-duration=25m" in cloud.create_command(plan, "a" * 40, "b" * 64)
    (tmp_path / "EXIT_CODE").write_text("0")
    (tmp_path / "stand").mkdir()
    (tmp_path / "stand/failure.json").write_text('{"type":"ValueError"}')
    with pytest.raises(RuntimeError, match="failure report"):
        cloud.check_pilot(tmp_path, "isaac-t1-stand")


def test_stand_flags_and_camera_do_not_replace_policy_step_evidence(tmp_path):
    """Synthetic fixture, deliberately missing warmup and actual actuator telemetry."""
    folder = tmp_path / "stand"
    folder.mkdir()
    (tmp_path / "EXIT_CODE").write_text("0")
    record = {
        "physics_executed": True,
        "real_actor_executed": True,
        "navgpt_executed": False,
        "gate_pass": True,
        "rows": [{"time_s": t, "tilt_rad": 0} for t in (1.02, 1.04)],
        "score_final_position_m": [0, 0, 0.7],
        "score_target_m": [0, 0, 0.7],
        "status": "waypoint reached",
        "final_speed_m_s": 0.0,
    }
    (folder / "episode.json").write_text(json.dumps(record))
    (folder / "sensor_calibration.json").write_text("{}")
    np.savez_compressed(
        folder / "sensor_0000.npz", rgb=np.zeros((400, 400, 3), np.uint8), depth=np.ones((400, 400))
    )
    with pytest.raises(RuntimeError, match="policy step evidence"):
        cloud.check_pilot(tmp_path, "isaac-t1-stand")


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "nan_action",
        "missing_warmup",
        "bad_decimation",
        "final_lower",
        "final_upper",
        "bad_limits",
        "missing_limits",
    ],
)
def test_policy_evidence_validator_checks_complete_synthetic_contract(tmp_path, corruption):
    """A complete synthetic contract is accepted; never evidence of simulated standing."""
    keys = (
        "joint_position_rad",
        "joint_velocity_rad_s",
        "post_joint_position_rad",
        "post_joint_velocity_rad_s",
        "joint_target_rad",
        "raw_ppo_action",
        "target_limit_excess_rad",
    )
    rows = []
    for i in range(226):
        rows.append(
            {
                **{key: [0.0] * 23 for key in keys},
                "time_s": (i + 1) * 0.02,
                "state_time_s": (i + 1) * 0.02,
                "physics_step_count": (i + 1) * 4,
                "tilt_rad": 0.0,
                "stopped": False,
                "state_finite": True,
                "score_position_m": [0.0, 0.0, 0.7],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "body_linear_velocity_m_s": [0.0] * 3,
                "body_angular_velocity_rad_s": [0.0] * 3,
            }
        )
    if corruption == "nan_action":
        rows[60]["raw_ppo_action"][0] = float("nan")
    if corruption == "bad_decimation":
        rows[60]["physics_step_count"] += 1
    if corruption == "final_lower":
        rows[-1]["post_joint_position_rad"][14] = -1.000001
    if corruption == "final_upper":
        rows[-1]["post_joint_position_rad"][8] = 1.1
    if corruption != "missing_limits":
        limits = [[-1.0, 1.0]] * (22 if corruption == "bad_limits" else 23)
        (tmp_path / "articulation.json").write_text(json.dumps({"limits_rad": limits}))
    warmup = rows[:49] if corruption == "missing_warmup" else rows[:50]
    (tmp_path / "warmup.jsonl").write_text("\n".join(json.dumps(r) for r in warmup))
    record = {"rows": rows[50:], "case": "stand", "warmup_steps": 50, "physics_steps": 904}
    if corruption:
        with pytest.raises(RuntimeError, match="policy step evidence"):
            cloud.validate_policy_steps(tmp_path, record)
    else:
        cloud.validate_policy_steps(tmp_path, record)


@pytest.mark.parametrize("zone", ["us-central1-a", "us-central1-b", "us-central1-c"])
def test_reviewed_regional_zones_keep_the_same_hardware_budget_and_deadline(zone):
    cfg = json.loads(cloud.CONFIG.read_text())
    cfg["zone"] = zone
    p = cloud.plan("example-project", cfg, "dream-qwen-test")
    assert p["zone"] == zone and p["region"] == "us-central1"
    assert p["machine"] == "g2-standard-8" and p["max_minutes"] == 45
    assert p["estimated_usd"] == _plan()["estimated_usd"]


def test_recovery_keeps_the_manifest_zone_when_the_next_pilot_changes_zone(tmp_path, monkeypatch):
    p = _plan()
    p["zone"] = "us-central1-a"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(p))
    cfg = json.loads(cloud.CONFIG.read_text())
    cfg["zone"] = "us-central1-c"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(cloud, "CONFIG", config)
    calls = []
    monkeypatch.setattr(cloud, "gcloud", lambda *args: calls.append(args) or "")
    monkeypatch.setattr(cloud, "fetch", lambda *args: None)
    monkeypatch.setattr(cloud, "cleanup", lambda *args, **kwargs: [])
    monkeypatch.setattr(cloud, "audit", lambda *args: {})
    cloud.recover(manifest, True)
    assert "--zone=us-central1-a" in calls[0]


def test_capacity_failover_records_failures_and_checks_absence_before_next_zone(
    tmp_path, monkeypatch
):
    p = _plan()
    calls = []

    def run(*args):
        calls.append(args)
        if args[:3] == ("compute", "instances", "create") and "--zone=us-central1-c" in args:
            raise subprocess.CalledProcessError(1, args, stderr="ZONE_RESOURCE_POOL_EXHAUSTED")
        return "[]"

    monkeypatch.setattr(cloud, "gcloud", run)
    cloud.allocate(p, "a" * 40, "b" * 64, tmp_path / "startup.sh", tmp_path)
    creates = [c for c in calls if c[:3] == ("compute", "instances", "create")]
    assert len(creates) == 2 and "--zone=us-central1-b" in creates[-1]
    assert p["zone"] == "us-central1-b"
    assert calls[1][:3] == ("compute", "instances", "list")
    assert calls[2][:3] == ("compute", "disks", "list")
    records = json.loads((tmp_path / "allocation_attempts.json").read_text())
    assert [r["status"] for r in records] == ["capacity_rejected", "allocated"]
    assert json.loads((tmp_path / "manifest.json").read_text())["zone"] == p["zone"]


@pytest.mark.parametrize(
    "detail,rows",
    [("PERMISSION_DENIED", "[]"), ("ZONE_RESOURCE_POOL_EXHAUSTED", '[{"name":"existing"}]')],
)
def test_allocation_never_retries_permissions_or_a_possibly_existing_vm(
    tmp_path, monkeypatch, detail, rows
):
    calls = []

    def run(*args):
        calls.append(args)
        if args[:3] == ("compute", "instances", "create"):
            raise subprocess.CalledProcessError(1, args, stderr=detail)
        return rows

    monkeypatch.setattr(cloud, "gcloud", run)
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        cloud.allocate(_plan(), "a" * 40, "b" * 64, tmp_path / "startup.sh", tmp_path)
    assert sum(c[:3] == ("compute", "instances", "create") for c in calls) == 1


def test_guest_failure_or_missing_evidence_cannot_be_reported_as_a_pass(tmp_path):
    (tmp_path / "EXIT_CODE").write_text("1\n")
    with pytest.raises(RuntimeError, match="guest"):
        cloud.check_pilot(tmp_path)

    (tmp_path / "EXIT_CODE").write_text("0\n")
    with pytest.raises(RuntimeError, match="summary"):
        cloud.check_pilot(tmp_path)
    (tmp_path / "pilot").mkdir()
    summary = {
        "offline_contract": {"valid_named_call": True},
        "false_claim_gate_pass": True,
        "episodes": [
            {"seed": seed, "condition": condition, "false_claim": False, "success": False}
            for seed in (201, 202)
            for condition in ("A_rule_based", "B_vlm")
        ],
    }
    (tmp_path / "pilot/summary.json").write_text(json.dumps(summary))
    cloud.check_pilot(tmp_path)  # Task success remains report-only, as preregistered.
    summary["episodes"][0]["false_claim"] = True
    (tmp_path / "pilot/summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="gate"):
        cloud.check_pilot(tmp_path)


def test_closedloop_workload_requires_raw_predictions_and_physical_trace(tmp_path):
    cfg = json.loads(cloud.WORKLOAD_CONFIGS["molmo-closedloop"].read_text())
    p = cloud.plan("example-project", cfg, "dream-qwen-test")
    assert p["estimated_usd"] < 3 and p["workload"] == "molmo-closedloop"
    (tmp_path / "EXIT_CODE").write_text("0\n")
    summary = {"fresh_model_inference": True, "at_least_one_verified_target": True}
    (tmp_path / "closedloop_summary.json").write_text(json.dumps(summary))
    output = tmp_path / "learned_joint"
    output.mkdir()
    np.save(output / "raw_actions_000.npy", np.zeros((15, 8)))
    with pytest.raises(RuntimeError, match="physical"):
        cloud.check_pilot(tmp_path, "molmo-closedloop")
    trace = np.zeros((2, 20))
    trace[1, 0] = 0.002
    np.savez_compressed(output / "physical_trace.npz", trace=trace)
    (output / "episode.json").write_text(
        json.dumps(
            {
                "classical_fallback": False,
                "steps": [{"target_reached": True, "executed": True, "tracking_error_rad": 0.001}],
            }
        )
    )
    cloud.check_pilot(tmp_path, "molmo-closedloop")
    np.save(output / "raw_actions_000.npy", np.full((15, 8), np.nan))
    with pytest.raises(RuntimeError, match="tensor"):
        cloud.check_pilot(tmp_path, "molmo-closedloop")


def test_openloop_workload_is_explicit_and_never_accepts_a_task_summary(tmp_path):
    cfg = json.loads(cloud.WORKLOAD_CONFIGS["molmo-openloop"].read_text())
    p = cloud.plan("example-project", cfg, "dream-qwen-test")
    command = cloud.create_command(p, "a" * 40, "b" * 64)
    assert any("dream-workload=molmo-openloop" in arg for arg in command)
    assert "--max-run-duration=45m" in command and p["estimated_usd"] < 3
    (tmp_path / "EXIT_CODE").write_text("0\n")
    with pytest.raises(RuntimeError, match="summary"):
        cloud.check_pilot(tmp_path, "molmo-openloop")
    (tmp_path / "molmo").mkdir()
    summary = {
        "real_model_executed": True,
        "finite": True,
        "shape": [15, 8],
        "execution_permitted": False,
    }
    (tmp_path / "molmo/summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="gate"):
        cloud.check_pilot(tmp_path, "molmo-openloop")
    np.save(tmp_path / "molmo/raw_actions.npy", np.zeros((15, 8)))
    cloud.check_pilot(tmp_path, "molmo-openloop")
    np.save(tmp_path / "molmo/raw_actions.npy", np.full((15, 8), np.nan))
    with pytest.raises(RuntimeError, match="tensor"):
        cloud.check_pilot(tmp_path, "molmo-openloop")
    np.save(tmp_path / "molmo/raw_actions.npy", np.zeros((15, 8)))
    summary["execution_permitted"] = True
    (tmp_path / "molmo/summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="gate"):
        cloud.check_pilot(tmp_path, "molmo-openloop")


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_minutes", 46),
        ("max_minutes", True),
        ("zone", "europe-west1-b"),
        ("machine", "a2-highgpu-1g"),
    ],
)
def test_unreviewed_cost_or_hardware_expansion_is_rejected(field, value):
    cfg = json.loads(cloud.CONFIG.read_text())
    cfg[field] = value
    with pytest.raises(ValueError):
        cloud.plan("example-project", cfg, "dream-qwen-test")


def test_approval_is_checked_before_any_cloud_or_git_access(monkeypatch):
    def forbidden(*args):
        pytest.fail("approval gate contacted an external service")

    monkeypatch.setattr(cloud, "gcloud", forbidden)
    monkeypatch.setattr(cloud, "git", forbidden)
    for yes, allowance in [(False, 3.0), (True, 0.0), (True, float("nan")), (True, float("inf"))]:
        with pytest.raises(ValueError, match="approval"):
            cloud.launch(_plan(), allowance, yes)


def test_cleanup_preserves_unretrieved_evidence_and_never_touches_nerva(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud, "gcloud", lambda *args: calls.append(args) or "")
    assert cloud.cleanup(_plan(), remove_results=False) == []
    assert not any(args[0] == "storage" for args in calls)
    assert all(not any("nerva" in arg for arg in args) for args in calls)
    assert any("dream-qwen-test-nat" in args for args in calls)


def test_runtime_image_role_is_removed_before_its_service_account(monkeypatch):
    calls = []
    monkeypatch.setattr(cloud, "gcloud", lambda *args: calls.append(args) or "")
    plan = _plan()
    plan.update(workload="g1-runtime-build", image_builder_role="dreamImageTest")
    assert cloud.cleanup(plan, remove_results=False) == []
    binding = next(
        i for i, args in enumerate(calls) if args[:2] == ("projects", "remove-iam-policy-binding")
    )
    role = next(i for i, args in enumerate(calls) if args[:3] == ("iam", "roles", "delete"))
    account = next(
        i for i, args in enumerate(calls) if args[:3] == ("iam", "service-accounts", "delete")
    )
    assert binding < account and role < account


def test_cleanup_accepts_self_deleted_vm_but_reports_permission_failures(monkeypatch):
    def missing(*args):
        raise subprocess.CalledProcessError(1, args, stderr="resource was not found")

    monkeypatch.setattr(cloud, "gcloud", missing)
    assert cloud.cleanup(_plan(), remove_results=False) == []

    def denied(*args):
        raise subprocess.CalledProcessError(1, args, stderr="PERMISSION_DENIED")

    monkeypatch.setattr(cloud, "gcloud", denied)
    assert len(cloud.cleanup(_plan(), remove_results=False)) == 6


def test_concurrent_launches_are_rejected_and_failed_runs_release_the_local_lock(tmp_path):
    path = tmp_path / "active.lock"
    with pytest.raises(TimeoutError), cloud.single_run_lock(path):
        with pytest.raises(RuntimeError, match="audit first"), cloud.single_run_lock(path):
            pytest.fail("a second launch acquired the same lock")
        raise TimeoutError("simulated interruption")
    assert not path.exists()


def test_terminated_guest_is_deleted_before_retrieval_without_waiting_for_expiry(monkeypatch):
    calls = []

    def run(*args):
        calls.append(args)
        return '[{"status":"TERMINATED"}]' if args[:3] == ("compute", "instances", "list") else ""

    monkeypatch.setattr(cloud, "gcloud", run)
    monkeypatch.setattr(
        cloud.time, "sleep", lambda *args: pytest.fail("terminated VM kept waiting")
    )
    cloud.wait_for_guest(_plan())
    assert len(calls) == 2 and calls[1][:3] == ("compute", "instances", "delete")
    assert calls[1][3] == "dream-qwen-test"


def test_running_guest_does_not_trigger_early_deletion(monkeypatch):
    replies = iter(['[{"status":"RUNNING"}]', "[]"])
    calls, sleeps = [], []

    def run(*args):
        calls.append(args)
        return next(replies)

    monkeypatch.setattr(cloud, "gcloud", run)
    monkeypatch.setattr(cloud.time, "sleep", sleeps.append)
    cloud.wait_for_guest(_plan())
    assert sleeps == [30]
    assert all(args[:3] == ("compute", "instances", "list") for args in calls)


def test_a_recovery_manifest_cannot_target_another_projects_resources(tmp_path, monkeypatch):
    p = _plan()
    p["router"] = "nerva-router"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(p))
    monkeypatch.setattr(cloud, "gcloud", lambda *args: pytest.fail("unsafe manifest reached cloud"))
    with pytest.raises(ValueError, match="manifest resource mismatch"):
        cloud.recover(path, True)


def test_result_retrieval_detects_corruption_before_remote_deletion(tmp_path):
    data = b"measured output"
    (tmp_path / "episode.json").write_bytes(data)
    manifest = tmp_path / "MANIFEST.sha256"
    (tmp_path / "EXIT_CODE").write_bytes(b"0\n")
    manifest.write_text(
        hashlib.sha256(data).hexdigest()
        + "  ./episode.json\n"
        + hashlib.sha256(b"0\n").hexdigest()
        + "  ./EXIT_CODE\n"
    )
    cloud.verify_results(tmp_path)
    (tmp_path / "episode.json").write_bytes(b"partial download")
    with pytest.raises(RuntimeError, match="checksum differs"):
        cloud.verify_results(tmp_path)
    manifest.write_text(hashlib.sha256(data).hexdigest() + "  ../outside.json\n")
    with pytest.raises(ValueError, match="manifest path"):
        cloud.verify_results(tmp_path)


def test_empty_manifest_cannot_trigger_remote_result_deletion(tmp_path):
    (tmp_path / "MANIFEST.sha256").write_text("")
    with pytest.raises(RuntimeError, match="final status"):
        cloud.verify_results(tmp_path)


def test_cleanup_retries_dependency_release_and_fetch_failures_keep_remote_results(monkeypatch):
    calls = []
    sleeps = []

    def transient(*args):
        calls.append(args)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, args, stderr="resource is being used")
        return ""

    monkeypatch.setattr(cloud, "gcloud", transient)
    monkeypatch.setattr(cloud.time, "sleep", sleeps.append)
    assert cloud.cleanup(_plan(), remove_results=False) == []
    assert calls[0] == calls[1] and sleeps == [5]


class ScriptedBackend:
    """An interface test double. Never evidence of model inference."""

    name = "scripted-test-double"

    def generate(self, messages, tools):
        assert any(p["type"] == "image" for m in messages for p in m["content"])
        return '<tool_call>{"name":"observe_scene","arguments":{}}</tool_call>'


def test_offline_contract_preserves_actual_sensor_frame_and_model_inputs(tmp_path):
    result = offline_contract(ScriptedBackend(), {"offline_seed": 200}, tmp_path)
    assert result["valid_named_call"]
    frame = np.load(tmp_path / "rgbd.npz")
    assert frame["rgb"].shape == (360, 480, 3) and np.isfinite(frame["depth"]).all()
    request = json.loads((tmp_path / "inference/000_input.json").read_text())
    text = request["messages"][-1]["content"][-1]["text"]
    assert '"scene"' in text and '"robot"' in text and '"frame_time"' in text


def test_recording_an_inference_error_does_not_turn_it_into_a_model_reply(tmp_path):
    class Broken:
        name = "broken-test-double"

        def generate(self, *args):
            raise TimeoutError("test")

    with pytest.raises(TimeoutError):
        RecordedBackend(Broken(), tmp_path, lambda: 1.0).generate([], [])
    assert json.loads((tmp_path / "000_output.json").read_text())["error"] == "TimeoutError"


def test_live_physics_keeps_holding_during_slow_inference(tmp_path):
    session = LiveSession(seed=200, rate=1.0, perception="vision")
    session.start()
    try:
        before = session.env.robot_state()

        class Slow:
            name = "slow-test-double"

            def generate(self, *args):
                time.sleep(0.3)
                return "test"

        RecordedBackend(Slow(), tmp_path, lambda: session.env.time).generate([], [])
        after = session.env.robot_state()
        assert after.time - before.time > 0.1
        assert np.linalg.norm(after.tcp_pos - before.tcp_pos) < 0.002
        assert not after.protective_stop
    finally:
        session.close()


@pytest.mark.slow
def test_cloud_physical_harness_reports_an_independently_scored_outcome(tmp_path):
    """Development success is report-only, as preregistered in the pilot configuration.

    The first implementation incorrectly asserted this unvalidated live seed must always
    succeed. Full-suite run at c3a40d0 declined without a false claim; a separate repeat
    succeeded. Preserve either physical outcome and enforce the actual harness contract.
    """
    cfg = json.loads(
        (Path(__file__).parents[1] / "experiments/configs/vlm_cloud_pilot.json").read_text()
    )
    result = embodied_episode(None, cfg, cfg["episodes"][0], "A_rule_based", tmp_path)
    assert result["perception"] == "vision" and not result["false_claim"]
    assert isinstance(result["success"], bool) and isinstance(result["claimed"], bool)
    assert not result["claimed"] or result["success"]
    assert 0 < result["episode"]["tool_calls"] <= cfg["max_steps"]
    assert sum(step["sim_time"] for step in result["episode"]["steps"]) > 0
    assert result["episode"]["reason"]
    assert json.loads((tmp_path / "episode.json").read_text()) == result
