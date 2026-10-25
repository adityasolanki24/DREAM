"""CPU contracts of the Isaac G1 path (no Isaac Sim here): the embodiment, camera, sensor
models, start rule and scoring must match the validated MuJoCo path they replace."""

from __future__ import annotations

import json
from math import cos, sin
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dream.eval.isaac_scene import LabelledObject, free_starts
from dream.robot.g1_wbc import N_BODY
from dream.sim.camera import MAX_RANGE, CameraSpec
from dream.sim.g1_mobile import HEAD_CAMERA, G1MobileEnv, build_mobile_g1
from dream.sim.isaac_g1 import (
    apertures,
    camera_axes,
    mjcf_actuators,
    quat_matrix,
    self_mask,
    sensor_depth,
    yaw_of,
)
from experiments.isaac_g1_kitchen import choose_start, handle_truth, score

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mobile():
    return build_mobile_g1()


def test_actuator_order_and_force_limits_match_mujoco(mobile):
    names, limits = mjcf_actuators()
    m = mobile
    assert len(names) == N_BODY == m.nu
    for i in range(m.nu):
        j = m.actuator(i).trnid[0]
        assert m.joint(j).name == names[i]
        assert m.jnt_actfrclimited[j]
        assert limits[i] == pytest.approx(max(-m.jnt_actfrcrange[j][0], m.jnt_actfrcrange[j][1]))


def test_head_camera_axes_match_the_mujoco_mount(mobile):
    d = mujoco.MjData(mobile)
    mujoco.mj_forward(mobile, d)
    torso = d.body("torso_link")
    cam = d.cam(HEAD_CAMERA.name)
    local = torso.xmat.reshape(3, 3).T @ cam.xmat.reshape(3, 3)
    np.testing.assert_allclose(camera_axes(HEAD_CAMERA), local, atol=1e-6)
    np.testing.assert_allclose(
        torso.xmat.reshape(3, 3).T @ (cam.xpos - torso.xpos), HEAD_CAMERA.position, atol=1e-6
    )


def test_apertures_reproduce_vertical_field_of_view():
    h, v = apertures(HEAD_CAMERA, focal=10.0)
    assert np.degrees(2 * np.arctan(v / 20.0)) == pytest.approx(HEAD_CAMERA.fovy_deg)
    assert h / v == pytest.approx(HEAD_CAMERA.width / HEAD_CAMERA.height)


def test_sensor_depth_drops_invalid_and_out_of_range_returns():
    spec = CameraSpec(depth_noise=(0.0, 0.0))
    raw = np.array([[1.0, np.inf, -1.0], [0.0, MAX_RANGE + 0.1, 2.0004]], np.float32)
    out = sensor_depth(raw, spec, np.random.default_rng(0))
    assert out[0, 0] == pytest.approx(1.0)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2]) and np.isnan(out[1, 0])
    assert np.isnan(out[1, 1])
    assert out[1, 2] == pytest.approx(2.0)  # 1 mm quantisation


def test_self_mask_uses_only_robot_prims():
    ids = np.array([[1, 2], [3, 0]])
    info = {"1": "/World/G1/Geometry/pelvis/x", "2": "/World/KitchenRoom/Floor",
            "3": "/World/G1Copy/thing"}  # fmt: skip
    np.testing.assert_array_equal(self_mask(ids, info), [[True, False], [False, False]])


def test_quaternion_helpers_match_mujoco():
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, q)
        np.testing.assert_allclose(quat_matrix(q), m.reshape(3, 3), atol=1e-12)
    yaw = 0.7
    assert yaw_of([cos(yaw / 2), 0, 0, sin(yaw / 2)]) == pytest.approx(yaw)


def test_odometry_model_is_shared_with_mujoco_env():
    """The Isaac env reuses G1MobileEnv's oracle-assisted integrator, not a copy."""
    src = (ROOT / "dream/sim/isaac_g1.py").read_text()
    assert "G1MobileEnv._integrate_odometry(self)" in src
    assert hasattr(G1MobileEnv, "_integrate_odometry")


def test_free_starts_respects_clearance_and_walls():
    boxes = [(np.array([-0.5, -0.5, 0]), np.array([0.5, 0.5, 1]))]
    cells = free_starts(boxes, np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 0.55)
    assert len(cells)
    d_box = np.hypot(
        np.maximum(np.abs(cells[:, 0]) - 0.5, 0), np.maximum(np.abs(cells[:, 1]) - 0.5, 0)
    )
    assert d_box.min() >= 0.55 - 1e-9
    assert np.abs(cells).max() <= 2.0 - 0.55 + 1e-9


def test_start_choice_is_seeded_and_clear():
    boxes = [(np.array([-0.5, -0.5, 0]), np.array([0.5, 0.5, 1]))]
    cells = free_starts(boxes, np.array([-2.0, -2.0]), np.array([2.0, 2.0]), 0.55)
    assert choose_start(cells, boxes, "explore", 1) == choose_start(cells, boxes, "explore", 1)
    assert choose_start(cells, boxes, "explore", 1) != choose_start(cells, boxes, "explore", 2)
    *_, run = choose_start(cells, boxes, "bringup", 0)
    assert run > 1.0


def _objects():
    cab = LabelledObject("/World/Base_00", "kitchen_cabinet", np.array([1.0, -0.3, 0.0]),
                         np.array([1.6, 0.3, 0.9]))  # fmt: skip
    pull = LabelledObject("/World/Base_00/Doors/Door_L/Pull", "cabinet_pull",
                          np.array([0.98, -0.05, 0.7]), np.array([1.0, 0.05, 0.8]))  # fmt: skip
    return [cab, pull]


def _env(pose, odom=None):
    return SimpleNamespace(
        true_pose=lambda: np.array(pose),
        odometry=lambda: np.array(odom if odom is not None else pose),
    )


def _tb(handle):
    cand = SimpleNamespace(handle=None if handle is None else np.array(handle), handle_views=3)
    return SimpleNamespace(memory=SimpleNamespace(candidates=[cand]))


def test_kitchen_score_accepts_a_true_inspection_pose():
    h = handle_truth(_objects())
    assert h[0]["cabinet"] == "/World/Base_00"
    r = score(_env([0.0, 0.0, 0.0]), _tb([0.99, 0.0, 0.75]), True, h)
    assert r["success"] and not r["false_claim"] and r["phantom_handles"] == 0


def test_kitchen_score_is_drift_free_and_rejects_wrong_poses():
    h = handle_truth(_objects())
    # Odometry drifted by 0.4 m; the report is consistent in the odometry frame: still correct.
    r = score(_env([0.0, 0.0, 0.0], [0.4, 0.0, 0.0]), _tb([1.39, 0.0, 0.75]), True, h)
    assert r["success"]
    # Behind the cabinet (wrong side) and facing it: false claim.
    r = score(_env([2.4, 0.0, np.pi]), _tb([0.99, 0.0, 0.75]), True, h)
    assert not r["success"] and r["false_claim"] and r["on_front_side"] is False
    # A handle reported 0.5 m away from any real one is a phantom and a false claim.
    r = score(_env([0.0, 0.0, 0.0]), _tb([0.99, 0.5, 0.75]), True, h)
    assert r["phantom_handles"] == 1 and r["false_claim"]
    # No claim: neither success nor false claim.
    r = score(_env([0.0, 0.0, 0.0]), _tb([0.99, 0.0, 0.75]), False, h)
    assert not r["success"] and not r["false_claim"]


def test_frozen_kitchen_config_keeps_development_and_evaluation_disjoint():
    cfg = json.loads((ROOT / "experiments/configs/isaac_g1_kitchen.json").read_text())
    assert not set(cfg["development_kitchens"]) & set(cfg["evaluation_kitchens"])
    assert cfg["explore"]["method"] == "gain"
    milestone_b = json.loads((ROOT / "experiments/configs/g1_explore_room.json").read_text())
    assert cfg["explore"]["time_budget_s"] == milestone_b["time_budget_s"]
    assert cfg["explore"]["max_steps"] == milestone_b["max_steps"]


def test_isaac_session_plan_is_capped_and_offline():
    from cloud.isaac_session import CONFIG, estimate, plan

    cfg = json.loads(CONFIG.read_text())
    p = plan("nerva-adityapersonal", cfg, "dream-qwen-isaac-20261025-000000-abcdef")
    assert p["estimated_usd"] == estimate(cfg, cfg["max_minutes"])
    assert p["estimated_usd"] <= p["per_session_ceiling_usd"] <= p["continuation_cap_usd"]
    with pytest.raises(ValueError):
        plan("nerva-adityapersonal", {**cfg, "max_minutes": 600}, p["name"])
    with pytest.raises(ValueError):
        plan("nerva-adityapersonal", cfg, "nerva-something")


def test_session_vm_status_retries_control_plane_timeouts(monkeypatch):
    """Regression: one 120 s `instances list` timeout aborted `stop` before cleanup."""
    import subprocess

    import cloud.isaac_session as S

    calls = []

    def fake(*args):
        calls.append(args)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("gcloud", 120)
        return json.dumps([{"status": "TERMINATED"}])

    monkeypatch.setattr(S, "gcloud", fake)
    assert S.vm_status({"project": "p", "name": "n"}) == "TERMINATED"
    assert len(calls) == 2


def test_guest_session_script_has_no_early_closing_pipes():
    """Regression: `ldd --version | head -1` under pipefail exited 141 before installation."""
    text = (ROOT / "cloud/isaac_g1_session.sh").read_text()
    assert "| head" not in text


def test_guest_runs_a_job_end_to_end(tmp_path, monkeypatch):
    """Executes cloud/isaac_session_guest.run with gsutil replaced by local copies: archive
    checksum, extraction, argument substitution, results/ copy, EXIT_CODE and manifest."""
    import hashlib
    import importlib
    import io
    import shutil
    import sys
    import tarfile

    monkeypatch.setattr(sys, "argv", ["guest", "bucket", "150", sys.executable, sys.executable])
    guest = importlib.import_module("cloud.isaac_session_guest")
    importlib.reload(guest)
    remote = tmp_path / "remote"
    (remote / "jobs/1").mkdir(parents=True)
    code = b"import pathlib,sys\npathlib.Path('results/demo').mkdir(parents=True)\n"
    code += b"pathlib.Path('results/demo/r.txt').write_text('ok')\n"
    code += b"pathlib.Path(sys.argv[1]).write_text('marker')\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("experiments/hello.py")
        info.size = len(code)
        tar.addfile(info, io.BytesIO(code))
    (remote / "jobs/1/code.tar.gz").write_bytes(buf.getvalue())
    spec = {"id": "1", "git_sha": "abc", "module": "experiments/hello.py",
            "args": ["{out}/marker.txt"], "timeout_s": 120, "env": "dream",
            "archive_sha256": hashlib.sha256(buf.getvalue()).hexdigest()}  # fmt: skip
    (remote / "jobs/1/job.json").write_text(json.dumps(spec))

    def fake_gs(*args, check=True):
        if args[0] == "cp":
            shutil.copy(remote / args[1].split("gs://bucket/")[1], args[2])
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(guest, "gs", fake_gs)
    monkeypatch.setattr(guest, "uptime", lambda: 0.0)
    monkeypatch.setattr(guest, "OUT", tmp_path / "out")
    monkeypatch.setattr(guest, "WORK", tmp_path / "work")
    guest.run("1")
    out = tmp_path / "out/1"
    assert (out / "EXIT_CODE").read_text().strip() == "0", (out / "job.log").read_text()
    assert (out / "marker.txt").read_text() == "marker"
    assert (out / "results/demo/r.txt").read_text() == "ok"
    listed = (out / "MANIFEST.sha256").read_text()
    assert "EXIT_CODE" in listed and "results" in listed


def test_guest_installs_isaac_only_from_the_offline_lock():
    """Regression (sessions 2-3): resolving on the VM failed twice; the lock was resolved
    offline with the same index flags the guest uses."""
    text = (ROOT / "cloud/isaac_g1_session.sh").read_text()
    lock = (ROOT / "cloud/isaac62.lock.txt").read_text()
    assert "-r /work/DREAM/cloud/isaac62.lock.txt" in text
    assert "--index-strategy unsafe-best-match" in text
    assert "'isaacsim[" not in text  # no unlocked Isaac requirement left
    for pin in ("isaacsim==6.2.0.0", "mujoco==3.12.0", "torch==2.11.0+cu128"):
        assert pin in lock
