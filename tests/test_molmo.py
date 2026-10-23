"""Sensor and action contracts only. No pretrained inference is simulated by these tests."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dream.runtime.molmo import REVISION, inspect_actions, observation, verify_checkpoint
from dream.sim.camera import CameraSpec
from dream.sim.tabletop import sample_tabletop
from dream.sim.tabletop_env import TabletopEnv


def test_wrist_camera_is_rigidly_mounted_and_reports_sensor_calibration():
    camera = CameraSpec(
        name="wrist",
        mount_body="hand",
        position=(0.05, 0, 0.03),
        target=(0, 0, 0.18),
        width=160,
        height=120,
    )
    env = TabletopEnv(
        sample_tabletop(np.random.default_rng(210)),
        np.random.default_rng(210),
        extra_cameras=(camera,),
    )
    try:
        frames = env.frame_sources()
        first = frames[1]()
        hand = env.data.body("hand")
        assert np.allclose(
            first.calibration.position, hand.xpos + hand.xmat.reshape(3, 3) @ camera.position
        )
        assert first.time == env.time and first.rgb.shape == (120, 160, 3)
        assert np.isfinite(first.depth).any() and first.rgb.std() > 1
        initial = first.calibration.position.copy()
        env.robot.set_tcp_target(env.robot.state().tcp_pos + [0, 0.03, 0])
        env.step(200)
        second = frames[1]()
        assert np.linalg.norm(second.calibration.position - initial) > 0.005
        hand = env.data.body("hand")
        assert np.allclose(
            second.calibration.position, hand.xpos + hand.xmat.reshape(3, 3) @ camera.position
        )
        request = observation(frames[0]().rgb, second.rgb, env.robot_state(), "move red")
        assert request["state"].shape == (8,) and len(request["images"]) == 3
        assert np.array_equal(request["images"][0], request["images"][1])
        stopped = replace(env.robot_state(), protective_stop=True)
        with pytest.raises(ValueError, match="stopped"):
            observation(first.rgb, second.rgb, stopped, "move red")
    finally:
        for cam in [env._camera, *env._extra_cameras.values()]:
            if cam is not None:
                cam.close()


def test_mounting_a_sensor_does_not_change_existing_physical_dynamics():
    from dream.sim.tabletop import build_model

    spec = sample_tabletop(np.random.default_rng(211))
    base = build_model(spec)
    mounted = build_model(spec, extra_cameras=(CameraSpec(name="wrist", mount_body="hand"),))
    assert mounted.nq == base.nq and mounted.nu == base.nu
    for field in ("body_mass", "body_inertia", "geom_size", "actuator_gainprm", "dof_armature"):
        assert np.array_equal(getattr(base, field), getattr(mounted, field))
    a, b = mujoco.MjData(base), mujoco.MjData(mounted)
    for _ in range(20):
        mujoco.mj_step(base, a)
        mujoco.mj_step(mounted, b)
    assert np.array_equal(a.qpos, b.qpos)


def test_action_inspection_preserves_values_and_never_authorizes_execution():
    limits = np.tile([-2.0, 2.0], (7, 1))
    actions = np.zeros((15, 8))
    actions[:, 7] = 0.5
    original = actions.copy()
    audit = inspect_actions(actions, limits)
    assert audit["arm_within_panda_joint_limits"] and audit["gripper_within_unit_interval"]
    assert not audit["execution_permitted"] and np.array_equal(actions, original)
    actions[2, 0] = 3
    assert not inspect_actions(actions, limits)["arm_within_panda_joint_limits"]
    actions[0, 7] = -0.1
    assert not inspect_actions(actions, limits)["gripper_within_unit_interval"]
    actions[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        inspect_actions(actions, limits)
    with pytest.raises(ValueError, match="15x8"):
        inspect_actions(np.zeros((1, 120)), limits)


def test_droid_training_contract_uses_normalized_aperture_not_a_knuckle_angle():
    from dream.runtime.molmo import decode_actions

    actions = np.zeros((15, 8))
    actions[:, 7] = np.linspace(-0.2, 1.2, 15)
    raw = actions.copy()
    decoded = decode_actions(actions, np.tile([-2.0, 2.0], (7, 1)))
    assert np.array_equal(actions, raw)
    assert decoded.width_m[0] == 0.08 and decoded.width_m[-1] == 0
    assert np.allclose(decoded.width_m, 0.08 * (1 - np.clip(raw[:, 7], 0, 1)))
    assert np.array_equal(decoded.saturated, (raw[:, 7] < 0) | (raw[:, 7] > 1))
    assert np.array_equal(decoded.joint_rad, raw[:, :7])  # absolute targets, not deltas
    actions[0, 0] = 3
    with pytest.raises(ValueError, match="joint"):
        decode_actions(actions, np.tile([-2.0, 2.0], (7, 1)))


def test_changed_checkpoint_source_is_rejected_before_custom_python_import(tmp_path):
    import hashlib

    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    source = snapshot / "modeling.py"
    source.write_bytes(b"reviewed source")
    hashes = {"modeling.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    verify_checkpoint(snapshot, hashes)
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(snapshot, hashes)


def test_checkpoint_download_includes_every_file_required_by_source_verification(
    tmp_path, monkeypatch
):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from experiments.cloud_molmo_openloop import download_checkpoint

    cfg = json.loads(
        (Path(__file__).parents[1] / "experiments/configs/molmo_openloop.json").read_text()
    )
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(tmp_path / REVISION)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    download_checkpoint(cfg, tmp_path)
    assert set(cfg["checkpoint_source_sha256"]) <= set(calls[0]["allow_patterns"])
    assert calls[0]["revision"] == REVISION
