"""A future adapted policy must learn an explicit physical coordinate contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from dream.sim.g1 import G1Env
from experiments.g1_dataset import aligned_samples, normalization_statistics
from experiments.g1_recording import EpisodeRecorder


def test_normalization_uses_all_supplied_training_rows_without_clipping():
    values = np.array([[0, 2], [2, 2], [10, 2]], dtype=np.float32)
    stats = normalization_statistics(values)
    assert set(stats) == {"mean", "std", "min", "max", "q01", "q99"}
    np.testing.assert_allclose(stats["mean"], [4, 2])
    np.testing.assert_allclose(stats["std"], values.std(axis=0))
    np.testing.assert_array_equal(stats["max"], [10, 2])
    assert stats["std"][1] == 0  # constant joints remain explicit, never invented spread
    np.testing.assert_allclose(stats["q01"], np.quantile(values, 0.01, axis=0))
    with pytest.raises(ValueError, match="finite"):
        normalization_statistics(np.array([[np.nan, 0]]))


def test_public_dataset_has_named_hand_order_not_driver_motor_order():
    info = json.loads(Path("results/g1_contract_resolution/nvidia_dataset_info.json").read_text())
    names = info["features"]["action"]["names"]
    assert names[22:29] == [
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
    ]
    env = G1Env()
    assert set(names) == set(env.robot.names)
    assert info["fps"] == 20


def test_next_reference_is_aligned_to_current_measured_state_without_offsets(tmp_path):
    env = G1Env()
    recorder = EpisodeRecorder(env, tmp_path / "episode", {})
    env.robot.command_joints({"left_elbow_joint": 0.1})
    for _ in range(100):
        env.step()
        recorder.sample("reach")
    recorder.save()
    root = recorder.root
    (root / "robot_contract.json").write_text(
        json.dumps(
            {
                "joint_names": list(env.robot.names),
                "joint_limits_rad": env.robot.limits.tolist(),
            }
        )
    )
    state, action, names, times = aligned_samples(root)
    index = names.index("left_elbow_joint")
    assert len(names) == 31 and state.shape == action.shape == (2, 31)
    assert state[0, index] == 0 and action[0, index] == pytest.approx(0.02)
    assert action[1, index] == pytest.approx(0.04)
    np.testing.assert_allclose(times, [0, 0.05], atol=1e-12)
    assert not any("hip" in name for name in names)
    with np.load(root / "snapshots.npz") as saved:
        values = dict(saved)
    values["reference_rad"][1, env.robot.names.index("left_elbow_joint")] = 999
    np.savez_compressed(root / "snapshots.npz", **values)
    with pytest.raises(ValueError, match="no clipping"):
        aligned_samples(root)
