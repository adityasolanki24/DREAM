"""Offline representation evidence must never authorize a physical mapping."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.groot_contract import decode_for_audit, require_actuation_contract

CONTRACT = Path("experiments/configs/groot_action_contract.json")
RUN = Path("results/groot_cuda_openloop/raw/groot")


def test_genuine_predictions_redecode_without_changing_input_or_adding_reference_twice():
    contract = json.loads(CONTRACT.read_text())
    normalized = np.load(RUN / "inference/normalized_actions.npy")
    original = normalized.copy()
    with np.load(RUN / "sensors/observation.npz") as state:
        decoded = decode_for_audit(normalized, dict(state), contract)
    with np.load(RUN / "inference/decoded_actions.npz") as saved:
        assert set(decoded) == set(saved.files)
        for key, value in decoded.items():
            np.testing.assert_allclose(value, saved[key], atol=2e-7, rtol=0)
    np.testing.assert_array_equal(normalized, original)


def test_joint_reference_is_last_observation_not_incremental_or_added_to_absolute_hand():
    contract = json.loads(CONTRACT.read_text())
    normalized = np.zeros((1, 40, 132), np.float32)
    state = {k: np.zeros((1, 2, d)) for k, d in contract["state_dimensions"].items()}
    for key in ("left_wrist_eef_9d", "right_wrist_eef_9d"):
        state[key][..., 3:] = [1, 0, 0, 0, 1, 0]
    before = decode_for_audit(normalized, state, contract)
    state["left_arm"][0, -1] = 0.25
    state["left_hand"][0, -1] = 0.8
    after = decode_for_audit(normalized, state, contract)
    np.testing.assert_allclose(after["left_arm"] - before["left_arm"], 0.25, atol=3e-8)
    np.testing.assert_array_equal(after["left_hand"], before["left_hand"])


def test_shapes_and_reconstructed_values_cannot_bypass_unresolved_physical_contract():
    with pytest.raises(ValueError, match="hand_joint_identity"):
        require_actuation_contract()


def test_decoder_rejects_invalid_tensor_before_processing():
    contract = json.loads(CONTRACT.read_text())
    with pytest.raises(ValueError, match="normalized"):
        decode_for_audit(np.full((1, 40, 132), np.nan), {}, contract)


def test_checkpoint_metadata_identity_matches_original_verified_download():
    contract = json.loads(CONTRACT.read_text())
    assets = json.loads(Path("results/groot_cuda_openloop/model_assets.json").read_text())
    for name, digest in contract["metadata_sha256"].items():
        original = next(
            item
            for item in assets["metadata"]
            if item["group"] == "checkpoint" and item["file"] == name
        )
        assert original["sha256"] == digest


def test_relative_wrist_translation_is_rotated_by_reference_orientation():
    from experiments.groot_contract import rotation_rows

    rotation = rotation_rows(np.array([0, -1, 0, 1, 0, 0]))
    np.testing.assert_array_equal(rotation @ [1, 0, 0], [0, 1, 0])
    np.testing.assert_array_equal(rotation @ rotation.T, np.eye(3))


def test_upstream_normalized_clipping_is_not_physical_target_clipping():
    contract = json.loads(CONTRACT.read_text())
    normalized = np.full((1, 40, 132), 2.0, np.float32)
    with np.load(RUN / "sensors/observation.npz") as state:
        decoded = decode_for_audit(normalized, dict(state), contract)
    # q99 is the decoder bound, which is NOT DREAM's provisional thumb_2 limit (0).
    assert decoded["right_hand"][0, 0, 2] == 1.5
    assert np.all(normalized == 2.0)
