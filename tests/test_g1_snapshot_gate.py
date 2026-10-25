"""Offline media is a new evidence mode; old frozen video gates must remain strict."""

import numpy as np
import pytest

from experiments.cloud_g1_qwen import validate_recording_evidence


def test_saved_states_do_not_silently_pass_an_old_video_gate(tmp_path):
    row = {
        "backend": "qwen",
        "case": "test",
        "recording": {"status": "captured", "frames": 2, "fps": 20},
    }
    with pytest.raises(ValueError, match="two-view"):
        validate_recording_evidence(tmp_path / "g1", row, "video")
    root = tmp_path / "recordings/qwen/qwen/test"
    root.mkdir(parents=True)
    for name in ("scene.mjb", "timeline.jsonl", "terminal.npz"):
        (root / name).write_bytes(b"contract fixture")
    np.savez(root / "snapshots.npz", time_s=[0, 0.05], integration_state=np.zeros((2, 3)))
    validate_recording_evidence(tmp_path / "g1", row, "snapshots")
    np.savez(root / "snapshots.npz", time_s=[0, 0.1], integration_state=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="chronology"):
        validate_recording_evidence(tmp_path / "g1", row, "snapshots")
