"""Explicit encoder/camera doubles; these tests provide no walking evidence."""

import json
import shutil

import numpy as np
import pytest

from experiments.t1_recording import FrameRecorder, encode


def test_frames_reject_empty_wrong_dtype_and_repeated_time(tmp_path):
    recorder = FrameRecorder(tmp_path)
    for frame in (np.zeros((0, 3, 4), np.uint8), np.zeros((4, 4, 3), float)):
        with pytest.raises(ValueError):
            recorder.capture(frame, 0, {})
    recorder.capture(np.zeros((4, 4, 4), np.uint8), 0, {})
    with pytest.raises(ValueError):
        recorder.capture(np.zeros((4, 4, 4), np.uint8), 0, {})


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="optional encoder"
)
def test_mock_encode_is_decodable_labelled_and_keeps_originals(tmp_path):
    recorder = FrameRecorder(tmp_path)
    for i in range(25):
        recorder.capture(np.full((180, 320, 4), i * 9, np.uint8), i * 0.04, {})
    with pytest.raises(FileNotFoundError):
        encode(tmp_path)  # Mock pixels cannot silently claim real physics.
    metadata = encode(tmp_path, mock=True)
    assert metadata["mock"] and not metadata["physical_gate_pass"]
    assert metadata["stream"]["nb_read_frames"] == "30"
    assert (tmp_path / "mock_encoder_test.mp4").stat().st_size > 1000
    assert len(list((tmp_path / "originals").glob("*.png"))) == 25
    assert not (tmp_path / "dream_t1_demo.mp4").exists()


def test_encoding_rejects_changed_frame_evidence(tmp_path):
    recorder = FrameRecorder(tmp_path)
    for i in range(2):
        recorder.capture(np.zeros((20, 20, 4), np.uint8), i * 0.04, {})
    (tmp_path / "originals/000000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        encode(tmp_path, mock=True)


def test_encoding_rejects_nonuniform_time(tmp_path):
    recorder = FrameRecorder(tmp_path)
    for time in (0, 0.04, 0.12):
        recorder.capture(np.zeros((20, 20, 4), np.uint8), time, {})
    with pytest.raises(ValueError, match="nonuniform"):
        encode(tmp_path, mock=True)
    assert len((tmp_path / "frames.jsonl").read_text().splitlines()) == 3
    assert json.loads((tmp_path / "frames.jsonl").read_text().splitlines()[-1])["time_s"] == 0.12
