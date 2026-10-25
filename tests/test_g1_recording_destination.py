"""Cloud retrieval must never download large evaluation footage into OneDrive."""

from pathlib import Path

import pytest

from cloud import launch


def test_recorded_cloud_job_uses_external_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    result = launch.run_destination("g1-autonomy", "example-run")
    assert result == tmp_path / "DREAM/cloud_runs/example-run"
    assert not result.is_relative_to(launch.ROOT)


def test_external_cloud_destination_rejects_onedrive_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "OneDrive"))
    with pytest.raises(ValueError, match="outside"):
        launch.run_destination("g1-autonomy", "example-run")


def test_old_nonrecording_recovery_location_is_preserved():
    assert (
        launch.run_destination("qwen-pilot", "old-run")
        == Path(launch.ROOT) / "results/cloud_runs/old-run"
    )
