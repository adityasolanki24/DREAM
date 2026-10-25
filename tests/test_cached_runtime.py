"""Future inference cannot allocate a cold or unvalidated environment."""

import hashlib
import json

import pytest

from cloud import cached_runtime, launch


def test_g1_without_cached_image_is_rejected_before_any_cloud_call(monkeypatch):
    monkeypatch.setattr(launch, "gcloud", lambda *args: pytest.fail("cloud call before cache gate"))
    with pytest.raises(ValueError, match="cached runtime"):
        launch.validate_cached_image({"workload": "g1-autonomy"})


def test_cached_image_must_match_immutable_id_and_current_dependency_lock(monkeypatch):
    p = {
        "workload": "g1-autonomy",
        "cached_image_id": "123",
        "image": "cached",
        "image_project": "test",
    }
    digest = hashlib.sha256((launch.ROOT / "cloud/requirements.lock.txt").read_bytes()).hexdigest()
    image = {
        "id": "123",
        "status": "READY",
        "description": json.dumps(
            {
                "requirements_lock_sha256": digest,
                "preflight_passed": True,
            }
        ),
    }
    monkeypatch.setattr(launch, "gcloud", lambda *args: json.dumps(image))
    launch.validate_cached_image(p)
    image["id"] = "replaced-image"
    with pytest.raises(ValueError, match="identity"):
        launch.validate_cached_image(p)


def test_runtime_preflight_checks_installed_versions_without_installing(tmp_path, monkeypatch):
    lock = tmp_path / "requirements.txt"
    lock.write_text("example==1.2 \\\n    --hash=sha256:abc\n")
    monkeypatch.setattr(cached_runtime.platform, "python_version", lambda: "3.12.12")
    monkeypatch.setattr(cached_runtime.importlib.metadata, "version", lambda name: "1.2")
    assert cached_runtime.validate(lock)["preflight_passed"]
    monkeypatch.setattr(cached_runtime.importlib.metadata, "version", lambda name: "1.3")
    with pytest.raises(ValueError, match="dependency"):
        cached_runtime.validate(lock)
