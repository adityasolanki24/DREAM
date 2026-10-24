"""Offline loading and credential-free artifact staging contracts; no model inference."""

import json

import pytest


def test_both_model_and_processor_use_local_backbone_without_touching_original(tmp_path):
    from experiments.groot_assets import localize_checkpoint

    checkpoint = tmp_path / "checkpoint"
    backbone = tmp_path / "backbone"
    checkpoint.mkdir()
    backbone.mkdir()
    original = {"action_horizon": 40, "model_name": "nvidia/Cosmos-Reason2-2B"}
    processor = {"processor_kwargs": {"modality_configs": {"keep": True}}}
    (checkpoint / "config.json").write_text(json.dumps(original))
    (checkpoint / "processor_config.json").write_text(json.dumps(processor))
    (checkpoint / "model.safetensors").write_bytes(b"untouched-weights")
    result = localize_checkpoint(checkpoint, backbone, tmp_path / "localized")
    assert json.loads((result / "config.json").read_text())["model_name"] == str(backbone.resolve())
    patched = json.loads((result / "processor_config.json").read_text())["processor_kwargs"]
    assert patched["model_name"] == str(backbone.resolve()) and patched["modality_configs"] == {
        "keep": True
    }
    assert json.loads((checkpoint / "config.json").read_text()) == original
    assert (result / "model.safetensors").read_bytes() == b"untouched-weights"


def test_offline_environment_drops_local_credentials(tmp_path, monkeypatch):
    from experiments.cloud_groot_openloop import offline_environment

    monkeypatch.setenv("HF_TOKEN", "do-not-upload")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "do-not-upload")
    env = offline_environment(tmp_path, tmp_path)
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env


def test_artifact_download_rejects_untrusted_host_and_hides_signed_urls(tmp_path):
    from experiments.groot_assets import download_weights

    entry = {
        "group": "backbone",
        "file": "model.safetensors",
        "url": "https://evil.invalid/secret",
        "sha256": "a" * 64,
        "size": 20,
        "expires": 9999999999,
    }
    with pytest.raises(ValueError, match="host") as error:
        download_weights([entry], tmp_path)
    assert "secret" not in str(error.value)
