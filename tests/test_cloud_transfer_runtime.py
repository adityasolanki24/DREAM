"""Recorded output transfers need bounded transfer time and explicit UTF-8 logs."""

from types import SimpleNamespace

from cloud import launch


def test_recorded_result_sync_uses_transfer_timeout_and_utf8(monkeypatch):
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout="complete\n")

    monkeypatch.setattr(launch.shutil, "which", lambda name: "gcloud")
    monkeypatch.setattr(launch.subprocess, "run", run)
    assert launch.gcloud("storage", "rsync", "--recursive", "source", "destination") == "complete"
    assert seen["timeout"] == 600
    assert seen["encoding"] == "utf-8"
    launch.gcloud("compute", "instances", "list")
    assert seen["timeout"] == 120


def test_vm_creation_gets_the_long_timeout_but_listing_stays_short(monkeypatch):
    """7 Nov 2026: creating the cached-image L4 VM took ~3 min; the 120 s limit aborted it."""
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(launch.shutil, "which", lambda name: "gcloud")
    monkeypatch.setattr(launch.subprocess, "run", run)
    launch.gcloud("compute", "instances", "create", "vm")
    assert seen["timeout"] == 600
    launch.gcloud("compute", "instances", "list")
    assert seen["timeout"] == 120
