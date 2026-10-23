"""Archive packaging regression; explicit actor stub, not policy-performance evidence."""

import json
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import isaac_t1_walk as runner


def test_archive_preflight_uses_supplied_revision_without_git(tmp_path, monkeypatch):
    (tmp_path / "env_cfg.json").write_text(json.dumps({"joint_names": [str(i) for i in range(23)]}))
    monkeypatch.setenv("DREAM_CODE_SHA", "a" * 40)
    monkeypatch.setattr(
        runner.T1Policy,
        "load",
        lambda *args: SimpleNamespace(policy=lambda obs: np.zeros(23, dtype=np.float32)),
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "test-double")
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *args, **kwargs: pytest.fail("archive called git")
    )
    result = runner.preflight(tmp_path, {"policy_hashes": {}})
    assert result["code_sha"] == "a" * 40 and result["physics_executed"] is False


def test_linux_guest_scripts_stay_lf_in_archive_with_windows_git_settings(tmp_path):
    archive = tmp_path / "source.tar"
    subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "archive",
            "--worktree-attributes",
            "--format=tar",
            "-o",
            str(archive),
            "HEAD",
        ],
        cwd=runner.ROOT,
        check=True,
    )
    with tarfile.open(archive) as source:
        for name in ("cloud/vm_startup.sh", "cloud/isaac_t1_job.sh"):
            contents = source.extractfile(name).read()
            assert b"\r\n" not in contents, f"Linux script archived with CRLF: {name}"
