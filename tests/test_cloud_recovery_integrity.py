"""Recovery cannot delete its source when a file is missing, corrupt or unexplained."""

import hashlib

import pytest

from experiments.verify_cloud_recovery import verify


def test_manifest_requires_every_file_and_rejects_unexplained_extras(tmp_path):
    status = tmp_path / "EXIT_CODE"
    status.write_bytes(b"124\n")
    (tmp_path / "MANIFEST.sha256").write_text(
        hashlib.sha256(status.read_bytes()).hexdigest() + "  ./EXIT_CODE\n"
    )
    # The manifest itself also needs an independent remote integrity check.
    with pytest.raises(ValueError, match="extra"):
        verify(tmp_path, {})
    status.unlink()
    with pytest.raises(ValueError, match="missing downloaded"):
        verify(tmp_path, {})
