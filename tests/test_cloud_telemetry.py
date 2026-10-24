"""Exercise the actual guest telemetry loop with an explicit transient driver double."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_gpu_sampling_survives_a_transient_management_library_failure(tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    source = (Path(__file__).parents[1] / "cloud/vm_startup.sh").read_text()
    begin = source.index("( while true; do\n    nvidia-smi --query-gpu=timestamp")
    end = source.index("  done ) &", begin) + len("  done ) &")
    loop = source[begin:end].replace("/work/out", ".")
    script = (
        """set -e
count=0
nvidia-smi() {
  count=$((count + 1))
  if [ "$count" -eq 1 ]; then echo 'transient management-library error' >&2; return 1; fi
  echo 'recovered GPU sample'
}
sleep() { if [ "$count" -ge 2 ]; then exit 0; fi; }
"""
        + loop
        + "\nwait\n"
    )
    result = subprocess.run(
        [executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "gpu.csv").read_text().splitlines() == ["recovered GPU sample"]
    assert "transient management-library error" in (tmp_path / "gpu_errors.log").read_text()
