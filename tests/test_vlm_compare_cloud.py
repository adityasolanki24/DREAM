import json

import pytest

from cloud import launch


def test_vlm_compare_startup_command_uses_shell_line_continuation():
    startup = (launch.ROOT / "cloud/vm_startup.sh").read_text()
    command = startup.split("  vlm-compare)\n", 1)[1].split(" ;;", 1)[0]
    assert '"${LEFT}s" \\\n      python experiments/vlm_compare.py qwen-local' in command


def test_vlm_compare_workload_is_whitelisted_and_under_ceiling():
    cfg = json.loads(launch.WORKLOAD_CONFIGS["vlm-compare"].read_text())
    p = launch.plan("nerva-adityapersonal", cfg, "dream-qwen-20261028-000000-abcdef")
    assert p["workload"] == "vlm-compare" and p["machine"] == "g2-standard-8"
    assert p["max_minutes"] <= 45 and p["estimated_usd"] < 6


def test_vlm_compare_check_requires_every_decision(tmp_path):
    (tmp_path / "EXIT_CODE").write_text("0")
    with pytest.raises(RuntimeError):
        launch.check_pilot(tmp_path, "vlm-compare")
    (tmp_path / "vlm").mkdir()
    (tmp_path / "vlm/run_qwen-local.json").write_text(
        json.dumps({"summary": {"n_attempted": 3, "n_planned": 11}}))
    with pytest.raises(RuntimeError):
        launch.check_pilot(tmp_path, "vlm-compare")
    (tmp_path / "vlm/run_qwen-local.json").write_text(
        json.dumps({"summary": {"n_attempted": 11, "n_planned": 11}}))
    launch.check_pilot(tmp_path, "vlm-compare")
