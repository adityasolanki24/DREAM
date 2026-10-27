import hashlib
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
        json.dumps({"summary": {"n_attempted": 3, "n_planned": 11}})
    )
    with pytest.raises(RuntimeError):
        launch.check_pilot(tmp_path, "vlm-compare")
    ids = [
        d["id"]
        for d in json.loads(
            (launch.ROOT / "results/phase1_vlm_compare/decisions.json").read_text()
        )["decisions"]
    ]
    frozen = json.loads((launch.ROOT / "results/phase1_vlm_compare/decisions.json").read_text())
    paths = ["results/phase1_vlm_compare/decisions.json"] + [
        "results/phase1_vlm_compare/" + d["image"] for d in frozen["decisions"]
    ]
    result = {
        "summary": {
            "n_attempted": 11,
            "n_planned": 11,
            "backend": "qwen3-vl:Qwen/Qwen3-VL-4B-Instruct",
            "provenance": {
                "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "device": "cuda",
            },
        },
        "selected_ids": ids,
        "rows": [{"id": i, "raw": "answer", "error": None} for i in ids],
        "sha256": {p: hashlib.sha256((launch.ROOT / p).read_bytes()).hexdigest() for p in paths},
    }
    path = tmp_path / "vlm/run_qwen-local.json"
    for change in ("empty", "errors", "duplicate", "cpu", "revision", "hash"):
        bad = json.loads(json.dumps(result))
        if change == "empty":
            bad["summary"].update(n_attempted=0, n_planned=0)
            bad["rows"] = []
        elif change == "errors":
            for row in bad["rows"]:
                row.update(error="failed", raw=None)
        elif change == "duplicate":
            bad["rows"][0]["id"] = bad["rows"][1]["id"]
        elif change == "cpu":
            bad["summary"]["provenance"]["device"] = "cpu"
        elif change == "revision":
            bad["summary"]["provenance"]["model_revision"] = None
        else:
            bad["sha256"][paths[0]] = "bad"
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):
            launch.check_pilot(tmp_path, "vlm-compare")
    path.write_text(json.dumps(result))
    launch.check_pilot(tmp_path, "vlm-compare")
