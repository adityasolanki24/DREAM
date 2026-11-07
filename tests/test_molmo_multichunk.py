"""Multi-row execution, the --config switch and the frozen multichunk config (stubs only)."""

import json
from pathlib import Path

import numpy as np
import pytest

import cloud.launch as cloud
from experiments.cloud_molmo_closedloop import resolve_config, run_condition

ROOT = Path(__file__).resolve().parents[1]
CFGS = ROOT / "experiments/configs"


class Creep:
    """Known safe stub: a 15-row chunk creeping joint 7, then a fresh inference."""

    def __init__(self):
        self.states = []

    def predict(self, request):
        self.states.append(request["state"].copy())
        raw = np.tile(request["state"], (15, 1))
        raw[:, 6] += 0.003 * np.arange(1, 16)
        raw[:, 7] = np.linspace(0, 0.02, 15)
        return raw


def cfg_for(rows, inferences):
    cfg = json.loads((CFGS / "molmo_multichunk_closeout.json").read_text())
    cfg["rows_per_inference"], cfg["max_inferences"] = rows, inferences
    return cfg, json.loads((CFGS / "molmo_openloop.json").read_text())


def test_rows_per_inference_executes_consecutive_rows_then_reobserves(tmp_path):
    cfg, sensors = cfg_for(3, 2)
    backend = Creep()
    result = run_condition(backend, cfg, sensors, "learned_joint", tmp_path / "t")
    steps = result["steps"]
    assert len(steps) == 6 and all(s["target_reached"] for s in steps)
    assert [(s["inference"], s["row"]) for s in steps] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert len(backend.states) == 2 and backend.states[1][6] - backend.states[0][6] > 0.006
    assert result["error"] is None and not result["classical_fallback"]
    assert (tmp_path / "t/raw_actions_000.npy").exists() and (
        tmp_path / "t/raw_actions_001.npy"
    ).exists()
    geometry = np.load(tmp_path / "t/object_geometry.npz")
    assert geometry["geometry"].shape[1] == 6 and np.all(geometry["geometry"][:, 4] > 0)
    assert np.load(tmp_path / "t/physical_trace.npz")["trace"].shape[1] == 20


def test_default_config_keeps_single_row_behaviour_and_no_geometry_file(tmp_path):
    cfg = json.loads((CFGS / "molmo_closedloop.json").read_text())
    cfg["max_inferences"] = 2
    sensors = json.loads((CFGS / "molmo_openloop.json").read_text())
    result = run_condition(Creep(), cfg, sensors, "learned_joint", tmp_path / "t")
    assert len(result["steps"]) == 2 and all("row" not in s for s in result["steps"])
    assert not (tmp_path / "t/object_geometry.npz").exists()


def test_invalid_rows_per_inference_rejected(tmp_path):
    cfg, sensors = cfg_for(16, 1)
    with pytest.raises(ValueError):
        run_condition(Creep(), cfg, sensors, "learned_joint", tmp_path / "t")


def test_config_argument_is_confined_to_experiment_configs():
    assert resolve_config(Path("experiments/configs/molmo_closedloop.json")).is_file()
    with pytest.raises(ValueError):
        resolve_config(Path("README.md"))
    with pytest.raises(ValueError):
        resolve_config(Path("../outside.json"))


def test_multichunk_config_differs_from_the_dev_config_only_where_declared():
    dev = json.loads((CFGS / "molmo_closedloop.json").read_text())
    new = json.loads((CFGS / "molmo_multichunk_closeout.json").read_text())
    assert dev["rows_per_inference"] == 1 and dev["max_inferences"] == 6  # frozen dev untouched
    for key in ("seed", "inference_seed", "instruction", "goal_object", "conditions", "gates"):
        assert new[key] == dev[key], key
    assert new["status"] == "frozen" and new["rows_per_inference"] == 15
    assert new["max_inferences"] >= 10 and new["max_chunk_sim_s"] <= 30


def test_multichunk_workload_is_registered_with_the_same_guard_and_lifetime():
    cfg = json.loads(cloud.WORKLOAD_CONFIGS["molmo-multichunk"].read_text())
    p = cloud.plan("nerva-adityapersonal", cfg, "dream-qwen-test-multichunk")
    assert p["workload"] == "molmo-multichunk" and p["max_minutes"] <= 45 and p["estimated_usd"] < 4
    startup = (ROOT / "cloud/vm_startup.sh").read_text()
    assert "molmo-closedloop|molmo-multichunk) LOCK=molmo.requirements.lock.txt" in startup
    assert "--config experiments/configs/molmo_multichunk_closeout.json" in startup
