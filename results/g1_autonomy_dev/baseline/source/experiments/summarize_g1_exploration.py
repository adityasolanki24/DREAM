"""Post-evaluation aggregation and initial-contact audit; never used by the agent.

All trials, including failed contact gates and an initially intersecting constrained scene,
remain in the totals. No statistical superiority or broad generalisation is inferred.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dream.sim.camera import CameraSpec
from dream.sim.g1 import G1Env, build_g1
from experiments.g1_dexterous import write_json


def summarize(root: Path) -> dict:
    freeze = json.loads((root / "g1_exploration_validation/freeze.json").read_text())
    for name, digest in freeze["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    methods = {}
    for method in ("eig", "fixed_short"):
        folder = root / f"g1_exploration_heldout_{method}"
        records = json.loads((folder / "summary.json").read_text())
        methods[method] = {
            "cases": records,
            "trials": len(records),
            "goal_successes": sum(r["goal_success"] for r in records.values()),
            "false_success_claims": sum(r["false_success_claim"] for r in records.values()),
            "classification_correct_raw": sum(
                r["classification_correct"] for r in records.values()
            ),
            "probes": {n: r["probes"] for n, r in records.items()},
            "penetration_gate_failures": [
                n for n, r in records.items() if not r["penetration_gate_passed"]
            ],
            "maximum_localization_error_m": max(
                r["maximum_localization_error_m"] or 0 for r in records.values()
            ),
            "failed_actions": sum(r["failed_actions"] for r in records.values()),
            "recovered_goal_successes": sum(
                r["goal_success"] and r["failed_actions"] > 0 for r in records.values()
            ),
        }
    config = json.loads((root / "g1_exploration_heldout_eig/metadata.json").read_text())["config"]
    initial = {}
    for case in config["cases"]:
        env = G1Env(
            build_g1(
                tuple(case["position_m"]),
                case["radius_m"],
                case["mass_kg"],
                constrained=case["constrained"],
                camera=CameraSpec(**config["camera"]),
                contact_timeconst=config["contact_timeconst_s"],
                target_position=tuple(config["target_position_m"]),
            )
        )
        geom = env.model.geom("test_object_geom").id
        penetration = max(
            (-float(c.dist) for c in env.data.contact if geom in (c.geom1, c.geom2)), default=0
        )
        initial[case["name"]] = {
            "initial_object_intersection_m": max(0, penetration),
            "invalid_initial_contact": penetration > config["gates"]["penetration_m"],
        }
    winner_case = next(
        c for c in config["cases"] if methods["eig"]["cases"][c["name"]]["goal_success"]
    )
    winner = root / "g1_exploration_heldout_eig" / winner_case["name"]
    rows = [json.loads(s) for s in (winner / "physics.jsonl").read_text().splitlines()]
    positions = np.array([r["object_position_m"] for r in rows[-51:]])
    summary = {
        "frozen_sources_match": True,
        "methods": methods,
        "initial_contact_audit": initial,
        "successful_episode_final_half_second_drift_m": float(
            np.linalg.norm(positions[-1] - positions[0])
        ),
        "successful_episode_final_sample_speed_m_s": float(
            np.linalg.norm((positions[-1] - positions[-2]) / 0.01)
        ),
        "interpretation": "One complete supported-base episode, not reliable general manipulation. "
        "The constrained held-out scene starts intersecting a finger and fails physical validity; "
        "its class result remains recorded but is not valid transfer evidence. "
        "The other free case develops excessive transient penetration. "
        "No successful recovery from a failed action was demonstrated. "
        "Four paired cases are too few for superiority claims; no held-out tuning performed.",
    }
    write_json(root / "g1_exploration_validation/comparison.json", summary)
    plot_success(winner, np.array(config["target_position_m"]), winner_case, rows)
    return summary


def plot_success(folder: Path, target: np.ndarray, case: dict, rows: list[dict]) -> None:
    """Measured simulation pose traces for independent scoring, not agent observations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["time_s"] for r in rows])
    positions = np.array([r["object_position_m"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, layout="constrained")
    for i, axis in enumerate(axes):
        axis.plot(t, positions[:, i], color="#217A68", label="Actual object pose")
        goal = target[i] if i < 2 else target[i] + case["radius_m"]
        axis.axhline(goal, color="#4169B1", linestyle="--", label="Target centre")
        axis.set_ylabel(f"{'xyz'[i]} (m)")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle(
        f"Supported G1: sensor-guided exploration and transfer, seed {case['seed']}\n"
        "Independent simulation scoring; deterministic Bayesian tool selection"
    )
    fig.savefig(folder / "object_trajectory.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    result = summarize(args.root)
    print(
        json.dumps(
            {k: {n: v for n, v in r.items() if n != "cases"} for k, r in result["methods"].items()},
            indent=2,
        )
    )
