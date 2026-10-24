"""Audit saved standing failure and real CPU actor targets; no simulated motion."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from cloud.launch import verify_results
from dream.robot.t1 import T1Policy


def inspect(out: Path, policy_dir: Path) -> dict:
    verify_results(out)
    preflight = json.loads((out / "stand/preflight.json").read_text())
    articulation = json.loads((out / "stand/articulation.json").read_text())
    failure = json.loads((out / "stand/failure.json").read_text())
    cfg = preflight["config"]
    policy_cfg = json.loads((policy_dir / "env_cfg.json").read_text())
    names = articulation["joint_names"]
    actor = T1Policy.load(policy_dir, cfg["policy_hashes"], names)
    limits = np.asarray(articulation["limits_rad"])
    original = []
    for prediction in preflight["actions"]:
        target = np.zeros(23)
        target[actor.indices] = actor.default + 0.5 * np.asarray(prediction["action"])
        excess = np.maximum(np.maximum(limits[:, 0] - target, target - limits[:, 1]), 0)
        original.append(
            {
                "command": prediction["command"],
                "violations": [
                    {
                        "joint": name,
                        "target_rad": float(target[i]),
                        "limits_rad": limits[i].tolist(),
                        "excess_rad": float(excess[i]),
                    }
                    for i, name in enumerate(names)
                    if excess[i] > 0
                ],
            }
        )
    # Controlled ideal proprioception, not reconstructed measurements from the failed guest.
    home = np.empty(23)
    home[actor.indices] = actor.default
    cpu = []
    for command in ([0, 0, 0], [0.2, 0, 0], [0.1, 0, 0.4]):
        actor.last[:] = 0
        target = actor.target(
            home, np.zeros(23), np.zeros(3), np.zeros(3), [0, 0, -1], command, limits
        )
        np.testing.assert_allclose(target[actor.indices], actor.default + 0.5 * actor.last)
        cpu.append(
            {
                "command": command,
                "raw_action": actor.last.tolist(),
                "target_rad": target.tolist(),
                "excess_rad": actor.target_limit_excess.tolist(),
            }
        )
    stages = [json.loads(s) for s in (out / "stand/stages.jsonl").read_text().splitlines()]
    return {
        "analysis_code_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "original_code_sha": preflight["code_sha"],
        "failure": failure,
        "original_stage_count": len(stages),
        "last_stage": stages[-1]["stage"],
        "joint_order_matches_training": names == policy_cfg["joint_names"],
        "original_preflight_excursions": original,
        "real_cpu_actor_at_home": cpu,
        "physics_executed_by_analysis": False,
        "standing_validated": False,
        "note": (
            "Original preflight uses zero offsets/ideal gravity, not actual failed-step sensors. "
            "CPU diagnostics preserve PPO targets and mechanical limits; "
            "only new physics can validate stability."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    args.summary.write_text(json.dumps(inspect(args.out, args.policy_dir), indent=2) + "\n")
