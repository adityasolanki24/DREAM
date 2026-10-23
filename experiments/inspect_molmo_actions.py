"""Describe a saved real action tensor without executing it or changing its values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect(actions_path: Path, observation_path: Path) -> dict:
    actions = np.load(actions_path, allow_pickle=False)
    observation = np.load(observation_path, allow_pickle=False)
    if actions.shape != (15, 8) or not np.isfinite(actions).all():
        raise ValueError("invalid action chunk")
    initial = observation["state"][:7]
    changes = np.diff(np.vstack([initial, actions[:, :7]]), axis=0)
    lo, hi = observation["joint_limits"].T
    return {
        "raw_actions_sha256": hashlib.sha256(actions_path.read_bytes()).hexdigest(),
        "observation_sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "min_per_column": actions.min(axis=0).tolist(),
        "max_per_column": actions.max(axis=0).tolist(),
        "gripper_outside_unit_interval_count": int(
            np.sum((actions[:, 7] < 0) | (actions[:, 7] > 1))
        ),
        "arm_joint_limit_violations": int(np.sum((actions[:, :7] < lo) | (actions[:, :7] > hi))),
        "initial_joint_target_delta_rad": (actions[0, :7] - initial).tolist(),
        "max_absolute_step_delta_rad": np.max(np.abs(changes), axis=0).tolist(),
        "physical_execution_permitted": False,
        "interpretation": (
            "Open-loop output only. No clipping; frequency and gripper mapping unvalidated. "
            "Joint range is not a collision, workspace or velocity check."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve earlier analysis")
    args.out.write_text(json.dumps(inspect(args.actions, args.observation), indent=2) + "\n")


if __name__ == "__main__":
    main()
