"""Offline diagnostics of genuine decoded predictions; never commands an actuator.

Decoded outputs are already absolute: the pinned StateActionProcessor.unapply_action
undoes relative joint/EEF representations using the final observed state. The slot-to-name
assignment below is explicitly conditional on DREAM's input order, not a verified training
convention. Mechanical compatibility cannot establish semantic compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.g1_dexterous import write_json
from experiments.groot_openloop import TAG, validate_actions


def joint_diagnostics(
    decoded: np.ndarray, reference: np.ndarray, limits: np.ndarray, speeds: np.ndarray
) -> dict:
    """Compare absolute decoded joint slots to provisional named mechanical limits."""
    q = np.asarray(decoded, float)
    ref = np.asarray(reference, float)
    bounds = np.asarray(limits, float)
    speed = np.asarray(speeds, float)
    if (
        q.ndim != 2
        or ref.shape != q.shape[1:]
        or bounds.shape != (q.shape[1], 2)
        or speed.shape != ref.shape
        or not all(np.isfinite(x).all() for x in (q, ref, bounds, speed))
        or np.any(speed <= 0)
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("invalid joint diagnostic contract")
    excess = np.maximum(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]), 0)
    return {
        "samples_outside_mechanical_range": int(np.count_nonzero(excess)),
        "steps_with_any_range_violation": int(np.count_nonzero(np.any(excess > 0, axis=1))),
        "max_range_excess_rad": float(excess.max()),
        "first_target_delta_rad": (q[0] - ref).tolist(),
        "first_target_slew_time_lower_bound_s": float(np.max(np.abs(q[0] - ref) / speed)),
        "per_step_slew_time_lower_bound_s": (
            np.max(np.abs(np.diff(q, axis=0)) / speed, axis=1).tolist()
        ),
        "hypothesis_only": "DREAM input slot order, signs and zero points equal training",
        "actuation_permitted": False,
    }


def reference_distribution(reference: np.ndarray, statistics: dict) -> dict:
    """Flag measured state slots outside the checkpoint's empirical training ranges."""
    values = {key: np.asarray(statistics[key], float) for key in ("min", "max", "q01", "q99")}
    if any(v.shape != reference.shape or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("training statistics differ from the measured slot contract")
    outside = (reference < values["min"]) | (reference > values["max"])
    return {
        "measured_reference": reference.tolist(),
        "training_statistics": {k: v.tolist() for k, v in values.items()},
        "reference_slots_outside_training_min_max": np.flatnonzero(outside).tolist(),
        "scope": "Statistical comparison under provisional slot order; not a verified mapping",
    }


def audit(run: Path, statistics_path: Path | None = None) -> dict:
    """Read saved model outputs and robot geometry only, without executing predictions."""
    from dream.sim.g1 import G1Env, build_g1

    sensor = run / "sensors"
    contract = json.loads((sensor / "contract.json").read_text())
    with np.load(run / "inference/decoded_actions.npz", allow_pickle=False) as saved:
        actions = {k: saved[k] for k in saved.files}
    shapes = validate_actions(actions)
    normalized = np.load(run / "inference/normalized_actions.npy", allow_pickle=False)
    if normalized.shape != (1, 40, 132) or not np.isfinite(normalized).all():
        raise ValueError("normalized REAL_G1 inference tensor violates frozen acceptance")
    env = G1Env(build_g1())
    robot = env.robot
    streams = {}
    statistics = json.loads(statistics_path.read_text())[TAG] if statistics_path else None
    with np.load(sensor / "observation.npz", allow_pickle=False) as observation:
        for stream, names in contract["joint_orders"].items():
            indices = [robot.names.index(name) for name in names]
            streams[stream] = {
                "provisional_joint_names": names,
                "mechanical_limits_rad": robot.limits[indices].tolist(),
                **joint_diagnostics(
                    actions[stream][0],
                    observation[stream][0, 0],
                    robot.limits[indices],
                    robot.speed[indices],
                ),
            }
            if statistics:
                streams[stream]["input_distribution"] = reference_distribution(
                    observation[stream][0, 0], statistics["state"][stream]
                )
    return {
        "scope": "Offline range diagnostics; no policy targets executed or clipped",
        "decoded_representation": "absolute after upstream relative-action postprocessing",
        "postprocessing_source": (
            "https://github.com/NVIDIA/Isaac-GR00T/blob/"
            "944fe70c4fd54e3fcb80a7d7258a328f9c061176/"
            "gr00t/data/state_action/state_action_processor.py"
        ),
        "shape_contract": shapes,
        "normalized_tensor": {
            "shape": list(normalized.shape),
            "all_finite": True,
            "min": float(normalized.min()),
            "max": float(normalized.max()),
        },
        "training_statistics_sha256": (
            hashlib.sha256(statistics_path.read_bytes()).hexdigest() if statistics_path else None
        ),
        "conditional_joint_diagnostics": streams,
        "unsupported_streams": ["base_height_command", "navigate_command"],
        "unverified": [
            "Training joint names/order, signs, zero points and exact hand revision",
            "Training wrist origin, reference frame and EEF/joint arbitration",
            "Training ego camera calibration and external-to-ego observation adaptation",
            "Training control frequency, history interval and chunk execution schedule",
            "Fixed-pelvis behavior versus whole-body base/navigation commands",
        ],
        "input_camera": contract["camera_semantics"],
        "actuation_permitted": False,
        "physical_policy_transfer_established": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--statistics", type=Path)
    args = parser.parse_args()
    write_json(args.out, audit(args.run, args.statistics))
