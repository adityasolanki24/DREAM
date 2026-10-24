"""Reproduce physical and inference measurements from an unchanged closed-loop pilot.

Reads saved evidence only. A tracked target, fresh inference and independently scored
task success are separate measurements; none implies held-out transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dream.session import make_scene


def inspect(root: Path) -> dict:
    meta = json.loads((root / "closedloop_meta.json").read_text())
    summary = json.loads((root / "closedloop_summary.json").read_text())
    conditions = {}
    for name in meta["config"]["conditions"]:
        folder = root / name
        episode = json.loads((folder / "episode.json").read_text())
        trace = np.load(folder / "physical_trace.npz", allow_pickle=False)["trace"]
        if trace.ndim != 2 or trace.shape[1] != 20 or len(trace) < 2:
            raise ValueError(
                "expected physical trace columns: time, q7, velocity7, TCP3, width, stop"
            )
        dt = np.diff(trace[:, 0])
        if not np.isfinite(trace).all() or not np.all(dt > 0):
            raise ValueError("physical trace must be finite with increasing simulation time")
        conditions[name] = {
            "success": episode["success"],
            "error": episode["error"],
            "protective_stop": episode["protective_stop"],
            "stop_reason": episode["stop_reason"],
            "link_collision": episode["link_collision"],
            "fresh_model_inference": episode["fresh_model_inference"],
            "classical_fallback": episode["classical_fallback"],
            "wall_s": episode["wall_s"],
            "simulation_span_s": float(trace[-1, 0] - trace[0, 0]),
            "peak_measured_joint_speed_rad_s": float(np.max(np.abs(trace[:, 8:15]))),
            "peak_measured_joint_accel_rad_s2": float(
                np.max(np.abs(np.diff(trace[:, 8:15], axis=0) / dt[:, None]))
            ),
            "joint_motion_range_rad": np.ptp(trace[:, 1:8], axis=0).tolist(),
            "trace_sha256": hashlib.sha256(
                (folder / "physical_trace.npz").read_bytes()
            ).hexdigest(),
        }
    outputs, previous = [], None
    folder = root / "learned_joint"
    scene = make_scene("tray", meta["config"]["seed"])
    obj = next(o for o in scene.objects if o.name == meta["config"]["goal_object"])
    initial_object = np.array([*obj.xy, obj.half_size])
    trace = np.load(folder / "physical_trace.npz", allow_pickle=False)["trace"]
    distances = np.linalg.norm(trace[:, 15:18] - initial_object, axis=1)
    conditions["learned_joint"].update(
        {
            "initial_object_center_reconstructed_for_diagnostics_only": initial_object.tolist(),
            "tcp_distance_to_initial_object_start_m": float(distances[0]),
            "tcp_distance_to_initial_object_end_m": float(distances[-1]),
            "tcp_distance_to_initial_object_min_m": float(distances.min()),
            "measured_gripper_width_range_m": [
                float(trace[:, 18].min()),
                float(trace[:, 18].max()),
            ],
            "limitation": (
                "Object trajectories not logged. Distances use initial scene geometry, "
                "not measured later object positions."
            ),
        }
    )
    steps = json.loads((folder / "episode.json").read_text())["steps"]
    for path in sorted(folder.glob("raw_actions_*.npy")):
        index = int(path.stem.rsplit("_", 1)[1])
        raw = np.load(path, allow_pickle=False)
        if raw.shape != (15, 8) or not np.isfinite(raw).all():
            raise ValueError("expected genuine finite 15x8 prediction")
        state = np.load(folder / f"input_{index:03}.npz", allow_pickle=False)["state"]
        sensor = json.loads((folder / f"input_{index:03}.json").read_text())
        outputs.append(
            {
                "index": index,
                "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "state_time_s": sensor["state_time"],
                "max_input_joint_change_since_previous_rad": None
                if previous is None
                else float(np.max(np.abs(state[:7] - previous[:7]))),
                "max_first_target_delta_from_observation_rad": float(
                    np.max(np.abs(raw[0, :7] - state[:7]))
                ),
                "raw_gripper_outside_unit_interval_count": int(
                    np.sum((raw[:, 7] < 0) | (raw[:, 7] > 1))
                ),
                "max_first_to_last_predicted_joint_delta_rad": float(
                    np.max(np.abs(raw[-1, :7] - raw[0, :7]))
                ),
                "raw_gripper_range": [float(raw[:, 7].min()), float(raw[:, 7].max())],
                "execution": next((s for s in steps if s["inference"] == index), None),
            }
        )
        previous = state
    return {
        "code_sha": meta["git_sha"],
        "config": meta["config"],
        "fresh_model_inference": summary["fresh_model_inference"],
        "verified_target_count": sum(bool(s.get("target_reached")) for s in steps),
        "raw_prediction_count": len(outputs),
        "conditions": conditions,
        "predictions": outputs,
        "interpretation": "Development scene; motion tracking is not task completion or transfer.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve earlier analyses; choose a new output path")
    result = inspect(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
