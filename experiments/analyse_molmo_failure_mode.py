"""Saved-evidence analysis of the MolmoAct2 closed-loop failure (no inference, no simulation).

    python experiments/analyse_molmo_failure_mode.py --out results/molmo_failure_mode_closeout

Reads results/molmo_cloud_feedback_dev, molmo_cloud_openloop and molmo_recorded_prefix_validated.
DROID_* constants are values read from the pinned upstream norm_stats.json (franka_droid) and
droid/franka/robot.py on 7 Nov 2026 through a page-summarising fetch, not a byte-checked
download; sources.json in results/molmo_contract_audit records their pinned hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dream.session import make_scene

ROOT = Path(__file__).resolve().parents[1]
FB = ROOT / "results/molmo_cloud_feedback_dev/raw"
# franka_droid state q01/q99 (7 joints) and action q01/q99, from upstream norm_stats.json.
STATE_Q01 = [-0.2793009809782748, -0.5873924424866738, -0.3058546817065916, -2.5639055042030354,
             -0.491431808753978, 1.7381500993283228, -0.5086147192989775]  # fmt: skip
STATE_Q99 = [0.3148177895778054, 0.7235689468221655, 0.2683897323238184, -1.530780071911146,
             0.415067150345451, 2.7863710743039887, 0.6952765173061115]  # fmt: skip
ACT_Q01 = [-0.2879620949867506, -0.5702219304684566, -0.31101638810433413, -2.5622234922052725,
           -0.5101021838814974, 1.7376836093987995, -0.5227783063045004]  # fmt: skip
GRIPPER_ACTION_STATS = {"mean": 0.439, "std": 0.442, "q01": 8.3e-05, "q99": 0.892}


def tcp_motion(trace: np.ndarray, obj: np.ndarray) -> dict:
    tcp = trace[:, 15:18]
    dist = np.linalg.norm(tcp - obj, axis=1)
    move = tcp[-1] - tcp[0]
    want = obj - tcp[0]
    cos = float(move @ want / (np.linalg.norm(move) * np.linalg.norm(want) + 1e-12))
    return {
        "sim_span_s": float(trace[-1, 0] - trace[0, 0]),
        "tcp_start": tcp[0].tolist(),
        "tcp_end": tcp[-1].tolist(),
        "dist_to_initial_object_start_m": float(dist[0]),
        "dist_to_initial_object_end_m": float(dist[-1]),
        "dist_to_initial_object_min_m": float(dist.min()),
        "tcp_displacement_m": float(np.linalg.norm(move)),
        "cosine_displacement_vs_direction_to_object": cos,
        "gripper_width_range_m": [float(trace[:, 18].min()), float(trace[:, 18].max())],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = json.loads((FB / "closedloop_meta.json").read_text())["config"]
    scene = make_scene("tray", meta["seed"])
    o = next(x for x in scene.objects if x.name == meta["goal_object"])
    obj = np.array([*o.xy, o.half_size])
    L = FB / "learned_joint"
    raws = [np.load(L / f"raw_actions_{i:03}.npy") for i in range(6)]
    states = [np.load(L / f"input_{i:03}.npz")["state"] for i in range(6)]
    grip = np.concatenate([r[:, 7] for r in raws])
    chunks = []
    for i, (r, s) in enumerate(zip(raws, states, strict=True)):
        z_state = [(s[j] - STATE_Q01[j]) for j in range(7)]
        chunks.append(
            {
                "inference": i,
                "input_state_joints": s[:7].tolist(),
                "input_state_gripper_closure": float(s[7]),
                "row0_target": r[0, :7].tolist(),
                "row0_minus_state": (r[0, :7] - s[:7]).tolist(),
                "state_below_droid_q01_rad": {
                    str(j): float(STATE_Q01[j] - s[j]) for j in range(7) if s[j] < STATE_Q01[j]
                },
                "state_above_droid_q99_rad": {
                    str(j): float(s[j] - STATE_Q99[j]) for j in range(7) if s[j] > STATE_Q99[j]
                },
                "row0_target_minus_droid_action_q01": {
                    str(j): float(r[0, j] - ACT_Q01[j]) for j in (5, 6)
                },
                "chunk_joint_span_first_to_last_rad": (r[-1, :7] - r[0, :7]).tolist(),
                "raw_gripper_min_max": [float(r[:, 7].min()), float(r[:, 7].max())],
                "_unused": z_state and None,
            }
        )
        del chunks[-1]["_unused"]
    classical = np.load(FB / "classical/physical_trace.npz")["trace"]
    learned = np.load(L / "physical_trace.npz")["trace"]
    prefix = {
        n: tcp_motion(
            np.load(ROOT / f"results/molmo_recorded_prefix_validated/trace_{n}.npz")["trace"], obj
        )
        for n in (1, 3, 15)
    }
    steps = json.loads((L / "episode.json").read_text())["steps"]
    cls_closed = classical[:, 18] < 0.04
    summary = {
        "inputs": [
            "results/molmo_cloud_feedback_dev/raw",
            "results/molmo_recorded_prefix_validated",
        ],
        "scene_seed": meta["seed"],
        "initial_object_centre": obj.tolist(),
        "gripper_channel": {
            "n_values": int(grip.size),
            "min": float(grip.min()),
            "max": float(grip.max()),
            "count_above_0_1": int((grip > 0.1).sum()),
            "count_above_0_5": int((grip > 0.5).sum()),
            "mapping_in_dream": "width = 0.08 * (1 - clip(closure, 0, 1)); continuous, no "
            "threshold; negatives clip to 0 (open)",
            "droid_upstream_convention": "0 = open (full width), 1 = closed; command clipped to "
            "[0,1]; gripper_position state = 1 - width/max_width (matches DREAM input mapping)",
            "droid_gripper_action_distribution": GRIPPER_ACTION_STATS,
            "input_state_closure_at_start": float(states[0][7]),
            "measured_width_range_m": [float(learned[:, 18].min()), float(learned[:, 18].max())],
        },
        "executed_rows": {
            "inferences": len(steps),
            "rows_executed_per_inference": 1,
            "rows_available_per_chunk": 15,
            "policy_time_executed_s": len(steps) / 15,
            "policy_time_available_s": len(steps) * 15 / 15,
            "sim_time_s": float(learned[-1, 0] - learned[0, 0]),
            "time_dilation": [s["time_dilation"] for s in steps],
        },
        "ee_geometry_learned_6_rows": tcp_motion(learned, obj),
        "ee_geometry_replayed_first_chunk": {str(k): v for k, v in prefix.items()},
        "ee_geometry_classical_reference": {
            **tcp_motion(classical, obj),
            "gripper_closed_ticks": int(cls_closed.sum()),
            "note": "same scene and frames; the classical condition grasped and placed the object",
        },
        "per_inference": chunks,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("gripper_channel", "executed_rows")}, indent=1))
    print(json.dumps(summary["ee_geometry_learned_6_rows"], indent=1))
    print(json.dumps(summary["ee_geometry_replayed_first_chunk"]["15"], indent=1))
    print(json.dumps(summary["ee_geometry_classical_reference"], indent=1))
    print(json.dumps(chunks[0], indent=1))


if __name__ == "__main__":
    main()
