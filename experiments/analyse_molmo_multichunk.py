"""Apply the pre-registered H4 rule to a saved multichunk run (read-only on the run folder).

    python experiments/analyse_molmo_multichunk.py RUN_DIR OUT.json

RUN_DIR holds closedloop_meta.json, closedloop_summary.json and learned_joint/ as written by
experiments/cloud_molmo_closedloop.py. The rule and its operational definitions are read from
the frozen config recorded in closedloop_meta.json; the thresholds below are those it states.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

NEAR_M, FAR_M = 0.05, 0.25
CLOSE_RAW, OPEN_RAW = 0.5, 0.1
MIN_INFERENCES = 10


def apply_rule(
    n_inferences: int, min_dist_m: float, max_raw: float, max_raw_near: float | None
) -> dict:
    """H4 rejected / stands / inconclusive, exactly as pre-registered."""
    if n_inferences < MIN_INFERENCES:
        return {"h4": "inconclusive", "why": f"only {n_inferences} fresh inferences completed"}
    if min_dist_m <= NEAR_M and max_raw_near is not None and max_raw_near > CLOSE_RAW:
        return {"h4": "rejected", "why": "TCP within 0.05 m and raw gripper above 0.5 near it"}
    if min_dist_m > FAR_M and max_raw <= OPEN_RAW:
        return {"h4": "stands", "why": "TCP never within 0.25 m and raw gripper never above 0.1"}
    return {"h4": "inconclusive", "why": "intermediate outcome (neither pre-registered branch)"}


def analyse(run: pathlib.Path) -> dict:
    meta = json.loads((run / "closedloop_meta.json").read_text())
    ep = json.loads((run / "learned_joint/episode.json").read_text())
    steps = ep["steps"]
    raw = [np.load(p) for p in sorted((run / "learned_joint").glob("raw_actions_*.npy"))]
    n_inf = len({s["inference"] for s in steps})
    geo = np.load(run / "learned_joint/object_geometry.npz")["geometry"]
    t, dist, width = geo[:, 0], geo[:, 4], geo[:, 5]
    rows_raw = np.array([s["raw_gripper"] for s in steps]) if steps else np.zeros(0)
    near = dist <= NEAR_M
    near_raw = None
    if near.any():
        spans = [(s.get("sim_start"), s.get("sim_end"), s["raw_gripper"]) for s in steps]
        vals = [
            g
            for tick in t[near]
            for a, b, g in spans
            if a is not None and b is not None and a <= tick <= b
        ]
        near_raw = max(vals) if vals else None
    all_raw_channel = np.concatenate([r[:, 7] for r in raw]) if raw else np.zeros(0)
    executed_max = float(rows_raw.max()) if len(rows_raw) else None
    i_min = int(np.argmin(dist))
    rule = apply_rule(
        n_inf, float(dist.min()), executed_max if executed_max is not None else 0.0, near_raw
    )
    return {
        "run": str(run),
        "git_sha": meta.get("git_sha"),
        "config_path": meta.get("config_path"),
        "inferences_completed": n_inf,
        "rows_executed": len(steps),
        "rows_target_reached": sum(bool(s.get("target_reached")) for s in steps),
        "episode_error": ep["error"],
        "protective_stop": ep.get("protective_stop"),
        "stop_reason": ep.get("stop_reason"),
        "task_success_independent": ep["success"],
        "tcp_object_distance_m": {
            "start": float(dist[0]),
            "end": float(dist[-1]),
            "min": float(dist[i_min]),
            "min_at_sim_s": float(t[i_min]),
        },  # fmt: skip
        "gripper_width_m": {"min": float(width.min()), "max": float(width.max())},
        "raw_gripper_all_predicted_values": {
            "max": float(all_raw_channel.max()) if len(all_raw_channel) else None,
            "min": float(all_raw_channel.min()) if len(all_raw_channel) else None,
        },
        "raw_gripper_executed_rows_max": executed_max,
        "raw_gripper_max_while_tcp_within_0.05m": near_raw,
        "preregistered_outcome": rule,
    }


def main() -> None:
    run, out = map(pathlib.Path, sys.argv[1:3])
    result = analyse(run)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
