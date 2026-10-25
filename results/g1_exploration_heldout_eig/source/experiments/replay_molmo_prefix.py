"""Offline prefix comparison using one saved real prediction. No fresh inference.

Frozen row counts 1/3/15, same original scene and controller limits. Report-only:
differences are replay diagnostics, not evidence of an improved feedback policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from dream.eval.joint import execute_chunk
from dream.eval.manipulation import object_in_tray, safety_summary
from dream.robot.joint import JointTargetController
from dream.runtime.molmo import SAMPLE_PERIOD_S, decode_actions
from dream.session import make_scene
from dream.sim.tabletop_env import TabletopEnv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results/molmo_cloud_feedback_dev/raw/learned_joint/raw_actions_000.npy"
    cfg = {"seed": 223, "row_counts": [1, 3, 15], "max_sim_s": 15}
    records = []
    for count in cfg["row_counts"]:
        env = TabletopEnv(make_scene("tray", cfg["seed"]), np.random.default_rng([223, 1]))
        env.robot = JointTargetController(env.model, env.data)
        decoded = decode_actions(
            np.load(source), np.stack([env.robot.ctrl_lo, env.robot.ctrl_hi], 1)
        )
        trace = []

        def record(*args, env=env, trace=trace):
            state = env.robot.state()
            trace.append(
                [
                    state.time,
                    *state.joint_pos,
                    *state.joint_vel,
                    *state.tcp_pos,
                    state.gripper_width,
                ]
            )

        env.step_hook = record
        result = execute_chunk(
            env,
            decoded.joint_rad,
            decoded.width_m,
            SAMPLE_PERIOD_S,
            max_targets=count,
            max_sim_s=cfg["max_sim_s"],
        )
        data = np.asarray(trace)
        np.savez_compressed(args.out / f"trace_{count}.npz", trace=data)
        records.append(
            {
                "requested_rows": count,
                **result,
                **safety_summary(env),
                "independent_task_success": object_in_tray(env, "red"),
                "final_tcp_m": env.robot.state().tcp_pos.tolist(),
                "peak_measured_accel_rad_s2": float(
                    np.max(
                        np.abs(np.diff(data[:, 8:15], axis=0) / np.diff(data[:, 0, None], axis=0))
                    )
                ),
            }
        )
    report = {
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": True,
        "config": cfg,
        "prediction_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256(
            (ROOT / "dream/robot/joint.py").read_bytes()
        ).hexdigest(),
        "records": records,
        "fresh_inference": False,
        "interpretation": "Recorded prefix replay; no fresh inference or feedback-policy claim",
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "requested_rows",
                        "completed",
                        "reason",
                        "independent_task_success",
                        "protective_stop",
                    ]
                }
                for r in records
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
