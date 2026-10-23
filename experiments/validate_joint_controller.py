"""Physical local validation: known targets first, then one genuine recorded model target."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dream.eval.joint import execute_target  # noqa: E402
from dream.eval.manipulation import safety_summary  # noqa: E402
from dream.robot.joint import JointTargetController  # noqa: E402
from dream.runtime.molmo import SAMPLE_PERIOD_S, decode_actions  # noqa: E402
from dream.sim.tabletop import sample_tabletop  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402


def run(seed: int, target: np.ndarray | None, width: float, out: Path, delta=0.0) -> dict:
    env = TabletopEnv(sample_tabletop(np.random.default_rng(seed)), np.random.default_rng(seed))
    env.robot = JointTargetController(env.model, env.data)
    if target is None:
        target = env.robot.state().joint_pos.copy()
        target[6] += delta
    trace = []

    def record(*args):
        state = env.robot.state()
        trace.append(
            [
                state.time,
                *state.joint_pos,
                *state.joint_vel,
                *env.robot.command_velocity,
                *env.robot.command_acceleration,
                *state.tcp_pos,
                state.gripper_width,
            ]
        )

    env.step_hook = record
    receipt = execute_target(env, target, width, SAMPLE_PERIOD_S)
    data = np.asarray(trace)
    np.savez_compressed(out / f"trace_{seed}.npz", trace=data)
    velocities = data[:, 8:15]
    result = {
        "seed": seed,
        "target": target.tolist(),
        "width_m": width,
        **receipt,
        **safety_summary(env),
        "peak_measured_speed_rad_s": float(np.max(np.abs(velocities))),
        "peak_measured_accel_rad_s2": float(np.max(np.abs(np.diff(velocities, axis=0))) / env.dt),
        "limits": dataclasses.asdict(env.robot.joint_limits),
    }
    (out / f"episode_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / "experiments/configs/joint_validation.json").read_text())
    records = [
        run(c["seed"], None, 0.07, args.out, c["joint7_delta_rad"]) for c in cfg["known_targets"]
    ]

    def passed(r):
        return (
            r.get("target_reached")
            and not r["protective_stop"]
            and r["tracking_error_rad"] < cfg["gates"]["max_tracking_error_rad"]
            and r["peak_measured_speed_rad_s"] < cfg["gates"]["max_measured_speed_rad_s"]
            and r["peak_measured_accel_rad_s2"] < cfg["gates"]["max_measured_accel_rad_s2"]
        )

    known_pass = all(passed(r) for r in records)
    if known_pass:
        raw = np.load(ROOT / cfg["recorded_prediction"]["path"], allow_pickle=False)
        observed = np.load(ROOT / "results/molmo_cloud_openloop/raw/observation/observation.npz")
        decoded = decode_actions(raw, observed["joint_limits"])
        r = run(
            cfg["recorded_prediction"]["seed"],
            decoded.joint_rad[0],
            float(decoded.width_m[0]),
            args.out,
        )
        r["gripper_saturated"] = bool(decoded.saturated[0])
        r["raw_gripper_prediction"] = float(raw[0, 7])
        (args.out / f"episode_{r['seed']}.json").write_text(json.dumps(r, indent=2) + "\n")
        records.append(r)
    result = {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "mujoco")},
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "dream/robot/joint.py",
                "dream/eval/joint.py",
                "dream/runtime/molmo.py",
                "experiments/validate_joint_controller.py",
                "dream/sim/tabletop_env.py",
                "experiments/configs/joint_validation.json",
            )
        },
        "config": cfg,
        "episodes": records,
        "gate_pass": known_pass and len(records) == 3 and all(passed(r) for r in records),
        "fresh_model_inference": False,
        "task_success_evaluated": False,
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
