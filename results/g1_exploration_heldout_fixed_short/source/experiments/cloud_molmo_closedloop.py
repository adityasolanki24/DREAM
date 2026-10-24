"""Approved-only experiment wiring: real fresh-observation policy feedback over joint control.

There is no stand-in policy mode. Known-target CPU validation precedes this paid workload.
Policy sees RGB and proprioception only. Ground-truth scoring is independent and withheld
until execution ends. Retiming and all safety interventions remain in the saved trace.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dream.eval.joint import execute_target  # noqa: E402
from dream.eval.manipulation import object_in_tray, safety_summary  # noqa: E402
from dream.robot.joint import JointTargetController  # noqa: E402
from dream.runtime.agent import TaskAgent  # noqa: E402
from dream.runtime.molmo import (  # noqa: E402
    SAMPLE_PERIOD_S,
    MolmoBackend,
    decode_actions,
    observation,
    verify_checkpoint,
)
from dream.session import LiveSession  # noqa: E402
from dream.sim.camera import CameraSpec  # noqa: E402
from dream.tools.goals import parse_goal  # noqa: E402
from experiments.cloud_molmo_openloop import download_checkpoint  # noqa: E402


def write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.tolist()) + "\n", encoding="utf-8"
    )


def run_condition(backend, cfg: dict, sensor_cfg: dict, condition: str, out: Path) -> dict:
    """Same physical scene/cameras, explicit controller choice, never a learned fallback."""
    if condition not in ("classical", "learned_joint"):
        raise ValueError("unsupported execution condition")
    out.mkdir()
    session = LiveSession(
        seed=cfg["seed"],
        rate=1,
        perception="vision",
        extra_cameras=(CameraSpec(**sensor_cfg["wrist_camera"]),),
    )
    records, trace = [], []
    inference_count = 0
    if condition == "learned_joint":
        session.env.robot = JointTargetController(session.env.model, session.env.data)

    def record(*args):
        state = session.env.robot.state()
        trace.append(
            [
                state.time,
                *state.joint_pos,
                *state.joint_vel,
                *state.tcp_pos,
                state.gripper_width,
                float(state.protective_stop),
            ]
        )

    session.env.step_hook = record
    session.start()
    start = time.monotonic()
    error = None
    try:
        if condition == "classical":
            ep = TaskAgent(session.toolbox).run(parse_goal(f"in {cfg['goal_object']} tray"))
            records.append({"controller": "classical", "episode": ep.to_dict()})
        else:
            for i in range(cfg["max_inferences"]):
                if time.monotonic() - start > cfg["max_episode_wall_s"]:
                    error = "episode wall budget"
                    break
                with session.env.lock:
                    session.env.robot.hold()
                    exterior, wrist = [source() for source in session.env.frame_sources()]
                    state = session.env.robot.state()
                request = observation(exterior.rgb, wrist.rgb, state, cfg["instruction"])
                np.savez_compressed(
                    out / f"input_{i:03}.npz",
                    exterior=exterior.rgb,
                    wrist=wrist.rgb,
                    state=request["state"],
                )
                write(
                    out / f"input_{i:03}.json",
                    {
                        "instruction": cfg["instruction"],
                        "state_time": state.time,
                        "external_calibration": dataclasses.asdict(exterior.calibration),
                        "wrist_calibration": dataclasses.asdict(wrist.calibration),
                    },
                )
                infer_start = time.monotonic()
                raw = backend.predict(request)
                inference_count += 1
                np.save(out / f"raw_actions_{i:03}.npy", raw, allow_pickle=False)
                if (
                    session.toolbox.cancel.is_set()
                    or time.monotonic() - start > cfg["max_episode_wall_s"]
                ):
                    error = "cancelled or budget expired during inference; no dispatch"
                    break
                decoded = decode_actions(
                    raw, np.stack([session.env.robot.ctrl_lo, session.env.robot.ctrl_hi], axis=1)
                )
                result = {
                    "inference": i,
                    "inference_s": time.monotonic() - infer_start,
                    "raw_gripper": float(raw[0, 7]),
                    "gripper_saturated": bool(decoded.saturated[0]),
                    "controller": "learned_joint",
                }
                result.update(
                    execute_target(
                        session.env,
                        decoded.joint_rad[0],
                        float(decoded.width_m[0]),
                        SAMPLE_PERIOD_S,
                        session.toolbox.cancel,
                    )
                )
                records.append(result)
                write(out / "steps.json", records)
                if not result.get("target_reached"):
                    error = result["reason"]
                    break
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        with session.env.lock:
            session.env.robot.hold()
        session.env.wait_until(lambda: False, 0.5)
        with session.env.lock:
            # Scoring after action termination; never fed into prediction or action selection.
            success = object_in_tray(session.env, cfg["goal_object"])
            safety = safety_summary(session.env)
        session.close()
        # These renderers were created on this thread. Release their GL contexts here,
        # rather than relying on cyclic session/trace closures and delayed collection.
        cameras = [session.env._camera, *session.env._extra_cameras.values()]
        for camera in cameras:
            if camera is not None:
                camera.close()
        np.savez_compressed(out / "physical_trace.npz", trace=np.asarray(trace))
    result = {
        "condition": condition,
        "seed": cfg["seed"],
        "instruction": cfg["instruction"],
        "success": success,
        "steps": records,
        "error": error,
        "wall_s": time.monotonic() - start,
        "fresh_model_inference": isinstance(backend, MolmoBackend) and inference_count > 0,
        "classical_fallback": False,
        **safety,
    }
    write(out / "episode.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "experiments/configs/molmo_closedloop.json").read_text())
    sensors = json.loads((ROOT / "experiments/configs/molmo_openloop.json").read_text())
    cache = Path(os.environ.get("HF_HOME", "/work/model-cache")).resolve()
    if ROOT in cache.parents or cache == ROOT or "onedrive" in str(cache).lower():
        raise ValueError("model cache outside repository and OneDrive required")
    model = Path(download_checkpoint(sensors, cache))
    verify_checkpoint(model, sensors["checkpoint_source_sha256"])
    backend = MolmoBackend(model, seed=cfg["inference_seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_sha": os.environ.get("DREAM_CODE_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": cfg,
        "sensor_config": sensors,
        "python": sys.version,
        "policy_seed": cfg["inference_seed"],
    }
    write(args.out / "closedloop_meta.json", meta)
    records = [run_condition(backend, cfg, sensors, c, args.out / c) for c in cfg["conditions"]]
    learned = next(r for r in records if r["condition"] == "learned_joint")
    summary = {
        "episodes": records,
        "fresh_model_inference": learned["fresh_model_inference"],
        "at_least_one_verified_target": any(s.get("target_reached") for s in learned["steps"]),
        "interpretation": "development pilot; task success report-only; no transfer claim",
    }
    write(args.out / "closedloop_summary.json", summary)


if __name__ == "__main__":
    main()
