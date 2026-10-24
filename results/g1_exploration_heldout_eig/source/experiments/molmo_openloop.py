"""Capture real Panda sensors or run a pinned learned-action inference, without moving on it.

Capture needs only DREAM's CPU environment. Inference requires an isolated GPU environment
and a separately acquired reviewed checkpoint, outside Git and OneDrive. No download occurs
here. Raw tensors are saved before validity gates, including negative results.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dream.runtime.molmo import MolmoBackend, inspect_actions, observation, verify_checkpoint  # noqa: E402
from dream.sim.camera import CameraSpec  # noqa: E402
from dream.sim.tabletop import sample_tabletop  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402


def write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=lambda x: x.tolist()) + "\n", encoding="utf-8"
    )


def capture(out: Path, cfg: dict) -> None:
    env = TabletopEnv(
        sample_tabletop(np.random.default_rng(cfg["seed"])),
        np.random.default_rng(cfg["seed"]),
        extra_cameras=(CameraSpec(**cfg["wrist_camera"]),),
        camera_seed=cfg["seed"],
    )
    try:
        # No physics thread here: both frames and proprioception have the same timestamp.
        exterior, wrist = [source() for source in env.frame_sources()]
        state = env.robot_state()
        request = observation(exterior.rgb, wrist.rgb, state, cfg["instruction"])
        np.savez_compressed(
            out / "observation.npz",
            external=exterior.rgb,
            wrist=wrist.rgb,
            state=request["state"],
            joint_limits=np.stack([env.robot.ctrl_lo, env.robot.ctrl_hi], axis=1),
        )
        from PIL import Image

        Image.fromarray(exterior.rgb).save(out / "external.png")
        Image.fromarray(wrist.rgb).save(out / "wrist.png")
        write(
            out / "sensors.json",
            {
                "robot": dataclasses.asdict(state),
                "external": dataclasses.asdict(exterior.calibration),
                "wrist": dataclasses.asdict(wrist.calibration),
                "time": exterior.time,
                "state_mapping": cfg["state_mapping"],
                "physical_execution_permitted": False,
            },
        )
    finally:
        for cam in [env._camera, *env._extra_cameras.values()]:
            if cam is not None:
                cam.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["capture", "infer"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--observation", type=Path)
    ap.add_argument("--model-path", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / "experiments/configs/molmo_openloop.json").read_text())
    write(
        args.out / "meta.json",
        {
            "config": cfg,
            "git_sha": os.environ.get("DREAM_CODE_SHA")
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": False
            if os.environ.get("DREAM_CODE_SHA")
            else bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "python": sys.version,
            "command": args.command,
            "real_model_executed": False,
        },
    )
    if args.command == "capture":
        capture(args.out, cfg)
        return
    if args.model_path is None or args.observation is None:
        ap.error("inference requires --model-path and --observation")
    model = args.model_path.resolve()
    if ROOT == model or ROOT in model.parents or "onedrive" in str(model).lower():
        raise ValueError("model weights must stay outside the repository and OneDrive")
    verify_checkpoint(model, cfg["checkpoint_source_sha256"])
    source = np.load(args.observation, allow_pickle=False)
    request = {
        "images": [source["external"], source["external"], source["wrist"]],
        "state": source["state"],
        "task": cfg["instruction"],
    }
    write(
        args.out / "input.json",
        {
            "observation_sha256": hashlib.sha256(args.observation.read_bytes()).hexdigest(),
            "instruction": cfg["instruction"],
            "state": source["state"],
        },
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    start = time.perf_counter()
    try:
        backend = MolmoBackend(model, seed=cfg["inference_seed"])
        start = time.perf_counter()
        actions = backend.predict(request)
        np.save(args.out / "raw_actions.npy", actions, allow_pickle=False)
        report = inspect_actions(actions, source["joint_limits"])
        report.update(inference_s=time.perf_counter() - start, real_model_executed=True)
        write(args.out / "summary.json", report)
    except Exception as error:
        write(
            args.out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "elapsed_s": time.perf_counter() - start,
                "physical_execution_permitted": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
