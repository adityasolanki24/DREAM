"""Inference-only REAL_G1 compatibility probe; predictions have no actuator connection.

The input is a measured DREAM snapshot with explicitly provisional joint/wrist conventions
and an external camera. Shape compatibility is not semantic or task compatibility. Requires
the pinned upstream checkout in a separate environment and authorized Cosmos backbone access.
No checkpoint, token, upstream code or cache is saved into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.g1_dexterous import write_json

STATE_DIMS = {
    "left_wrist_eef_9d": 9,
    "right_wrist_eef_9d": 9,
    "left_hand": 7,
    "right_hand": 7,
    "left_arm": 7,
    "right_arm": 7,
    "waist": 3,
}
ACTION_DIMS = {**STATE_DIMS, "base_height_command": 1, "navigate_command": 3}
TAG = "real_g1_relative_eef_relative_joints"


def load_observation(path: Path, instruction: str) -> dict:
    """Validate the actual array contract without importing a learned-policy package."""
    with np.load(path, allow_pickle=False) as arrays:
        video = arrays["video"]
        if (
            video.dtype != np.uint8
            or video.ndim != 5
            or video.shape[:2] != (1, 2)
            or video.shape[-1] != 3
        ):
            raise ValueError("REAL_G1 requires two uint8 RGB frames in (1,2,H,W,3)")
        states = {key: arrays[key].copy() for key in STATE_DIMS}
        for key, dimension in STATE_DIMS.items():
            if (
                states[key].dtype != np.float32
                or states[key].shape != (1, 1, dimension)
                or not np.isfinite(states[key]).all()
            ):
                raise ValueError(f"invalid REAL_G1 state {key}")
    return {
        "video": {"ego_view": video},
        "state": states,
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def validate_actions(actions: dict) -> dict:
    if set(actions) != set(ACTION_DIMS):
        raise ValueError("REAL_G1 decoded action keys differ from checkpoint contract")
    report = {}
    for key, dimension in ACTION_DIMS.items():
        array = np.asarray(actions[key])
        if array.shape != (1, 40, dimension) or not np.isfinite(array).all():
            raise ValueError(f"invalid REAL_G1 action tensor: {key}")
        report[key] = {
            "shape": list(array.shape),
            "min": float(array.min()),
            "max": float(array.max()),
            "dtype": str(array.dtype),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=73001)
    ap.add_argument(
        "--instruction",
        default="Investigate the red object and move it to the blue target if possible.",
    )
    args = ap.parse_args()
    observation = load_observation(args.input, args.instruction)
    args.out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json(
        args.out / "input.json",
        {
            "file": str(args.input),
            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "instruction": args.instruction,
            "tag": TAG,
            "started_utc": started,
            "seed": args.seed,
            "state_shapes": {k: list(v.shape) for k, v in observation["state"].items()},
            "semantics_verified": False,
            "actuation_permitted": False,
            "scope": "measured DREAM input; provisional conventions/domain shift; inference only",
        },
    )
    import torch
    from gr00t.policy import Gr00tPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback or substitute model")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = Gr00tPolicy(
        embodiment_tag="REAL_G1", model_path=str(args.model_path), device="cuda:0", strict=True
    )
    if any(p.device.type != "cuda" for p in policy.model.parameters()):
        raise RuntimeError("GR00T checkpoint is not fully on CUDA")
    original = policy.model.get_action

    def recorded(**kwargs):
        prediction = original(**kwargs)
        np.save(
            args.out / "normalized_actions.npy",
            prediction["action_pred"].detach().float().cpu().numpy(),
        )
        return prediction

    policy.model.get_action = recorded
    start = time.perf_counter()
    actions, _ = policy.get_action(observation)
    torch.cuda.synchronize()
    np.savez_compressed(args.out / "decoded_actions.npz", **actions)
    # Save first, then validate: malformed predictions remain genuine failure evidence.
    report = validate_actions(actions)
    write_json(
        args.out / "summary.json",
        {
            "genuine_inference": True,
            "physical_execution": False,
            "actuator_mapping": False,
            "input_semantics_verified": False,
            "latency_s": time.perf_counter() - start,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "device": "cuda:0",
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "git_sha": os.environ.get("DREAM_CODE_SHA"),
            "actions": report,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    )


if __name__ == "__main__":
    main()
