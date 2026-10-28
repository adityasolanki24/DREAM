"""Capture supported G1 sensors for an inference-only, provisional REAL_G1 contract probe.

No model actions are executed. Joint order is explicitly DREAM's named order; wrist pose is
the robot wrist-yaw body in pelvis coordinates, not a verified upstream EEF convention.
The fixed external camera is NOT an ego camera. Twenty sample indices span one second at
our declared 20 Hz acquisition rate, which is not a claim about training frequency.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np

from dream.robot.g1 import ARM, hand_names
from dream.sim.camera import CameraSpec, RGBDCamera
from dream.sim.g1 import G1Env, build_g1
from experiments.g1_dexterous import write_json


def calibration_record(calibration) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in dataclasses.asdict(calibration).items()
    }


def capture(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path("experiments/configs/g1_qwen_comparison.json").read_text())
    case = cfg["cases"][2]  # Existing free/heavier development scene, not a new success case.
    camera_spec = CameraSpec(**cfg["camera"])
    env = G1Env(
        build_g1(
            tuple(case["position_m"]),
            case["radius_m"],
            case["mass_kg"],
            camera=camera_spec,
            contact_timeconst=cfg["contact_timeconst_s"],
            target_position=tuple(cfg["target_position_m"]),
        )
    )
    camera = RGBDCamera(env.model, camera_spec, np.random.default_rng(case["seed"]))
    frames = []
    try:
        frames.append(camera.capture(env.data))
        for _ in range(round(1.0 / env.model.opt.timestep)):
            env.step()
        mujoco.mj_forward(env.model, env.data)
        frames.append(camera.capture(env.data))
        state = env.robot.state()
        if state.protective_stop:
            raise RuntimeError(state.stop_reason)
        named = dict(zip(state.joint_names, state.joint_pos, strict=True))
        orders = {
            f"{side}_arm": [f"{side}_{name}_joint" for name in ARM] for side in ("left", "right")
        }
        orders.update({f"{side}_hand": list(hand_names(side)) for side in ("left", "right")})
        orders["waist"] = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        streams = {
            key: np.array([[[named[n] for n in names]]], np.float32)
            for key, names in orders.items()
        }
        root = env.data.body("pelvis")
        root_r = root.xmat.reshape(3, 3)
        for side in ("left", "right"):
            wrist = env.data.body(f"{side}_wrist_yaw_link")
            position = root_r.T @ (wrist.xpos - root.xpos)
            rotation = root_r.T @ wrist.xmat.reshape(3, 3)
            # Exact upstream pose.py ROT6D helper uses the first two rows flattened.
            streams[f"{side}_wrist_eef_9d"] = np.concatenate(
                (position, rotation[:2].ravel())
            ).astype(np.float32)[None, None]
        np.savez_compressed(
            out / "observation.npz", video=np.stack([f.rgb for f in frames])[None], **streams
        )
        for i, frame in enumerate(frames):
            np.savez_compressed(
                out / f"sensor_{i}.npz",
                rgb=frame.rgb,
                depth=frame.depth,
                self_mask=frame.self_mask,
                time_s=frame.time,
            )
        write_json(
            out / "contract.json",
            {
                "joint_orders": orders,
                "joint_unit": "rad",
                "wrist_position_unit": "m",
                "wrist_frame": "pelvis; wrist_yaw_link origin; rotation first two rows",
                "camera": calibration_record(frames[0].calibration),
                "camera_semantics": "external fixed, not ego; domain shift unresolved",
                "sample_times_s": [f.time for f in frames],
                "declared_sampling_hz": 20,
                "history_indices": [-20, 0],
                "trained_frequency_verified": False,
                "upstream_joint_order_and_frame_verified": False,
                "motor_policy_actions_executed": False,
                "physics_steps": 1000,
            },
        )
    finally:
        camera.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    capture(parser.parse_args().out)
