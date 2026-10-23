"""Isolated ordinary T1 walking validation. No NavGPT or manipulation inference.

--preflight runs the genuine reviewed ONNX actor on CPU, not simulated walking.
Execution requires a separate Isaac Sim 5.1 environment and approved RTX resources.
No downloads or paid resource allocation occur in this runner.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from dream.navigation.control import LocalWaypoint, WaypointTracker
from dream.robot.t1 import T1Policy

ROOT = Path(__file__).resolve().parents[1]


def preflight(folder: Path, cfg: dict) -> dict:
    if ROOT == folder.resolve() or ROOT in folder.resolve().parents:
        raise ValueError("policy must stay outside repository and OneDrive")
    if "onedrive" in str(folder.resolve()).lower():
        raise ValueError("policy must stay outside OneDrive")
    policy_cfg = json.loads((folder / "env_cfg.json").read_text())
    actor = T1Policy.load(folder, cfg["policy_hashes"], policy_cfg["joint_names"])
    obs = np.zeros(81, dtype=np.float32)
    obs[8] = -1
    actions = []
    for command in ([0, 0, 0], [0.2, 0, 0], [0.1, 0, 0.4]):
        obs[9:12] = command
        action = actor.policy(obs)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError("real actor contract failed")
        actions.append({"command": command, "action": action.tolist()})
    return {
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": cfg,
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "real_actor_executed": True,
        "physics_executed": False,
        "actor_input_dimension": 81,
        "actor_output_dimension": 23,
        "actions": actions,
        "isaacsim_installed": importlib.util.find_spec("isaacsim") is not None,
    }


def execute(folder: Path, cfg: dict, case: str, out: Path) -> dict:
    """Run actual PhysX; missing simulator raises instead of substituting a test double."""
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 720})
    rows, frames = [], []
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.storage.native import get_assets_root_path

        from dream.sim.isaac_t1 import IsaacT1

        world = World(stage_units_in_meters=1, physics_dt=0.005, rendering_dt=0.02)
        world.scene.add_default_ground_plane()
        # Fixed, empty development room; no semantic targets or privileged route hints.
        # Walls give the depth camera finite returns. Open-space no-return sensing is not
        # silently treated as free space by the conservative navigation guard.
        for i, (position, scale) in enumerate(
            [
                ([5, 0, 1.5], [0.1, 10, 3]),
                ([-5, 0, 1.5], [0.1, 10, 3]),
                ([0, 5, 1.5], [10, 0.1, 3]),
                ([0, -5, 1.5], [10, 0.1, 3]),
            ]
        ):
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/wall_{i}",
                    name=f"wall_{i}",
                    position=np.array(position),
                    scale=np.array(scale),
                )
            )
        body = IsaacT1(
            world,
            get_assets_root_path() + cfg["asset_relative_url"],
            folder,
            cfg["policy_hashes"],
        )
        # Stand initialization under the actor, never direct state writes after initialization.
        for _ in range(50):
            body.advance(0, 0)
        rgb, depth = body.capture()
        captured_time = body.time
        start_state = body.navigation_state(depth, captured_time)
        calibration = {
            "intrinsics": body.camera.get_intrinsics_matrix().tolist(),
            "mount_frame": "H2",
            "translation_m": [0.10, 0, 0.16],
            "orientation_wxyz_world_axes": [1, 0, 0, 0],
            "hfov_deg": 120,
            "width": 400,
            "height": 400,
            "depth_unit": "metres",
            "proprioception": "ideal simulated body velocity/gyro/orientation, integrated odometry",
        }
        (out / "sensor_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        tracker = WaypointTracker()
        goal = cfg["cases"][case]
        tracker.submit(
            LocalWaypoint(
                goal["forward_m"],
                goal["left_m"],
                goal["heading_rad"],
                body.time,
                0,
            ),
            start_state,
        )
        # Independent score frame: never passed to tracker or locomotion policy.
        start_position, start_quat = body.art.get_world_pose()
        from dream.sim.isaac_t1 import rotation_wxyz

        target = np.asarray(start_position) + rotation_wxyz(start_quat) @ np.array(
            [
                goal["forward_m"],
                goal["left_m"],
                0,
            ]
        )
        begin, status = time.monotonic(), "not started"
        for k in range(int(cfg["max_sim_s"] / 0.02)):
            if time.monotonic() - begin >= cfg["max_wall_s"]:
                status = "wall timeout"
                break
            if k % 10 == 0:
                rgb, depth = body.capture()
                captured_time = body.time
                frames.append({"time_s": body.time, "index": k})
                np.savez_compressed(out / f"sensor_{k:04}.npz", rgb=rgb, depth=depth)
            result = tracker.update(body.navigation_state(depth, captured_time))
            row = body.advance(result.forward_m_s, result.yaw_rad_s)
            pos, _ = body.art.get_world_pose()
            row.update(score_position_m=np.asarray(pos).tolist(), status=result.status)
            rows.append(row)
            status = result.status
            if body.stopped or status != "tracking":
                break
        body.command[:] = 0
        for _ in range(25):
            if body.sensors()["tilt"] > np.deg2rad(50):
                break
            body.advance(0, 0)
        final, _ = body.art.get_world_pose()
        error = float(np.linalg.norm(np.asarray(final)[:2] - target[:2]))
        final_speed = float(np.linalg.norm(body.art.get_linear_velocity()))
        return {
            "case": case,
            "real_actor_executed": True,
            "physics_executed": True,
            "navgpt_executed": False,
            "controller": "deterministic_waypoint_to_real_PPO",
            "position_error_m": error,
            "score_target_m": target.tolist(),
            "score_final_position_m": np.asarray(final).tolist(),
            "final_speed_m_s": final_speed,
            "status": status,
            "rows": rows,
            "frames": frames,
            "gate_pass": bool(
                rows
                and status == "waypoint reached"
                and not body.stopped
                and error < cfg["gates"]["final_position_error_m"]
                and final_speed < 0.1  # conservative prototype settled-stop gate, fixed pre-run
                and min(r["score_position_m"][2] for r in rows) >= 0.45
            ),
            "collision_count": None,
            "collision_note": "Contact scoring not implemented; flat ground only",
        }
    finally:
        (out / "partial_trace.json").write_text(json.dumps(rows, indent=2) + "\n")
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--case", choices=("stand", "forward", "left"), default="forward")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "experiments/configs/t1_walk.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = preflight(args.policy_dir, cfg)
    (args.out / "preflight.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.preflight:
        if not metadata["isaacsim_installed"]:
            raise RuntimeError("Isaac Sim unavailable; preflight is not physical walking")
        result = execute(args.policy_dir, cfg, args.case, args.out)
        (args.out / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "actions"}, indent=2))


if __name__ == "__main__":
    main()
