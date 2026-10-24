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
import os
import subprocess
import time
import traceback
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
        "code_sha": os.environ.get("DREAM_CODE_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": cfg,
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "real_actor_executed": True,
        "physics_executed": False,
        "actor_input_dimension": 81,
        "actor_output_dimension": 23,
        "actions": actions,
        "isaacsim_installed": importlib.util.find_spec("isaacsim") is not None,
    }


def start_application():
    """Start Kit, then explicitly enable sensor/storage modules before importing them."""
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 1280,
            "height": 720,
            "extra_args": ["--allow-root"] if hasattr(os, "geteuid") and os.geteuid() == 0 else [],
        }
    )
    try:
        from isaacsim.core.utils.extensions import enable_extension

        for name in ("isaacsim.sensors.camera", "isaacsim.storage.native"):
            enable_extension(name)
    except BaseException:
        # Kit may exit before execute can see an extension failure.
        traceback.print_exc()
        raise
    return app


def mark_stage(out: Path, stage: str) -> None:
    """Persist progress before Isaac Sim can shut down the process."""
    record = {"stage": stage, "timestamp": time.time()}
    with (out / "stages.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")
        log.flush()
        os.fsync(log.fileno())
    print(f"DREAM_T1_STAGE: {stage}", flush=True)


def last_stage(out: Path) -> str | None:
    path = out / "stages.jsonl"
    return json.loads(path.read_text().splitlines()[-1])["stage"] if path.exists() else None


def record_physical_step(body, row: dict, out: Path, phase: str) -> None:
    """Independent score telemetry; this world pose is never passed to the controller."""
    position, orientation = body.art.get_world_pose()
    row.update(
        score_position_m=np.asarray(position).tolist(),
        score_orientation_wxyz=np.asarray(orientation).tolist(),
        phase=phase,
    )
    with (out / "robot_trajectory.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(row, allow_nan=False) + "\n")
        log.flush()


def execute(folder: Path, cfg: dict, case: str, out: Path) -> dict:
    """Persist outcomes before Kit shutdown, which may terminate the Python process."""
    rows = []
    app = None
    try:
        mark_stage(out, "simulation_starting")
        app = start_application()
        mark_stage(out, "simulation_started")
        result = run_episode(folder, cfg, case, out, rows)
        (out / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
        mark_stage(out, "episode_saved")
        if cfg.get("record_video"):
            (out / "spectator/episode.json").write_text(json.dumps(result, indent=2) + "\n")
            (out / "spectator/source.json").write_text(
                json.dumps(
                    {
                        "code_sha": os.environ.get("DREAM_CODE_SHA")
                        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        "config": cfg,
                        "camera_role": "spectator only; never policy input",
                    },
                    indent=2,
                )
                + "\n"
            )
        return result
    except BaseException as error:
        (out / "failure.json").write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "last_stage": last_stage(out),
                },
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        raise
    finally:
        (out / "partial_trace.json").write_text(json.dumps(rows, indent=2) + "\n")
        if app is not None:
            app.close()


def run_episode(folder: Path, cfg: dict, case: str, out: Path, rows: list) -> dict:
    """Run actual PhysX; missing simulator raises instead of using a test double."""
    frames = []
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.storage.native import get_assets_root_path

    from dream.sim.isaac_t1 import IsaacT1

    world = World(stage_units_in_meters=1, physics_dt=0.005, rendering_dt=0.02)
    mark_stage(out, "world_created")
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
    mark_stage(out, "t1_initialization_started")
    body = IsaacT1(
        world,
        get_assets_root_path() + cfg["asset_relative_url"],
        folder,
        cfg["policy_hashes"],
        checkpoint=lambda stage: mark_stage(out, stage),
        diagnostic_dir=out,
    )
    mark_stage(out, "t1_initialized")
    # Stand initialization under the actor, never direct state writes after initialization.
    mark_stage(out, "warmup_started")
    recorder = camera = None
    if cfg.get("record_video"):
        from experiments.t1_recording import FrameRecorder, spectator_camera

        camera = spectator_camera()
        recorder = FrameRecorder(out / "spectator")

    def record(phase: str) -> None:
        if recorder is None or body.time - recorder.last_time < 0.04 - 1e-6:
            return
        world.render()
        rgba = camera.get_rgba()
        if rgba is None:
            raise RuntimeError("spectator frame unavailable")
        position, _ = body.art.get_world_pose()
        recorder.capture(
            rgba,
            body.time,
            {
                "phase": phase,
                "command": body.command.tolist(),
                "position_m": np.asarray(position).tolist(),
                "speed_m_s": float(np.linalg.norm(body.art.get_linear_velocity())),
            },
        )

    warmup = []
    with (out / "warmup.jsonl").open("w", encoding="utf-8") as log:
        for step in range(50):
            try:
                row = body.advance(0, 0)
                record_physical_step(body, row, out, "warmup")
                warmup.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if body.stopped:
                    raise RuntimeError("tilt stop during standing warmup")
                record("standing warmup")
            except BaseException as error:
                report = {
                    "failed_step": step,
                    "completed_steps": len(warmup),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "sim_time_s": body.time,
                }
                try:
                    state = body.sensors()
                    report["joint_names"] = list(body.art.dof_names)
                    report["joint_limits_rad"] = body.limits.tolist()
                    report["proprioception"] = {
                        key: np.asarray(value).tolist() for key, value in state.items()
                    }
                    report["last_accepted_ppo_action"] = body.actor.last.tolist()
                    report["last_target_limit_excess_rad"] = body.actor.target_limit_excess.tolist()
                except Exception as diagnostic_error:
                    report["diagnostic_error"] = str(diagnostic_error)
                try:
                    (out / "warmup_failure.json").write_text(
                        json.dumps(report, indent=2) + "\n",
                        encoding="utf-8",
                    )
                except Exception:
                    traceback.print_exc()
                raise
    mark_stage(out, "warmup_completed")
    body.initialize_camera()
    mark_stage(out, "camera_capture_started")
    rgb, depth = body.capture()
    mark_stage(out, "camera_captured")
    np.savez_compressed(out / "sensor_initial.npz", rgb=rgb, depth=depth)
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
    mark_stage(out, "walking_started")
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
        row.update(status=result.status)
        record_physical_step(body, row, out, case)
        rows.append(row)
        record(case)
        status = result.status
        holding_stand = case == "stand" and k < 150 and status == "waypoint reached"
        if body.stopped or (status != "tracking" and not holding_stand):
            break
    mark_stage(out, "walking_loop_exited")
    body.command[:] = 0
    for _ in range(25):
        if body.sensors()["tilt"] > np.deg2rad(50):
            break
        row = body.advance(0, 0)
        row.update(status="settling")
        record_physical_step(body, row, out, "settling")
        rows.append(row)
        record("controlled stop")
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
        "warmup_steps": len(warmup),
        "physics_steps": world.current_time_step_index,
        "frames": frames,
        "gate_pass": bool(
            rows
            and status == "waypoint reached"
            and not body.stopped
            and error < cfg["gates"]["final_position_error_m"]
            and final_speed < 0.1  # conservative prototype settled-stop gate, fixed pre-run
            and len(warmup) == 50
            and min(r["score_position_m"][2] for r in warmup + rows) >= 0.45
            and max(r["tilt_rad"] for r in warmup + rows) <= np.deg2rad(25)
        ),
        "collision_count": None,
        "collision_note": "Contact scoring not implemented; flat ground only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--case", choices=("stand", "forward", "left"), default="forward")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "experiments/configs/t1_walk.json").read_text())
    if args.record_video:
        if "onedrive" in str(args.out.resolve()).lower() or ROOT in args.out.resolve().parents:
            raise ValueError("video output must stay outside Git and OneDrive")
        cfg["record_video"] = True
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = preflight(args.policy_dir, cfg)
    (args.out / "preflight.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.preflight:
        if not metadata["isaacsim_installed"]:
            raise RuntimeError("Isaac Sim unavailable; preflight is not physical walking")
        execute(args.policy_dir, cfg, args.case, args.out)
    print(json.dumps({k: v for k, v in metadata.items() if k != "actions"}, indent=2))


if __name__ == "__main__":
    main()
