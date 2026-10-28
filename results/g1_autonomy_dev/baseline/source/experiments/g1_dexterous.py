"""Reproducible supported G1 close/lift/hold/release through physical actuators.

python -m experiments.g1_dexterous --out results/g1_dex3_dev --media <external-directory>
Scene setup uses known robot geometry, not perception. All cases are development cases.
Original RGB frames stay outside OneDrive; traces and checksummed metadata stay in results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from dream.eval.g1 import object_evidence, score_episode
from dream.robot.g1 import ARM
from dream.sim.g1 import G1Env, build_g1


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(config: Path, out: Path, media: Path | None = None, *, recording: bool = True) -> dict:
    cfg = json.loads(config.read_text())
    if cfg["sim_dt_s"] != 0.001 or cfg["sample_hz"] != 100:
        raise ValueError("unsupported sampling contract")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    repo = Path(__file__).resolve().parents[1]
    sources = [
        *repo.glob("dream/**/g1.py"),
        Path(__file__),
        config,
        repo / "dream/tools/dexterous.py",
        repo / "dream/types.py",
    ]
    source = {
        str(p.resolve().relative_to(repo)).replace("\\", "/"): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sources
    }
    write_json(
        out / "metadata.json",
        {
            "base_git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "working_tree_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff"])
            ).hexdigest(),
            "source_sha256": source,
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "seeds": [],
            "randomness": "none; deterministic scene and controller",
            "asset_manifest": "third_party/unitree_g1/SHA256SUMS.json",
            "asset_sha": "15ebe53f4cd77e7d745ae83b9d7ff90e685e8998",
            "support": "fixed pelvis; no root free joint; all 43 hinges physical",
        },
    )
    probe = G1Env()
    palm, _ = probe.robot.palm_pose("left")
    summaries = {}
    for case in cfg["cases"]:
        folder = out / case["name"]
        folder.mkdir()
        position = palm + np.array(case["offset_from_palm_m"])
        env = G1Env(build_g1(tuple(position), case["radius_m"], case["mass_kg"]))
        robot = env.robot
        recorder = None
        if recording:
            from experiments.g1_recording import EpisodeRecorder, external_recording_root

            recorder = EpisodeRecorder(
                env,
                external_recording_root(media) / out.name / case["name"],
                {
                    "config": cfg,
                    "case": case,
                    "seed": None,
                    "base_git_sha": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "scope": "classical commissioning; no learned motor calls",
                },
            )
        write_json(
            folder / "joint_mapping.json",
            [
                {
                    "name": n,
                    "actuator_index": i,
                    "joint_id": int(robot.joints[i]),
                    "qpos_index": int(robot.qadr[i]),
                    "qvel_index": int(robot.dadr[i]),
                    "limits_rad": robot.limits[i].tolist(),
                    "effort_limits_nm": robot.effort[i].tolist(),
                }
                for i, n in enumerate(robot.names)
            ],
        )
        renderer = None
        frames = []
        media_folder = None
        if media is not None:
            media_folder = media / case["name"]
            media_folder.mkdir(parents=True, exist_ok=False)
            try:
                renderer = mujoco.Renderer(env.model, height=480, width=640)
            except Exception as error:
                write_json(
                    folder / "render_failure.json",
                    {"type": type(error).__name__, "error": str(error)},
                )
        camera = mujoco.MjvCamera()
        camera.lookat[:] = palm + [0.06, -0.02, 0.02]
        camera.distance, camera.azimuth, camera.elevation = 0.65, 145, -20
        rows = []
        step = 0
        wall_start = time.perf_counter()
        try:
            with (folder / "trace.jsonl").open("w", encoding="utf-8") as log:
                for phase, seconds in cfg["phases_s"].items():
                    if phase in ("open", "release", "close"):
                        for side in ("left", "right"):
                            robot.set_hand(side, float(phase == "close"))
                    if phase == "lift":
                        q = np.zeros(len(ARM))
                        q[3] = cfg["lift_elbow_rad"]
                        robot.move_arm("left", q)
                    phase_steps = round(seconds / cfg["sim_dt_s"])
                    for phase_step in range(phase_steps):
                        env.step()
                        if recorder is not None:
                            recorder.sample(phase)
                        step += 1
                        if step % 10 == 0 or robot.stop_reason:
                            state = robot.state()
                            row = {
                                "phase": phase,
                                "physics_step": step,
                                "time_s": state.time,
                                "joint_pos_rad": state.joint_pos.tolist(),
                                "joint_vel_rad_s": state.joint_vel.tolist(),
                                "actuator_control_rad": env.data.ctrl.tolist(),
                                "joint_target_rad": robot.target.tolist(),
                                "actuator_force_nm": env.data.actuator_force.tolist(),
                                "joint_limit_excess_rad": np.maximum(
                                    np.maximum(
                                        robot.limits[:, 0] - state.joint_pos,
                                        state.joint_pos - robot.limits[:, 1],
                                    ),
                                    0,
                                ).tolist(),
                                "root_position_m": state.root_position.tolist(),
                                "root_quaternion_wxyz": state.root_quaternion.tolist(),
                                "palm_position_m": robot.palm_pose("left")[0].tolist(),
                                "stop_reason": state.stop_reason,
                                **object_evidence(env.model, env.data),
                            }
                            log.write(json.dumps(row, allow_nan=False) + "\n")
                            rows.append(row)
                        if renderer is not None and phase_step == phase_steps - 1:
                            from PIL import Image

                            renderer.update_scene(env.data, camera=camera)
                            path = media_folder / f"{len(frames):05}.png"
                            Image.fromarray(renderer.render()).save(path)
                            frames.append(
                                {
                                    "file": str(path.resolve()),
                                    "time_s": float(env.data.time),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                }
                            )
                        if robot.stop_reason:
                            break
                    if robot.stop_reason:
                        break
        finally:
            if renderer is not None:
                renderer.close()
            if recorder is not None:
                recorder.save()
        summary = score_episode(rows, cfg["gates"])
        summary["wall_s"] = time.perf_counter() - wall_start
        summary["frame_count"] = len(frames)
        write_json(folder / "frames.json", frames)
        write_json(folder / "summary.json", summary)
        if recorder is not None:
            summary["recording"] = recorder.finish(
                folder, summary["grasp_success"] and summary["release_drop_observed"]
            )
            write_json(folder / "summary.json", summary)
        summaries[case["name"]] = summary
    write_json(out / "summary.json", summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/g1_dex3_dev.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.out, args.media), indent=2))
