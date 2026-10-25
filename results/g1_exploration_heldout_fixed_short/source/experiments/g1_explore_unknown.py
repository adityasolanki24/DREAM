"""RGB-D -> TaskAgent -> exploration tools -> real G1 physics -> independent score.

python -m experiments.g1_explore_unknown --out results/g1_exploration_v1
No object construction parameters are passed to a decision component. The deterministic
backend receives a caller-bound structured objective, not a claim of language understanding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from dream.eval.g1 import object_evidence
from dream.perception.sphere import SpherePerception, detect_sphere
from dream.runtime.agent import TaskAgent
from dream.runtime.exploration import ExplorationPlanner
from dream.sim.camera import CameraSpec, RGBDCamera
from dream.sim.g1 import G1Env, build_g1
from dream.tools.goals import InRegion
from dream.tools.object_exploration import ExplorationToolbox
from dream.types import CameraCalibration, RGBDFrame
from experiments.g1_dexterous import write_json


def source_metadata(config: dict) -> dict:
    paths = [p for directory in ("dream", "experiments") for p in Path(directory).rglob("*.py")]
    return {
        "base_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "config": config,
        "reasoning": config["backend"],
        "new_paid_resources": False,
        "vlm_available": {
            k: importlib.util.find_spec(k) is not None for k in ("torch", "transformers")
        },
        "vlm_note": "No local Qwen checkpoint found in inspected Hugging Face cache; "
        "no configured inference endpoint. No download or paid inference.",
        "support": "fixed pelvis; original G1 hinges, effort/joint/slew limits retained",
        "contact_change": "5 ms critically damped solref, converged at 1/0.5 ms dt "
        "in separate numerical audit; not material calibration",
    }


def run_case(case: dict, config: dict, folder: Path, model_path: Path | None = None) -> dict:
    from PIL import Image

    folder.mkdir()
    (folder / "frames").mkdir()
    camera_spec = CameraSpec(**config["camera"])
    model = build_g1(
        tuple(case["position_m"]),
        case["radius_m"],
        case["mass_kg"],
        constrained=case["constrained"],
        camera=camera_spec,
        contact_timeconst=config["contact_timeconst_s"],
        target_position=tuple(config["target_position_m"]),
    )
    env = G1Env(model)
    camera = RGBDCamera(model, camera_spec, np.random.default_rng(case["seed"]))
    write_json(
        folder / "robot_contract.json",
        {
            "joint_names": env.robot.names,
            "qpos_indices": env.robot.qadr.tolist(),
            "qvel_indices": env.robot.dadr.tolist(),
            "joint_limits_rad": env.robot.limits.tolist(),
            "effort_limits_nm": env.robot.effort.tolist(),
            "physics_dt_s": model.opt.timestep,
            "control": "implicit position PD, rate-limited robot setpoints",
            "trace_hz": 100,
            "object_penetration_peak_hz": 1000,
        },
    )
    frames = []
    rows = []
    physics_step = 0
    peak_object_penetration = 0.0
    object_geom = model.geom("test_object_geom").id  # independent scoring only
    tools = None
    with (folder / "physics.jsonl").open("w", encoding="utf-8") as physics:

        def step():
            nonlocal physics_step, peak_object_penetration
            if env.data.time >= config["gates"]["max_episode_s"]:
                raise ValueError("episode simulation budget exhausted")
            env.step()
            physics_step += 1
            for contact in env.data.contact:
                if object_geom in (contact.geom1, contact.geom2):
                    peak_object_penetration = max(peak_object_penetration, -float(contact.dist))
            if physics_step % 10 == 0 or env.robot.stop_reason:
                state = env.robot.state()
                row = {
                    "time_s": state.time,
                    "physics_step": physics_step,
                    "tool": tools.action if tools is not None else "initialization",
                    "joint_pos_rad": state.joint_pos.tolist(),
                    "joint_vel_rad_s": state.joint_vel.tolist(),
                    "target_rad": env.robot.target.tolist(),
                    "control_rad": env.data.ctrl.tolist(),
                    "actuator_force_nm": env.data.actuator_force.tolist(),
                    "stop_reason": state.stop_reason,
                    "root_position_m": state.root_position.tolist(),
                    "root_quaternion_wxyz": state.root_quaternion.tolist(),
                    "grasp_frame_position_m": env.robot.grasp_pose("left")[0].tolist(),
                    **object_evidence(model, env.data),
                }
                physics.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)

        def capture():
            # Refresh derived poses at the current integrated state; no qpos/qvel writes.
            mujoco.mj_forward(model, env.data)
            frame = camera.capture(env.data)
            path = folder / "frames" / f"{len(frames):04}.npz"
            cal = frame.calibration
            np.savez_compressed(
                path,
                rgb=frame.rgb,
                depth=frame.depth,
                self_mask=frame.self_mask,
                time_s=frame.time,
                position=cal.position,
                rotation=cal.rotation,
                fovy_deg=cal.fovy_deg,
            )
            Image.fromarray(frame.rgb).save(path.with_suffix(".png"))
            # Independent scoring data is stored separately and never returned to perception.
            frames.append(
                {
                    "file": path.name,
                    "time_s": frame.time,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "score_object_position_m": env.data.body("test_object").xpos.tolist(),
                }
            )
            return frame

        perception = SpherePerception(capture, lambda: env.robot.state().time)
        tools = ExplorationToolbox(env.robot, step, perception, config["strategy"])
        goal = InRegion("red", "target")
        try:
            if config["backend"] == "deterministic":
                episode = TaskAgent(tools, ExplorationPlanner(), max_steps=config["max_steps"]).run(
                    goal
                )
            elif config["backend"] == "qwen":
                if model_path is None or not model_path.is_dir():
                    raise ValueError(
                        "genuine Qwen inference requires an existing local checkpoint path"
                    )
                from dream.runtime.qwen import QwenTransformersBackend
                from dream.runtime.vlm import VLMAgent

                backend = QwenTransformersBackend(str(model_path), device="cpu")
                episode = VLMAgent(
                    tools, backend, camera=lambda: capture().rgb, max_steps=config["max_steps"]
                ).run(config["instruction"], expected_goal=goal)
            else:
                raise ValueError("unknown reasoning backend")
            episode_data = episode.to_dict()
        except Exception as error:
            env.robot.hold()
            episode_data = {"success_claimed": False, "reason": f"{type(error).__name__}: {error}"}
            write_json(folder / "failure.json", episode_data)
        finally:
            camera.close()

    write_json(folder / "episode.json", episode_data)
    write_json(folder / "tools.json", tools.events)
    write_json(folder / "belief.json", tools.belief.evidence)
    # Re-run deterministic perception on saved sensor arrays for independent error scoring.
    errors = []
    for record in frames:
        with np.load(folder / "frames" / record["file"]) as arrays:
            rgb = arrays["rgb"]
            cal = CameraCalibration(
                rgb.shape[1],
                rgb.shape[0],
                float(arrays["fovy_deg"]),
                arrays["position"],
                arrays["rotation"],
            )
            frame = RGBDFrame(
                float(arrays["time_s"]), rgb, arrays["depth"], arrays["self_mask"], cal
            )
            estimate = detect_sphere(frame)
        record["estimate"] = None if estimate is None else estimate.to_dict()
        if estimate is not None:
            error = float(np.linalg.norm(estimate.position - record["score_object_position_m"]))
            record["localization_error_m"] = error
            errors.append(error)
    write_json(folder / "frames.json", frames)
    mujoco.mj_forward(model, env.data)
    final = env.data.body("test_object").xpos.copy()
    target = np.array(config["target_position_m"])
    inside = bool(np.max(abs(final[:2] - target[:2])) < 0.065 - case["radius_m"])
    resting = bool(abs(final[2] - case["radius_m"] - target[2]) < config["gates"]["rest_height_m"])
    penetration = peak_object_penetration
    safety = env.robot.state().protective_stop or any(
        e["status"] == "stopped" for e in tools.events
    )
    grasp_samples = 0
    longest = 0
    for row in rows:
        chains = {
            c["body"].split("_hand_")[1].split("_")[0]
            for c in row["contacts"]
            if "_hand_" in c["body"] and c["normal_force_n"] > 0.1
        }
        lifted = row["object_position_m"][2] > case["position_m"][2] + 0.02
        grasp_samples = grasp_samples + 1 if lifted and len(chains) >= 2 else 0
        longest = max(longest, grasp_samples)
    contact_ok = penetration <= config["gates"]["penetration_m"]
    p = tools.belief.p_movable
    classification = (
        "movable" if p >= 0.95 else "constrained_under_probe" if p <= 0.02 else "uncertain"
    )
    correct = (classification == "movable" and not case["constrained"]) or (
        classification == "constrained_under_probe" and case["constrained"]
    )
    success = (
        inside
        and resting
        and tools.released
        and tools.holding is None
        and contact_ok
        and not safety
    )
    summary = {
        "physics_executed": physics_step > 0,
        "physics_steps": physics_step,
        "simulated_s": env.data.time,
        "backend": config["backend"],
        "instruction": config["instruction"],
        "classification": classification,
        "classification_correct": correct,
        "p_movable": p,
        "probes": tools.probes,
        "grasp_independently_verified": longest * 0.01 >= 0.3 and contact_ok and not safety,
        "maximum_sustained_lift_contact_s": longest * 0.01,
        "goal_geometric": inside and resting,
        "goal_success": success,
        "success_claimed": episode_data["success_claimed"],
        "false_success_claim": episode_data["success_claimed"] and not success,
        "maximum_contact_penetration_m": penetration,
        "penetration_gate_passed": contact_ok,
        "safety_stop": safety,
        "reason": episode_data["reason"],
        "detection_frames": len(errors),
        "total_frames": len(frames),
        "maximum_localization_error_m": max(errors, default=None),
        "localization_gate_passed": bool(errors)
        and max(errors) <= config["gates"]["localization_m"],
        "final_object_position_m": final.tolist(),
        "failed_actions": sum(e["status"] not in ("succeeded", "rejected") for e in tools.events),
        "scope": "fixed pelvis, red sphere/blue platform, classical perception and control; "
        "no VLM inference unless backend=qwen",
    }
    write_json(folder / "summary.json", summary)
    return summary


def run(config_path: Path, out: Path, model_path: Path | None = None) -> dict:
    config = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    metadata = source_metadata(config)
    write_json(out / "metadata.json", metadata)
    for name in metadata["source_sha256"]:
        path = out / "source" / name
        path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(name, path)
    summaries = {}
    for case in config["cases"]:
        summaries[case["name"]] = run_case(case, config, out / case["name"], model_path)
        write_json(out / "summary.json", summaries)
        print(case["name"], json.dumps(summaries[case["name"]]), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/g1_explore_dev.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    run(args.config, args.out, args.model_path)
