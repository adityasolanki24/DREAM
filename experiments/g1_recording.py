"""Record actual evaluation states; render them offline without executing physics again.

Snapshots are evidence, not robot observations. mj_setState below operates only on a
separate rendering MjData and never on the running controller. No resimulation, object
motion edits, interpolated frames or speed changes are used to create footage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image

from experiments.g1_dexterous import write_json

FPS = 20  # Recording/data contract, within 15–24 FPS; not an inferred REAL_G1 control rate.
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def external_recording_root(value: str | Path | None = None) -> Path:
    """Large footage stays outside this checkout, OneDrive and model/cache directories."""
    repo = Path(__file__).resolve().parents[1]
    if value is None:
        value = os.environ.get("DREAM_RECORDINGS_ROOT")
    if value is None:
        value = (
            "/work/out/recordings"
            if os.environ.get("DREAM_CODE_SHA")
            else Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "DREAM" / "recordings"
        )
    root = Path(value).resolve()
    forbidden = {"onedrive", "huggingface", ".cache", "models", "snapshots"}
    if root.is_relative_to(repo) or any(
        part.lower().startswith("onedrive") or part.lower() in forbidden for part in root.parts
    ):
        raise ValueError("recordings must be outside the repository, OneDrive and model caches")
    return root


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EpisodeRecorder:
    """Copy 20 Hz integration states and setpoints; no renderer or encoder during control."""

    def __init__(self, env, root: Path, metadata: dict) -> None:
        self.env = env
        self.root = external_recording_root(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict] = []
        self.states: list[np.ndarray] = []
        self.period = round(1 / (FPS * env.model.opt.timestep))
        if not np.isclose(self.period * env.model.opt.timestep, 1 / FPS):
            raise ValueError("recording requires an integer physics-step frame interval")
        write_json(
            self.root / "metadata.json",
            {
                **metadata,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "fps": FPS,
                "resolution": [1280, 720],
                "mujoco": mujoco.__version__,
                "joint_names": env.robot.names,
                "state_spec": int(STATE),
                "physics_dt_s": env.model.opt.timestep,
                "method": "actual evaluation snapshots, offline rendering only, original speed",
                "model_serialization": "MuJoCo binary model, visualization evidence only",
                "controller": "DREAM classical PD/exploration; no learned motor calls",
            },
        )
        mujoco.mj_saveModel(env.model, str(self.root / "scene.mjb"))
        self.sample("initialization")

    def sample(self, action: str) -> None:
        step = round(self.env.data.time / self.env.model.opt.timestep)
        if step % self.period or (self.rows and step <= self.rows[-1]["physics_step"]):
            return
        state = np.empty(mujoco.mj_stateSize(self.env.model, STATE))
        mujoco.mj_getState(self.env.model, self.env.data, state, STATE)
        self.states.append(state)
        robot = self.env.robot
        self.rows.append(
            {
                "time_s": float(self.env.data.time),
                "physics_step": step,
                "utc": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "stop_reason": robot.stop_reason,
                "joint_pos_rad": self.env.data.qpos[robot.qadr].copy(),
                "reference_rad": robot.reference.copy(),
                "target_rad": robot.target.copy(),
            }
        )

    def save(self) -> None:
        """Preserve all collected state samples before attempting any rendering."""
        np.savez_compressed(
            self.root / "snapshots.npz",
            integration_state=np.array(self.states),
            **{
                key: np.array([row[key] for row in self.rows])
                for key in ("time_s", "joint_pos_rad", "reference_rad", "target_rad")
            },
        )
        terminal = np.empty(mujoco.mj_stateSize(self.env.model, STATE))
        mujoco.mj_getState(self.env.model, self.env.data, terminal, STATE)
        np.savez_compressed(self.root / "terminal.npz", integration_state=terminal)
        write_json(
            self.root / "terminal.json",
            {
                "time_s": float(self.env.data.time),
                "stop_reason": self.env.robot.stop_reason,
                "last_uniform_frame_time_s": self.rows[-1]["time_s"],
                "note": "Terminal state retained even between video frame times; no padded frames.",
            },
        )
        with (self.root / "timeline.jsonl").open("w", encoding="utf-8") as log:
            for row in self.rows:
                log.write(
                    json.dumps({k: v for k, v in row.items() if not isinstance(v, np.ndarray)})
                    + "\n"
                )

    def replay_data(self, index: int) -> mujoco.MjData:
        """Restore one recorded state into a separate rendering buffer; never mj_step."""
        data = mujoco.MjData(self.env.model)
        mujoco.mj_setState(self.env.model, data, self.states[index], STATE)
        mujoco.mj_forward(self.env.model, data)
        return data

    def finish(self, episode_folder: Path, success: bool, *, render_media: bool = True) -> dict:
        self.save()
        for name in (
            "physics.jsonl",
            "trace.jsonl",
            "joint_mapping.json",
            "tools.json",
            "belief.json",
            "episode.json",
            "summary.json",
            "robot_contract.json",
            "frames.json",
        ):
            path = episode_folder / name
            if path.is_file():
                shutil.copyfile(path, self.root / name)
        if (episode_folder / "inference").exists():
            shutil.copytree(episode_folder / "inference", self.root / "inference")
        if (episode_folder / "frames").exists():
            shutil.copytree(episode_folder / "frames", self.root / "sensor_observations")
        if not render_media:
            result = {
                "status": "captured",
                "frames": len(self.rows),
                "fps": FPS,
                "physics_reexecuted": False,
                "rendering_deferred": True,
            }
        elif len(self.rows) < 2:
            result = {"status": "no_physics_segment", "frames": len(self.rows)}
        else:
            try:
                result = self.render(success)
            except Exception as error:
                result = {"status": "render_failed", "error": f"{type(error).__name__}: {error}"}
        result["path"] = str(self.root)
        write_json(self.root / "recording.json", result)
        return result

    def render(self, success: bool) -> dict:
        """Render both fixed spectator views; full footage survives failed evaluations."""
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg required; snapshots and episode evidence preserved")
        model = mujoco.MjModel.from_binary_path(str(self.root / "scene.mjb"))
        model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
        data = mujoco.MjData(model)
        cameras = {}
        specifications = {}
        for name, lookat, distance, azimuth, elevation in (
            ("wide", [0.12, 0, 0.78], 2.3, 135, -15),
            ("hands", [0.37, 0.10, 0.92], 0.60, 135, -25),
        ):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
            cameras[name] = camera
            specifications[name] = {
                "lookat_m": lookat,
                "distance_m": distance,
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
                "vertical_fov_deg": float(model.vis.global_.fovy),
                "resolution": [1280, 720],
                "frame": "world; fixed spectator camera",
            }
            (self.root / name / "originals").mkdir(parents=True, exist_ok=False)
        write_json(self.root / "cameras.json", specifications)
        frame_manifest = []
        with mujoco.Renderer(model, height=720, width=1280) as renderer:
            # Spectator graphics only. No change to sensor rendering or physics;
            # mesh shadows/reflections dominate CPU render time on local hardware.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            for i, state in enumerate(self.states):
                mujoco.mj_setState(model, data, state, STATE)
                mujoco.mj_forward(model, data)
                for name, camera in cameras.items():
                    renderer.update_scene(data, camera=camera)
                    path = self.root / name / "originals" / f"{i:06}.png"
                    Image.fromarray(renderer.render()).save(path)
                    frame_manifest.append(
                        {
                            "view": name,
                            "index": i,
                            "time_s": self.rows[i]["time_s"],
                            "sha256": digest(path),
                        }
                    )
        write_json(self.root / "frames_manifest.json", frame_manifest)
        videos = {}
        for name in cameras:
            movie = self.root / name / "episode.mp4"
            subprocess.run(
                [
                    executable,
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-framerate",
                    str(FPS),
                    "-i",
                    str(self.root / name / "originals" / "%06d.png"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(movie),
                ],
                check=True,
                capture_output=True,
            )
            videos[name] = {"file": str(movie), "sha256": digest(movie)}
            if success:
                # Keep the complete observation-to-verification sequence of this bounded episode.
                highlight = self.root / name / "complete_success.mp4"
                shutil.copyfile(movie, highlight)
                videos[name]["highlight"] = str(highlight)
        return {
            "status": "complete",
            "frames_per_view": len(self.states),
            "fps": FPS,
            "duration_s": len(self.states) / FPS,
            "videos": videos,
            "goal_success": success,
            "highlight_created": success,
            "physics_reexecuted": False,
            "speed_factor": 1.0,
            "spectator_graphics": "shadows and reflections disabled; original geometry/states",
        }


def render_saved(root: Path) -> dict:
    """Recover footage from retained states, preserving any earlier incomplete render.

    No environment/controller is constructed and mj_step is never called. The saved
    binary model and integration states are the sole source of motion.
    """
    root = external_recording_root(root)
    previous = root / ("partial_render_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    for name in ("wide", "hands"):
        path = root / name
        if path.exists():
            destination = previous / name
            if not path.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(
                root
            ):
                raise ValueError("render recovery paths escaped recording root")
            previous.mkdir(exist_ok=True)
            path.rename(destination)
    recorder = EpisodeRecorder.__new__(EpisodeRecorder)
    recorder.root = root
    with np.load(root / "snapshots.npz", allow_pickle=False) as arrays:
        recorder.states = list(arrays["integration_state"])
    recorder.rows = [
        json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()
    ]
    if len(recorder.rows) != len(recorder.states):
        raise ValueError("snapshot/timeline length mismatch")
    recorder.env = SimpleNamespace()  # render() deliberately needs no live simulation
    success = json.loads((root / "summary.json").read_text())["goal_success"]
    result = {**recorder.render(success), "path": str(root), "recovered_from_saved_states": True}
    write_json(root / "recording.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-saved", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render_saved(args.render_saved), indent=2))
