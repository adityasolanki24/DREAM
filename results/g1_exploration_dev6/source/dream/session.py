"""Live simulation session: a continuously running robot that accepts goals while it runs.

Threads:
- physics: steps the world at ``rate`` x real time and keeps the controller running. The robot
  holds its pose when idle, independent of how long planning takes;
- agent: runs one submitted goal at a time through ``TaskAgent``;
- caller (CLI or tests): submits goals, pauses or cancels, and reads status.

Pausing or cancelling interrupts the current tool at its next wait. The robot then holds its
current pose (the safe state), and ``resume`` replans from the current observed state. Each
episode and a 10 Hz state trajectory (for replay) are written to ``log_dir``.

This module wires the simulator to the agent, like ``dream.eval.episode`` does, and is not
decision-making code.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from dream.perception.vision import VisionPerception
from dream.runtime.agent import Episode, TaskAgent
from dream.sim.camera import CameraSpec
from dream.sim.tabletop import TabletopSpec, sample_compartment_scene, sample_tabletop
from dream.sim.tabletop_env import TabletopEnv
from dream.tools.goals import Goal, parse_goal
from dream.tools.toolbox import Toolbox


@dataclass
class Status:
    goal: str | None = None
    state: str = "idle"  # idle | running | paused | done | failed | cancelled
    current_action: str | None = None
    last_result: str | None = None
    message: str = ""
    events: list[dict] = field(default_factory=list)
    discovery: dict | None = None  # latest investigate_physics step (for displays)


SCENES = ("tray", "drawer", "lid", "fixed")


def make_scene(scene: str, seed: int) -> TabletopSpec:
    """The tray scene, or a compartment scene with a hidden mechanism of the given kind."""
    if scene == "tray":
        return sample_tabletop(np.random.default_rng(seed))
    return sample_compartment_scene(scene, np.random.default_rng([seed, 0]))


class LiveSession:
    def __init__(
        self,
        seed: int = 0,
        rate: float = 1.0,
        viewer: bool = False,
        log_dir: str | pathlib.Path | None = None,
        scene: str = "tray",
        perception: str = "oracle",
        extra_cameras: tuple[CameraSpec, ...] = (),
    ) -> None:
        """``perception``: "oracle" (noisy ground-truth trackers) or "vision" (the RGB-D camera
        pipeline). Logs record which one ran."""
        if scene not in SCENES:
            raise ValueError(f"scene must be one of {SCENES}")
        if perception not in ("oracle", "vision"):
            raise ValueError("perception must be 'oracle' or 'vision'")
        self.seed = seed
        self.scene_kind = scene
        self.rate = rate  # simulated seconds per wall second (0 = as fast as possible)
        self.env = TabletopEnv(
            make_scene(scene, seed),
            np.random.default_rng([seed, 1]),
            camera_seed=seed,
            extra_cameras=extra_cameras,
        )
        self.perception = (
            self.env
            if perception == "oracle"
            else VisionPerception(
                self.env.capture_frame, lambda: self.env.time, np.random.default_rng([seed, 3])
            )
        )
        self.toolbox = Toolbox(
            self.env.embodiment(), self.perception, rng=np.random.default_rng([seed, 2])
        )
        self.toolbox.on_discovery_step = self._on_discovery_step
        self.agent = TaskAgent(self.toolbox)
        self.status = Status()
        self.episodes: list[Episode] = []
        self.log_dir = pathlib.Path(log_dir) if log_dir else None
        self._trajectory: list[dict] = []
        self._running = threading.Event()
        self._physics: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._paused_goal: Goal | None = None
        self._viewer = None
        self._want_viewer = viewer

    # ----- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        self.env.background = True
        self._running.set()
        self._physics = threading.Thread(target=self._physics_loop, name="physics", daemon=True)
        self._physics.start()

    def close(self) -> None:
        self.cancel()
        if self._worker:
            self._worker.join(timeout=10)
        self._running.clear()
        if self._physics:
            self._physics.join(timeout=5)
        self.env.background = False
        if self._viewer is not None:
            self._viewer.close()
        self._write_logs()

    # ----- commands ------------------------------------------------------------------------

    def submit(self, goal: str | dict | Goal) -> str:
        """Start working on a goal. Returns an error message, or '' on acceptance."""
        if self.busy:
            return "busy: pause or cancel the current goal first"
        try:
            g = goal if not isinstance(goal, (str, dict)) else parse_goal(goal)
        except ValueError as e:
            return str(e)
        self.toolbox.cancel.clear()
        self.status = Status(goal=repr(g), state="running")
        self._worker = threading.Thread(target=self._run_goal, args=(g,), name="agent", daemon=True)
        self._worker.start()
        return ""

    def pause(self) -> None:
        """Interrupt the current tool. The robot holds still, and ``resume`` continues the goal."""
        if self.busy:
            self._paused_goal = self._current_goal
            self.toolbox.cancel.set()
            self._worker.join(timeout=30)
            self.status.state = "paused"

    def resume(self) -> str:
        if self._paused_goal is None:
            return "nothing to resume"
        g, self._paused_goal = self._paused_goal, None
        return self.submit(g)

    def cancel(self) -> None:
        self._paused_goal = None
        if self.busy:
            self.toolbox.cancel.set()
            self._worker.join(timeout=30)
            self.status.state = "cancelled"

    def reset_safety(self) -> None:
        with self.env.lock:
            self.env.robot.reset_safety()

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def wait(self, timeout: float = 300.0) -> bool:
        """Block until the current goal finishes (for scripts and tests)."""
        if self._worker:
            self._worker.join(timeout)
        return not self.busy

    def scene(self) -> dict:
        return self.toolbox.call("observe_scene").data

    # ----- threads ------------------------------------------------------------------------

    def _physics_loop(self) -> None:
        chunk = 10  # physics steps per iteration (20 ms of simulated time)
        last_log = -1.0
        while self._running.is_set():
            t_wall = time.perf_counter()
            self.env.step(chunk)
            if self._want_viewer:
                self._sync_viewer()
            if self.env.time - last_log >= 0.1:
                last_log = self.env.time
                with self.env.lock:
                    self._trajectory.append(
                        {
                            "t": round(self.env.time, 3),
                            "qpos": np.round(self.env.data.qpos, 5).tolist(),
                            "overlay": self._overlay(),
                        }
                    )
            if self.rate > 0:
                slack = chunk * self.env.dt / self.rate - (time.perf_counter() - t_wall)
                if slack > 0:
                    time.sleep(slack)

    def _run_goal(self, goal: Goal) -> None:
        self._current_goal = goal

        def on_event(kind: str, data: dict) -> None:
            self.status.events.append({"t": round(self.env.time, 2), "kind": kind, **data})
            if kind == "action":
                self.status.current_action = f"{data['tool']}({data['args']})"
            elif kind == "result":
                self.status.last_result = (
                    f"{data['tool']}: {data['status']} {data['message']}".strip()
                )
                self.status.current_action = None

        ep = self.agent.run(goal, on_event)
        self.episodes.append(ep)
        self.status.message = ep.reason
        if self.toolbox.cancel.is_set():
            self.status.state = "paused" if self._paused_goal is not None else "cancelled"
        else:
            self.status.state = "done" if ep.success_claimed else "failed"

    def _on_discovery_step(self, step: dict, belief) -> None:
        fit = belief.fit_check()
        info = {
            "probe": step["step"],
            "direction": step["probe"]["direction"],
            "max_force_n": step["probe"]["max_force"],
            "type_probs": step["type_probs"],
            "predicted_mm": round(1000 * step["predicted_displacement_m"], 1),
            "observed_mm": round(1000 * step["observed_displacement_m"], 1),
            "prediction_error_mm": round(1000 * step["pre_action_pred_err_m"], 1),
            "chi2_per_dof": round(fit["chi2_per_dof"], 3),
            "model_adequate": fit["model_adequate"],
        }
        self.status.discovery = info
        self.status.events.append({"t": round(self.env.time, 2), "kind": "probe", **info})

    def _overlay(self) -> dict:
        """What a display shows at this instant (logged with the trajectory for replay)."""
        st = self.status
        return {
            "goal": st.goal,
            "state": st.state,
            "action": st.current_action,
            "last_result": st.last_result,
            "discovery": st.discovery,
            "message": st.message,
            "perception": self._perception_overlay(),
        }

    def _perception_overlay(self) -> dict | None:
        """The latest camera frame's detections, labelled with world-model names (vision only)."""
        vis = self.perception
        if not isinstance(vis, VisionPerception) or vis.last_frame is None:
            return None
        names = vis.world.names()
        tracks = {id(t.detection): t for t in vis.world.tracks}
        dets = []
        for d in vis.last_detections:
            t = tracks.get(id(d))
            dets.append(
                {
                    "label": names.get(t.id, d.kind) if t else d.kind,
                    "kind": d.kind,
                    "bbox": list(d.bbox),
                    "position": np.round(d.position, 3).tolist(),
                    "std_mm": round(1000 * d.position_std, 1),
                    "occluded": d.occluded,
                }
            )
        hidden = [names[t.id] for t in vis.world.tracks if not t.visible]
        return {"frame_time": round(vis.last_frame.time, 2), "detections": dets, "hidden": hidden}

    def _sync_viewer(self) -> None:
        import mujoco.viewer

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        with self.env.lock:
            self._viewer.sync()

    def _write_logs(self) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "episodes.jsonl", "w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep.to_dict()) + "\n")
        (self.log_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "seed": self.seed,
                    "scene": self.scene_kind,
                    "perception": self.perception.backend,
                    "states": self._trajectory,
                }
            )
        )
