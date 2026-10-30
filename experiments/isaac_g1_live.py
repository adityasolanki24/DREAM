"""Live Isaac G1 phone-API session with every task decision made by Qwen3-VL.

Isaac/PhysX owns the robot and scene. The pretrained WBC remains the bounded motor
controller; Qwen selects one validated mobile tool per observation. Ground-truth kitchen
geometry is used only for a collision-clear initial pose. Agent observations are RGB-D,
proprioception, estimated odometry and camera-derived memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


class PairedPhoneAPI:
    """Allow one private-IAP bootstrap to bind the laptop's existing random API token."""

    def __new__(cls, *args, **kwargs):
        from dream.runtime.phone_api import PhoneAPI

        class API(PhoneAPI):
            bootstrapped = False

            def handle(self, method, url, auth, body):
                if method == "POST" and urlparse(url).path == "/_bootstrap":
                    if self.bootstrapped:
                        return 403, {"error": "already paired"}
                    try:
                        token = json.loads(body).get("token")
                    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                        return 400, {"error": "invalid pairing request"}
                    if not isinstance(token, str) or len(token) < 24:
                        return 400, {"error": "invalid pairing token"}
                    self.token = token
                    self.bootstrapped = True
                    return 200, {"ready": True}
                return super().handle(method, url, auth, body)

        return API(*args, **kwargs)


class DecisionRecorder:
    """Keep each genuine Qwen input image, raw tool-call output and timing."""

    def __init__(self, backend, out: pathlib.Path, sim_time, capture=None):
        self.backend, self.out, self.sim_time = backend, out, sim_time
        self.folder = out / "qwen_decisions"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.capture = capture

    @property
    def name(self):
        return self.backend.name

    def set_cancel_event(self, event):
        self.backend.set_cancel_event(event)

    def generate(self, messages, tools):
        import numpy as np
        from PIL import Image

        index = self.index
        self.index += 1
        if self.capture is not None:
            self.capture(index)
        image_sha = None
        text_parts = []
        for message in messages:
            for part in message.get("content", []):
                if part.get("type") == "text":
                    text_parts.append({"role": message.get("role"), "text": part["text"]})
                elif part.get("type") == "image":
                    rgb = np.asarray(part["image"], dtype=np.uint8)
                    image_path = self.folder / f"{index:04d}_observation.png"
                    Image.fromarray(rgb).save(image_path)
                    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        t0 = time.perf_counter()
        record = {
            "index": index,
            "sim_time_s": float(self.sim_time()),
            "model": self.backend.model,
            "revision": QWEN_REVISION,
            "image_sha256": image_sha,
            "messages": text_parts,
            "tools": tools,
            "status": "started",
        }
        path = self.folder / f"{index:04d}_decision.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        try:
            output = self.backend.generate(messages, tools)
            record.update(output=output, usage=self.backend.last_usage, status="completed")
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            record["inference_s"] = round(time.perf_counter() - t0, 3)
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return output


def start_qwen(args, out: pathlib.Path):
    # A timed-out Isaac job can leave its separate, pinned model server alive.
    # Reuse that verified server rather than loading a second copy on the same GPU.
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.qwen_port}/healthz", timeout=2
        ) as response:
            health = json.loads(response.read())
        if health.get("ready") and health.get("revision") == QWEN_REVISION:
            (out / "qwen_reused.json").write_text(json.dumps(health), encoding="utf-8")
            return None, None
    except (OSError, ValueError):
        pass
    log = (out / "qwen_server.log").open("w", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    command = [
        os.environ.get("DREAM_QWEN_PYTHON", "/opt/dream-runtime/bin/python"),
        "-m",
        "experiments.qwen_tool_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.qwen_port),
        "--revision",
        QWEN_REVISION,
    ]
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + args.model_start_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.flush()
            raise RuntimeError(f"Qwen server exited during load; inspect {out / 'qwen_server.log'}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.qwen_port}/healthz", timeout=2
            ) as r:
                health = json.loads(r.read())
            if health.get("ready") and health.get("revision") == QWEN_REVISION:
                return proc, log
        except Exception:
            time.sleep(2)
    proc.terminate()
    proc.wait(timeout=30)
    log.close()
    raise TimeoutError("Qwen3-VL did not finish its pinned CUDA preflight before the deadline")


def run_tasks(app, scheduler, sleep=time.sleep, keep_running=None):
    """Serve tasks while Isaac is open; only a tool may advance controlled physics."""
    while app.is_running() and (keep_running is None or keep_running()):
        if scheduler.step() is None:
            # Kit updates tick a playing PhysX timeline without a WBC policy tick.
            # Keep this synchronous simulation frozen during user/model waiting.
            sleep(0.05)


def serve(args, cfg):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.storage.native import get_assets_root_path

    from dream.perception.spatial import SpatialMemory
    from dream.robot.g1_wbc import WBCLocomotion
    from dream.runtime.agent import TaskAgent
    from dream.runtime.mobile_model import MobileModelPlanner, interpret_mobile_goal
    from dream.runtime.scheduler import TaskScheduler
    from dream.sim.g1_mobile import DEFAULT_ODOMETRY_NOISE, HEAD_CAMERA, G1MobileBase
    from dream.sim.isaac_g1 import IsaacG1Env
    from dream.tools.goals import NaturalLanguageGoal
    from dream.tools.mobile import MobileToolbox
    from experiments.isaac_g1_kitchen import (
        FLOOR,
        Evidence,
        choose_start,
        free_space,
        kitchen_url,
    )
    from experiments.phase1_interface import (
        build_providers,
        make_snapshot,
        make_validator,
        mobile_goal,
    )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(out)
    qwen_process = qwen_log = api_server = scheduler = camera = media = None
    stopped = False

    def request_stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    try:
        assets = get_assets_root_path()
        url = kitchen_url(assets, args.kitchen)
        evidence.stage("asset_server", kitchen=url)
        if not open_stage(url):
            raise RuntimeError(f"could not open the existing NVIDIA kitchen {url}")
        app.update()
        world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
        cells, boxes, lo, hi = free_space(world.stage, cfg)
        x, y, yaw, clear_run = choose_start(cells, boxes, "explore", args.start)
        rng = np.random.default_rng([args.start, 1])
        env = IsaacG1Env(
            world,
            args.robot_usd,
            rng,
            DEFAULT_ODOMETRY_NOISE,
            HEAD_CAMERA,
            (FLOOR,),
            checkpoint=evidence.stage,
        )
        env.reset(x, y, yaw, floor_z=float(hi[2]))
        base = G1MobileBase(env, WBCLocomotion.from_cache())
        # Use the same validated ten-second WBC settling phase as kitchen bring-up.
        # The first live session fell before the first model-selected action.
        stand_s = float(cfg["bringup"]["stand_s"])
        base.walk(0.0, 0.0, 0.0, stand_s)
        if base.stop_reason is not None:
            raise RuntimeError(f"G1 failed standing gate: {base.stop_reason}")
        memory = SpatialMemory(np.random.default_rng([args.start, 2]))
        events_path = out / "tool_events.jsonl"

        def tool_event(kind, text, **data):
            row = {"sim_time_s": env.time, "kind": kind, "text": text, **data}
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, default=str) + "\n")
                stream.flush()
            if api_server is not None:
                api_server.notify(kind, row)

        toolbox = MobileToolbox(base, memory, on_event=tool_event, navigator="rpp")
        from dream.eval.isaac_scene import labelled_objects
        from dream.runtime.camera_stream import CameraStream
        from dream.runtime.media_runtime import MediaRuntime, install_binary
        from experiments.isaac_g1_kitchen import handle_truth, score

        truth_handles = handle_truth(labelled_objects(world.stage))  # evaluation only
        trajectory = out / "trajectory.jsonl"
        capture_stats = []
        if args.media:
            binary, media_lock = install_binary(
                ROOT / "experiments/configs/media_runtime.json",
                pathlib.Path("/work/cache/mediamtx"),
            )
            media = MediaRuntime(out / "media", binary)
            evidence.write("media_runtime_lock.json", media_lock)
        camera = CameraStream(on_rgb=None if media is None else media.submit)
        last_media = last_path = -float("inf")

        def sample():
            nonlocal last_media, last_path
            if env.time - last_path >= 0.1:
                with trajectory.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "sim_time_s": env.time,
                                "wall_unix_s": time.time(),
                                "odometry": base.odometry().tolist(),
                                "true_pose_evaluation_only": env.true_pose().tolist(),
                                "height_evaluation_only": env.true_height(),
                            }
                        )
                        + "\n"
                    )
                last_path = env.time
            if time.monotonic() - last_media >= 0.1:
                start = time.perf_counter()
                captured_unix = time.time()
                frame = env.capture_media()
                camera.submit(frame, captured_unix_s=captured_unix)
                capture_stats.append(
                    {"sim_time_s": env.time, "capture_ms": (time.perf_counter() - start) * 1000}
                )
                last_media = time.monotonic()

        base.on_tick = sample
        sample()
        toolbox.places = {"entrance": np.zeros(2)}
        evidence.write(
            "isaac_session.json",
            {
                "simulator": "NVIDIA Isaac Sim 6.2.0.0 / PhysX / RTX head camera",
                "scene": args.kitchen,
                "scene_source": url,
                "robot": "Unitree G1 under the unchanged NVIDIA Decoupled WBC",
                "model": "Qwen/Qwen3-VL-4B-Instruct",
                "model_revision": QWEN_REVISION,
                "model_decisions": "live tool calls from current Isaac camera observations",
                "motor_policy": "pretrained WBC velocity controller; typed-tool planner above it",
                "odometry": "oracle-assisted, as documented for this Milestone B adapter",
                "seed": args.start,
                "initial_pose": [x, y, yaw],
                "initial_clear_run_m": clear_run,
                "start_selection": "scene geometry only for a collision-clear reset pose",
            },
        )
        evidence.stage(
            "robot_ready",
            x=x,
            y=y,
            yaw=yaw,
            standing_s=stand_s,
            sim_time_s=env.time,
            stop_reason=base.stop_reason,
        )

        qwen_process, qwen_log = start_qwen(args, out)
        providers_args = argparse.Namespace(
            qwen_url=f"http://127.0.0.1:{args.qwen_port}", openai_token_file=None, openai_model=None
        )
        providers = build_providers(providers_args)
        qwen_choice = providers["qwen"]
        backend = qwen_choice.backend_for(qwen_choice.default_model)

        def capture_decision(index):
            frame = memory.last_frame
            if frame is None:
                return
            np.savez_compressed(
                out / f"sensor_{index:04d}.npz",
                rgb=frame.rgb,
                depth=frame.depth,
                self_mask=frame.self_mask,
                camera_position=frame.calibration.position,
                camera_rotation=frame.calibration.rotation,
                camera_fovy_deg=frame.calibration.fovy_deg,
                occupancy=memory.grid.logodds,
            )

        recorder = DecisionRecorder(backend, out, lambda: env.time, capture_decision)
        qwen_choice.backend_for = lambda _model: recorder
        evidence.stage(
            "qwen_ready", model=qwen_choice.default_model, revision=QWEN_REVISION, gpu="NVIDIA L4"
        )

        api_ref = {"api": None}
        task_finished_at = None

        def run_task(task, history, preempt, cancel):
            nonlocal task_finished_at
            toolbox.cancel, toolbox.preempt = cancel, preempt
            if task.meta.get("provider") != "qwen":
                raise RuntimeError("this live Isaac session accepts Qwen VLM tasks only")
            if base.stop_reason is not None:
                raise RuntimeError(f"robot requires a fresh session: {base.stop_reason}")
            recorder.set_cancel_event(cancel)
            if isinstance(task.goal, NaturalLanguageGoal):
                observed = toolbox.call("observe_scene")
                if not observed.ok:
                    raise RuntimeError("goal interpretation observation failed")
                task.goal = interpret_mobile_goal(
                    recorder,
                    task.text,
                    toolbox.places,
                    observed.data,
                    lambda: None if memory.last_frame is None else memory.last_frame.rgb.copy(),
                )
            planner = MobileModelPlanner(
                recorder,
                task.text,
                lambda: None if memory.last_frame is None else memory.last_frame.rgb.copy(),
            )

            def emit(kind, data):
                row = {"id": task.id, **data}
                api_ref["api"].notify(kind, row)
                with events_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps({"sim_time_s": env.time, "kind": kind, **row}, default=str)
                        + "\n"
                    )
                    stream.flush()

            from experiments.g1_explore_room import Budgeted

            episode = TaskAgent(
                toolbox, Budgeted(planner, base, 180), max_steps=cfg["explore"]["max_steps"]
            ).run(
                task.goal,
                on_event=emit,
                history=history,
                preempt=preempt,
            )
            verdict = toolbox.call("verify_goal", goal=task.goal).data
            candidate = verdict.get("candidate")
            result = score(
                env, toolbox, episode.success_claimed, truth_handles, candidate_id=candidate
            )
            evidence.write(
                f"task_{task.id}_evaluation.json",
                {
                    "episode": episode.to_dict(),
                    "agent_verdict": verdict,
                    "independent_verification": result,
                },
            )
            task_finished_at = time.monotonic()
            return episode

        store = out / "tasks.sqlite"
        scheduler = TaskScheduler(str(store), run_task, mobile_goal)
        token = secrets.token_urlsafe(32)
        api = PairedPhoneAPI(
            scheduler,
            token,
            status=lambda: {
                "robot_time_s": round(base.time, 2),
                "odometry": base.odometry().tolist(),
                "stop_reason": base.stop_reason,
            },
            snapshot=lambda: {
                **make_snapshot(api, scheduler, base, memory, providers=providers),
                "environment": "NVIDIA Isaac Sim 6.2 / Unitree G1 / PhysX",
            },
            provider="qwen",
            validate_task=make_validator(providers, "qwen"),
            camera=camera,
            media=media,
        )
        api.camera_managed = True
        api.provider_label = qwen_choice.label
        api_ref["api"] = api
        scheduler.notify = api.notify
        initial = toolbox.call("observe_scene")
        if not initial.ok:
            raise RuntimeError("initial camera observation failed")
        evidence.stage("initial_observation_ready", sim_time_s=env.time)
        api_server = api.serve(host="0.0.0.0", port=args.api_port)
        print(
            f"Isaac G1 live API ready on port {args.api_port}; one-time IAP pairing enabled",
            flush=True,
        )
        evidence.stage("phone_api_ready", port=args.api_port)
        deadline = time.monotonic() + args.duration_s
        run_tasks(
            app,
            scheduler,
            keep_running=lambda: (
                not stopped
                and time.monotonic() < deadline
                and (
                    not args.single_task
                    or task_finished_at is None
                    or time.monotonic() - task_finished_at < 30
                )
            ),
        )
    finally:
        if "env" in locals():
            evidence.write(
                "physical_outcome.json",
                {
                    "simulation_time_s": env.time,
                    "final_true_pose_for_evaluation_only": env.true_pose().tolist(),
                    "final_height_m_for_evaluation_only": env.true_height(),
                    "contact_steps": env.contact_steps,
                    "contacts": [c.__dict__ for c in env.contacts],
                    "physics_steps": env.physics_steps,
                    "control_tick_ms": env.control_timings_ms,
                },
            )
        if api_server is not None:
            evidence.write(
                "camera_delivery.json",
                {
                    "server": camera.metadata(),
                    "client": api.client_metrics,
                    "capture_samples": capture_stats,
                    "media": None if media is None else media.status(),
                },
            )
            api_server.shutdown()
        if camera is not None:
            camera.close()
        if media is not None:
            media.close()
        if scheduler is not None:
            scheduler.db.close()
        if qwen_process is not None:
            qwen_process.terminate()
            qwen_process.wait(timeout=30)
        if qwen_log is not None:
            qwen_log.close()
        app.close(wait_for_replicator=False, skip_cleanup=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=pathlib.Path, default=ROOT / "experiments/configs/isaac_g1_kitchen.json"
    )
    parser.add_argument("--kitchen", default="l_shape")
    parser.add_argument("--start", type=int, default=7)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--robot-dir", type=pathlib.Path, default=pathlib.Path("/work/g1_usd"))
    parser.add_argument("--api-port", type=int, default=8765)
    parser.add_argument("--qwen-port", type=int, default=8767)
    parser.add_argument("--model-start-timeout-s", type=int, default=1200)
    parser.add_argument("--duration-s", type=int, default=1200)
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--single-task", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.kitchen not in cfg["development_kitchens"]:
        parser.error("live voice demo is restricted to the existing development kitchen")
    receipt = json.loads((args.robot_dir / "receipt.json").read_text(encoding="utf-8"))
    usd = receipt.get("usda")
    args.robot_usd = str(args.robot_dir / "usd" / (usd or receipt["usd"]))
    if not pathlib.Path(args.robot_usd).is_file():
        parser.error("prepared G1 USD from the existing Isaac session is missing")
    serve(args, cfg)


if __name__ == "__main__":
    main()
