"""Local web bridge for the existing mobile scheduler; no direct motor commands.

Run: python -m experiments.phase1_interface --seed 4100
The structured planner is not an LLM. Odometry is oracle-assisted. The map and
object candidates are session observations, not persistent home memory.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np
from PIL import Image

from dream.perception.occupancy import CELL
from dream.perception.spatial import SpatialMemory
from dream.robot.g1_wbc import WBCLocomotion
from dream.runtime.agent import TaskAgent
from dream.runtime.phone_api import PhoneAPI, load_or_create_token, token_file
from dream.runtime.scheduler import TaskScheduler
from dream.runtime.search import PlacePlanner, RoomSearchPlanner
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1
from dream.sim.room import add_room, sample_room
from dream.tools.goals import GoToPlace, Inspect, parse_goal
from dream.tools.mobile import MobileToolbox
from experiments.g1_explore_room import Budgeted

ROOT = pathlib.Path(__file__).resolve().parents[1]


def mobile_goal(text):
    goal = parse_goal(text)
    if not isinstance(goal, (GoToPlace, Inspect)):
        raise ValueError("mobile session supports 'inspect cabinet' and 'goto <place>' only")
    return goal


STRUCTURED_LABEL = "Structured goal parser (no language model)"
STRUCTURED_SUPPORTS = ["inspect cabinet", "goto <place>"]
MODEL_SUPPORTS = ["inspect cabinet"]


@dataclass
class Provider:
    """One provider that can be chosen per task. ``backend_for(model)`` is None for the
    structured parser; model providers return one backend per model."""

    id: str
    label: str
    available: bool
    reason: str | None = None
    models: list[str] = field(default_factory=list)
    default_model: str | None = None
    supports: list[str] = field(default_factory=list)
    backend_for: Callable | None = None

    def describe(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "available": self.available,
            "reason": self.reason,
            "models": list(self.models),
            "default_model": self.default_model,
            "supports": list(self.supports),
        }


def chatgpt_models(token_file, openai_model):
    """(models, reason): one GET /v1/models catalogue call at start-up. No inference."""
    from dream.runtime.openai_plan import list_models

    try:
        record = json.loads(pathlib.Path(token_file).read_text(encoding="utf-8"))
        if "chatgpt.tokens.use.direct" not in record.get("scopes", []):
            return [], "credential does not grant chatgpt.tokens.use.direct; sign in again"
        if record.get("expires_at", 0) <= time.time():
            return [], "SIWC access token expired; sign in again"
        status, _, listed = list_models(record.get("access_token", ""))
    except ValueError as error:  # our own messages; never the token itself
        return [], str(error)
    except Exception as error:  # network or file errors: report the type only
        return [], f"model list unavailable: {type(error).__name__}"
    if not 200 <= status < 300:
        return [], f"model list request failed: HTTP {status}"
    models = []
    for item in listed:
        name = item.get("slug") or item.get("id")
        if isinstance(name, str) and name not in models:
            models.append(name)
    return models, None


def chatgpt_backend_factory(token_file):
    """One ChatGPTPlanBackend per model, created on first use."""
    from dream.runtime.openai_plan import ChatGPTPlanBackend

    cache = {}

    def backend_for(model):
        if model not in cache:
            cache[model] = ChatGPTPlanBackend.from_token_file(token_file, model)
        return cache[model]

    return backend_for


def build_providers(args):
    """Every provider in display order; unconfigured ones are listed with the reason."""
    providers = {
        "structured": Provider("structured", STRUCTURED_LABEL, True, supports=STRUCTURED_SUPPORTS)
    }
    qwen_label = "Qwen3-VL (configured server)"
    if args.qwen_url:
        from dream.runtime.qwen import OpenAICompatibleBackend

        qwen = OpenAICompatibleBackend(
            args.qwen_url, api_key=os.environ.get("DREAM_QWEN_KEY"), timeout_s=60
        )
        providers["qwen"] = Provider(
            "qwen",
            qwen_label,
            True,
            models=[qwen.model],
            default_model=qwen.model,
            supports=MODEL_SUPPORTS,
            backend_for=lambda model: qwen,
        )
    else:
        providers["qwen"] = Provider(
            "qwen",
            qwen_label,
            False,
            reason="start the backend with --qwen-url",
            supports=MODEL_SUPPORTS,
        )
    chatgpt_label = "ChatGPT account (configured model)"
    if args.openai_token_file:
        models, reason = chatgpt_models(args.openai_token_file, args.openai_model)
        default = args.openai_model or (models[0] if models else None)
        providers["openai_chatgpt_plan"] = Provider(
            "openai_chatgpt_plan",
            chatgpt_label,
            reason is None,
            reason=reason,
            models=models,
            default_model=default,
            supports=MODEL_SUPPORTS,
            backend_for=chatgpt_backend_factory(args.openai_token_file),
        )
    else:
        providers["openai_chatgpt_plan"] = Provider(
            "openai_chatgpt_plan",
            chatgpt_label,
            False,
            reason=(
                "sign in with experiments/openai_plan_login.py and start with --openai-token-file"
            ),
            supports=MODEL_SUPPORTS,
        )
    return providers


def make_validator(providers, default):
    """Checks one submission against the registry; returns the task meta or raises ValueError."""

    def validate(text, provider, model):
        pid = default if provider is None else provider
        choice = providers.get(pid)
        if choice is None:
            raise ValueError(f"unknown provider: {pid}")
        if not choice.available:
            raise ValueError(f"provider is not available: {choice.reason}")
        if choice.backend_for is None:
            if model is not None:
                raise ValueError("model must be omitted for structured")
            mobile_goal(text)
            return {"provider": pid, "model": None}
        chosen = choice.default_model if model is None else model
        if not chosen:
            raise ValueError(f"model is required for {pid}")
        if choice.models and chosen not in choice.models:
            raise ValueError(f"model is not available for {pid}: {chosen}")
        if not isinstance(mobile_goal(text), Inspect):
            raise ValueError("model mobile planner currently supports 'inspect cabinet' only")
        return {"provider": pid, "model": chosen}

    return validate


def read_home_memory(path):
    """Read last stored sightings without creating or altering the home-memory database."""
    if path is None:
        return None
    try:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            rows = db.execute(
                "SELECT o.id,o.label,o.map_id,v.time,v.x,v.y,v.z,v.confidence,v.source "
                "FROM objects o JOIN observations v ON v.id=(SELECT id FROM observations "
                "WHERE object_id=o.id ORDER BY time DESC LIMIT 1) ORDER BY v.time DESC LIMIT 100"
            ).fetchall()
        finally:
            db.close()
        return {
            "available": True,
            "source": "home memory: last recorded sightings; current state not assessed",
            "objects": [
                {
                    "id": r[0],
                    "label": r[1],
                    "map_id": r[2],
                    "last_seen": r[3],
                    "position": list(r[4:7]),
                    "confidence_at_observation": r[7],
                    "observation_source": r[8],
                }
                for r in rows
            ],
        }
    except sqlite3.Error:
        return {"available": False, "error": "home memory could not be read"}


def make_snapshot(api, scheduler, base, memory, home_path=None, providers=None):
    frame = memory.last_frame
    camera = None
    if frame is not None:
        buffer = io.BytesIO()
        Image.fromarray(frame.rgb.copy()).save(buffer, format="PNG")
        camera = {
            "mime": "image/png",
            "data_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
            "time_s": frame.time,
        }
    grid = memory.grid
    cells = np.zeros_like(grid.logodds, dtype=np.uint8)
    cells[grid.free()] = 1
    cells[grid.occupied()] = 2
    known = np.argwhere(cells != 0)
    map_data = None
    if len(known):
        lo, hi = known.min(axis=0), known.max(axis=0) + 1
        map_data = {
            "resolution_m": CELL,
            "origin_m": (lo * CELL - grid.size / 2).tolist(),
            "cells": cells[lo[0] : hi[0], lo[1] : hi[1]].tolist(),
            "axes": "cells[x][y]",
            "odometry": "oracle-assisted",
        }
    tasks = scheduler.tasks()
    latest = tasks[-1] if tasks else None
    events = api.handle("GET", "/events?after=0", f"Bearer {api.token}", b"")[1]
    task_id = scheduler.running or (latest["id"] if latest else None)
    actions = [e for e in events if e.get("id") == task_id and e["kind"] == "action"]
    nodes = [{"id": str(e["n"]), "label": e["tool"], "args": e["args"]} for e in actions]
    graph = (
        {
            "kind": "observed execution sequence",
            "task_id": task_id,
            "nodes": nodes,
            "edges": [
                {"source": a["id"], "target": b["id"]}
                for a, b in zip(nodes, nodes[1:], strict=False)
            ],
        }
        if nodes
        else None
    )
    return {
        "environment": "MuJoCo simulation (Unitree G1)",
        "provider": api.provider,
        "provider_label": api.provider_label,
        "provider_choices": [p.describe() for p in (providers or {}).values()],
        "robot_time_s": round(base.time, 2),
        "odometry": base.odometry().tolist(),
        "stop_reason": base.stop_reason,
        "camera": camera,
        "map": map_data,
        "objects": [c.summary() for c in list(memory.candidates)],
        "persistent_memory": read_home_memory(home_path),
        "task_graph": graph,
        "tool_executions": [e for e in events if e["kind"] in ("action", "result")],
        "outcome": (
            {"state": latest["state"], "reason": latest["result"]}
            if latest and latest["state"] in ("succeeded", "failed", "cancelled")
            else None
        ),
        "history": tasks,
    }


def make_run_task(toolbox, base, memory, providers, emit):
    """Planner per task from its meta: structured planners unless a model provider is named
    and the goal is an Inspect. ``emit(kind, data)`` receives the task's own events."""

    def run_task(task, history, preempt, cancel):
        toolbox.preempt, toolbox.cancel = preempt, cancel
        meta = task.meta or {}
        provider_id, model = meta.get("provider", "structured"), meta.get("model")
        if provider_id != "structured" and provider_id not in providers:
            raise RuntimeError(f"provider not configured: {provider_id}")
        choice = providers.get(provider_id)
        if choice is not None and choice.backend_for is not None and isinstance(task.goal, Inspect):
            from dream.runtime.mobile_model import MobileModelPlanner

            backend = choice.backend_for(model)
            backend.set_cancel_event(cancel)
            planner = MobileModelPlanner(
                backend,
                task.text,
                lambda: None if memory.last_frame is None else memory.last_frame.rgb.copy(),
            )
        else:
            planner = (
                PlacePlanner()
                if isinstance(task.goal, GoToPlace)
                else Budgeted(RoomSearchPlanner("gain"), base, base.time + 240)
            )

        def event(kind, data):
            emit(kind, {"id": task.id, **data})

        return TaskAgent(toolbox, planner, max_steps=60).run(
            task.goal, on_event=event, history=history, preempt=preempt
        )

    return run_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store", type=pathlib.Path)
    parser.add_argument(
        "--memory", type=pathlib.Path, help="existing home memory SQLite (read-only)"
    )
    parser.add_argument(
        "--provider",
        choices=("structured", "qwen", "openai_chatgpt_plan"),
        default="structured",
        help="default provider for tasks that do not name one",
    )
    parser.add_argument("--qwen-url", help="configured Qwen OpenAI-compatible server URL")
    parser.add_argument("--openai-token-file", type=pathlib.Path)
    parser.add_argument("--openai-model", help="default model for the ChatGPT plan provider")
    args = parser.parse_args()
    if args.memory is not None and not args.memory.is_file():
        parser.error("--memory must name an existing home memory database")
    if args.provider == "qwen" and not args.qwen_url:
        parser.error("--provider qwen requires --qwen-url")
    if args.provider == "openai_chatgpt_plan" and not args.openai_token_file:
        parser.error("--provider openai_chatgpt_plan requires --openai-token-file")
    providers = build_providers(args)
    for choice in providers.values():
        state = "available" if choice.available else f"unavailable ({choice.reason})"
        print(f"provider {choice.id}: {state}", flush=True)
    room = sample_room(np.random.default_rng([args.seed, 0]))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([args.seed, 1])
    )
    env.reset(*room.start)
    base = G1MobileBase(env, WBCLocomotion.from_cache())
    memory = SpatialMemory(np.random.default_rng([args.seed, 2]))
    tb = MobileToolbox(base, memory, navigator="rpp")
    tb.places = {"entrance": np.zeros(2)}
    store = (
        args.store
        or pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home())) / "DREAM/phase1"
    )
    store.mkdir(parents=True, exist_ok=True)
    api = None

    run_task = make_run_task(tb, base, memory, providers, lambda kind, data: api.notify(kind, data))
    sched = TaskScheduler(str(store / "tasks.sqlite"), run_task, mobile_goal)
    api = PhoneAPI(
        sched,
        load_or_create_token(),
        status=lambda: {
            "robot_time_s": round(base.time, 2),
            "odometry": base.odometry().tolist(),
            "stop_reason": base.stop_reason,
        },
        snapshot=lambda: make_snapshot(api, sched, base, memory, args.memory, providers),
        web_root=ROOT / "web" / "out",
        provider=args.provider,
        validate_task=make_validator(providers, args.provider),
    )
    api.provider_label = providers[args.provider].label
    sched.notify = api.notify
    server = api.serve(port=args.port)
    print(
        f"Serving http://127.0.0.1:{server.server_address[1]}/ ; token file: {token_file()}",
        flush=True,
    )
    try:
        while True:
            if sched.step() is None:
                time.sleep(0.1)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
