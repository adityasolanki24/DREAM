"""Actual GPU inference, then the existing embodied runtime. No stand-in model mode.

The first request uses a recorded RGB-D sensor frame. Only a valid model-generated named
tool call permits the physical phase. Both planners then run in LiveSession's continuous
physics thread, which holds safely during model latency. Results include model inputs,
raw outputs, camera frames, timestamps, GPU usage and independent physical scoring.
Weights and caches live on the ephemeral VM, outside the repository and result directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dream.eval.manipulation import object_in_tray, safety_summary  # noqa: E402
from dream.runtime.agent import TaskAgent  # noqa: E402
from dream.runtime.qwen import QwenTransformersBackend  # noqa: E402
from dream.runtime.vlm import FINISH_SPEC, SYSTEM_PROMPT, VLMAgent, parse_tool_call, tool_schemas  # noqa: E402
from dream.session import LiveSession  # noqa: E402
from dream.tools.goals import parse_goal  # noqa: E402
from dream.tools.toolbox import SPECS, _validate  # noqa: E402


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.tolist()), encoding="utf-8")


class RecordedBackend:
    """Audit wrapper around a real backend; records the exact messages passed to it."""

    def __init__(self, backend, directory: Path, clock):
        self.backend, self.directory, self.clock = backend, directory, clock
        self.name = backend.name
        self.n = 0
        directory.mkdir(parents=True, exist_ok=True)

    def set_cancel_event(self, cancel):
        self.backend.set_cancel_event(cancel)

    def generate(self, messages, tools):
        from PIL import Image

        index = self.n
        self.n += 1
        saved = []
        for i, message in enumerate(messages):
            parts = []
            for j, part in enumerate(message["content"]):
                if part["type"] == "image":
                    filename = f"{index:03d}_{i}_{j}.png"
                    Image.fromarray(part["image"]).save(self.directory / filename)
                    parts.append(
                        {
                            "type": "image",
                            "file": filename,
                            "rgb_sha256": hashlib.sha256(part["image"].tobytes()).hexdigest(),
                        }
                    )
                else:
                    parts.append(part)
            saved.append({**message, "content": parts})
        record = {
            "sim_time_before": self.clock(),
            "messages": saved,
            "tools": tools,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(self.directory / f"{index:03d}_input.json", record)
        start = time.perf_counter()
        try:
            text = self.backend.generate(messages, tools)
        except Exception as error:
            write_json(
                self.directory / f"{index:03d}_output.json",
                {
                    "error": type(error).__name__,
                    "detail": str(error),
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "latency_s": time.perf_counter() - start,
                },
            )
            raise
        write_json(
            self.directory / f"{index:03d}_output.json",
            {
                "text": text,
                "utc": datetime.now(timezone.utc).isoformat(),
                "latency_s": time.perf_counter() - start,
                "sim_time_after": self.clock(),
            },
        )
        return text


def offline_contract(backend, cfg: dict, out: Path) -> dict:
    session = LiveSession(seed=cfg["offline_seed"], perception="vision")
    try:
        frame = session.env.capture_frame()
        np.savez_compressed(
            out / "rgbd.npz", rgb=frame.rgb, depth=frame.depth, self_mask=frame.self_mask
        )
        write_json(out / "calibration.json", dataclasses.asdict(frame.calibration))
        scene = session.toolbox.call("observe_scene").data
        state = session.toolbox.call("get_robot_state").data
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame.rgb},
                    {
                        "type": "text",
                        "text": "Pick up the red object and put it in the tray.\n"
                        + json.dumps({"frame_time": frame.time, "scene": scene, "robot": state}),
                    },
                ],
            },
        ]
        recorded = RecordedBackend(backend, out / "inference", lambda: session.env.time)
        text = recorded.generate(messages, tool_schemas())
        call, error = parse_tool_call(text)
        allowed = {s["function"]["name"] for s in tool_schemas()}
        valid = call is not None and call["name"] in allowed
        if valid:
            spec = FINISH_SPEC if call["name"] == "finish" else SPECS[call["name"]]
            error = _validate(spec["parameters"], call["arguments"])
            valid = not bool(error)
        return {"valid_named_call": valid, "call": call, "error": error}
    finally:
        session.close()


def embodied_episode(backend, cfg: dict, item: dict, condition: str, out: Path) -> dict:
    session = LiveSession(seed=item["seed"], perception="vision", rate=1.0)
    assert session.perception.backend == "vision"
    goal = parse_goal(item["goal"])
    session.start()
    try:
        if condition == "A_rule_based":
            ep = TaskAgent(session.toolbox).run(goal)
        else:
            recorded = RecordedBackend(backend, out / "inference", lambda: session.env.time)
            agent = VLMAgent(
                session.toolbox,
                recorded,
                camera=lambda: session.env.capture_frame().rgb,
                max_steps=cfg["max_steps"],
            )
            ep = agent.run(item["instruction"], expected_goal=goal)
        session.env.wait_until(lambda: False, 0.5)
        with session.env.lock:
            success = object_in_tray(session.env, goal.object)
            safety = safety_summary(session.env)
        result = {
            "condition": condition,
            "seed": item["seed"],
            "instruction": item["instruction"],
            "goal": item["goal"],
            "perception": "vision",
            "success": success,
            "claimed": ep.success_claimed,
            "false_claim": ep.success_claimed and not success,
            "episode": ep.to_dict(),
            **safety,
        }
        write_json(out / "episode.json", result)
        return result
    finally:
        session.close()


def main() -> None:
    import torch
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / "experiments/configs/vlm_cloud_pilot.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA inference is required; there is no CPU/test-double fallback")
    cache = Path(os.environ.get("HF_HOME", "/work/model-cache")).resolve()
    if (
        cache == ROOT
        or ROOT in cache.parents
        or cache == args.out.resolve()
        or args.out.resolve() in cache.parents
    ):
        raise ValueError("model cache must be outside the repository and result tree")
    path = snapshot_download(
        cfg["checkpoint"],
        revision=cfg["revision"],
        cache_dir=cache,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
    meta = {
        "git_sha": os.environ["DREAM_CODE_SHA"],
        "config": cfg,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "checkpoint_revision": cfg["revision"],
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    write_json(args.out / "meta.json", meta)
    backend = QwenTransformersBackend(path, device="cuda:0", timeout_s=cfg["timeout_s"])
    offline = args.out / "offline"
    offline.mkdir()
    contract = offline_contract(backend, cfg, offline)
    write_json(offline / "contract.json", contract)
    if not contract["valid_named_call"]:
        write_json(
            args.out / "summary.json", {"offline_contract": contract, "physical_phase": "not run"}
        )
        raise SystemExit("offline contract failed; physical phase not run")
    records = []
    for item in cfg["episodes"]:
        for condition in ("A_rule_based", "B_vlm"):
            out = args.out / f"{item['seed']}_{condition}"
            out.mkdir()
            result = embodied_episode(backend, cfg, item, condition, out)
            records.append(result)
            write_json(out / "episode.json", result)
    summary = {
        "offline_contract": contract,
        "episodes": records,
        "false_claim_gate_pass": not any(r["false_claim"] for r in records),
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "interpretation": "development pilot, no generalisation or learning claim",
    }
    write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
