"""One CUDA checkpoint, paired G1 episodes, existing tools and independent scoring.

This is synchronous closed-loop simulation, not continuous physics during inference.
All four previously observed cases are retained as development comparisons, including
the invalid initial-overlap challenge. Failures are results, never classical fallbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

from experiments.g1_dexterous import write_json
from experiments.g1_explore_unknown import load_backend, run_case, source_metadata

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/g1_qwen_comparison.json"


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_evidence(out: Path) -> None:
    """Require real inference/physics records; task and safety failures stay reportable."""
    cfg = json.loads(CONFIG.read_text())
    summary = json.loads((out / "comparison.json").read_text())
    expected = {(c["name"], b) for c in cfg["cases"] for b in ("qwen", "deterministic")}
    rows = summary.get("episodes", [])
    observed = {(r.get("case"), r.get("backend")) for r in rows}
    meta = json.loads((out / "model.json").read_text())
    if observed != expected or len(rows) != len(expected) or meta.get("device") != "cuda:0":
        raise ValueError("G1 paired episodes or CUDA provenance incomplete")
    if meta.get("revision") != cfg["model"]["revision"]:
        raise ValueError("G1 checkpoint revision differs")
    decisions = 0
    for row in rows:
        folder = out / row["backend"] / row["case"]
        if not (folder / "summary.json").is_file() or not (folder / "physics.jsonl").is_file():
            raise ValueError("G1 physical record missing")
        if not (folder / "frames.json").is_file() or not (folder / "tools.json").is_file():
            raise ValueError("G1 observation or tool record missing")
        if row["backend"] == "qwen":
            outputs = sorted((folder / "inference").glob("*_output.json"))
            texts = [json.loads(p.read_text()).get("text", "") for p in outputs]
            if not texts or not any(texts):
                raise ValueError("G1 genuine inference output missing")
            calls = json.loads((folder / "tools.json").read_text())
            inputs = sorted((folder / "inference").glob("*_input.json"))
            times = [json.loads(p.read_text())["sim_time_before"] for p in inputs]
            physical_actions = {
                "reach_object",
                "probe_object",
                "grasp_object",
                "move_object",
                "release_object",
            }
            moved = any(c["tool"] in physical_actions for c in calls)
            belief = json.loads((folder / "belief.json").read_text())
            updated = any(e.get("update_accepted") and e.get("contact_valid") for e in belief)
            feedback = any(
                "probe_object result:" in p.get("text", "") and '"posterior"' in p["text"]
                for path in inputs
                if json.loads(
                    path.with_name(path.name.replace("_input", "_output")).read_text()
                ).get("text")
                for message in json.loads(path.read_text()).get("messages", [])
                for p in message["content"]
                if p["type"] == "text"
            )
            if (
                len(texts) >= 2
                and len(times) >= 2
                and max(times) > min(times)
                and row.get("physics_executed")
                and moved
                and updated
                and feedback
                and (folder / "physics.jsonl").stat().st_size > 0
            ):
                decisions += 1
    if decisions == 0:
        raise ValueError("G1 probe/belief/next-decision cycle not demonstrated")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text())
    metadata = source_metadata({**cfg, "backend": "paired"})
    write_json(
        args.out / "freeze.json",
        {
            **metadata,
            "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "acceptance": "all paired records; CUDA model; actual outputs, physical probe, "
            "accepted tactile/displacement belief update and feedback in a later model input. "
            "Success, false claims, safety and recovery "
            "reported separately; no tuning after this freeze.",
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU or classical fallback")
    cache = Path(os.environ.get("HF_HOME", "/work/model-cache")).resolve()
    if cache == ROOT or ROOT in cache.parents or args.out.resolve() in (cache, *cache.parents):
        raise ValueError("checkpoint cache must be outside source and results")
    model = cfg["model"]
    path = Path(
        snapshot_download(
            model["checkpoint"],
            revision=model["revision"],
            cache_dir=cache,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
    )
    backend = load_backend(model, path)
    device = str(backend.model.device)
    if device != "cuda:0" or any(p.device.type != "cuda" for p in backend.model.parameters()):
        raise RuntimeError("checkpoint did not fully load onto CUDA")
    write_json(
        args.out / "model.json",
        {
            **model,
            "device": device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "checkpoint_files_sha256": {
                p.name: file_hash(p) for p in path.iterdir() if p.is_file()
            },
        },
    )
    episodes = []
    for condition in ("qwen", "deterministic"):
        (args.out / condition).mkdir()
    for case in cfg["cases"]:
        for condition in ("qwen", "deterministic"):
            record = run_case(
                case,
                {**cfg, "backend": condition, **model},
                args.out / condition / case["name"],
                backend=backend,
            )
            episodes.append({**record, "case": case["name"], "seed": case["seed"]})
            write_json(
                args.out / "comparison.json",
                {
                    "scope": cfg["scope"],
                    "episodes": episodes,
                    "classical_fallback": False,
                    "task_success": {
                        b: sum(r["goal_success"] for r in episodes if r["backend"] == b)
                        for b in ("qwen", "deterministic")
                    },
                },
            )
    write_json(
        args.out / "gpu_memory.json",
        {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    )
    validate_evidence(args.out)


if __name__ == "__main__":
    main()
