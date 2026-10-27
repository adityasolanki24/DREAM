"""Run the pre-registered VLM decision set (results/phase1_vlm_compare/decisions.json).

Every backend gets identical observations, prompts and DREAM's real mobile tool schemas
(dream.runtime.mobile_model.SCHEMAS), parsed and validated by MobileModelPlanner. One call per
decision, no retries (ChatGPT plan limits are shared with the owner's own use).

    python experiments/vlm_compare.py gpt --model gpt-6-luna --reasoning-effort none `
        --token-file <credential-file> --out <new-output-directory>
    python experiments/vlm_compare.py qwen-http --base-url http://HOST:8000
    python experiments/vlm_compare.py qwen-local            # needs the vlm extra and a GPU
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.runtime.agent import Finish, ToolCall  # noqa: E402
from dream.runtime.mobile_model import MobileModelPlanner  # noqa: E402

OUT = ROOT / "results" / "phase1_vlm_compare"
GOAL = "Inspect(kind='cabinet')"


class Recorder:
    """Wraps a backend: records raw text, latency, usage and errors for each generate()."""

    def __init__(self, backend):
        self.backend, self.last = backend, {}

    def generate(self, messages, tools):
        t0 = time.monotonic()

        def canonical(value):
            if isinstance(value, np.ndarray):
                return {
                    "image_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            if isinstance(value, dict):
                return {k: canonical(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [canonical(v) for v in value]
            return value

        def digest(value):
            encoded = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(encoded.encode()).hexdigest()

        rec = {
            "error": None,
            "raw": None,
            "messages_sha256": digest(messages),
            "tools_sha256": digest(tools),
        }
        try:
            rec["raw"] = self.backend.generate(messages, tools)
        except Exception as exc:  # recorded, never retried
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        rec["latency_s"] = round(time.monotonic() - t0, 3)
        for k in (
            "last_http_status",
            "last_request_id",
            "last_response_id",
            "last_usage",
            "last_rate_limits",
            "last_error_body",
            "last_stream_items",
        ):
            rec[k] = getattr(self.backend, k, None)
        self.last = rec
        if rec["error"]:
            raise RuntimeError(rec["error"])
        return rec["raw"]


def make_backend(args):
    if args.backend == "gpt":
        from dream.runtime.openai_plan import ChatGPTPlanBackend

        if not args.token_file:
            raise ValueError("GPT requires an explicit --token-file")
        return ChatGPTPlanBackend.from_token_file(
            args.token_file, args.model, reasoning_effort=args.reasoning_effort
        )
    if args.backend == "qwen-http":
        from dream.runtime.qwen import OpenAICompatibleBackend

        return OpenAICompatibleBackend(args.base_url, timeout_s=120.0)
    from dream.runtime.qwen import QwenTransformersBackend

    return QwenTransformersBackend(
        device=args.device,
        timeout_s=180.0,
        model_path=args.model_path,
        revision=args.model_revision,
    )


def classify(action, scene):
    """Map the planner's validated action back to the model-level tool and args."""
    if isinstance(action, Finish):
        if action.reason.startswith(("invalid", "model success rejected")):
            return None, {}, action.reason
        return "finish", {"success": action.success}, ""
    if isinstance(action, ToolCall):
        if action.name == "go_to":
            for i, f in enumerate(scene.get("frontiers", [])):
                if list(f["xy"]) == [action.args["x"], action.args["y"]]:
                    return "explore_frontier", {"index": i}, ""
        return action.name, dict(action.args), ""
    return None, {}, "unknown action"


def correct(tool, args, accept):
    return any(
        a["tool"] == tool and all(args.get(k) == v for k, v in a["args"].items()) for a in accept
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["gpt", "qwen-http", "qwen-local"])
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--base-url")
    ap.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--model-revision")
    ap.add_argument("--tag")
    ap.add_argument("--token-file", type=pathlib.Path)
    ap.add_argument("--ids", nargs="+", help="Preselected decision IDs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    tag = args.tag or (args.backend if args.backend != "gpt" else "gpt-" + args.model)
    result_path = args.out / f"run_{tag}.json"
    if result_path.exists():
        ap.error(f"result already exists: {result_path}")
    cfg = json.loads((OUT / "decisions.json").read_text())
    if args.ids:
        selected = [d for d in cfg["decisions"] if d["id"] in args.ids]
        if len(selected) != len(set(args.ids)):
            ap.error("decision IDs must be unique and present in the frozen set")
        cfg["decisions"] = selected
    backend = make_backend(args)
    rec = Recorder(backend)
    rows = []
    for dec in cfg["decisions"]:
        image = np.asarray(Image.open(OUT / dec["image"]).convert("RGB"))
        planner = MobileModelPlanner(rec, cfg["instruction"], lambda image=image: image)
        history = [
            type(
                "S",
                (),
                {
                    "call": ToolCall(h["tool"], h["args"]),
                    "result": type("R", (), {"status": h["status"], "message": h["message"]})(),
                },
            )()
            for h in dec["history"]
        ]
        rec.last = {}
        try:
            action = planner.next_action(GOAL, dec["scene"], dec["verdict"], history)
            tool, targs, why = classify(action, dec["scene"])
        except RuntimeError:  # backend error already recorded in rec.last
            tool, targs, why = None, {}, "backend error"
        valid = tool is not None
        row = {
            "id": dec["id"],
            "category": dec["category"],
            "valid_tool_call": valid,
            "tool": tool,
            "args": targs,
            "correct": valid and correct(tool, targs, dec["accept"]),
            "reject_reason": why,
            **rec.last,
        }
        rows.append(row)
        print(
            dec["id"],
            "valid" if valid else "INVALID",
            "correct" if row["correct"] else "wrong",
            tool,
            targs,
            row["latency_s"],
            row["error"] or "",
            flush=True,
        )
        if row["error"] and (
            "subscription_sharing" in row["error"] or 400 <= (row["last_http_status"] or 0) < 500
        ):
            print("stopping after an API rejection (no retries)", flush=True)
            break
    lat = [r["latency_s"] for r in rows if not r["error"]]
    usage = [r["last_usage"] for r in rows if r.get("last_usage")]
    summary = {
        "backend": getattr(backend, "name", args.backend),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort if args.backend == "gpt" else None,
        "provenance": getattr(backend, "provenance", None),
        "n_attempted": len(rows),
        "n_planned": len(cfg["decisions"]),
        "valid_rate": sum(r["valid_tool_call"] for r in rows) / max(len(rows), 1),
        "correct_rate": sum(r["correct"] for r in rows) / max(len(rows), 1),
        "latency_p50_s": statistics.median(lat) if lat else None,
        "latency_max_s": max(lat) if lat else None,
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "errors": [r["error"] for r in rows if r["error"]],
        "by_category": {
            c: {
                "n": sum(r["category"] == c for r in rows),
                "correct": sum(r["correct"] for r in rows if r["category"] == c),
            }
            for c in sorted({r["category"] for r in rows})
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "dream/runtime/openai_plan.py",
        ROOT / "dream/runtime/qwen.py",
        ROOT / "dream/runtime/mobile_model.py",
        ROOT / "experiments/vlm_compare.py",
        OUT / "decisions.json",
    ]
    sources += [OUT / d["image"] for d in cfg["decisions"]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = json.dumps(
        {
            "summary": summary,
            "rows": rows,
            "sha256": hashes,
            "source_commit": source_commit,
            "selected_ids": [d["id"] for d in cfg["decisions"]],
        },
        indent=1,
        default=str,
    )
    with result_path.open("x", encoding="utf-8") as f:
        f.write(payload)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
