"""Model-driven agent evaluation: condition A (rule-based planner) vs B (Qwen3-VL planner).

    python experiments/run_vlm.py experiments/configs/vlm.json --model-path /path/to/checkpoint
    python experiments/run_vlm.py experiments/configs/vlm.json --server http://host:8000

Both conditions use the same scenes, perception backend, toolbox, skills and safety limits.
- A receives the *structured* goal: it cannot read language, so it is an upper bound on goal
  understanding.
- B receives the natural-language instruction and the camera image, and must choose tools
  itself.
Instructions are deliberately phrased differently (``instructions`` in the config). They are
the model's input, never matched against a list. The structured goal is used by condition A,
by B's success verifier (withheld from the model) and by the independent evaluator. Success is
ground truth (dream/eval/manipulation.py).

This runner refuses to run condition B without a real model backend. There is no stand-in.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

if "DISPLAY" not in os.environ:  # offscreen GL for the camera, chosen when MuJoCo is imported
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.adequacy import wilson_interval  # noqa: E402
from dream.eval.manipulation import object_in_tray, object_on_top, safety_summary  # noqa: E402
from dream.perception.vision import VisionPerception  # noqa: E402
from dream.runtime.agent import RuleBasedPlanner, TaskAgent  # noqa: E402
from dream.runtime.vlm import VLMAgent  # noqa: E402
from dream.sim.tabletop import sample_tabletop  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402
from dream.tools.goals import parse_goal  # noqa: E402
from dream.tools.toolbox import Toolbox  # noqa: E402


def make_backend(args):
    if args.server:
        from dream.runtime.qwen import OpenAICompatibleBackend

        return OpenAICompatibleBackend(args.server, model=args.model_id)
    if args.model_path:
        from dream.runtime.qwen import QwenTransformersBackend

        return QwenTransformersBackend(args.model_path, device=args.device)
    return None


def episode(seed: int, item: dict, condition: str, backend, perception: str) -> dict:
    spec = sample_tabletop(np.random.default_rng([seed, 0]))
    env = TabletopEnv(spec, np.random.default_rng([seed, 1]), camera_seed=seed)
    sensing = env
    if perception == "vision":
        sensing = VisionPerception(env.capture_frame, lambda: env.time, np.random.default_rng(seed))
    tb = Toolbox(env.embodiment(), sensing, rng=np.random.default_rng([seed, 2]))
    goal = parse_goal(item["goal"])
    t0 = time.perf_counter()
    if condition == "A_rule_based":
        ep = TaskAgent(tb, RuleBasedPlanner()).run(goal)
        record = {
            "tool_calls": [s.call.name for s in ep.steps],
            "claimed": ep.success_claimed,
            "reason": ep.reason,
            "invalid_outputs": 0,
            "rejected_calls": 0,
            "model_s": [],
        }
    else:
        agent = VLMAgent(tb, backend, camera=lambda: env.capture_frame().rgb)
        ep = agent.run(item["instruction"], expected_goal=goal)
        record = {
            "tool_calls": ep.tool_calls,
            "claimed": ep.success_claimed,
            "reason": ep.reason,
            "invalid_outputs": ep.invalid_outputs,
            "rejected_calls": ep.rejected_calls,
            "model_s": [s.model_s for s in ep.steps],
            "episode": ep.to_dict(),
        }
    env.step(250)
    if item.get("impossible"):  # the correct outcome is a refusal, not a success
        success = False
        record["correct_refusal"] = not record["claimed"]
    elif goal.__class__.__name__ == "InRegion":
        success = object_in_tray(env, goal.object)
    else:
        success = object_on_top(env, goal.object, goal.support)
    return {
        "seed": seed,
        "condition": condition,
        "instruction": item["instruction"],
        "goal": item["goal"],
        "success": success,
        "false_claim": record["claimed"] and not success,
        "wall_s": time.perf_counter() - t0,
        "perception": sensing.backend,
        **record,
        **safety_summary(env),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model-path")
    ap.add_argument("--server")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    backend = make_backend(args)
    if backend is None and "B_vlm" in cfg["conditions"]:
        sys.exit("condition B needs a real model: pass --model-path or --server (no stand-in)")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    eps = []
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as f:  # sequential: one model instance
        for seed in range(cfg["seed0"], cfg["seed0"] + cfg["n"]):
            for item in cfg["instructions"]:
                for cond in cfg["conditions"]:
                    e = episode(seed, item, cond, backend, cfg.get("perception", "vision"))
                    eps.append(e)
                    f.write(json.dumps(e) + "\n")
                    f.flush()
    summary = {}
    for cond in cfg["conditions"]:
        all_es = [e for e in eps if e["condition"] == cond]
        es = [e for e in all_es if "correct_refusal" not in e]
        k = sum(e["success"] for e in es)
        lat = [x for e in es for x in e["model_s"]]
        summary[cond] = {
            "n": len(es),
            "success": k,
            "success_ci95": wilson_interval(k, len(es)),
            "false_claims": sum(e["false_claim"] for e in all_es),
            "correct_refusals": sum(e.get("correct_refusal", False) for e in all_es),
            "invalid_outputs": sum(e["invalid_outputs"] for e in es),
            "rejected_calls": sum(e["rejected_calls"] for e in es),
            "protective_stops": sum(e["protective_stop"] for e in es),
            "mean_tool_calls": float(np.mean([len(e["tool_calls"]) for e in es])),
            "model_latency_s_median": float(np.median(lat)) if lat else None,
        }
    meta = {
        "config": cfg,
        "git_sha": sha,
        "mujoco": mujoco.__version__,
        "backend": getattr(backend, "name", None),
        "machine": platform.platform(),
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
