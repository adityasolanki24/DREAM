"""Camera perception evaluation: perception accuracy and its effect on task success.

    python experiments/run_vision.py experiments/configs/vision.json [--workers 4]

Paired design: for each condition, seed and task, every method runs the same scene, goal,
controller, skills and planner; only the perception backend differs.
  oracle               noisy ground-truth tracker (the previous baseline, an upper bound)
  vision               camera pipeline with the persistent world model and normal retries
  vision_no_retry      the same with one attempt per action (no recovery)
  vision_single_frame  the same without world-model memory (entities not in the latest frame
                       are forgotten): a simpler visual baseline
Conditions vary the scene or the sensor (sizes, colours, lighting, calibration, noise,
clutter); see the config. Perception accuracy is scored on the first frame of each vision
episode against ground truth (dream/eval/vision.py). Task success is read from ground truth
(dream/eval/manipulation.py), independently of the agent's claims.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

if "DISPLAY" not in os.environ:  # offscreen GL for the camera, chosen when MuJoCo is imported
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.adequacy import mcnemar_p, wilson_interval  # noqa: E402
from dream.eval.manipulation import object_in_tray, object_on_top, safety_summary  # noqa: E402
from dream.eval.vision import score_detections  # noqa: E402
from dream.perception.vision import VisionPerception  # noqa: E402
from dream.runtime.agent import RuleBasedPlanner, TaskAgent  # noqa: E402
from dream.sim.camera import CameraSpec  # noqa: E402
from dream.sim.tabletop import sample_tabletop  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402
from dream.tools.goals import InRegion, OnTop  # noqa: E402
from dream.tools.toolbox import Toolbox  # noqa: E402


def episode(args: tuple) -> dict:
    cond_name, cond, seed, task, method = args
    scene = dict(cond.get("scene", {}))
    colors = tuple(scene.pop("colors", ("red", "blue", "green")))
    spec = sample_tabletop(np.random.default_rng([seed, 0]), colors=colors, **scene)
    env_kw = dict(cond.get("env", {}))
    camera = CameraSpec(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in env_kw.pop("camera", {}).items()}
    )
    env = TabletopEnv(
        spec, np.random.default_rng([seed, 1]), camera=camera, camera_seed=seed, **env_kw
    )
    out = {"condition": cond_name, "seed": seed, "task": task, "method": method}
    if method == "oracle":
        perception = env
    else:
        perception = VisionPerception(
            env.capture_frame,
            lambda: env.time,
            np.random.default_rng([seed, 2]),
            memory=method != "vision_single_frame",
        )
        perception.refresh()
        out["perception"] = score_detections(env, perception.last_detections)
    out["backend"] = perception.backend
    attempts = 1 if method == "vision_no_retry" else 3
    tb = Toolbox(env.embodiment(), perception, rng=np.random.default_rng([seed, 3]))
    goal = InRegion(colors[0], "tray") if task == "tray" else OnTop(colors[0], colors[1])
    t0 = time.perf_counter()
    ep = TaskAgent(tb, RuleBasedPlanner(max_attempts=attempts)).run(goal)
    wall = time.perf_counter() - t0
    env.step(250)
    if task == "tray":
        success = object_in_tray(env, colors[0])
    else:
        success = object_on_top(env, colors[0], colors[1])
    grasps = [s for s in ep.steps if s.call.name == "grasp"]
    out.update(
        success=success,
        claimed=ep.success_claimed,
        false_claim=ep.success_claimed and not success,
        reason=ep.reason,
        steps=ep.to_dict()["steps"],
        grasp_attempts=len(grasps),
        failed_actions=ep.failed_actions,
        recovered=ep.recovered,
        frames=getattr(perception, "frames", 0),
        robot_s=float(sum(s.result.sim_time for s in ep.steps)),
        wall_s=wall,
        **safety_summary(env),
    )
    return out


def summarise(eps: list[dict]) -> dict:
    out: dict = {"tasks": {}, "perception": {}, "paired": {}}
    keys = sorted({(e["condition"], e["task"], e["method"]) for e in eps})
    for c, t, m in keys:
        es = [e for e in eps if (e["condition"], e["task"], e["method"]) == (c, t, m)]
        k = sum(e["success"] for e in es)
        out["tasks"][f"{c}/{t}/{m}"] = {
            "n": len(es),
            "success": k,
            "success_ci95": wilson_interval(k, len(es)),
            "false_claims": sum(e["false_claim"] for e in es),
            "mean_grasp_attempts": float(np.mean([e["grasp_attempts"] for e in es])),
            "recovered": sum(e["recovered"] for e in es),
            "mean_frames": float(np.mean([e["frames"] for e in es])),
            "mean_robot_s": float(np.mean([e["robot_s"] for e in es])),
            "mean_wall_s": float(np.mean([e["wall_s"] for e in es])),
            "protective_stops": sum(e["protective_stop"] for e in es),
            "link_collisions": sum(e["link_collision"] for e in es),
            "failures": [e["reason"][:80] for e in es if not e["success"]],
        }
    for c in sorted({e["condition"] for e in eps}):
        ps = [e["perception"] for e in eps if e["condition"] == c and e["method"] == "vision"]
        if not ps:
            continue
        m = [x for p in ps for x in p["matches"]]
        err = np.array([x["position_error_m"] for x in m])
        z = np.array([x["z_score"] for x in m])
        out["perception"][c] = {
            "frames": len(ps),
            "recall": float(np.mean([p["recall"] for p in ps])),
            "precision": float(np.mean([p["precision"] for p in ps])),
            "position_error_mm_median": 1000 * float(np.median(err)) if len(err) else None,
            "position_error_mm_p90": 1000 * float(np.percentile(err, 90)) if len(err) else None,
            "position_error_mm_max": 1000 * float(err.max()) if len(err) else None,
            "size_error_mm_median": 1000 * float(np.median([abs(x["size_error_m"]) for x in m])),
            "yaw_error_deg_median": float(np.median([x["yaw_error_deg"] for x in m])),
            "color_accuracy": float(np.mean([x["color_correct"] for x in m])),
            "within_2_sigma": float(np.mean(z < 2.0)) if len(z) else None,
            "tray_detected": float(np.mean([p.get("tray_detected", False) for p in ps])),
            "tray_centre_error_mm_median": 1000
            * float(
                np.median([p["tray_centre_error_m"] for p in ps if "tray_centre_error_m" in p])
            ),
        }
    for c, t in sorted({(e["condition"], e["task"]) for e in eps}):
        ref = {
            e["seed"]: e
            for e in eps
            if (e["condition"], e["task"], e["method"]) == (c, t, "oracle")
        }
        for m in ("vision", "vision_no_retry", "vision_single_frame"):
            other = {
                e["seed"]: e for e in eps if (e["condition"], e["task"], e["method"]) == (c, t, m)
            }
            seeds = sorted(set(ref) & set(other))
            if not seeds:
                continue
            b = sum(ref[s]["success"] and not other[s]["success"] for s in seeds)
            cc = sum(other[s]["success"] and not ref[s]["success"] for s in seeds)
            out["paired"][f"{c}/{t}/oracle_vs_{m}"] = {
                "n": len(seeds),
                "only_oracle": b,
                "only_vision": cc,
                "mcnemar_p": mcnemar_p(b, cc),
            }
    return out


def check_gates(summary: dict, gates: dict) -> dict:
    res = {}
    nominal = [
        v for k, v in summary["tasks"].items() if k.startswith("nominal/") and k.endswith("/vision")
    ]
    if "vision_nominal_success_min" in gates and nominal:
        rate = sum(v["success"] for v in nominal) / sum(v["n"] for v in nominal)
        res["vision_nominal_success"] = {
            "value": rate,
            "gate": gates["vision_nominal_success_min"],
            "pass": rate >= gates["vision_nominal_success_min"],
        }
    if "max_false_claims" in gates:
        n = sum(v["false_claims"] for v in summary["tasks"].values())
        res["false_claims_all"] = {
            "value": n,
            "gate": gates["max_false_claims"],
            "pass": n <= gates["max_false_claims"],
        }
    p = summary["perception"].get("nominal")
    if p and "nominal_recall_min" in gates:
        res["nominal_recall"] = {
            "value": p["recall"],
            "gate": gates["nominal_recall_min"],
            "pass": p["recall"] >= gates["nominal_recall_min"],
        }
    if p and "nominal_median_position_error_mm_max" in gates:
        v = p["position_error_mm_median"]
        res["nominal_median_position_error_mm"] = {
            "value": v,
            "gate": gates["nominal_median_position_error_mm_max"],
            "pass": v <= gates["nominal_median_position_error_mm_max"],
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain", "dream"], cwd=ROOT, text=True)
    )
    jobs = []
    for name, cond in cfg["conditions"].items():
        seeds = range(cfg["seed0"], cfg["seed0"] + cond.get("n", cfg["n"]))
        for s in seeds:
            for t in cfg["tasks"]:
                for m in cond["methods"]:
                    jobs.append((name, cond, s, t, m))
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = []
    with ProcessPoolExecutor(args.workers) as pool, open(out / "episodes.jsonl", "w") as f:
        for i, e in enumerate(pool.map(episode, jobs), 1):
            eps.append(e)
            f.write(json.dumps(e) + "\n")
            if i % 25 == 0:
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s]", flush=True)
    summary = summarise(eps)
    summary["gates"] = check_gates(summary, cfg.get("gates", {}))
    meta = {
        "config": cfg,
        "git_sha": sha,
        "dirty": dirty,
        "mujoco": mujoco.__version__,
        "wall_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    print(json.dumps({k: summary[k] for k in ("perception", "gates")}, indent=1))
    for k, v in summary["tasks"].items():
        shown = (
            f"{v['success']}/{v['n']} false={v['false_claims']} "
            f"grasps={v['mean_grasp_attempts']:.2f} frames={v['mean_frames']:.1f} "
            f"stops={v['protective_stops']}"
        )
        print(f"{k:45s} {shown}")


if __name__ == "__main__":
    main()
