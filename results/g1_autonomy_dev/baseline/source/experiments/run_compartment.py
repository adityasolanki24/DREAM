"""Milestone 11 evaluation: "put the red cube in the compartment" with an unknown mechanism.

    python experiments/run_compartment.py experiments/configs/compartment.json [--workers 4]

Every episode is one continuous simulation: the Panda investigates the closed compartment, opens
it (or declines), picks the red cube, places it inside and verifies, through the same runtime.
Methods share the scene, seeds, controller, skills, sensing, probe set, probe budget, force cap
and planner (paired design); only the discovery component differs:
  dream_eig     investigate_physics with expected-information-gain probing, then open along
                the learned constraint
  random_probe  the same, with probes drawn uniformly from the same candidate set
  fixed_pull    no discovery: always pull straight out of the visible front (a drawer
                strategy), up to the same number of attempts
Scenes are held out: generated from ``seed0`` onward, disjoint from every development seed
(0-9 and 100-109). Success is read from ground truth (dream.eval.manipulation), independently
of the agent's claims. One physical interaction is one compliant motion of the gripped handle:
a probe, an opening segment, or a fixed pull.
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

if "DISPLAY" not in os.environ:  # offscreen GL for the cameras, chosen when MuJoCo is imported
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.adequacy import mcnemar_p, wilson_interval  # noqa: E402
from dream.eval.manipulation import (  # noqa: E402
    compartment_opening,
    object_in_compartment,
    safety_summary,
)
from dream.perception.vision import VisionPerception  # noqa: E402
from dream.runtime.agent import RuleBasedPlanner, TaskAgent  # noqa: E402
from dream.sim.camera import SIDE_CAMERA, CameraSpec  # noqa: E402
from dream.sim.tabletop import sample_compartment_scene  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402
from dream.tools.goals import InRegion  # noqa: E402
from dream.tools.toolbox import Toolbox  # noqa: E402

METHODS = {
    "dream_eig": {"exploration": "eig", "opening": "learned"},
    "random_probe": {"exploration": "random", "opening": "learned"},
    "fixed_pull": {"exploration": "eig", "opening": "fixed_pull"},  # never investigates
}
TRUE_CLASS = {"drawer": "prismatic", "lid": "revolute", "fixed": "fixed"}
GOAL = InRegion("red", "compartment")


def episode(args: tuple) -> dict:
    kind, seed, method, budget, scene_kwargs, env_kwargs, perception = args
    spec = sample_compartment_scene(kind, np.random.default_rng([seed, 0]), **scene_kwargs)
    if perception == "vision":  # both cameras: the lid hides its cavity from the default one
        env_kwargs = {**env_kwargs, "extra_cameras": (CameraSpec(**SIDE_CAMERA),)}
        env_kwargs.setdefault("camera_seed", seed)
    env = TabletopEnv(spec, np.random.default_rng([seed, 1]), **env_kwargs)
    sensing = env
    if perception == "vision":
        sensing = VisionPerception(
            env.frame_sources(), lambda: env.time, np.random.default_rng([seed, 4])
        )
    tb = Toolbox(env.embodiment(), sensing, rng=np.random.default_rng([seed, 2]))
    planner = RuleBasedPlanner(probe_budget=budget, **METHODS[method])
    t0 = time.perf_counter()
    ep = TaskAgent(tb, planner, max_steps=14).run(GOAL)
    wall = time.perf_counter() - t0
    env.step(250)  # let everything settle before judging
    comp = spec.compartment
    out = {
        "kind": kind,
        "looks_like": comp.looks_like,
        "seed": seed,
        "method": method,
        "perception": sensing.backend,
        "success": object_in_compartment(env, "red"),
        "claimed": ep.success_claimed,
        "reason": ep.reason,
        "tool_calls": ep.tool_calls,
        "failed_actions": ep.failed_actions,
        "recovered": ep.recovered,
        "steps": ep.to_dict()["steps"],
        "opening_true": compartment_opening(env),
        "upper_true": comp.upper,
        "wall_s": wall,
        "robot_s": float(sum(s.result.sim_time for s in ep.steps)),
        **safety_summary(env),
    }
    out["false_claim"] = out["claimed"] and not out["success"]
    probes = segments = pulls = 0
    explore_s = 0.0
    for s in ep.steps:
        if s.call.name == "investigate_physics":
            probes += int(s.result.data.get("interactions", 0))
            explore_s += s.result.sim_time
        elif s.call.name == "open_articulated":
            if s.call.args.get("strategy") == "fixed_pull":
                pulls += 1
            else:
                segments += len(s.result.data.get("segments", []))
    out.update(
        probes=probes,
        opening_segments=segments,
        fixed_pulls=pulls,
        interactions=probes + segments + pulls,
        exploration_robot_s=explore_s,
        open_calls=sum(1 for s in ep.steps if s.call.name == "open_articulated"),
    )
    inv = tb.investigations.get("compartment")
    if inv is not None:
        out["investigation"] = {k: v for k, v in inv.summary().items() if k not in ("reason",)} | {
            "reason": inv.reason
        }
        out["identified_correctly"] = (
            inv.status == "identified" and inv.mechanism == TRUE_CLASS[kind]
        )
        out["max_probe_slip_m"] = max((st["slip_m"] for st in inv.steps), default=0.0)
        out["max_probe_force_n"] = max((st["force_n"] for st in inv.steps), default=0.0)
        if kind != "fixed" and inv.constraint.get("axis") is not None:
            est = np.asarray(inv.constraint["axis"])
            cos = float(np.clip(est @ comp.joint_axis_world(), -1, 1))
            out["axis_error_deg"] = float(np.degrees(np.arccos(cos)))
        out["pre_action_pred_err_m"] = inv.prediction_error_m
    out["correct_rejection"] = kind == "fixed" and out["open_calls"] == 0 and not out["claimed"]
    return out


def _bootstrap_ratio(num: np.ndarray, den: np.ndarray, rng, n: int = 2000) -> list[float]:
    idx = rng.integers(0, len(num), size=(n, len(num)))
    ratios = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return np.percentile(ratios, [2.5, 97.5]).tolist()


def _bootstrap_mean(x: np.ndarray, rng, n: int = 2000) -> list[float]:
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist()


def summarise(eps: list[dict], methods: list[str]) -> dict:
    rng = np.random.default_rng(0)
    out: dict = {"by_method": {}, "paired": {}}
    by = {
        m: sorted([e for e in eps if e["method"] == m], key=lambda e: (e["kind"], e["seed"]))
        for m in methods
    }
    for m, es in by.items():
        openable = [e for e in es if e["kind"] != "fixed"]
        fixed = [e for e in es if e["kind"] == "fixed"]
        k = sum(e["success"] for e in openable)
        inter = np.array([e["interactions"] for e in openable], float)
        succ = np.array([e["success"] for e in openable], float)
        r: dict = {
            "openable_n": len(openable),
            "openable_success": k,
            "openable_success_rate": k / max(len(openable), 1),
            "openable_success_ci95": wilson_interval(k, len(openable)),
            "success_per_10_interactions": 10 * succ.sum() / max(inter.sum(), 1),
            "success_per_10_interactions_ci95": (
                10 * np.array(_bootstrap_ratio(succ, inter, rng))
            ).tolist(),
            "mean_interactions_openable": float(inter.mean()) if len(inter) else None,
            # Over every scene: success where it can open, a correct rejection where it cannot.
            "correct_outcomes_per_10_interactions": 10
            * sum(e["success"] or e["correct_rejection"] for e in es)
            / max(sum(e["interactions"] for e in es), 1),
            "mean_probes_openable": float(np.mean([e["probes"] for e in openable])),
            "mean_exploration_robot_s": float(np.mean([e["exploration_robot_s"] for e in es])),
            "mean_robot_s": float(np.mean([e["robot_s"] for e in es])),
            "fixed_n": len(fixed),
            "correct_rejections": sum(e["correct_rejection"] for e in fixed),
            "correct_rejection_ci95": wilson_interval(
                sum(e["correct_rejection"] for e in fixed), len(fixed)
            ),
            "mean_interactions_fixed": float(np.mean([e["interactions"] for e in fixed]))
            if fixed
            else None,
            "false_claims": sum(e["false_claim"] for e in es),
            "protective_stops": sum(e["protective_stop"] for e in es),
            "link_collisions": sum(e["link_collision"] for e in es),
            "mean_failed_actions": float(np.mean([e["failed_actions"] for e in es])),
            "recovered": sum(e["recovered"] for e in es),
        }
        for kind in ("drawer", "lid", "fixed"):
            ke = [e for e in es if e["kind"] == kind]
            if not ke:
                continue
            ks = sum(e["success"] for e in ke)
            row = {
                "n": len(ke),
                "success": ks,
                "success_ci95": wilson_interval(ks, len(ke)),
                "mean_interactions": float(np.mean([e["interactions"] for e in ke])),
                "mean_probes": float(np.mean([e["probes"] for e in ke])),
            }
            inv = [e for e in ke if "investigation" in e]
            if inv:
                row["identified_correctly"] = sum(e["identified_correctly"] for e in inv)
                row["investigation_status"] = {
                    s: sum(e["investigation"]["status"] == s for e in inv)
                    for s in sorted({e["investigation"]["status"] for e in inv})
                }
                ax = [e["axis_error_deg"] for e in inv if "axis_error_deg" in e]
                if ax:
                    row["median_axis_error_deg"] = float(np.median(ax))
                row["mean_pre_action_pred_err_mm"] = 1000 * float(
                    np.nanmean([e["pre_action_pred_err_m"] for e in inv])
                )
                row["max_probe_slip_mm"] = 1000 * max(e["max_probe_slip_m"] for e in inv)
                row["max_probe_force_n"] = max(e["max_probe_force_n"] for e in inv)
            r[kind] = row
        out["by_method"][m] = r
    ref = by.get("dream_eig")
    for m, es in by.items():
        if m == "dream_eig" or ref is None:
            continue
        pairs = [(a, b) for a, b in zip(ref, es, strict=True) if a["kind"] != "fixed"]
        assert all(a["seed"] == b["seed"] and a["kind"] == b["kind"] for a, b in pairs)
        b_ = sum(a["success"] and not o["success"] for a, o in pairs)
        c_ = sum(o["success"] and not a["success"] for a, o in pairs)
        d_inter = np.array([a["interactions"] - o["interactions"] for a, o in pairs], float)
        out["paired"][f"dream_eig_vs_{m}"] = {
            "n_pairs_openable": len(pairs),
            "only_dream_succeeded": b_,
            "only_other_succeeded": c_,
            "mcnemar_p": mcnemar_p(b_, c_),
            "mean_interaction_difference": float(d_inter.mean()),
            "mean_interaction_difference_ci95": _bootstrap_mean(d_inter, rng),
        }
    return out


def check_gates(summary: dict, gates: dict) -> dict:
    a = summary["by_method"].get("dream_eig", {})
    res = {}
    if "dream_openable_success_min" in gates:
        res["dream_openable_success"] = {
            "value": a["openable_success_rate"],
            "gate": gates["dream_openable_success_min"],
            "pass": a["openable_success_rate"] >= gates["dream_openable_success_min"],
        }
    if "dream_correct_rejection_min" in gates and a.get("fixed_n"):
        rate = a["correct_rejections"] / a["fixed_n"]
        res["dream_correct_rejection"] = {
            "value": rate,
            "gate": gates["dream_correct_rejection_min"],
            "pass": rate >= gates["dream_correct_rejection_min"],
        }
    if "max_false_claims" in gates:
        n = sum(r["false_claims"] for r in summary["by_method"].values())
        res["false_claims_all_methods"] = {
            "value": n,
            "gate": gates["max_false_claims"],
            "pass": n <= gates["max_false_claims"],
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    seeds = range(cfg["seed0"], cfg["seed0"] + cfg["n_per_kind"])
    # Optional shifted or challenge conditions: passed to the scene sampler and the environment.
    scene_kwargs = cfg.get("scene_kwargs", {})
    env_kwargs = cfg.get("env_kwargs", {})
    perception = cfg.get("perception", "oracle")  # "vision": camera pipeline, two cameras
    jobs = [
        (kind, s, m, cfg["probe_budget"], scene_kwargs, env_kwargs, perception)
        for kind in cfg["kinds"]
        for s in seeds
        for m in cfg["methods"]
    ]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain", "dream"], cwd=ROOT, text=True)
    )
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = []
    with ProcessPoolExecutor(args.workers) as pool, open(out / "episodes.jsonl", "w") as f:
        for i, e in enumerate(pool.map(episode, jobs), 1):
            eps.append(e)
            f.write(json.dumps(e) + "\n")
            if i % 20 == 0:
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s]", flush=True)
    summary = summarise(eps, cfg["methods"])
    summary["gates"] = check_gates(summary, cfg.get("gates", {}))
    meta = {
        "config": cfg,
        "git_sha": sha,  # read at the start: the code that ran
        "dirty": dirty,
        "mujoco": mujoco.__version__,
        "wall_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
