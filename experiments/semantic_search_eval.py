"""Object-search benchmark: semantic belief-guided search against frontier baselines in
procedurally generated multi-room homes (dream/sim/home.py), scored on the original request.

    python experiments/semantic_search_eval.py experiments/configs/<config>.json --split dev
        [--arms proposed,A_gain_frontier] [--workers 3] [--base kinematic|g1] [--out DIR]

Wiring only (like dream/eval): it builds the scene, the robot and its sensors, runs TaskAgent
with the arm's planner, and scores each episode with dream/eval/semantic_scoring.py. Every arm
in a run shares homes (paired by seed), detector noise, navigator, budgets and verifier; only
the arm's declared differences change. Labels that apply to every row:
- odometry is ORACLE-ASSISTED (true increments plus noise);
- the semantic detector is ORACLE-ASSISTED (rendered segmentation plus an assumed noise
  model, dream/sim/semantic_sensor.py), standing in for a neural detector;
- ``base=kinematic`` is a KINEMATIC SURROGATE (no legged dynamics); ``base=g1`` is the
  pretrained whole-body controller in MuJoCo;
- remembered sightings (scenario I) are placed in the robot's odometry frame using the true
  start pose (oracle-assisted initial localisation).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.semantic_scoring import (  # noqa: E402
    TrueInstance,
    odometry_to_world,
    score_episode,
    score_rooms,
)
from dream.memory.home import HomeMemory  # noqa: E402
from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.runtime.agent import Finish, TaskAgent  # noqa: E402
from dream.runtime.semantic_search import (  # noqa: E402
    SemanticSearchPlanner,
    UtilityWeights,
    stop_outcome,
)
from dream.semantics.belief import BeliefConfig  # noqa: E402
from dream.semantics.knowledge import load_knowledge, uniform_knowledge  # noqa: E402
from dream.semantics.verify import VerifierThresholds  # noqa: E402
from dream.semantics.world import SemanticWorld  # noqa: E402
from dream.sim import home as homes  # noqa: E402
from dream.tools.goals import UnsupportedGoal  # noqa: E402
from dream.tools.intent import (  # noqa: E402
    INSPECTED,
    LOCALIZED,
    REACHED,
    Capabilities,
    bind_request,
    default_vocabulary,
    ground_request,
)
from dream.tools.mobile import MobileToolbox  # noqa: E402

GRID_SIZE = 24.0  # m: homes reach 11 m from the start (dream/sim/home.EXTENT_LIMIT)
MEMORY_MAP = "episode-odometry"


class TimeBudget:
    """Ends the run when the robot-time budget is spent (an episode limit, not a strategy)."""

    def __init__(self, planner, base, seconds: float) -> None:
        self.planner, self.base, self.seconds = planner, base, seconds

    def next_action(self, goal, scene, verdict, history):
        if not verdict.get("satisfied") and self.base.time > self.seconds:
            return Finish(False, f"time budget ({self.seconds:.0f} s) spent")
        return self.planner.next_action(goal, scene, verdict, history)


class SubstitutingPlanner:
    """FAULT INJECTION for the goal-consistency ablation, not a method: after ``after_s`` of
    robot time without success it walks to the region whose inferred room is the target's
    most typical one and claims success there, the 7 Nov 2026 failure mode ("the kitchen was
    reached, so the refrigerator was found"). With the original request recorded, TaskAgent
    must reject that claim; without it, nothing does."""

    def __init__(self, inner, toolbox, target: str, after_s: float) -> None:
        self.inner, self.toolbox, self.target, self.after_s = inner, toolbox, target, after_s
        self.trace = getattr(inner, "trace", [])
        self.tries = 0

    def next_action(self, goal, scene, verdict, history):
        from dream.runtime.agent import ToolCall

        base, world = self.toolbox.base, self.toolbox.semantic
        if verdict.get("satisfied") or base.time < self.after_s:
            return self.inner.next_action(goal, scene, verdict, history)
        room = world.knowledge.typical_rooms(self.target, 1)[0][0]
        world.refresh()
        regions = [r for r in world.regions.values() if r.leading()[0] == room]
        if not regions or self.tries >= 3:
            return self.inner.next_action(goal, scene, verdict, history)
        here = world.segmenter.region_of(world.grid, base.odometry()[:2])
        if any(r.id == here for r in regions):
            return Finish(True, f"reached the {room}; a {self.target} is usually there")
        r = max(regions, key=lambda r: r.weights.get(room, 0.0))
        self.tries += 1
        return ToolCall("go_to", {"x": float(r.centroid[0]), "y": float(r.centroid[1])})


class PathMeter:
    """Wraps a MobileBase to measure the true path length (evaluation only)."""

    def __init__(self, base, true_pose) -> None:
        self.inner, self.true_pose, self.path_m = base, true_pose, 0.0
        self.frames = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def time(self):
        return self.inner.time

    @property
    def stop_reason(self):
        return self.inner.stop_reason

    def odometry(self):
        return self.inner.odometry()

    def capture(self):
        self.frames += 1
        return self.inner.capture()

    def walk(self, forward, lateral, turn, seconds):
        a = self.true_pose()[:2]
        ok = self.inner.walk(forward, lateral, turn, seconds)
        self.path_m += float(np.linalg.norm(self.true_pose()[:2] - a))
        return ok


def vocabularies(knowledge):
    objects = default_vocabulary(
        [*homes.CLASSES, "kettle"], {k: v for k, v in knowledge.synonyms.items()}
    )
    rooms = default_vocabulary(knowledge.categories)
    return objects, rooms


def to_odometry(start, xy):
    """World point -> the robot's odometry frame (origin and heading at the true start)."""
    x0, y0, yaw = start
    d = np.asarray(xy, float)[:2] - (x0, y0)
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def memory_hints(home, target: str, enabled: bool, folder: str) -> tuple[list[dict], HomeMemory]:
    """Seed persistent memory with the scenario's earlier sightings and read them back."""
    mem = HomeMemory(os.path.join(folder, "home.sqlite"), clock=lambda: 1_000_000.0)
    for cls, xyz, age in home.memory_prior:
        p = to_odometry(home.start, xyz[:2])
        mem.observe(cls, [p[0], p[1], xyz[2]], MEMORY_MAP, time=1_000_000.0 - age,
                    source="scenario prior sighting")  # fmt: skip
    if not enabled:
        return [], mem
    hints = []
    for b in mem.where_is(target, MEMORY_MAP):
        if b.state == "missing":
            continue
        # A remembered sighting is a place to check, not a weighted fact: HomeMemory's 6 h
        # confidence half-life is designed for movable objects, so it is reported, not used.
        hints.append({"xy": b.position[:2].tolist(), "confidence": 1.0, "id": b.object_id,
                      "age_s": b.age_s, "last_seen": b.last_seen})  # fmt: skip
    return hints, mem


def write_back(mem: HomeMemory, world, target: str, hints, verdict, now: float) -> dict:
    """Persist what this episode established: a verified target sighting, and absences where
    a remembered sighting was looked at closely and nothing was found."""
    out = {"sighting": None, "absences": []}
    if verdict.get("status") == "target_found_verified":
        p = verdict["position"]
        out["sighting"] = mem.observe(target, p, MEMORY_MAP, time=now, source="verified search")
    for h in hints:
        i, j = world.grid.cell(np.asarray(h["xy"], float))
        r = int(0.5 / 0.05)
        cov = world.map.coverage[max(i - r, 0) : i + r + 1, max(j - r, 0) : j + r + 1]
        near = [t for t in world.map.candidates(target)
                if np.linalg.norm(t.position[:2] - np.asarray(h["xy"])) < 1.0]  # fmt: skip
        if cov.size and float(cov.max()) >= 0.6 and not near:
            out["absences"] = mem.record_absence(h["xy"], 0.8, MEMORY_MAP, time=now,
                                                 source="searched, not seen")  # fmt: skip
    return out


def build_base(kind: str, home, seed: int, noise):
    from dream.sim.semantic_sensor import DEGRADED, NOMINAL

    preset = {"nominal": NOMINAL, "degraded": DEGRADED}[noise]
    if kind == "kinematic":
        from dream.sim.kinematic_base import KinematicBase

        base = KinematicBase(home, np.random.default_rng([seed, 1]), semantic=preset)
        return base, base.true_pose, lambda: len(base.contact_log())
    from dream.robot.g1_wbc import WBCLocomotion
    from dream.sim.g1_mobile import HEAD_CAMERA, G1MobileBase, G1MobileEnv, build_mobile_g1
    from dream.sim.semantic_sensor import SemanticCamera, SemanticG1Base

    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: homes.add_home(s, home)), np.random.default_rng([seed, 1])
    )
    env.reset(*home.start)
    sensor = SemanticCamera(env.model, HEAD_CAMERA, homes.semantic_index(env.model, home),
                            np.random.default_rng([seed, 3]), preset)  # fmt: skip
    base = SemanticG1Base(G1MobileBase(env, WBCLocomotion.from_cache()), sensor)
    return base, env.true_pose, lambda: None


def episode(job: dict) -> dict:
    cfg, arm = job["cfg"], job["arm"]
    row = {k: job[k] for k in ("scenario", "target", "generator", "noise", "seed", "arm_name",
                               "base")}  # fmt: skip
    t0 = time.perf_counter()
    try:
        rng = np.random.default_rng([job["seed"], 0])
        home = homes.sample_home(rng, job["scenario"], job["target"], job["generator"])
    except Exception as error:  # the generator has no valid home: a recorded scene failure
        return {**row, "scene_failure": repr(error), "result": "scene_failure"}
    try:
        return {**row, **_run(home, job, cfg, arm, t0)}
    except Exception as error:  # an infrastructure crash is a recorded failure, not a pass
        return {**row, "home_seed": home.seed, "crash": repr(error),
                "traceback": traceback.format_exc()[-2000:], "result": "crash"}  # fmt: skip


def _run(home, job, cfg, arm, t0) -> dict:
    knowledge = load_knowledge() if arm.get("semantic", True) else uniform_knowledge()
    words = load_knowledge()  # synonyms for reading requests are not search knowledge
    objects, rooms = vocabularies(words)
    caps = Capabilities(
        outcomes=frozenset({LOCALIZED, REACHED, INSPECTED}),
        object_classes=frozenset(homes.CLASSES),
        outcome_classes={INSPECTED: frozenset({"cabinet"})},
    )
    spec = ground_request(home.request, objects, rooms)
    goal = bind_request(spec, caps)
    base, true_pose, contacts = build_base(job["base"], home, job["seed"], job["noise"])
    meter = PathMeter(base, true_pose)
    belief = BeliefConfig(
        semantic=arm.get("semantic", True),
        room_inference=arm.get("room_inference", True),
        memory=arm.get("memory", True),
    )
    thresholds = VerifierThresholds(**{**cfg.get("verifier", {}), **arm.get("verifier", {})})
    memory = SpatialMemory(np.random.default_rng([job["seed"], 2]), GRID_SIZE, boxes=False)
    world = SemanticWorld(memory.grid, knowledge, belief, thresholds)
    toolbox = MobileToolbox(meter, memory, navigator=cfg["navigator"], semantic=world)
    with tempfile.TemporaryDirectory() as folder:
        target = getattr(goal, "target", spec.target_class) or ""
        hints, mem = memory_hints(home, target, belief.memory, folder)
        weights = UtilityWeights(**{**cfg.get("weights", {}), **arm.get("weights", {})})
        planner = SemanticSearchPlanner(toolbox, arm["policy"], weights, memory_hints=hints)
        chosen = planner
        if "substitute_after_s" in arm:  # fault injection (goal-consistency ablation)
            chosen = SubstitutingPlanner(planner, toolbox, target, arm["substitute_after_s"])
        agent = TaskAgent(toolbox, TimeBudget(chosen, meter, cfg["time_budget_s"]),
                          max_steps=cfg["max_steps"])  # fmt: skip
        episode = agent.run(goal, spec=spec if arm.get("gate", True) else None)
        verdict = toolbox.call("verify_goal", {"goal": goal}).data
        stored = write_back(mem, world, target, hints, verdict, 1_000_000.0 + meter.time)
        mem.close()
    if isinstance(goal, UnsupportedGoal):
        completion = "goal_unsupported"
    else:
        completion = stop_outcome(toolbox, target, episode.completion, episode.reason)
        if episode.success_claimed:
            completion = "target_found_verified"
    claim_xy = verdict.get("position") if episode.success_claimed else None
    if episode.success_claimed and claim_xy is None:  # a claim without a verified object
        claim_xy = meter.odometry()[:2].tolist()
    world.refresh()
    to_world = odometry_to_world(true_pose(), meter.odometry())
    instances = [
        TrueInstance(o.instance_id, o.cls, np.array(o.xy, float), np.array(o.half[:2], float),
                     float(o.yaw), getattr(homes.room_at(home, o.xy), "type", "none"))
        for o in home.objects
    ]  # fmt: skip
    supported = home.scenario != "L"
    score = score_episode(
        target=target, outcome=getattr(goal, "outcome", spec.required_outcome),
        supported_expected=supported, claimed=episode.success_claimed, completion=completion,
        claim_xy_odom=claim_xy,
        instances=instances, to_world=to_world, true_pose_end=true_pose(),
    )  # fmt: skip

    def room_type(xy):
        r = homes.room_at(home, xy)
        return None if r is None else r.type

    rooms_scored = score_rooms([r.summary() for r in world.regions.values()], to_world, room_type)
    steps = episode.steps
    return {
        "home_seed": home.seed, "variant": home.scenario, "request": home.request,
        "spec": spec.to_dict(),
        "goal": repr(goal), "claimed": episode.success_claimed, "reason": episode.reason,
        "completion": completion, **score,
        "sim_s": round(meter.time, 2), "wall_s": round(time.perf_counter() - t0, 1),
        "path_m": round(meter.path_m, 2), "frames": meter.frames,
        "tool_calls": len(steps), "look_arounds": sum(s.call.name == "look_around" for s in steps),
        "failed_go_to": sum(s.call.name == "go_to" and not s.result.ok for s in steps),
        "model_calls": 0, "contact_substeps": contacts(), "fell": bool(meter.stop_reason),
        "explored_m2": round(float(memory.grid.free().sum()) * 0.05**2, 1),
        "tracks": len(world.map.tracks), "regions": len(world.regions), "rooms": rooms_scored,
        "verdict": {k: verdict.get(k) for k in ("status", "reason", "track", "position")},
        "candidates_seen": len(world.map.candidates(target)) if target else 0,
        "memory_hints": len(hints), "memory_write_back": stored,
        "trace": planner.trace[-40:],
    }  # fmt: skip


def jobs_for(cfg: dict, split: str, arms: list[str], base: str) -> list[dict]:
    s = cfg["splits"][split]
    out = []
    for cond in s["conditions"]:
        for seed in range(*cond["seeds"]):
            for arm_name in arms:
                arm = cfg["arms"][arm_name]
                if cond["scenario"] not in arm.get("scenarios", cond["scenario"]):
                    continue
                out.append({"cfg": cfg, "arm": arm, "arm_name": arm_name, "seed": seed,
                            "scenario": cond["scenario"], "target": cond.get("target",
                            "refrigerator"), "generator": cond.get("generator", "nominal"),
                            "noise": cond.get("noise", "nominal"), "base": base})  # fmt: skip
    return out


def summarise(rows: list[dict]) -> dict:
    from collections import Counter, defaultdict

    by = defaultdict(list)
    for r in rows:
        by[(r["arm_name"], r["scenario"])].append(r)
    table = {}
    for (arm, scen), rs in sorted(by.items()):
        res = Counter(r.get("result") for r in rs)
        times = sorted(r["sim_s"] for r in rs if r.get("result") == "success")
        table[f"{arm}|{scen}"] = {
            "n": len(rs),
            **dict(res),
            "median_success_sim_s": times[len(times) // 2] if times else None,
            "mean_path_m": round(float(np.mean([r.get("path_m", 0) for r in rs])), 1),
        }
    totals = defaultdict(Counter)
    for r in rows:
        totals[r["arm_name"]][r.get("result")] += 1
    return {"by_arm_scenario": table, "totals": {a: dict(c) for a, c in totals.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", help="comma-separated arm names (default: the split's arms)")
    ap.add_argument("--base", choices=["kinematic", "g1"], default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, help="run only the first N jobs (smoke)")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    split = cfg["splits"][args.split]
    arms = args.arms.split(",") if args.arms else split["arms"]
    base = args.base or split.get("base", "kinematic")
    jobs = jobs_for(cfg, args.split, arms, base)[: args.limit]
    out = pathlib.Path(args.out or ROOT / "results" / cfg["name"] / args.split)
    out.mkdir(parents=True, exist_ok=True)
    from experiments.code_version import code_dirty, code_sha

    meta = {"config": cfg, "split": args.split, "arms": arms, "base": base,
            "git_sha": code_sha(ROOT), "dream_dirty_at_start": code_dirty(ROOT),
            "renderer": os.environ.get("MUJOCO_GL"), "machine": platform.platform(),
            "jobs": len(jobs)}  # fmt: skip
    rows, t0 = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool, open(out / "episodes.jsonl", "w") as f:
        for r in pool.map(episode, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"[{len(rows)}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {r['arm_name']} "
                  f"{r['scenario']} {r['target']} s{r['seed']}: {r.get('result')} "
                  f"{r.get('completion')} sim={r.get('sim_s')} wall={r.get('wall_s')}",
                  flush=True)  # fmt: skip
    meta["wall_s"] = round(time.perf_counter() - t0, 1)
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summarise(rows)}, indent=2,
                                                 default=float))  # fmt: skip
    print(json.dumps(summarise(rows)["totals"], indent=2))


if __name__ == "__main__":
    main()
