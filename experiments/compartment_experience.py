"""Tests E/F: explore only when needed. Reuse a verified mechanism model; distrust it on failure.

    python experiments/compartment_experience.py experiments/configs/compartment_experience.json

The validated Panda task "put the red cube in the compartment" (dream_eig planner, MuJoCo,
oracle-assisted handle tracker as in the held-out compartment results), three sessions per
scene, each in a fresh simulation, with a home memory that survives between them:
1. ``learn``: no memory. Investigate (EIG probes), open, place. A successful investigation is
   stored with its belief as a checksummed artifact.
2. ``reuse`` (Test F): same scene after a restart. The remembered procedure is recalled from
   the handle's observed position; expected: no probes, success, less robot time.
3. ``changed`` (Test E): the same-looking compartment is now immovable (kind "fixed",
   identical appearance). Expected: the remembered model makes no progress, is withdrawn,
   the object is investigated afresh, and the task is declined without a false claim.
   ``changed_control``: the same scene without memory.
Success, opening and claims are scored from ground truth (dream.eval.manipulation).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.manipulation import compartment_opening, object_in_compartment  # noqa: E402
from dream.memory.experience import ExperienceToolbox, remember_investigation  # noqa: E402
from dream.memory.home import HomeMemory  # noqa: E402
from dream.runtime.agent import RuleBasedPlanner, TaskAgent  # noqa: E402
from dream.sim.tabletop import sample_compartment_scene  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402
from dream.skills import articulation as art  # noqa: E402
from dream.tools.goals import InRegion  # noqa: E402

GOAL = InRegion("red", "compartment")
MAP = "panda-table"


def session(spec, seed, cfg, memory, artifacts, clock_offset, recall: bool) -> tuple[dict, object]:
    env = TabletopEnv(spec, np.random.default_rng([seed, 1]))
    tb = ExperienceToolbox(env.embodiment(), env, rng=np.random.default_rng([seed, 2]),
                           memory=memory, artifacts=artifacts, map_id=MAP,
                           clock=lambda: clock_offset + env.time)  # fmt: skip
    handle = np.asarray(art.find_articulated(env, "compartment").handle_pos, float)
    recalled = tb.recall("compartment") if recall else False
    planner = RuleBasedPlanner(probe_budget=cfg["probe_budget"], exploration="eig",
                               opening="learned")  # fmt: skip
    t0 = time.perf_counter()
    ep = TaskAgent(tb, planner, max_steps=cfg["max_steps"]).run(GOAL)
    env.step(250)
    probes = sum(int(s.result.data.get("interactions", 0)) for s in ep.steps
                 if s.call.name == "investigate_physics")  # fmt: skip
    out = {"recalled": recalled, "claimed": ep.success_claimed, "reason": ep.reason,
           "success": object_in_compartment(env, "red"), "opening_true": compartment_opening(env),
           "probes": probes, "investigations": sum(s.call.name == "investigate_physics"
                                                   for s in ep.steps),
           "robot_s": float(sum(s.result.sim_time for s in ep.steps)),
           "wall_s": time.perf_counter() - t0, "tools": [s.call.name for s in ep.steps],
           "experience": tb.experience_log}  # fmt: skip
    out["false_claim"] = out["claimed"] and not out["success"]
    return out, (tb, handle, env)


def scene(args) -> dict:
    kind, seed, cfg, store = args
    spec = sample_compartment_scene(kind, np.random.default_rng([seed, 0]))
    changed = dataclasses.replace(spec, compartment=dataclasses.replace(
        spec.compartment, kind="fixed", looks_like=kind, upper=0.0))  # fmt: skip
    path = pathlib.Path(store) / f"home_{kind}_{seed}.sqlite"
    artifacts = pathlib.Path(store) / f"artifacts_{kind}_{seed}"
    path.unlink(missing_ok=True)
    res = {"kind": kind, "seed": seed}
    mem = HomeMemory(path)
    res["learn"], (tb, handle, env) = session(spec, seed, cfg, mem, artifacts, 0.0, False)
    inv = tb.investigations.get("compartment")
    res["stored"] = bool(res["learn"]["success"] and inv is not None and inv.status == "identified")
    if res["stored"]:
        remember_investigation(mem, artifacts, "compartment", inv, handle, MAP, env.time,
                               {"task_success_claimed": res["learn"]["claimed"]})  # fmt: skip
    mem.close()  # restart
    mem = HomeMemory(path)
    res["reuse"], _ = session(spec, seed, cfg, mem, artifacts, 1e5, True)
    res["changed"], _ = session(changed, seed, cfg, mem, artifacts, 2e5, True)
    mem.close()
    res["changed_control"], _ = session(changed, seed, cfg, None, artifacts, 2e5, False)
    return res


def summarise(rows: list[dict]) -> dict:
    stored = [r for r in rows if r["stored"]]

    def med(cond, key, rs):
        return float(np.median([r[cond][key] for r in rs])) if rs else None

    return {
        "scenes": len(rows), "stored_after_learning": len(stored),
        "learn": {"success": sum(r["learn"]["success"] for r in rows),
                  "median_probes": med("learn", "probes", rows),
                  "median_robot_s": med("learn", "robot_s", rows)},
        "reuse": {"recalled": sum(r["reuse"]["recalled"] for r in stored),
                  "success": sum(r["reuse"]["success"] for r in stored),
                  "zero_probe_successes": sum(r["reuse"]["success"] and r["reuse"]["probes"] == 0
                                              for r in stored),
                  "median_probes": med("reuse", "probes", stored),
                  "median_robot_s": med("reuse", "robot_s", stored),
                  "false_claims": sum(r["reuse"]["false_claim"] for r in stored)},
        "changed": {"withdrawn": sum(any(e["event"] == "recalled_procedure_withdrawn"
                                         for e in r["changed"]["experience"]) for r in stored),
                    "reinvestigated": sum(r["changed"]["investigations"] > 0 for r in stored),
                    "correct_refusals": sum(not r["changed"]["claimed"] for r in stored),
                    "false_claims": sum(r["changed"]["false_claim"] for r in stored),
                    "median_robot_s": med("changed", "robot_s", stored)},
        "changed_control": {"correct_refusals": sum(not r["changed_control"]["claimed"]
                                                    for r in stored),
                            "median_robot_s": med("changed_control", "robot_s", stored)},
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    from experiments.provenance import code_revision

    sha, dirty = code_revision()
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    store = pathlib.Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "DREAM/experience" / cfg["name"]
    store.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s, cfg, str(store)) for k in cfg["kinds"] for s in range(*cfg["seeds"])]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool, open(out / "scenes.jsonl", "w") as f:
        for r in pool.map(scene, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"{r['kind']} {r['seed']}: stored={r['stored']} "
                  f"learn={r['learn']['success']}/{r['learn']['probes']}p "
                  f"reuse={r['reuse']['success']}/{r['reuse']['probes']}p "
                  f"changed claimed={r['changed']['claimed']} inv={r['changed']['investigations']}",
                  flush=True)  # fmt: skip
    summary = summarise(rows)
    meta = {
        "config": cfg,
        "git_sha": sha,
        "dream_dirty_at_start": dirty,
        "perception": "oracle-assisted handle tracker (held-out compartment setting)",
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
