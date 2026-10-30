"""CPU development acceptance of the actual RGB-D/WBC/navigation/verification loop.

The deterministic planner here is a test baseline, never described as model inference.
Original room generation, controller, sensor noise and independent evaluator are reused.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from dream.perception.spatial import SpatialMemory
from dream.robot.g1_wbc import WBCLocomotion
from dream.runtime.agent import Finish, TaskAgent
from dream.runtime.mobile_model import available_actions
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1
from dream.sim.room import add_room, sample_room
from dream.tools.goals import Inspect
from dream.tools.mobile import MobileToolbox
from experiments.g1_explore_room import Budgeted, Monitor, score


class RecoveryBaseline:
    def next_action(self, goal, scene, verdict, history):
        if verdict.get("satisfied"):
            return Finish(True, "measured inspection complete")
        actions = available_actions(goal, scene, verdict, history)
        if not actions:
            return Finish(False, "admissible search exhausted")
        inspections = [a for a in actions if a.name == "inspect"]
        if inspections:
            return inspections[0]
        inspection_vps = [v for v in scene["viewpoints"] if v["purpose"] == "inspection"]
        for v in sorted(inspection_vps, key=lambda v: v["cost_m"]):
            a = next(
                (
                    a
                    for a in actions
                    if a.name == "approach"
                    and a.args == {"candidate": v["candidate"], "side": v["side"]}
                ),
                None,
            )
            if a is not None:
                return a
        if any(a.name == "look_around" for a in actions):
            return next(a for a in actions if a.name == "look_around")
        return actions[0]


def run(seed, out):
    out.mkdir(parents=True, exist_ok=False)
    room = sample_room(np.random.default_rng([seed, 0]))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([seed, 1])
    )
    env.reset(*room.start)
    monitor = Monitor(env)
    path = []

    def tick():
        monitor.tick()
        if not path or env.time - path[-1]["sim_s"] >= 0.1:
            path.append({"sim_s": env.time, "pose": env.true_pose().tolist()})

    base = G1MobileBase(env, WBCLocomotion.from_cache(), tick)
    memory = SpatialMemory(np.random.default_rng([seed, 2]))
    records = []

    def event(kind, text, **data):
        records.append({"sim_s": base.time, "kind": kind, "text": text, **data})

    toolbox = MobileToolbox(base, memory, event, navigator="rpp")
    start = time.perf_counter()
    ep = TaskAgent(toolbox, Budgeted(RecoveryBaseline(), base, 180), max_steps=24).run(
        Inspect("cabinet"), on_event=lambda kind, data: event(kind, "", **data)
    )
    result = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "local_modified_tree": True,
        "seed": seed,
        "model_calls": 0,
        "scope": "CPU development physics acceptance; deterministic test baseline",
        "episode": ep.to_dict(),
        "independent_verification": score(env, room, toolbox, ep.success_claimed),
        "sim_s": base.time,
        "wall_s": time.perf_counter() - start,
        "stop_reason": base.stop_reason,
        "contact_steps": monitor.contact_steps,
        "camera_frames": memory.frames,
        "path_m": monitor.path_m,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "events.json").write_text(json.dumps(records, indent=2, default=str))
    (out / "path.json").write_text(json.dumps(path, indent=2))
    env.renderer.close() if getattr(env, "renderer", None) else None
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    args = parser.parse_args()
    for seed in args.seeds:
        print(json.dumps(run(seed, args.out / f"seed-{seed}")), flush=True)


if __name__ == "__main__":
    main()
