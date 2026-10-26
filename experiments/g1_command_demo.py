"""The end-to-end command scenario (experiments/command_scenario.py) in a MuJoCo room, recorded.

    python experiments/g1_command_demo.py --seed 4004 --out results/<name> [--record <dir>]

A demonstration on a development room, not an evaluation. The Isaac Sim kitchen version is
``experiments/isaac_g1_kitchen.py --mode commands``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tempfile

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.memory.home import HomeMemory  # noqa: E402
from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from experiments.command_scenario import run_commands  # noqa: E402
from experiments.g1_explore_room import Monitor, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4004)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--record", help="external directory for the full-state recording")
    ap.add_argument("--time-budget-s", type=float, default=240)
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()
    cfg = {"time_budget_s": args.time_budget_s, "max_steps": args.max_steps}
    room = sample_room(np.random.default_rng([args.seed, 0]))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([args.seed, 1])
    )
    env.reset(*room.start)
    monitor = Monitor(env)
    rec = None
    if args.record:
        from experiments.mobile_recording import MobileRecorder

        rec = MobileRecorder(env, pathlib.Path(args.record) / f"commands_{args.seed}",
                             {"experiment": "g1_command_demo", "seed": args.seed})  # fmt: skip

    def on_tick():
        monitor.tick()
        if rec:
            rec.sample()

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick)
    events = []

    def log(kind, text, **data):
        events.append({"time_s": env.time, "kind": kind, "text": text, **data})
        if rec:
            rec.event(kind, text)

    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([args.seed, 2])),
                       navigator="rpp", on_event=lambda k, t, **_: log(k, t))  # fmt: skip

    def scorer(task, claimed):
        if task == "goto entrance":
            x, y, _ = env.true_pose()
            err = math.hypot(x - room.start[0], y - room.start[1])
            return {"success": claimed and err <= 0.3, "entrance_error_m": err}
        return score(env, room, tb, claimed)

    with tempfile.TemporaryDirectory() as store:
        home = HomeMemory(pathlib.Path(store) / "home.sqlite")
        result = run_commands(env, base, tb, home, f"odom:room{args.seed}", cfg, scorer, log,
                              pathlib.Path(store))  # fmt: skip
        home.close()
    result.update(seed=args.seed, sim_s=env.time, fell=base.stop_reason is not None,
                  contact_steps=monitor.contact_steps, path_m=monitor.path_m)  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    with open(args.out / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, default=float) + "\n")
    if rec:
        rec.save({k: v for k, v in result.items() if k != "tasks"})
    print(json.dumps({k: result[k] for k in ("verdicts", "sim_s", "fell", "contact_steps")},
                     indent=2, default=float))  # fmt: skip


if __name__ == "__main__":
    main()
