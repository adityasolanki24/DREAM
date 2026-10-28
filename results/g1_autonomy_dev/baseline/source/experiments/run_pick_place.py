"""Milestone 7 acceptance test: pick a named cube and place it in the tray, on held-out scenes.

    python experiments/run_pick_place.py [--n 50] [--seed0 1000] [--workers 4]

Seeds 0-39 were used while developing the controller and skills. The default seeds start at
1000. Success is judged by dream.eval.manipulation from true object poses, independently of
the skills' own success flags. Writes results/pick_place/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.adequacy import wilson_interval  # noqa: E402
from dream.eval.manipulation import object_in_tray, placement_error, safety_summary  # noqa: E402
from dream.sim.tabletop import sample_tabletop  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402
from dream.skills.manipulation import pick, place  # noqa: E402


def episode(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    spec = sample_tabletop(rng)
    env = TabletopEnv(spec, np.random.default_rng([seed, 1]))
    robot = env.embodiment()
    name = str(rng.choice([o.name for o in spec.objects]))
    centre, _ = env.fixtures()["tray"]
    target = centre + [0, 0, 0.02]  # 4 cm cubes; centre z is the floor top
    t0 = time.perf_counter()
    grasp = pick(robot, env, name)
    out = {"seed": seed, "object": name, "grasp_ok": grasp.ok, "grasp_reason": grasp.reason}
    if grasp.ok:
        placed = place(robot, env, name, target)
        out.update(place_ok=placed.ok, place_reason=placed.reason)
    env.step(250)  # let everything settle before judging
    out.update(
        success=object_in_tray(env, name),
        placement_error_m=placement_error(env, name, target),
        sim_time_s=env.time,
        wall_time_s=time.perf_counter() - t0,
        **safety_summary(env),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    seeds = list(range(args.seed0, args.seed0 + args.n))
    with ProcessPoolExecutor(args.workers) as pool:
        eps = list(pool.map(episode, seeds))
    k = sum(e["success"] for e in eps)
    errs = [e["placement_error_m"] for e in eps if e["success"]]
    summary = {
        "n": len(eps),
        "success": k,
        "success_rate": k / len(eps),
        "success_ci95": wilson_interval(k, len(eps)),
        "grasp_success_rate": float(np.mean([e["grasp_ok"] for e in eps])),
        "median_placement_error_mm": 1000 * float(np.median(errs)) if errs else None,
        "max_placement_error_mm": 1000 * float(np.max(errs)) if errs else None,
        "protective_stops": int(sum(e["protective_stop"] for e in eps)),
        "link_collisions": int(sum(e["link_collision"] for e in eps)),
        "mean_sim_time_s": float(np.mean([e["sim_time_s"] for e in eps])),
        "failures": [e for e in eps if not e["success"]],
    }
    out = ROOT / "results" / "pick_place"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "seeds": [seeds[0], seeds[-1]],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "mujoco": mujoco.__version__,
    }
    (out / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2))
    (out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    print(json.dumps({**meta, **{k: v for k, v in summary.items() if k != "failures"}}, indent=2))
    for f in summary["failures"]:
        print("FAIL", f)


if __name__ == "__main__":
    main()
