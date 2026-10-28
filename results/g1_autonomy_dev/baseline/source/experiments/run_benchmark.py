"""Benchmark exploration strategies on identical mechanisms with identical probe budgets.

Usage:
    python experiments/run_benchmark.py experiments/configs/main.json [--workers 4]

Writes results/<config name>/: episodes.jsonl (every step of every episode), summary.json,
meta.json (git SHA, versions, config) and plots.
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

from dream.agent.exploration import (  # noqa: E402
    EIGSelector,
    RandomSelector,
    axis_sweep_selector,
    candidate_probes,
    goal_directed_selector,
)
from dream.eval.episode import run_episode  # noqa: E402
from dream.eval.plots import plot_summary  # noqa: E402
from dream.eval.summary import probes_to_identification, summarise  # noqa: E402
from dream.sim.mechanisms import sample_mechanism  # noqa: E402


def make_selector(name: str):
    probes = candidate_probes()
    return {
        "eig": lambda: EIGSelector(probes),
        "random": lambda: RandomSelector(probes),
        "goal_directed": goal_directed_selector,
        "axis_sweep": axis_sweep_selector,
    }[name]()


def _job(args: tuple) -> dict:
    family, mech_seed, strategy, cfg = args
    spec = sample_mechanism(
        family, np.random.default_rng([cfg["seed"], mech_seed]), **cfg["mechanism"]
    )
    t0 = time.perf_counter()
    log = run_episode(
        spec,
        make_selector(strategy),
        budget=cfg["budget"],
        seed=1000 * cfg["seed"] + mech_seed,
        n_particles=cfg["n_particles"],
        prior_kwargs=cfg.get("prior", {}),
    )
    log["mechanism_seed"] = mech_seed
    log["wall_s"] = time.perf_counter() - t0
    return log


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": cfg,
        "git_sha": _git_sha(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    jobs = [
        (fam, s, strat, cfg)
        for fam in cfg["families"]
        for s in range(cfg["n_mechanisms"])
        for strat in cfg["strategies"]
    ]
    t0 = time.time()
    logs = []
    with ProcessPoolExecutor(args.workers) as pool, open(out / "episodes.jsonl", "w") as f:
        for i, log in enumerate(pool.map(_job, jobs), 1):
            logs.append(log)
            f.write(json.dumps(log) + "\n")
            f.flush()
            print(
                f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {log['family']} "
                f"#{log['mechanism_seed']} {log['strategy']}: "
                f"identified at {probes_to_identification(log)}",
                flush=True,
            )
    summary = summarise(logs, cfg["budget"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_summary(logs, summary, cfg["budget"], out)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
