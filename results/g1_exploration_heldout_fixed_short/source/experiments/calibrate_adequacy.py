"""Calibrate the model-adequacy threshold on development mechanism families only.

    python experiments/calibrate_adequacy.py [--workers 4]

Runs EIG and random episodes on in-class development families (a seed disjoint from the
benchmarks) and records the chi2/dof of the best explanation after every probe. The threshold is
the largest value observed times a 1.25 margin, so there are no false alarms on calibration data
by construction. Copy the printed value into ``dream.agent.belief.ADEQUACY_THRESHOLD``. Held-out
and out-of-class mechanisms are never used here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.agent.exploration import EIGSelector, RandomSelector, candidate_probes  # noqa: E402
from dream.eval.episode import run_episode  # noqa: E402
from dream.sim.mechanisms import DEVELOPMENT_FAMILIES, sample_mechanism  # noqa: E402

SEED = 7  # disjoint from experiments/configs/*.json
N_MECHANISMS = 6
BUDGET = 8
MARGIN = 1.25


def _job(args: tuple) -> dict:
    family, i, strategy = args
    spec = sample_mechanism(family, np.random.default_rng([SEED, i]))
    sel = (EIGSelector if strategy == "eig" else RandomSelector)(candidate_probes())
    chi2 = []
    run_episode(
        spec,
        sel,
        BUDGET,
        seed=1000 * SEED + i,
        evaluate=False,
        on_step=lambda step, belief, info: chi2.append(belief.fit_check()["chi2_per_dof"]),
    )
    return {"family": family, "mechanism": i, "strategy": strategy, "chi2_per_dof": chi2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [
        (f, i, s)
        for f in DEVELOPMENT_FAMILIES
        for i in range(N_MECHANISMS)
        for s in ("eig", "random")
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(_job, jobs))
    values = np.concatenate([r["chi2_per_dof"] for r in runs])
    threshold = float(MARGIN * values.max())
    out = ROOT / "results" / "adequacy_calibration"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": SEED,
        "families": list(DEVELOPMENT_FAMILIES),
        "n_episodes": len(runs),
        "n_values": int(values.size),
        "percentiles": {str(q): float(np.percentile(values, q)) for q in (50, 90, 99, 100)},
        "margin": MARGIN,
        "threshold": threshold,
        "runs": runs,
    }
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
