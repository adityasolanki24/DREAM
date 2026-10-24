"""Calibrate the model-adequacy threshold for Panda probes on in-class development scenes.

    python experiments/calibrate_panda_adequacy.py [--workers 4]

Same procedure as the gantry's (experiments/calibrate_adequacy.py): run investigations with
EIG and random probing on development compartment scenes whose mechanism is in the model class
(drawer, lid), with no early stopping and the adequacy check disabled, and record the chi2/dof
of the best explanation after every probe from ``MIN_EVIDENCE_PROBES`` on (the check is never
applied earlier). The threshold is the largest value times a 1.25 margin. Copy it into
``dream.tools.discovery.PANDA_ADEQUACY_THRESHOLD``. Evaluation scenes are never used here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.agent.belief import MIN_EVIDENCE_PROBES  # noqa: E402
from dream.sim.tabletop import sample_compartment_scene  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402
from dream.tools.discovery import investigate  # noqa: E402

DEV_SEEDS = range(100, 110)  # development scenes, disjoint from experiments/configs/*.json
KINDS = ("drawer", "lid")
BUDGET = 10
MARGIN = 1.25


def _job(args: tuple) -> dict:
    kind, seed, strategy = args
    spec = sample_compartment_scene(kind, np.random.default_rng(seed))
    env = TabletopEnv(spec, np.random.default_rng([seed, 1]))
    chi2 = []
    inv = investigate(
        env.embodiment(),
        env,
        "compartment",
        BUDGET,
        np.random.default_rng([seed, 2]),
        strategy=strategy,
        on_step=lambda step, belief: chi2.append(step["chi2_per_dof"]),
        adequacy_threshold=np.inf,
        stop_early=False,
    )
    return {
        "kind": kind,
        "seed": seed,
        "strategy": strategy,
        "status": inv.status,
        "mechanism": inv.mechanism,
        "chi2_per_dof": chi2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [(k, s, m) for k in KINDS for s in DEV_SEEDS for m in ("eig", "random")]
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(_job, jobs))
    values = np.concatenate([r["chi2_per_dof"][MIN_EVIDENCE_PROBES - 1 :] for r in runs])
    threshold = float(MARGIN * values.max())
    out = ROOT / "results" / "panda_adequacy_calibration"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "mujoco": mujoco.__version__,
            "dev_seeds": list(DEV_SEEDS),
            "kinds": list(KINDS),
            "budget": BUDGET,
            "min_evidence_probes": MIN_EVIDENCE_PROBES,
        },
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
