"""Test a precommitted minimum-evidence gate on unseen mechanism instances.

Usage:
    python experiments/validate_adequacy.py --workers 4

Do NOT tune THRESHOLD or MIN_PROBES on this validation data. Uses fresh
random streams and the SAME frozen source-code threshold as the published
calibration, and writes all results including false alarms.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.agent.belief import ADEQUACY_THRESHOLD  # noqa: E402
from dream.agent.exploration import EIGSelector, candidate_probes  # noqa: E402
from dream.eval.adequacy import summarize_detection  # noqa: E402
from dream.eval.episode import run_episode  # noqa: E402
from dream.sim.mechanisms import DEVELOPMENT_FAMILIES, sample_mechanism  # noqa: E402

SEED = 14819  # never used in earlier calibration/development results
N_PER_FAMILY = 12
BUDGET = 8
MIN_PROBES = 3  # fixed before running this validation, not optimized here


def _job(args: tuple[str, int]) -> dict:
    family, index = args
    spec = sample_mechanism(family, np.random.default_rng([SEED, index, len(family)]))
    log = run_episode(
        spec,
        EIGSelector(candidate_probes()),
        BUDGET,
        seed=SEED * 100 + index + 1000 * len(family),
        evaluate=False,
    )
    return {
        "family": family,
        "mechanism_index": index,
        "out_of_class": family == "rising_hinge",
        "scores": [float(s["chi2_per_dof"]) for s in log["steps"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    families = (*DEVELOPMENT_FAMILIES, "sash", "pivot", "rising_hinge")
    jobs = [(family, i) for family in families for i in range(N_PER_FAMILY)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(_job, jobs))

    output = {
        "seed": SEED,
        "n_per_family": N_PER_FAMILY,
        "budget": BUDGET,
        "threshold": ADEQUACY_THRESHOLD,
        "new_min_probes": MIN_PROBES,
        "unmodified": summarize_detection(cases, ADEQUACY_THRESHOLD, 1),
        "proposed_gate": summarize_detection(cases, ADEQUACY_THRESHOLD, MIN_PROBES),
        "cases": cases,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    destination = ROOT / "results" / "fresh_adequacy_validation"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
