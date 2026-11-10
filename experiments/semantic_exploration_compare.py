"""Paired comparison of search arms with and without the unexplored-room option.

    python experiments/semantic_exploration_compare.py DIR [DIR ...] [--out compare.json]
        [--budget 600] [--pairs a:b,c:d]

Read-only analysis of episodes.jsonl files (rows from several directories are pooled; an arm
name must be unique per pooled home). Pairs are identical homes (scenario, target, generator,
noise, seed) where the target exists, scenarios B and D. Time to verified find is censored at
the budget for misses. For each pair ``a:b``: successes, only-a / only-b finds, median and mean
paired time difference (negative: a faster), a bootstrap 95% interval of the median difference
(2000 resamples, seed 0), the one-sided Wilcoxon p that a is faster and the two-sided p, the
mean path length and the share of homes where the two runs were identical to the second.

This is the analysis the K2 primary gate uses (experiments/configs/semantic_exploration_k2.json)
and the development comparison of the same options; it reports and does not decide."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.semantic_knowledge_gates import holm, one_sided_wilcoxon, pair_arrays  # noqa: E402
from experiments.summarise_semantic_search import load  # noqa: E402

DEFAULT_PAIRS = (
    "sem_none:A_frontier", "sem_conceptnet:A_frontier", "sem_qwen:A_frontier",
    "sem_combined:A_frontier", "sem_conceptnet_unexp:sem_conceptnet",
    "sem_qwen_unexp:sem_qwen", "sem_combined_unexp:sem_combined",
    "sem_conceptnet_unexp:A_frontier", "sem_qwen_unexp:A_frontier",
    "sem_combined_unexp:A_frontier", "sem_qwen_unexp:sem_none", "sem_conceptnet_unexp:sem_none",
)  # fmt: skip


def bootstrap_median(diff: np.ndarray, n: int = 2000, seed: int = 0) -> list[float] | None:
    if not len(diff):
        return None
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(diff, len(diff)))) for _ in range(n)]
    return [round(float(np.percentile(meds, 2.5)), 1), round(float(np.percentile(meds, 97.5)), 1)]


def compare(rows: list[dict], a: str, b: str, budget: float, scenarios=("B", "D")) -> dict:
    ta, tb, sa, sb = pair_arrays(rows, a, b, budget, scenarios)
    diff = ta - tb
    paths = {}
    for arm in (a, b):
        xs = [r.get("path_m", 0.0) for r in rows if r["arm_name"] == arm
              and r.get("target_exists") and r["scenario"] in scenarios]  # fmt: skip
        paths[arm] = round(float(np.mean(xs)), 1) if xs else None
    nonzero = diff[diff != 0]
    two = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    return {
        "pairs": len(diff), "found": [int(sa.sum()), int(sb.sum())],
        "only_a_found": int((sa & ~sb).sum()), "only_b_found": int((~sa & sb).sum()),
        "median_paired_diff_s": round(float(np.median(diff)), 1) if len(diff) else None,
        "mean_paired_diff_s": round(float(np.mean(diff)), 1) if len(diff) else None,
        "median_diff_ci95": bootstrap_median(diff),
        "a_faster": int((diff < 0).sum()), "b_faster": int((diff > 0).sum()),
        "identical": int((diff == 0).sum()),
        "p_one_sided_a_faster": one_sided_wilcoxon(diff) if len(diff) else 1.0,
        "p_two_sided": two, "mean_path_m": paths,
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--pairs", help="comma-separated a:b pairs (default: the standard set)")
    args = ap.parse_args()
    rows = load(args.dirs)
    present = {r["arm_name"] for r in rows}
    pairs = [p.split(":") for p in (args.pairs.split(",") if args.pairs else DEFAULT_PAIRS)]
    pairs = [(a, b) for a, b in pairs if a in present and b in present]
    out = {"arms": sorted(present), "episodes": len(rows), "budget_s": args.budget,
           "pairs": {f"{a} vs {b}": compare(rows, a, b, args.budget)
                     for a, b in pairs}}  # fmt: skip
    raw = {k: v["p_one_sided_a_faster"] for k, v in out["pairs"].items()}
    out["holm_over_listed_pairs"] = holm(raw, 0.05) if raw else {}
    text = json.dumps(out, indent=1, default=float)
    if args.out:
        pathlib.Path(args.out).write_text(text)
    for k, v in out["pairs"].items():
        print(
            f"{k:42s} n={v['pairs']} found={v['found']} med={v['median_paired_diff_s']} "
            f"mean={v['mean_paired_diff_s']} ci={v['median_diff_ci95']} "
            f"faster={v['a_faster']}/{v['b_faster']} same={v['identical']} "
            f"p1={v['p_one_sided_a_faster']:.3f} path={v['mean_path_m']}"
        )


if __name__ == "__main__":
    main()
