"""Aggregate episode logs into the benchmark's metrics and statistical comparisons."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def probes_to_identification(log: dict) -> int | None:
    """Smallest t such that the mechanism stays identified from step t to the end of the budget."""
    ids = [s["identified"] for s in log["steps"]]
    if not ids[-1]:
        return None
    t = len(ids) - 1
    while t > 0 and ids[t - 1]:
        t -= 1
    return t


def _censored(log: dict, budget: int) -> int:
    n = probes_to_identification(log)
    return budget + 1 if n is None else n


def _curve(logs: list[dict], key: str) -> list[float | None]:
    n_steps = len(logs[0]["steps"])
    out = []
    for t in range(n_steps):
        vals = [lg["steps"][t][key] for lg in logs if lg["steps"][t].get(key) is not None]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip test on paired differences a - b (H0: mean difference is 0)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(d == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def bootstrap_ci(x: np.ndarray, stat=np.mean, n: int = 5000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    boots = [stat(rng.choice(x, size=len(x))) for _ in range(n)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _strategy_stats(logs: list[dict], budget: int) -> dict:
    n_id = np.array([_censored(lg, budget) for lg in logs])
    return {
        "n_episodes": len(logs),
        "median_probes_to_id": float(np.median(n_id)),
        "mean_probes_to_id": float(n_id.mean()),
        "mean_probes_to_id_ci95": bootstrap_ci(n_id),
        "identified_by_budget": float(np.mean(n_id <= budget)),
        "id_rate_curve": _curve(logs, "identified"),
        "task_success_curve": _curve(logs, "task_success"),
        "heldout_err_mm_curve": [
            None if v is None else 1000 * v for v in _curve(logs, "heldout_pos_err_m")
        ],
        "type_entropy_curve": _curve(logs, "type_entropy"),
        "p_true_type_curve": _curve(logs, "p_true_type"),
        "mean_select_s": float(
            np.mean([s.get("select_s", 0) for lg in logs for s in lg["steps"][1:]])
        ),
    }


def _family_stats(logs: list[dict], budget: int) -> dict:
    final = [lg["steps"][-1] for lg in logs]
    return {
        "mean_probes_to_id": float(np.mean([_censored(lg, budget) for lg in logs])),
        "identified_by_budget": float(
            np.mean([probes_to_identification(lg) is not None for lg in logs])
        ),
        "final_task_success": float(np.mean([s["task_success"] for s in final])),
        "final_flagged_inadequate": float(
            np.mean([not s.get("model_adequate", True) for s in final])
        ),
        "final_max_type_prob": float(np.mean([s.get("max_type_prob", np.nan) for s in final])),
    }


def summarise(logs: list[dict], budget: int, reference: str = "eig") -> dict:
    """Per-strategy metrics overall and per family, plus paired tests against ``reference``.

    Probes-to-identification is censored at ``budget + 1`` for episodes never identified.
    """
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for lg in logs:
        by_strat[lg["strategy"]].append(lg)
    overall = {s: _strategy_stats(lgs, budget) for s, lgs in by_strat.items()}

    families = sorted({lg["family"] for lg in logs})
    per_family = {
        fam: {
            s: _family_stats([lg for lg in lgs if lg["family"] == fam], budget)
            for s, lgs in by_strat.items()
        }
        for fam in families
    }

    comparisons = {}
    if reference in by_strat:
        key = lambda lg: (lg["family"], lg["mechanism_seed"])  # noqa: E731
        ref = {key(lg): _censored(lg, budget) for lg in by_strat[reference]}
        for s, lgs in by_strat.items():
            if s == reference:
                continue
            pairs = [(ref[key(lg)], _censored(lg, budget)) for lg in lgs if key(lg) in ref]
            a, b = np.array(pairs).T
            comparisons[f"{reference}_vs_{s}"] = {
                "n_pairs": len(pairs),
                f"mean_{reference}": float(a.mean()),
                f"mean_{s}": float(b.mean()),
                "median_reduction_pct": float(100 * (np.median(b) - np.median(a)) / np.median(b)),
                "mean_reduction_pct": float(100 * (b.mean() - a.mean()) / b.mean()),
                "p_value_paired_permutation": paired_permutation_test(a, b),
                f"{reference}_better": int(np.sum(a < b)),
                "ties": int(np.sum(a == b)),
                f"{reference}_worse": int(np.sum(a > b)),
            }
    return {"overall": overall, "per_family": per_family, "comparisons": comparisons}
