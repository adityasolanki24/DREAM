"""Pure, preregistered summaries of mechanism-model inadequacy detection.

The input is one chi-squared-per-degree-of-freedom value at step 0 (no probes)
and after each executed probe. This module contains no simulator imports and
does not make deployment decisions; it only scores validation experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import median


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score 95% interval for a binomial proportion k/n (supplementary reporting)."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def alert_flags(scores: Sequence[float], threshold: float, min_probes: int = 3) -> list[bool]:
    """Mark a model inadequate only after enough independent physical probes.

    A minimum of three observations is a *proposed* fix for early false alarms,
    not a calibrated guarantee. Step zero never generates an alert.
    """
    if min_probes < 1:
        raise ValueError("min_probes must be >= 1")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return [i >= min_probes and value > threshold for i, value in enumerate(scores)]


def summarize_detection(cases: Sequence[dict], threshold: float, min_probes: int) -> dict:
    """Aggregate independent mechanism instances, not individual timesteps.

    Each case contains 'scores' (including step zero) and 'out_of_class'. The
    primary statistic is the fraction of entire episodes with any alarm.
    """
    results = {}
    for is_ood, label in ((False, "in_class"), (True, "out_of_class")):
        subset = [case for case in cases if bool(case["out_of_class"]) == is_ood]
        if not subset:
            results[label] = None
            continue
        counts = []
        detected = []
        final = []
        first = []
        for case in subset:
            flags = alert_flags(case["scores"], threshold, min_probes)
            counts.append(sum(flags[1:]))
            detected.append(any(flags))
            final.append(flags[-1])
            first.append(next((i for i, flag in enumerate(flags) if flag), None))
        n_steps = sum(len(case["scores"]) - 1 for case in subset)
        latencies = [f for f in first if f is not None]
        results[label] = {
            "n_cases": len(subset),
            "n_probe_steps": n_steps,
            "episode_alert_rate": sum(detected) / len(subset),
            "episode_alert_rate_ci95": wilson_interval(sum(detected), len(subset)),
            "median_first_alert_probe": median(latencies) if latencies else None,
            "final_step_alert_rate": sum(final) / len(subset),
            "probe_step_alert_rate": sum(counts) / n_steps if n_steps else None,
            "first_alert_probe": first,
        }
    return results
