"""Pure, preregistered summaries of mechanism-model inadequacy detection.

The input is one chi-squared-per-degree-of-freedom value at step 0 (no probes)
and after each executed probe. This module contains no simulator imports and
does not make deployment decisions; it only scores validation experiments.
"""

from __future__ import annotations

from collections.abc import Sequence


def alert_flags(
    scores: Sequence[float], threshold: float, min_probes: int = 3
) -> list[bool]:
    """Mark a model inadequate only after enough independent physical probes.

    A minimum of three observations is a *proposed* fix for early false alarms,
    not a calibrated guarantee. Step zero never generates an alert.
    """
    if min_probes < 1:
        raise ValueError("min_probes must be >= 1")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return [i >= min_probes and value > threshold for i, value in enumerate(scores)]


def summarize_detection(
    cases: Sequence[dict], threshold: float, min_probes: int
) -> dict:
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
        results[label] = {
            "n_cases": len(subset),
            "n_probe_steps": n_steps,
            "episode_alert_rate": sum(detected) / len(subset),
            "final_step_alert_rate": sum(final) / len(subset),
            "probe_step_alert_rate": sum(counts) / n_steps if n_steps else None,
            "first_alert_probe": first,
        }
    return results
