"""Bayesian bounded-lift response, not the gantry mechanism likelihood or mass estimation.

Two hypotheses: object follows an established finger contact, or no displacement is
resolved under this bounded action. A negative result never establishes universal immobility.
Gaussian response widths include perception and contact/model uncertainty; contact validity
is checked separately. This belief is session state, not persistent experience learning.
"""

from dataclasses import dataclass, field

import numpy as np


def entropy(p: float) -> float:
    return float(-p * np.log(max(p, 1e-15)) - (1 - p) * np.log(max(1 - p, 1e-15)))


@dataclass
class MovabilityBelief:
    p_movable: float = 0.5
    evidence: list[dict] = field(default_factory=list)
    inadequate: bool = False

    def posterior(self, displacement: np.ndarray | float, expected: float, sigma: float):
        y = np.asarray(displacement)
        log_odds = np.log(max(self.p_movable, 1e-12) / max(1 - self.p_movable, 1e-12))
        log_odds += (y**2 - (y - expected) ** 2) / (2 * sigma**2)
        return 1 / (1 + np.exp(-np.clip(log_odds, -30, 30)))

    def select(self, sigma: float, strategy: str = "eig") -> dict:
        # The same candidate set and budget for adaptive and fixed-short baselines.
        candidates = [0.012, 0.025]
        nodes, weights = np.polynomial.hermite.hermgauss(24)
        scores = []
        for amplitude in candidates:
            expected_entropy = 0.0
            for mean, prior in ((0, 1 - self.p_movable), (amplitude, self.p_movable)):
                y = mean + np.sqrt(2) * sigma * nodes
                post = self.posterior(y, amplitude, sigma)
                h = -post * np.log(np.maximum(post, 1e-15))
                h -= (1 - post) * np.log(np.maximum(1 - post, 1e-15))
                expected_entropy += prior * float(weights @ h / np.sqrt(np.pi))
            eig = max(0.0, entropy(self.p_movable) - expected_entropy)
            scores.append(
                {
                    "distance_m": amplitude,
                    "eig_nats": eig,
                    "utility": eig - 0.02 * amplitude / 0.025,
                }
            )
        best = max(scores, key=lambda s: s["utility"]) if strategy == "eig" else scores[0]
        return {
            **best,
            "candidates": scores,
            "strategy": strategy,
            "expected_displacement_m": {"responsive": best["distance_m"], "unresolved": 0},
            "sigma_m": sigma,
        }

    def update(self, measured: float, expected: float, sigma: float, contact_valid: bool) -> dict:
        before = self.p_movable
        if contact_valid and expected > 2 * sigma:
            if min(abs(measured), abs(measured - expected)) > 3 * sigma:
                self.inadequate = True
            else:
                self.p_movable = float(self.posterior(measured, expected, sigma))
        row = {
            "prior": before,
            "posterior": self.p_movable,
            "measured_displacement_m": measured,
            "expected_responsive_displacement_m": expected,
            "sigma_m": sigma,
            "contact_valid": contact_valid,
            "model_inadequate": self.inadequate,
            "update_accepted": contact_valid and expected > 2 * sigma and not self.inadequate,
        }
        self.evidence.append(row)
        return row
