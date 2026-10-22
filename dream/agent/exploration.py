"""Probe selection: expected information gain (EIG) versus a random baseline.

Both selectors choose from the same finite candidate set, so any difference between them comes
from *which* experiment is chosen, not from what experiments are available.

EIG estimator: draw M hypotheses from the posterior and predict each one's outcome y_i under
probe a. The outcome distribution is then a Gaussian mixture, and the mutual information
I(Y; theta) is estimated with the pairwise-distance estimator of Kolchinsky & Tracey (2017):
    I_hat = -1/M sum_i log( 1/M sum_j exp(-D(p_i || p_j)) ),
with D the KL divergence between equal-covariance Gaussians, 0.5 * ||(y_i - y_j) / sigma||^2.
The estimate is an upper bound on I that is tight when components are well separated, and it is
capped at log M.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from dream.agent.belief import Belief
from dream.agent.model import Hypotheses, ProbeController, predict_outcome
from dream.types import Probe


def candidate_probes(
    distances: tuple[float, ...] = (0.03, 0.08),
    forces: tuple[float, ...] = (6.0, 15.0),
) -> list[Probe]:
    """26 directions (cube faces, edges, corners) x distances x force limits."""
    dirs = [np.array(d, float) for d in itertools.product((-1, 0, 1), repeat=3) if any(d)]
    return [
        Probe(direction=d / np.linalg.norm(d), distance=dist, max_force=f)
        for d in dirs
        for dist in distances
        for f in forces
    ]


def predict_for_probes(
    hyps: Hypotheses,
    h0: np.ndarray,
    handle_now: np.ndarray,
    ee_now: np.ndarray,
    probes: list[Probe],
    controller: ProbeController | None = None,
) -> np.ndarray:
    """Predicted outcomes and start states for every (probe, hypothesis) pair, each (K, M, 6)."""
    m, k = len(hyps), len(probes)
    tiled = hyps.take(np.tile(np.arange(m), k))
    dirs = np.repeat(np.stack([p.direction for p in probes]), m, axis=0)
    dist = np.repeat([p.distance for p in probes], m)
    force = np.repeat([p.max_force for p in probes], m)
    y, _, y_start = predict_outcome(
        tiled, h0, handle_now, ee_now, dirs, dist, force, controller=controller
    )
    return y.reshape(k, m, 6), y_start.reshape(k, m, 6)


def pairwise_mixture_mi(y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Kolchinsky-Tracey estimator of I(Y; theta) for each probe.

    ``y`` and ``sigma`` are (K, M, D): predicted outcomes and their per-sample noise widths.
    For two components with different widths, the pairwise divergence is approximated by
    0.5 * sum_d (y_i - y_j)^2 / ((s_i^2 + s_j^2) / 2), a symmetrised KL that ignores the
    variance-ratio term.
    """
    out = np.empty(y.shape[0])
    for k in range(y.shape[0]):
        diff2 = (y[k, :, None, :] - y[k, None, :, :]) ** 2
        var = 0.5 * (sigma[k, :, None, :] ** 2 + sigma[k, None, :, :] ** 2)
        log_kernel = -0.5 * np.sum(diff2 / var, axis=-1)
        mx = log_kernel.max(axis=-1, keepdims=True)
        lse = np.log(np.mean(np.exp(log_kernel - mx), axis=-1)) + mx[:, 0]
        out[k] = -np.mean(lse)
    return out


@dataclass
class EIGSelector:
    """Pick the probe with the highest expected information gain, minus a small effort cost.

    ``effort_cost`` is in nats per (max_force / 15 N) x (distance / 0.08 m). It breaks ties in
    favour of gentle, short probes, which is the report's force and motion penalty.
    """

    probes: list[Probe]
    n_samples: int = 128
    effort_cost: float = 0.05
    name: str = "eig"

    def select(
        self, belief: Belief, handle_now: np.ndarray, ee_now: np.ndarray, rng: np.random.Generator
    ) -> tuple[Probe, dict]:
        hyps = belief.sample(self.n_samples, rng)
        y, y_start = predict_for_probes(
            hyps, belief.h0, handle_now, ee_now, self.probes, belief.controller
        )
        eig = pairwise_mixture_mi(y, belief.noise.sigma_for(y, y_start))
        effort = np.array([p.max_force / 15.0 * p.distance / 0.08 for p in self.probes])
        score = eig - self.effort_cost * effort
        best = int(np.argmax(score))
        return self.probes[best], {"eig": float(eig[best]), "eig_all": eig}


@dataclass
class RandomSelector:
    """Uniformly random probe from the same candidate set (the baseline)."""

    probes: list[Probe]
    name: str = "random"

    def select(
        self, belief: Belief, handle_now: np.ndarray, ee_now: np.ndarray, rng: np.random.Generator
    ) -> tuple[Probe, dict]:
        return self.probes[int(rng.integers(len(self.probes)))], {}


@dataclass
class ScriptedSelector:
    """A fixed probe sequence, repeated (a non-adaptive baseline).

    ``goal_directed``: always pull straight out of the panel, which is a direct attempt to
    open it (the report's "direct goal-seeking manipulation" baseline).
    ``axis_sweep``: cycle through the six signed axis directions at full force. This is a
    sensible hand-designed exploration script.
    """

    sequence: list[Probe]
    name: str

    def select(
        self, belief: Belief, handle_now: np.ndarray, ee_now: np.ndarray, rng: np.random.Generator
    ) -> tuple[Probe, dict]:
        probe = self.sequence[len(belief.history) % len(self.sequence)]
        return probe, {}


def goal_directed_selector(distance: float = 0.08, max_force: float = 15.0) -> ScriptedSelector:
    return ScriptedSelector(
        [Probe(np.array([0.0, -1.0, 0.0]), distance, max_force)], "goal_directed"
    )


def axis_sweep_selector(distance: float = 0.08, max_force: float = 15.0) -> ScriptedSelector:
    order = ([0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0])
    return ScriptedSelector(
        [Probe(np.array(d, float), distance, max_force) for d in order], "axis_sweep"
    )
