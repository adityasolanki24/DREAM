"""Bayesian belief over mechanism hypotheses.

Structure: one sequential Monte Carlo (SMC) particle set per joint type, each with its own
running estimate of the model evidence p(data | type). The type posterior is
``prior(type) * evidence(type)``. Keeping the types in separate particle sets means a type
cannot vanish through resampling degeneracy, so a probability like P(fixed) stays meaningful.

Each new observation is assimilated with adaptive likelihood tempering. A record's likelihood
is applied in fractions that keep the effective sample size above a threshold, with resampling
and Metropolis-Hastings rejuvenation (targeting prior x full-history likelihood) between the
fractions. This is standard resample-move SMC (Chopin 2002; Del Moral et al. 2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dream.agent.model import FIXED, PRISMATIC, REVOLUTE, TYPE_NAMES, Hypotheses, predict_outcome
from dream.types import PartGeometry


@dataclass(frozen=True)
class Record:
    """One probe's evidence: the start state, the command, and the settled observation."""

    handle_before: np.ndarray  # (3,) observed
    ee_before: np.ndarray  # (3,) observed
    direction: np.ndarray  # (3,)
    distance: float
    max_force: float
    y_obs: np.ndarray  # (6,) handle position and rotation vector relative to the closed state


@dataclass(frozen=True)
class NoiseModel:
    """Observation likelihood: Gaussian with a broad outlier component.

    The sigmas cover both sensor noise (1 mm, 0.5 deg) and measured model mismatch (about
    0.2-2 mm, with rare ~1 cm outliers; see tests/test_model.py).
    """

    sigma_pos: float = 0.003  # m
    sigma_rot: float = np.deg2rad(2.0)  # rad
    outlier_prob: float = 0.03
    outlier_scale: float = 10.0

    @property
    def sigma(self) -> np.ndarray:
        return np.array([self.sigma_pos] * 3 + [self.sigma_rot] * 3)

    def loglik(self, y_pred: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        z2 = np.sum(((y_pred - y_obs) / self.sigma) ** 2, axis=-1)
        d = y_obs.shape[-1]
        log_in = -0.5 * z2 + np.log1p(-self.outlier_prob)
        log_out = -0.5 * z2 / self.outlier_scale**2 + np.log(self.outlier_prob)
        log_out -= d * np.log(self.outlier_scale)
        norm = -0.5 * d * np.log(2 * np.pi) - np.sum(np.log(self.sigma))
        return np.logaddexp(log_in, log_out) + norm


@dataclass(frozen=True)
class Prior:
    """Uninformative prior over hypotheses, given only the visible part geometry."""

    geometry: PartGeometry
    prismatic_upper: tuple[float, float] = (0.05, 0.4)  # m
    revolute_upper: tuple[float, float] = (0.3, 2.0)  # rad
    friction: tuple[float, float] = (0.0, 5.0)  # N
    origin_margin: float = 0.1  # m beyond the panel in x and z, and in front of or behind it in y

    def _origin_box(self) -> tuple[np.ndarray, np.ndarray]:
        c, half = self.geometry.panel_center, self.geometry.panel_half_extents
        span = np.array(
            [half[0] + self.origin_margin, self.origin_margin, half[2] + self.origin_margin]
        )
        return c - span, c + span

    def _upper_range(self, kind: int) -> tuple[float, float]:
        return self.prismatic_upper if kind == PRISMATIC else self.revolute_upper

    def sample(self, kind: int, n: int, rng: np.random.Generator) -> Hypotheses:
        axis = rng.normal(size=(n, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        lo, hi = self._origin_box()
        upper = rng.uniform(*self._upper_range(kind), size=n) if kind != FIXED else np.zeros(n)
        return Hypotheses(
            kind=np.full(n, kind),
            axis=axis if kind != FIXED else np.zeros((n, 3)),
            origin=rng.uniform(lo, hi, size=(n, 3)),
            upper=upper,
            friction=rng.uniform(*self.friction, size=n),
        )

    def in_support(self, h: Hypotheses) -> np.ndarray:
        """Indicator of the (uniform) prior support. Densities are otherwise constant."""
        ok = (h.friction >= self.friction[0]) & (h.friction <= self.friction[1])
        for kind in (PRISMATIC, REVOLUTE):
            lo, hi = self._upper_range(kind)
            ok &= (h.kind != kind) | ((h.upper >= lo) & (h.upper <= hi))
        lo, hi = self._origin_box()
        in_box = np.all((h.origin >= lo) & (h.origin <= hi), axis=1)
        return ok & ((h.kind != REVOLUTE) | in_box)


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m)))) if np.isfinite(m) else -np.inf


def _normalise(logw: np.ndarray) -> np.ndarray:
    return logw - _logsumexp(logw)


def _ess(logw: np.ndarray) -> float:
    w = np.exp(_normalise(logw))
    return float(1.0 / np.sum(w * w))


def _systematic_resample(logw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    w = np.exp(_normalise(logw))
    n = len(w)
    positions = (rng.uniform() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), positions), n - 1)


class TypeFilter:
    """SMC particle set for one joint type, with a running log-evidence estimate."""

    def __init__(self, kind: int, prior: Prior, n: int, rng: np.random.Generator) -> None:
        self.kind = kind
        self.prior = prior
        self.rng = rng
        self.particles = prior.sample(kind, 1 if kind == FIXED else n, rng)
        self.logw = _normalise(np.zeros(len(self.particles.kind)))
        self.ll_hist = np.zeros(len(self.particles.kind))
        self.log_evidence = 0.0
        self.mh_acceptance: list[float] = []

    def update(
        self,
        record: Record,
        history: list[Record],
        h0: np.ndarray,
        noise: NoiseModel,
        ess_frac: float = 0.5,
        mh_steps: int = 4,
    ) -> None:
        ll_new = log_likelihood(self.particles, [record], h0, noise)
        if self.kind == FIXED:  # no free parameters: the evidence is exact
            self.log_evidence += float(ll_new[0])
            self.ll_hist += ll_new
            return
        n = len(self.logw)
        beta = 0.0
        while beta < 1.0:
            delta = self._next_increment(ll_new, 1.0 - beta, ess_frac * n)
            logw = self.logw + delta * ll_new
            self.log_evidence += _logsumexp(logw)  # self.logw is normalised
            self.logw = _normalise(logw)
            beta += delta
            if beta < 1.0 or _ess(self.logw) < ess_frac * n:
                idx = _systematic_resample(self.logw, self.rng)
                self.particles = self.particles.take(idx)
                self.ll_hist, ll_new = self.ll_hist[idx], ll_new[idx]
                self.logw = _normalise(np.zeros(n))
                ll_new = self._rejuvenate(history, record, ll_new, beta, h0, noise, mh_steps)
        self.ll_hist += ll_new

    def _next_increment(self, ll_new: np.ndarray, remaining: float, ess_target: float) -> float:
        if _ess(self.logw + remaining * ll_new) >= ess_target:
            return remaining
        lo, hi = 0.0, remaining
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _ess(self.logw + mid * ll_new) >= ess_target else (lo, mid)
        return max(lo, 1e-6)

    def _rejuvenate(
        self,
        history: list[Record],
        record: Record,
        ll_new: np.ndarray,
        beta: float,
        h0: np.ndarray,
        noise: NoiseModel,
        steps: int,
    ) -> np.ndarray:
        """Random-walk MH moves that leave prior x L(history) x L(record)^beta invariant."""
        p = self.particles
        for _ in range(steps):
            prop = self._propose(p)
            ok = self.prior.in_support(prop)
            ll_hist_prop = log_likelihood(prop, history, h0, noise) if history else 0.0 * ok
            ll_new_prop = log_likelihood(prop, [record], h0, noise)
            log_ratio = (ll_hist_prop + beta * ll_new_prop) - (self.ll_hist + beta * ll_new)
            accept = ok & (np.log(self.rng.uniform(size=len(ok))) < log_ratio)
            self.mh_acceptance.append(float(accept.mean()))
            p = _where(accept, prop, p)
            self.ll_hist = np.where(accept, ll_hist_prop, self.ll_hist)
            ll_new = np.where(accept, ll_new_prop, ll_new)
        self.particles = p
        return ll_new

    def _propose(self, p: Hypotheses) -> Hypotheses:
        """Symmetric random-walk proposal scaled to the current population spread."""
        n = len(p.kind)
        rng = self.rng
        s_axis = np.clip(0.5 * p.axis.std(axis=0).mean(), 0.005, 0.3)
        axis = p.axis + rng.normal(scale=s_axis, size=(n, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        s_origin = np.clip(0.5 * p.origin.std(axis=0), 0.001, 0.05)
        s_upper = max(0.5 * p.upper.std(), 1e-3)
        s_fric = max(0.5 * p.friction.std(), 0.02)
        return Hypotheses(
            kind=p.kind,
            axis=axis,
            origin=p.origin + rng.normal(size=(n, 3)) * s_origin,
            upper=p.upper + rng.normal(scale=s_upper, size=n),
            friction=p.friction + rng.normal(scale=s_fric, size=n),
        )

    def weights(self) -> np.ndarray:
        return np.exp(self.logw)


def _where(mask: np.ndarray, a: Hypotheses, b: Hypotheses) -> Hypotheses:
    m = mask[:, None]
    return Hypotheses(
        kind=a.kind,
        axis=np.where(m, a.axis, b.axis),
        origin=np.where(m, a.origin, b.origin),
        upper=np.where(mask, a.upper, b.upper),
        friction=np.where(mask, a.friction, b.friction),
    )


def log_likelihood(
    h: Hypotheses, records: list[Record], h0: np.ndarray, noise: NoiseModel
) -> np.ndarray:
    """Sum over ``records`` of log p(y_obs | hypothesis, probe). Vectorised over N x R."""
    n, r = len(h.kind), len(records)
    if r == 0:
        return np.zeros(n)
    tiled = h.take(np.tile(np.arange(n), r))  # record-major: index = record * n + particle

    def per_pair(field: str) -> np.ndarray:
        values = np.stack(
            [np.atleast_1d(np.asarray(getattr(rec, field), float)) for rec in records]
        )
        return np.repeat(values, n, axis=0)

    y_pred, _ = predict_outcome(
        tiled,
        h0,
        per_pair("handle_before"),
        per_pair("ee_before"),
        per_pair("direction"),
        per_pair("distance")[:, 0],
        per_pair("max_force")[:, 0],
    )
    return noise.loglik(y_pred, per_pair("y_obs")).reshape(r, n).sum(axis=0)


class Belief:
    """Posterior over mechanism type and parameters given probe records."""

    def __init__(
        self,
        geometry: PartGeometry,
        rng: np.random.Generator,
        n_particles: int = 1000,
        noise: NoiseModel | None = None,
        prior: Prior | None = None,
    ) -> None:
        self.geometry = geometry
        self.h0 = np.asarray(geometry.handle_pos, dtype=float)
        self.noise = noise or NoiseModel()
        self.prior = prior or Prior(geometry)
        self.rng = rng
        self.filters = [
            TypeFilter(k, self.prior, n_particles, rng) for k in (FIXED, PRISMATIC, REVOLUTE)
        ]
        self.log_type_prior = np.log(np.full(3, 1 / 3))
        self.history: list[Record] = []

    def update(self, record: Record) -> None:
        for f in self.filters:
            f.update(record, self.history, self.h0, self.noise)
        self.history.append(record)

    def type_probs(self) -> np.ndarray:
        logp = self.log_type_prior + np.array([f.log_evidence for f in self.filters])
        return np.exp(_normalise(logp))

    def type_entropy(self) -> float:
        p = self.type_probs()
        return float(-np.sum(p * np.log(np.clip(p, 1e-300, None))))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> Hypotheses:
        """Draw ``n`` hypotheses from the full mixture posterior."""
        rng = rng or self.rng
        counts = rng.multinomial(n, self.type_probs())
        parts = [
            f.particles.take(rng.choice(len(f.logw), size=c, p=f.weights()))
            for f, c in zip(self.filters, counts, strict=True)
            if c > 0
        ]
        return Hypotheses.concat(parts)

    def map_type(self) -> int:
        return int(np.argmax(self.type_probs()))

    def summary(self) -> dict:
        """Human-readable posterior summary (type probabilities, best-type parameter means)."""
        probs = self.type_probs()
        out = {"type_probs": dict(zip(TYPE_NAMES, probs.round(4).tolist(), strict=True))}
        k = self.map_type()
        out["map_type"] = TYPE_NAMES[k]
        if k != FIXED:
            est = self.estimate(k)
            out.update({key: np.round(v, 4).tolist() for key, v in est.items()})
        return out

    def estimate(self, kind: int) -> dict[str, np.ndarray]:
        """Weighted parameter means and axis spread for one type's particle set."""
        f = self.filters[kind]
        w = f.weights()
        p = f.particles
        axis = np.sum(w[:, None] * p.axis, axis=0)
        axis_conc = float(np.linalg.norm(axis))  # 1 means every particle agrees on the axis
        axis = axis / max(axis_conc, 1e-12)
        est = {
            "axis": axis,
            "axis_concentration": np.array(axis_conc),
            "upper": np.array(np.sum(w * p.upper)),
            "friction": np.array(np.sum(w * p.friction)),
        }
        if kind == REVOLUTE:
            # Report the axis line's point closest to the handle (origin along the axis is free).
            rel = p.origin - self.h0
            foot = self.h0 + rel - np.sum(rel * p.axis, 1, keepdims=True) * p.axis
            est["origin"] = np.sum(w[:, None] * foot, axis=0)
        return est


def make_record(
    before_handle: np.ndarray,
    before_ee: np.ndarray,
    probe_direction: np.ndarray,
    probe_distance: float,
    probe_max_force: float,
    after_handle: np.ndarray,
    after_rotvec: np.ndarray,
) -> Record:
    return Record(
        handle_before=np.asarray(before_handle, float),
        ee_before=np.asarray(before_ee, float),
        direction=np.asarray(probe_direction, float),
        distance=float(probe_distance),
        max_force=float(probe_max_force),
        y_obs=np.concatenate([after_handle, after_rotvec]),
    )
