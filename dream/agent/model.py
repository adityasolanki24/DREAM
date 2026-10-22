"""Mechanism hypotheses and a quasi-static model of what a compliant probe does to them.

A hypothesis is (type, axis, origin, upper limit, friction). Types:
    0 fixed      handle never moves
    1 prismatic  x(q) = h0 + q * axis
    2 revolute   x(q) = origin + Rot(axis, q) (h0 - origin)
with q in [0, upper] and h0 the handle position in the closed state (observed by the agent).

Probe model (an engineering approximation that the tests check against MuJoCo):
the gripper is a per-axis spring F = clip(kp (target - x), +-F_max) attached to the handle. The
handle slides along its constraint while the generalised force |F . dx/dq| exceeds Coulomb
friction ``friction * |dx/dq|``, and stops at the first equilibrium or at a joint limit. Inertia,
damping and constraint compliance are ignored, so this predicts *settled* outcomes only.

All functions are vectorised over a leading batch dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dream.types import GANTRY_KP

FIXED, PRISMATIC, REVOLUTE = 0, 1, 2
TYPE_NAMES = ("fixed", "prismatic", "revolute")


@dataclass
class Hypotheses:
    """A batch of N mechanism hypotheses (struct of arrays)."""

    kind: np.ndarray  # (N,) int
    axis: np.ndarray  # (N, 3) unit
    origin: np.ndarray  # (N, 3) m; only meaningful for revolute
    upper: np.ndarray  # (N,) m or rad
    friction: np.ndarray  # (N,) N, Coulomb resistance referred to the handle

    def __len__(self) -> int:
        return len(self.kind)

    def take(self, idx: np.ndarray) -> Hypotheses:
        return Hypotheses(
            self.kind[idx], self.axis[idx], self.origin[idx], self.upper[idx], self.friction[idx]
        )

    def copy(self) -> Hypotheses:
        return self.take(np.arange(len(self)))

    @staticmethod
    def concat(parts: list[Hypotheses]) -> Hypotheses:
        return Hypotheses(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in Hypotheses._fields())
        )

    @staticmethod
    def _fields() -> tuple[str, ...]:
        return ("kind", "axis", "origin", "upper", "friction")


def rotate(axis: np.ndarray, angle: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of vectors ``v`` about unit ``axis`` by ``angle`` (batched)."""
    c = np.cos(angle)[..., None]
    s = np.sin(angle)[..., None]
    dot = np.sum(axis * v, axis=-1, keepdims=True)
    return v * c + np.cross(axis, v) * s + axis * dot * (1 - c)


def handle_at(h: Hypotheses, q: np.ndarray, h0: np.ndarray) -> np.ndarray:
    """Handle position for joint coordinate ``q`` under each hypothesis. Shape (N, 3)."""
    h0 = np.broadcast_to(h0, h.axis.shape)
    pris = h0 + q[:, None] * h.axis
    rev = h.origin + rotate(h.axis, q, h0 - h.origin)
    out = np.where((h.kind == PRISMATIC)[:, None], pris, h0)
    return np.where((h.kind == REVOLUTE)[:, None], rev, out)


def tangent(h: Hypotheses, x: np.ndarray) -> np.ndarray:
    """dx/dq at handle position ``x``. Zero for fixed. Shape (N, 3)."""
    rev = np.cross(h.axis, x - h.origin)
    out = np.where((h.kind == PRISMATIC)[:, None], h.axis, 0.0)
    return np.where((h.kind == REVOLUTE)[:, None], rev, out)


def project(h: Hypotheses, x_obs: np.ndarray, h0: np.ndarray) -> np.ndarray:
    """Joint coordinate that best explains an observed handle position (clipped to range)."""
    x_obs = np.broadcast_to(x_obs, h.axis.shape)
    h0 = np.broadcast_to(h0, h.axis.shape)
    q_pris = np.sum((x_obs - h0) * h.axis, axis=-1)
    r0 = h0 - h.origin
    r1 = x_obs - h.origin
    r0 = r0 - np.sum(r0 * h.axis, -1, keepdims=True) * h.axis
    r1 = r1 - np.sum(r1 * h.axis, -1, keepdims=True) * h.axis
    q_rev = np.arctan2(np.sum(h.axis * np.cross(r0, r1), -1), np.sum(r0 * r1, -1))
    q = np.where(h.kind == PRISMATIC, q_pris, np.where(h.kind == REVOLUTE, q_rev, 0.0))
    return np.clip(q, 0.0, h.upper)


def settle(
    h: Hypotheses,
    q_start: np.ndarray,
    h0: np.ndarray,
    target: np.ndarray,
    max_force: np.ndarray,
    kp: float = GANTRY_KP,
    n_iter: int = 60,
    max_step: float = 0.005,
) -> np.ndarray:
    """Quasi-static equilibrium joint coordinate after the gripper target moves to ``target``.

    Iterates friction-limited Newton steps along the constraint, with each step capped to
    ``max_step`` metres of handle travel so that saturated springs do not overshoot.
    """
    q = q_start.astype(float).copy()
    max_force = np.broadcast_to(max_force, q.shape)[:, None]
    moving = h.kind != FIXED
    for _ in range(n_iter):
        x = handle_at(h, q, h0)
        j = tangent(h, x)
        jn2 = np.sum(j * j, -1) + 1e-12
        force = np.clip(kp * (target - x), -max_force, max_force)
        gen = np.sum(force * j, -1)
        resist = h.friction * np.sqrt(jn2)
        excess = np.sign(gen) * np.maximum(np.abs(gen) - resist, 0.0)
        dq = excess / (kp * jn2)
        dq = np.clip(dq, -max_step / np.sqrt(jn2), max_step / np.sqrt(jn2))
        q = np.where(moving, np.clip(q + dq, 0.0, h.upper), q)
    return q


def handle_rotation(h: Hypotheses, q: np.ndarray) -> np.ndarray:
    """Handle rotation vector relative to the closed state. Shape (N, 3)."""
    return np.where((h.kind == REVOLUTE)[:, None], h.axis * q[:, None], 0.0)


def predict_outcome(
    h: Hypotheses,
    h0: np.ndarray,
    handle_before: np.ndarray,
    ee_before: np.ndarray,
    direction: np.ndarray,
    distance: np.ndarray,
    max_force: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted settled outcome of a probe.

    Returns ``(y, q, y_start)``. ``y`` is (N, 6), the handle position followed by the handle
    rotation vector relative to the closed state. ``q`` is the predicted joint coordinate (N,),
    and ``y_start`` is each hypothesis's own reconstruction of the pre-probe state (N, 6).

    Probe arguments broadcast against the hypothesis batch, so one call can evaluate many
    hypotheses under one probe, or matched (hypothesis, probe) pairs.
    """
    q0 = project(h, handle_before, h0)
    target = np.asarray(ee_before) + np.asarray(direction) * np.asarray(distance)[..., None]
    target = np.broadcast_to(target, h.axis.shape)
    q1 = settle(h, q0, h0, target, np.asarray(max_force, dtype=float))
    y = np.concatenate([handle_at(h, q1, h0), handle_rotation(h, q1)], axis=-1)
    y_start = np.concatenate([handle_at(h, q0, h0), handle_rotation(h, q0)], axis=-1)
    return y, q1, y_start


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions."""
    aw, av = a[0], a[1:]
    bw, bv = b[0], b[1:]
    return np.concatenate([[aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)])


def relative_rotvec(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking orientation ``q_ref`` to ``q``."""
    conj = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])
    r = quat_mul(q, conj)
    if r[0] < 0:
        r = -r
    s = np.linalg.norm(r[1:])
    if s < 1e-12:
        return np.zeros(3)
    return r[1:] / s * 2 * np.arctan2(s, r[0])
