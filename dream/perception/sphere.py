"""RGB-D sphere baseline: red foreground + robust surface fit, not open-world vision.

Only calibrated pixels enter this module. Assumes one separated red, opaque, approximately
spherical object of radius 1-6 cm. Pose orientation is unobservable for a sphere. Covariance
includes a 2 mm systematic floor, not just the much smaller iid pixel fitting error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dream.perception.detect import _hue_sat, components, min_area_rect
from dream.perception.geometry import backproject
from dream.types import RGBDFrame


@dataclass(frozen=True)
class SphereEstimate:
    time: float
    position: np.ndarray
    covariance: np.ndarray
    radius: float
    fit_rms: float
    pixels: int

    def to_dict(self) -> dict:
        return {
            "time_s": self.time,
            "position_m": self.position.tolist(),
            "covariance_m2": self.covariance.tolist(),
            "radius_m": self.radius,
            "fit_rms_m": self.fit_rms,
            "pixels": self.pixels,
            "source": "rendered RGB-D, red segmentation, sphere surface fit",
            "orientation": "unobservable for sphere",
        }


def detect_sphere(frame: RGBDFrame) -> SphereEstimate | None:
    points = backproject(frame)
    hue, sat = _hue_sat(frame.rgb)
    mask = ((hue < 15) | (hue > 345)) & (sat > 0.45)
    mask &= np.isfinite(points).all(axis=-1) & ~frame.self_mask
    labels = components(mask, points, frame.rgb)
    ids, counts = np.unique(labels[mask], return_counts=True)
    if not len(ids) or counts.max() < 40:
        return None
    order = np.argsort(counts)
    if len(order) > 1 and counts[order[-2]] > 0.3 * counts[order[-1]]:
        return None  # ambiguous multiple targets, do not invent an identity
    cloud = points[labels == ids[order[-1]]]
    origin = cloud.mean(axis=0)
    p = cloud - origin  # condition the least-squares fit near zero
    a = np.column_stack([2 * p, np.ones(len(p))])
    if np.linalg.matrix_rank(a) < 4:
        return None
    x = np.linalg.lstsq(a, np.sum(p * p, axis=1), rcond=None)[0]
    centre = x[:3]
    radius = np.sqrt(max(x[3] + centre @ centre, 0))
    # Geometric (rather than algebraic) refinement reduces partial-surface bias.
    for _ in range(8):
        diff = p - centre
        distance = np.linalg.norm(diff, axis=1)
        residual = distance - radius
        jac = np.column_stack([-diff / np.maximum(distance[:, None], 1e-9), -np.ones(len(p))])
        weights = np.minimum(1, 0.003 / np.maximum(abs(residual), 1e-9))
        step = np.linalg.lstsq(jac * weights[:, None], -residual * weights, rcond=None)[0]
        centre += step[:3]
        radius += step[3]
    rms = float(np.sqrt(np.mean(residual**2)))
    if not 0.01 < radius < 0.06 or rms > 0.004 or np.linalg.cond(jac.T @ jac) > 1e5:
        return None
    covariance = (rms**2 * np.linalg.pinv(jac.T @ jac))[:3, :3]
    covariance += np.eye(3) * 0.002**2  # systematic floor, checked on independent RGB-D scenes
    return SphereEstimate(frame.time, centre + origin, covariance, float(radius), rms, len(p))


def detect_target(frame: RGBDFrame) -> tuple[np.ndarray, float] | None:
    """Blue rectangular horizontal support, inferred from its visible top RGB-D surface."""
    hue, sat = _hue_sat(frame.rgb)
    points = backproject(frame)
    mask = (hue > 195) & (hue < 265) & (sat > 0.45) & ~frame.self_mask
    mask &= np.isfinite(points).all(axis=-1)
    # A task-independent supported-arm workspace prior rejects the similarly blue
    # ground plane. These are the controller's vertical workspace bounds, not a
    # fixture pose supplied by scene construction.
    mask &= (points[..., 2] > 0.74) & (points[..., 2] < 1.20)
    cloud = points[mask]
    if len(cloud) < 40:
        return None
    height = float(np.percentile(cloud[:, 2], 90))
    top = cloud[np.abs(cloud[:, 2] - height) < 0.004]
    if len(top) < 20:
        return None
    centre, extent, _ = min_area_rect(top[:, :2])
    return np.array([*centre, height]), float(min(extent) / 2)


class SpherePerception:
    """Frame cache at the robot clock; every physical action requires a new RGB-D frame."""

    backend = "classical RGB-D sphere baseline"

    def __init__(self, source: Callable[[], RGBDFrame], clock: Callable[[], float]) -> None:
        self.source, self.clock = source, clock
        self.frame: RGBDFrame | None = None
        self.object: SphereEstimate | None = None
        self.target: tuple[np.ndarray, float] | None = None

    def observe(self) -> SphereEstimate | None:
        if self.frame is None or self.frame.time < self.clock():
            self.frame = self.source()
            self.object = detect_sphere(self.frame)
            self.target = detect_target(self.frame)
        return self.object
