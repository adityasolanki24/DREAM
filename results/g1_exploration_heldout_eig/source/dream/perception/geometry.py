"""Pixel <-> world geometry for a calibrated pinhole RGB-D camera, and table-plane fitting."""

from __future__ import annotations

import numpy as np

from dream.types import CameraCalibration, RGBDFrame


def backproject(frame: RGBDFrame) -> np.ndarray:
    """World coordinates of every pixel, (H, W, 3); NaN where depth is missing.

    MuJoCo cameras look along -z with x right and y up; image rows grow downward.
    """
    cal = frame.calibration
    h, w = frame.depth.shape
    f = cal.focal
    u = np.arange(w) + 0.5 - w / 2
    v = np.arange(h) + 0.5 - h / 2
    d = frame.depth.astype(np.float64)
    cam = np.stack([u[None, :] * d / f, -v[:, None] * d / f, -d], axis=-1)
    return cam @ cal.rotation.T + cal.position


def project(cal: CameraCalibration, points: np.ndarray) -> np.ndarray:
    """Pixel coordinates (u, v) of world points, (..., 2); for overlays."""
    cam = (np.asarray(points) - cal.position) @ cal.rotation
    d = -cam[..., 2]
    u = cam[..., 0] * cal.focal / d + cal.width / 2
    v = -cam[..., 1] * cal.focal / d + cal.height / 2
    return np.stack([u, v], axis=-1)


def fit_plane(
    points: np.ndarray, rng: np.random.Generator, iters: int = 200, tol: float = 0.004
) -> tuple[np.ndarray, float]:
    """RANSAC plane through the dominant surface: (unit normal pointing up, offset), n.x = c."""
    best, best_n = (np.array([0.0, 0.0, 1.0]), 0.0), -1
    if len(points) < 3:
        return best
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm * np.sign(n[2] if n[2] != 0 else 1.0)
        c = float(n @ p[0])
        count = int(np.sum(np.abs(points @ n - c) < tol))
        if count > best_n:
            best, best_n = (n, c), count
    n, c = best
    inliers = points[np.abs(points @ n - c) < tol]
    if len(inliers) >= 3:  # least-squares refinement on the inliers
        centre = inliers.mean(axis=0)
        n = np.linalg.svd(inliers - centre, full_matrices=False)[2][-1]
        n = n * np.sign(n[2])
        c = float(n @ centre)
    return n, c
