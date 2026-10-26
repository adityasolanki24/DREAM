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


OCCLUSION_MARGIN = 0.15  # m: a return this much nearer than the point means something blocks it


def visibility(frame: RGBDFrame, point: np.ndarray, margin_px: int = 8,
               max_range: float = 2.5) -> str:  # fmt: skip
    """Could this camera frame have seen ``point``? From the frame's own depth, not geometry.

    ``visible``: in the image (``margin_px`` from the border), off the robot's own body, with
    a depth return no more than OCCLUSION_MARGIN nearer than the point (median of a 5x5
    window) and the point within ``max_range``. Otherwise the reason: ``behind``,
    ``out_of_image``, ``self_occluded``, ``too_far``, ``no_return`` or ``occluded``. Only a
    ``visible`` look can support "it is not there"; anything else says nothing.
    """
    cal = frame.calibration
    cam = (np.asarray(point, float) - cal.position) @ cal.rotation
    expected = -cam[2]
    if expected <= 0.05:
        return "behind"
    u, v = project(cal, np.asarray(point, float))
    h, w = frame.depth.shape
    if not (margin_px <= u < w - margin_px and margin_px <= v < h - margin_px):
        return "out_of_image"
    i, j = int(v), int(u)
    if frame.self_mask[max(i - 2, 0) : i + 3, max(j - 2, 0) : j + 3].any():
        return "self_occluded"
    if expected > max_range:
        return "too_far"
    window = frame.depth[max(i - 2, 0) : i + 3, max(j - 2, 0) : j + 3]
    window = window[np.isfinite(window)]
    if window.size < 5:
        return "no_return"
    if float(np.median(window)) < expected - OCCLUSION_MARGIN:
        return "occluded"
    return "visible"


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
