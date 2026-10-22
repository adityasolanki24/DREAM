"""Object and container detection from one RGB-D frame: a classical RGB-D baseline.

Pipeline: back-project depth to 3D -> drop the robot's own pixels (self mask) -> fit the table
plane (RANSAC) -> keep points more than ``FG_HEIGHT`` above it -> connect neighbouring pixels
that are close in 3D *and* in colour -> describe each component geometrically.

Assumptions (this is not general object understanding):
- objects rest on (or are held above) one dominant horizontal support plane;
- objects are box-like: a footprint rectangle and a height describe them;
- adjacent objects differ in colour (two touching same-coloured objects merge);
- colour names come from a fixed hue vocabulary (``COLOR_NAMES``), not from learning;
- an "open container" is a large component whose interior is much lower than its rim.
Everything is inferred from the image arrays and the camera calibration.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from dream.perception.geometry import backproject, fit_plane
from dream.types import RGBDFrame

FG_HEIGHT = 0.008  # m above the table: ~3 sigma of depth noise at 1 m
NEIGHBOUR_3D = 0.015  # m between adjacent pixels of one surface
# Adjacent pixels belong to one object if both are unsaturated (grey) or both saturated with a
# hue difference below this. Hue is nearly invariant to the shading difference between a lit top
# face and a shaded side (raw RGB and chromaticity split cubes in two).
NEIGHBOUR_HUE_DEG = 25.0
SATURATED = 0.3
MIN_PIXELS = 20
CONTAINER_MIN_SIDE = 0.10  # m: footprints larger than this may be containers
RESTING_BOTTOM = 0.015  # m: a component reaching this close to the table rests on it
# Hue bands in degrees; low saturation is grey. A fixed vocabulary, not learned.
COLOR_NAMES = (
    ("red", 0, 15),
    ("orange", 15, 40),
    ("yellow", 40, 70),
    ("green", 70, 170),
    ("cyan", 170, 195),
    ("blue", 195, 265),
    ("purple", 265, 320),
    ("pink", 320, 345),
    ("red", 345, 361),
)


@dataclass(frozen=True)
class Detection:
    kind: str  # "object" | "container"
    color: str
    position: np.ndarray  # object: box centre; container: inner floor centre (floor-top z)
    yaw: float  # rad, rectangle orientation (mod pi/2 for a square)
    extent: np.ndarray  # (side_a, side_b, height) m of the visible box
    position_std: float  # m, 1-sigma
    occluded: bool  # touches the robot, the image border or a nearer surface
    resting: bool  # bottom at the support plane (False: held or stacked high)
    inner_half: float = 0.0  # container: axis-aligned inner half width
    pixels: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # (u0, v0, u1, v1) for overlays

    @property
    def size(self) -> float:
        return float(np.median(self.extent))


def color_name(rgb: np.ndarray) -> str:
    r, g, b = (np.asarray(rgb, float) / 255.0).tolist()
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.3 or v < 0.15:
        return "grey"
    deg = 360.0 * h
    return next(name for name, lo, hi in COLOR_NAMES if lo <= deg < hi)


def components(fg: np.ndarray, pts: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Label 4-connected foreground pixels joined by 3D and colour continuity (-1: none)."""
    h, w = fg.shape
    idx = np.arange(h * w).reshape(h, w)
    hue, sat = _hue_sat(rgb)
    colored = sat > SATURATED
    edges = []
    for dv, du in ((0, 1), (1, 0)):
        a = (slice(0, h - dv), slice(0, w - du))
        b = (slice(dv, h), slice(du, w))
        ok = fg[a] & fg[b]
        ok &= np.linalg.norm(pts[a] - pts[b], axis=-1) < NEIGHBOUR_3D
        dh = np.abs(hue[a] - hue[b])
        dh = np.minimum(dh, 360.0 - dh)
        same = (colored[a] & colored[b] & (dh < NEIGHBOUR_HUE_DEG)) | (~colored[a] & ~colored[b])
        ok &= same
        edges.append(np.stack([idx[a][ok], idx[b][ok]], axis=1))
    e = np.concatenate(edges)
    label = np.where(fg.ravel(), np.arange(h * w), -1)
    if len(e) == 0:
        return label.reshape(h, w)
    while True:  # min-label propagation over the edges until stable
        lo = np.minimum(label[e[:, 0]], label[e[:, 1]])
        new = label.copy()
        np.minimum.at(new, e[:, 0], lo)
        np.minimum.at(new, e[:, 1], lo)
        new[new >= 0] = new[new[new >= 0]]  # pointer jumping speeds convergence
        if np.array_equal(new, label):
            return label.reshape(h, w)
        label = new


def _hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel hue (deg) and saturation of an RGB image (vectorised HSV)."""
    c = rgb.astype(np.float32) / 255.0
    mx, mn = c.max(axis=-1), c.min(axis=-1)
    delta = mx - mn
    safe = np.where(delta > 1e-6, delta, 1.0)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    hue = np.where(
        mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)
    )
    sat = np.where(mx > 1e-6, delta / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return 60.0 * hue, sat


def min_area_rect(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(centre, (side_a, side_b), angle) of the smallest rectangle around 2D points (1 deg)."""
    best = None
    for a in np.deg2rad(np.arange(0.0, 90.0, 1.0)):
        c, s = np.cos(a), np.sin(a)
        r = xy @ np.array([[c, -s], [s, c]])
        lo, hi = np.percentile(r, 1, axis=0), np.percentile(r, 99, axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            mid = (lo + hi) / 2
            centre = mid @ np.array([[c, s], [-s, c]])
            best = (area, centre, hi - lo, a)
    return best[1], best[2], best[3]


def detect(
    frame: RGBDFrame, rng: np.random.Generator, plane: tuple[np.ndarray, float] | None = None
) -> tuple[list[Detection], tuple[np.ndarray, float]]:
    """Detections in one frame, and the support plane used (pass it back to reuse it)."""
    pts = backproject(frame)
    valid = np.isfinite(pts).all(axis=-1) & ~frame.self_mask
    if plane is None:
        sample = pts[valid]
        sample = sample[rng.choice(len(sample), min(4000, len(sample)), replace=False)]
        plane = fit_plane(sample, rng)
    n, c = plane
    height = np.where(valid, pts @ n - c, np.nan)
    fg = valid & (height > FG_HEIGHT)
    labels = components(fg, pts, frame.rgb)
    out = []
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    for lab in ids[counts >= MIN_PIXELS]:
        mask = labels == lab
        d = _describe(mask, pts, height, frame, valid)
        if d is not None:
            out.append(d)
    return _absorb_container_fragments(out), plane


def _absorb_container_fragments(dets: list[Detection]) -> list[Detection]:
    """Drop same-coloured pieces lying on a container's wall band (walls broken up by shading
    or occlusion). Assumes objects contrast in colour with the container they are in."""
    containers = [d for d in dets if d.kind == "container"]
    keep = []
    for d in dets:
        part_of = False
        if d.kind == "object":
            for c in containers:
                ca, sa = np.cos(c.yaw), np.sin(c.yaw)
                local = (d.position[:2] - c.position[:2]) @ np.array([[ca, -sa], [sa, ca]])
                to_edge = np.min(c.extent[:2] / 2 - np.abs(local))
                part_of |= d.color == c.color and -0.015 < to_edge < 0.02
        if not part_of:
            keep.append(d)
    return keep


def _occluded(mask: np.ndarray, frame: RGBDFrame, jump: float = 0.02) -> bool:
    """Something is in front of part of the component: a bordering pixel belongs to the robot,
    is nearer to the camera than the component pixel beside it, or lies on the image border."""
    h, w = mask.shape
    depth = np.nan_to_num(frame.depth, nan=np.inf)
    own = np.where(mask, depth, np.inf)
    for axis in (0, 1):
        for shift in (1, -1):
            beside = np.roll(own, shift, axis=axis)  # the component's depth next to each pixel
            nb = ~mask & np.isfinite(beside)
            if (nb & frame.self_mask).any() or (nb & (depth < beside - jump)).any():
                return True
    vs, us = np.nonzero(mask)
    return bool(vs.min() <= 1 or us.min() <= 1 or vs.max() >= h - 2 or us.max() >= w - 2)


def _describe(mask, pts, height, frame, valid) -> Detection | None:
    vs, us = np.nonzero(mask)
    p, hgt = pts[mask], height[mask]
    top = float(np.percentile(hgt, 97))
    bottom = float(np.percentile(hgt, 3))
    centre, sides, angle = min_area_rect(p[:, :2])
    occluded = _occluded(mask, frame)
    bbox = (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))
    depth = float(np.nanmedian(frame.depth[mask]))
    pixel = depth / frame.calibration.focal  # m per pixel at this range
    if max(sides) > CONTAINER_MIN_SIDE:
        return _container(
            mask, pts, height, valid, centre, sides, angle, top, occluded, bbox, frame
        )
    resting = bottom < RESTING_BOTTOM
    z_lo = 0.0 if resting else bottom
    top_face = hgt > top - 0.006
    rgb = frame.rgb[mask][top_face].mean(axis=0) if top_face.any() else frame.rgb[mask].mean(0)
    # Rectangle edges are quantised by the pixel footprint; occlusion can hide whole faces.
    std = 0.5 * pixel / np.sqrt(max(int(top_face.sum()), 1)) + 0.5 * pixel
    if occluded:  # the hidden part can be as large as the object: up to half its extent off
        std = max(std, 0.5 * float(max(*sides, top - z_lo)))
    return Detection(
        kind="object",
        color=color_name(rgb),
        position=np.array([centre[0], centre[1], (z_lo + top) / 2]),
        yaw=float(angle),
        extent=np.array([sides[0], sides[1], top - z_lo]),
        position_std=float(std),
        occluded=occluded,
        resting=resting,
        pixels=int(mask.sum()),
        bbox=bbox,
    )


def _container(mask, pts, height, valid, centre, sides, angle, top, occluded, bbox, frame):
    """An open container if its interior floor is far below its rim, else None (a box)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    local = (pts[..., :2] - centre) @ rot  # every pixel in the rectangle's frame
    half = np.asarray(sides) / 2
    inside = valid & np.all(np.abs(local) < half - 0.02, axis=-1)
    if inside.sum() < MIN_PIXELS:
        return None
    floor = float(np.nanmedian(height[inside]))
    if floor > 0.4 * top:  # a solid block, not something to put things into
        return None
    # Wall thickness: the band of rim-height points along the rectangle's edges.
    rim = mask & (height > 0.7 * top)
    edge_dist = np.min(half - np.abs(local[rim]), axis=-1)
    wall = float(np.clip(2 * np.median(np.clip(edge_dist, 0, None)), 0.003, 0.03))
    inner_local = float(min(half)) - wall
    inner_half = inner_local / (abs(c) + abs(s))  # axis-aligned square inside the rotated one
    floor_z = float(np.nanmedian(pts[..., 2][inside]))
    frame_rgb_mean = frame.rgb[mask].mean(axis=0)
    return Detection(
        kind="container",
        color=color_name(frame_rgb_mean),
        position=np.array([centre[0], centre[1], floor_z]),
        yaw=float(angle),
        extent=np.array([sides[0], sides[1], top]),
        position_std=0.003,
        occluded=occluded,
        resting=True,
        inner_half=float(inner_half),
        pixels=int(mask.sum()),
        bbox=bbox,
    )
