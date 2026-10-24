"""Object and container detection from one RGB-D frame: a classical RGB-D baseline.

Pipeline: back-project depth to 3D -> drop the robot's own pixels (self mask) -> fit the table
plane (RANSAC) -> keep points more than ``FG_HEIGHT`` above it -> connect neighbouring pixels
that are close in 3D *and* in colour -> describe each component geometrically.

Assumptions (this is not general object understanding):
- objects rest on (or are held above) one dominant horizontal support plane;
- objects are box-like: a footprint rectangle and a height describe them;
- adjacent objects differ in colour (two touching same-coloured objects merge);
- colour names come from a fixed hue vocabulary (``COLOR_NAMES``), not from learning;
- an "open container" is a large component whose interior is much lower than its rim; any
  other large component is a "box", whose opening (if any) is the largest region of its
  footprint well below its rim;
- a "handle" is a small component protruding from a box's side, no higher than its top.
Everything is inferred from the image arrays and the camera calibration.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, replace

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
MIN_SIDE = 0.006  # m: thinner components are fragments, not objects
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
    kind: str  # "object" | "container" | "box" | "handle"
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
    # box: the largest visible opening, (floor-top centre, axis-aligned half width), if any
    opening: tuple[np.ndarray, float] | None = None
    # small objects: centre from the topmost points (top-centre of a knob or bar handle)
    top_centre: np.ndarray | None = None

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
    hue, sat = _hue_sat(frame.rgb)
    ctx = _Context(pts, height, valid, fg, hue, sat, frame)
    masks = []
    for lab in ids[counts >= MIN_PIXELS]:
        mask = labels == lab
        d = _describe(mask, ctx)
        if d is not None:
            out.append(d)
            masks.append(mask)
    kept = _absorb_container_fragments(out)
    # Second pass: receptacle openings, knowing which pixels are other objects (contents).
    objects = [
        m
        for d, m in zip(out, masks, strict=True)
        if d.kind == "object" and any(d is k for k in kept)
    ]
    if objects and any(d.kind in ("box", "container") for d in out):
        ctx = replace(ctx, contents=np.any(objects, axis=0))
        out = [
            _describe(m, ctx) if d.kind in ("box", "container") else d
            for d, m in zip(out, masks, strict=True)
        ]
        kept = _absorb_container_fragments(out)
    return _assign_handles(kept), plane


HANDLE_REACH = 0.07  # m: how far outside a box's side a handle may protrude


def _assign_handles(dets: list[Detection]) -> list[Detection]:
    """Small components just outside a box's footprint and no higher than its top are its
    handle candidates (assumption: nothing else is placed touching a box's side)."""
    boxes = [d for d in dets if d.kind == "box"]
    out = []
    for d in dets:
        if d.kind == "object" and d.top_centre is not None:
            for b in boxes:
                ca, sa = np.cos(b.yaw), np.sin(b.yaw)
                local = (d.top_centre[:2] - b.position[:2]) @ np.array([[ca, -sa], [sa, ca]])
                outside = np.abs(local) - b.extent[:2] / 2
                beside = outside.max() > 0 and outside.max() < HANDLE_REACH and outside.min() < 0
                if beside and d.top_centre[2] <= b.extent[2] + 0.01:
                    d = replace(d, kind="handle", position=d.top_centre)
                    break
        out.append(d)
    return out


def _absorb_container_fragments(dets: list[Detection]) -> list[Detection]:
    """Merge pieces of receptacles (open containers and boxes):
    - same-coloured objects on a container's wall band, or inside a box's grouped footprint
      (walls broken up by shading or occlusion, a drawer's separate side pieces);
    - same-coloured boxes whose centres are within 0.1 m (one box seen as two components):
      the one with more pixels is kept, with the largest opening any of them showed (each
      piece groups a slightly different footprint; an opening seen in one is open space, as
      in ``fuse`` across cameras).
    Assumes objects contrast in colour with the receptacle they are in or next to."""
    recs = sorted([d for d in dets if d.kind in ("container", "box")], key=lambda d: -d.pixels)
    kept_recs: list[Detection] = []
    for r in recs:
        dup = next(
            (
                i
                for i, k in enumerate(kept_recs)
                if k.color == r.color and np.linalg.norm(k.position[:2] - r.position[:2]) < 0.1
            ),
            None,
        )
        if dup is None:
            kept_recs.append(r)
            continue
        k = kept_recs[dup]
        if r.opening is not None and (k.opening is None or r.opening[1] > k.opening[1]):
            kept_recs[dup] = replace(k, opening=r.opening)
    keep = list(kept_recs)
    for d in dets:
        if d.kind in ("container", "box"):
            continue
        part_of = False
        for c in kept_recs:
            ca, sa = np.cos(c.yaw), np.sin(c.yaw)
            local = (d.position[:2] - c.position[:2]) @ np.array([[ca, -sa], [sa, ca]])
            to_edge = np.min(c.extent[:2] / 2 - np.abs(local))
            lo = -0.015 if c.kind == "container" else -0.01
            hi = 0.02 if c.kind == "container" else np.inf
            part_of |= d.color == c.color and lo < to_edge < hi
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


@dataclass
class _Context:
    """Per-frame arrays shared by the component descriptions."""

    pts: np.ndarray
    height: np.ndarray
    valid: np.ndarray
    fg: np.ndarray
    hue: np.ndarray
    sat: np.ndarray
    frame: RGBDFrame
    contents: np.ndarray | None = None  # pixels of separately detected objects (second pass)


BOX_GROUP_RADIUS = 0.35  # m: same-coloured foreground this close belongs to one box


def _describe(mask, ctx: _Context) -> Detection | None:
    pts, height, frame, valid = ctx.pts, ctx.height, ctx.frame, ctx.valid
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
        container = _container(
            mask, pts, height, valid, centre, sides, angle, top, occluded, bbox, frame
        )
        if container is not None:
            rim = float(np.median(height[mask]))
            return replace(container, opening=_opening(ctx, centre, sides, angle, rim))
        return _box(mask, ctx, top, occluded, bbox)
    if min(sides) < MIN_SIDE:  # a sliver (a shaded edge of something larger), not an object
        return None
    resting = bottom < RESTING_BOTTOM
    z_lo = 0.0 if resting else bottom
    top_face = hgt > top - 0.006
    rgb = frame.rgb[mask][top_face].mean(axis=0) if top_face.any() else frame.rgb[mask].mean(0)
    # Rectangle edges are quantised by the pixel footprint; occlusion can hide whole faces.
    std = 0.5 * pixel / np.sqrt(max(int(top_face.sum()), 1)) + 0.5 * pixel
    if occluded:  # the hidden part can be as large as the object: up to half its extent off
        std = max(std, 0.5 * float(max(*sides, top - z_lo)))
    highest = hgt > top - 0.003
    top_centre = np.array([*p[highest][:, :2].mean(axis=0), top - 0.5 * float(min(sides))])
    return Detection(
        kind="object",
        top_centre=top_centre,
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


OPENING_CELL = 0.01  # m
OPENING_BELOW_RIM = 0.012  # m: a cell this far below the box's top is open
# m: the median-height rim estimate reads 7-15 mm below the true rim of an open box
# (development scenes, drawer and lid, seed 3), so contents may reach this far above it.
CONTENTS_ABOVE_RIM = 0.02


def _box(mask, ctx: _Context, top, occluded, bbox):
    """A solid-looking large component, grouped with same-coloured foreground nearby (a drawer
    pulled out of its housing shows up as separate pieces of the same colour). Its opening, if
    any, is the largest square of *observed* cells in the group's footprint whose surface is
    well below the rim (a depth well). Cells with no observation are never counted as open."""
    pts, height, frame = ctx.pts, ctx.height, ctx.frame
    hue0 = float(np.median(ctx.hue[mask]))
    colored = float(np.median(ctx.sat[mask])) > SATURATED
    dh = np.abs(ctx.hue - hue0)
    dh = np.minimum(dh, 360.0 - dh)
    same = (ctx.sat > SATURATED) & (dh < NEIGHBOUR_HUE_DEG) if colored else ctx.sat <= SATURATED
    centre0 = pts[mask][:, :2].mean(axis=0)
    near = np.linalg.norm(pts[..., :2] - centre0, axis=-1) < BOX_GROUP_RADIUS
    group = mask | (ctx.fg & same & near)
    centre, sides, angle = min_area_rect(pts[group][:, :2])
    top = max(top, float(np.percentile(height[group], 97)))
    mask = group
    # The rim is the group's median height, not its top: a lid raised a little sticks up above
    # the box and would make its own surface near the hinge look "below the rim".
    opening = _opening(ctx, centre, sides, angle, float(np.median(height[mask])))
    return Detection(
        kind="box",
        color=color_name(frame.rgb[mask].mean(axis=0)),
        position=np.array([centre[0], centre[1], top / 2]),
        yaw=float(angle),
        extent=np.array([sides[0], sides[1], top]),
        position_std=0.003,
        occluded=occluded,
        resting=True,
        pixels=int(mask.sum()),
        bbox=bbox,
        opening=opening,
    )


def _opening(ctx: _Context, centre, sides, angle, top) -> tuple[np.ndarray, float] | None:
    """Largest square of observed cells inside a footprint whose surface lies well below the
    rim: (floor-top centre, axis-aligned half width), or None. Unobserved cells are not open.

    A separately detected object inside the footprint and no higher than the rim is
    *contents* resting in the cavity: its cells count as open, as the oracle tracker counts
    them (an object low inside the cavity). Otherwise a cube put into the cavity makes its own
    cells look closed and the opening moves off it. An object well above the rim, such as a
    cube lying on a closed lid or a knob on a raised lid, still closes its cells.
    """
    pts, height, valid = ctx.pts, ctx.height, ctx.valid
    c, s = np.cos(angle), np.sin(angle)
    local = (pts[..., :2] - centre) @ np.array([[c, -s], [s, c]])
    half = np.asarray(sides) / 2
    inside = valid & np.all(np.abs(local) < half, axis=-1)
    if not inside.any():
        return None
    contents = np.zeros_like(inside) if ctx.contents is None else ctx.contents
    contents = inside & contents & (height < top + CONTENTS_ABOVE_RIM)
    surface = inside & ~contents
    nx, ny = (np.ceil(2 * half / OPENING_CELL)).astype(int)
    loc = np.where(inside[..., None], local, 0.0)
    ix = np.clip(((loc[..., 0] + half[0]) / OPENING_CELL).astype(int), 0, nx - 1)
    iy = np.clip(((loc[..., 1] + half[1]) / OPENING_CELL).astype(int), 0, ny - 1)
    cell_max = np.full((nx, ny), -np.inf)
    np.maximum.at(cell_max, (ix[surface], iy[surface]), height[surface])
    held = np.zeros((nx, ny), bool)
    held[ix[contents], iy[contents]] = True
    seen = np.isfinite(cell_max)
    below = _shadow_bound(ctx, contents, centre, c, s, half, cell_max)
    open_cells = (
        (seen & (cell_max < top - OPENING_BELOW_RIM))
        | (held & ~seen)
        | (~seen & ~held & (below < top + CONTENTS_ABOVE_RIM))  # floor or more contents
    )
    n, i0, j0 = _largest_square(open_cells, prefer=held)
    if n < 3:
        return None
    lc = np.array([(i0 + n / 2) * OPENING_CELL - half[0], (j0 + n / 2) * OPENING_CELL - half[1]])
    world = centre + lc @ np.array([[c, s], [-s, c]])
    cells = np.zeros_like(open_cells)
    cells[i0 : i0 + n, j0 : j0 + n] = True
    sel = surface & cells[ix, iy]
    if not sel.any():  # the floor is entirely covered by contents
        sel = inside & cells[ix, iy]
    floor_z = float(np.nanpercentile(pts[..., 2][sel], 20))
    return np.array([world[0], world[1], floor_z]), n * OPENING_CELL / 2 / (abs(c) + abs(s))


SHADOW_STEP = 0.005  # m along the ray: half an opening cell
SHADOW_REACH = 0.15  # m: farther than any cavity is wide


def _shadow_bound(ctx: _Context, contents, centre, c, s, half, cell_max) -> np.ndarray:
    """Per cell, an upper bound on the hidden surface in cells the contents hide from the camera.

    A cell nobody sees, behind a cube in the cavity, holds nothing above the lines of sight
    that graze the cube and pass over it (else those rays would have hit it). So the hidden
    surface lies below the extension of the camera rays beyond the contents. If that bound is
    within the contents limit, whatever is hidden there is floor or more contents: open. Each ray is
    followed until it reaches the table or enters an observed cell whose surface it would have
    hit. Cells no such ray reaches stay unbounded (+inf): unobserved is still not open.
    """
    bound = np.full(cell_max.shape, -np.inf)
    if not contents.any():
        return np.full(cell_max.shape, np.inf)
    cam = ctx.frame.calibration.position
    p, h = ctx.pts[contents], ctx.height[contents]
    d = p - cam
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = np.arange(1, int(SHADOW_REACH / SHADOW_STEP) + 1) * SHADOW_STEP
    xy = p[:, None, :2] + t[None, :, None] * d[:, None, :2]
    z = h[:, None] + t[None, :] * d[:, None, 2]
    local = (xy - centre) @ np.array([[c, -s], [s, c]])
    nx, ny = cell_max.shape
    i = np.floor((local[..., 0] + half[0]) / OPENING_CELL).astype(int)
    j = np.floor((local[..., 1] + half[1]) / OPENING_CELL).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (z > 0)
    ic, jc = np.clip(i, 0, nx - 1), np.clip(j, 0, ny - 1)
    blocked = ok & (cell_max[ic, jc] >= z)
    ok &= ~(np.cumsum(blocked | ~ok, axis=1) > 0)  # stop at the first hit or exit
    np.maximum.at(bound, (ic[ok], jc[ok]), z[ok])
    return np.where(np.isfinite(bound), bound, np.inf)


def _largest_square(grid: np.ndarray, prefer: np.ndarray | None = None) -> tuple[int, int, int]:
    """Side (cells) and lower-left index of the largest all-True square (dynamic programming).

    An open area longer than it is wide holds many largest squares. Take the one covering the
    most ``prefer`` cells (contents already in the receptacle: its region should contain what
    is in it), then the one nearest the middle of the open cells, so the region does not jump
    along the area from frame to frame with noise at its far ends.
    """
    n, m = grid.shape
    dp = np.zeros((n + 1, m + 1), int)
    for i in range(n):
        for j in range(m):
            if grid[i, j]:
                dp[i + 1, j + 1] = 1 + min(dp[i, j + 1], dp[i + 1, j], dp[i, j])
    best = int(dp.max())
    if best == 0:
        return 0, 0, 0
    mid = np.argwhere(grid).mean(axis=0)
    ends = np.argwhere(dp[1:, 1:] == best)  # upper-right cell of each largest square
    starts = ends - best + 1
    dist = np.linalg.norm(starts + (best - 1) / 2 - mid, axis=1)
    covered = np.zeros(len(starts))
    if prefer is not None:
        cs = np.pad(np.cumsum(np.cumsum(prefer, 0), 1), ((1, 0), (1, 0)))
        a, b = starts[:, 0], starts[:, 1]
        covered = cs[a + best, b + best] - cs[a, b + best] - cs[a + best, b] + cs[a, b]
    k = int(np.lexsort((dist, -covered))[0])
    return best, int(starts[k, 0]), int(starts[k, 1])
