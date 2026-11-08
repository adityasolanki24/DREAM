"""Where the requested object might be: a search belief over the robot's own map, and what a
look from a candidate viewpoint is expected to reveal.

The belief is a HYPOTHESIS layer. It is never written into the factual object tracks or the
scene graph's observed relations, and nothing in it is an observation.

Model (classical probabilistic search, as in Koopman's search theory and Chung & Burdick's
review of search with uncertain detection):

- Hypothesis space: the target stands at one *candidate cell*: an occupied cell (a mapped
  surface; a refrigerator in front of a wall is hit before the wall is) or an unknown cell
  connected to known free floor through unknown space within ``UNKNOWN_REACH`` (unexplored
  floor, rooms behind doorways, pockets behind furniture). Free floor is not a hypothesis.
- Prior density per cell: ``s(cell)``, the semantic factor of its region: the expected
  knowledge likelihood ratio of the target under the region's heuristic room weights (1 for
  unexplored space and for unknown rooms), plus a bump around remembered sightings. With
  ``semantic=False`` every cell has factor 1 (pure coverage search).
- Negative evidence: ``coverage(cell)`` (dream/perception/semantic.py) is the best assumed
  detection quality reached on that surface. Mass is ``prior * (1 - coverage)``, the
  posterior under the assumed detector model when the coverage looks are treated as one
  look. The absolute values are not calibrated; only their ratios drive decisions.
- A look from viewpoint ``v`` (a full turn in place) raises coverage on the cells its rays
  reach before the first occupied cell, unknown cells transparent for at most
  ``UNKNOWN_SEE_THROUGH``. Its expected detection value is the mass it would explain,
  ``sum m(c) * (q' - q) / (1 - q)``, in units of ``DETECTION_UNIT`` neutral cells (about one
  room's surfaces at factor 1). Not a share of the total: much of the total sits in unknown
  space no reachable view can see (outside unobserved stretches of outer wall), and dividing
  by it made every semantic difference vanish on development home 100-B. Its information
  gain is the unknown area its rays reach (m^2).

``unobserved_pockets`` measures what the search could not see: unknown space inside the
explored area large enough to hide an object. A search is only "exhausted" without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from dream.perception.occupancy import CELL, OccupancyGrid
from dream.perception.semantic import SemanticMap, Track, detection_quality
from dream.semantics.knowledge import UNKNOWN_ROOM, SemanticKnowledge
from dream.semantics.rooms import Region, RoomSegmenter, category_weights, label_confidence

UNKNOWN_REACH = 2.5  # m: unexplored space credited with mass (the frontier gain's sense range)
UNKNOWN_SEE_THROUGH = 1.5  # m of unknown cells a hypothetical ray may cross (optimistic)
VIEW_RANGE = 3.0  # m: the depth camera's range
RAYS = 180  # one ray per 2 degrees of a full turn
NEAR_OBJECTS = 2.5  # m: neighbourhood used instead of regions when room_inference=False
MEMORY_PEAK = 5.0  # prior density at a remembered sighting relative to a neutral cell
MEMORY_SIGMA = 0.5  # m
DETECTION_UNIT = 1000.0  # neutral cells (~2.5 m^2 at 5 cm): a room's walls and furniture faces
POCKET_RADIUS = 0.2  # m: an unseen region must fit a 0.4 m object (a toilet's width) to count


@dataclass(frozen=True)
class BeliefConfig:
    """Which knowledge the belief may use (ablations switch one off at a time)."""

    semantic: bool = True  # knowledge-based room priors
    room_inference: bool = True  # False: object co-occurrence within NEAR_OBJECTS instead
    memory: bool = True  # remembered sightings from persistent memory


@dataclass
class ViewValue:
    xy: np.ndarray
    detection: float  # expected target mass explained, in DETECTION_UNITs
    gain_m2: float  # unknown area reached
    cells: int


@dataclass
class TargetBelief:
    target: str
    knowledge: SemanticKnowledge
    config: BeliefConfig = field(default_factory=BeliefConfig)
    memory_hints: list[dict] = field(default_factory=list)  # {"xy", "confidence", "id"}
    candidate: np.ndarray | None = None  # (n, n) bool
    prior: np.ndarray | None = None  # (n, n) float
    mass: np.ndarray | None = None  # prior * (1 - coverage) on candidate cells
    region_factor: dict[int, float] = field(default_factory=dict)

    # ----- update ------------------------------------------------------------------------------
    def update(
        self,
        grid: OccupancyGrid,
        semantic: SemanticMap,
        segmenter: RoomSegmenter,
        regions: dict[int, Region],
    ) -> None:
        occ, unknown, free = grid.occupied(), grid.unknown(), grid.free()
        reach = int(UNKNOWN_REACH / CELL)
        # Unknown space reachable from known floor through unknown space only (not through
        # walls), within UNKNOWN_REACH.
        grown = free.copy()
        frontier_unknown = np.zeros_like(free)
        for _ in range(reach):
            nxt = ndimage.binary_dilation(grown, structure=np.ones((3, 3), bool)) & unknown
            nxt &= ~frontier_unknown
            if not nxt.any():
                break
            frontier_unknown |= nxt
            grown = nxt
        cand = occ | frontier_unknown
        factor = np.ones(occ.shape, np.float32)
        lr = self.knowledge.likelihood_ratio(self.target)
        cats = self.knowledge.categories
        self.region_factor = {}
        if self.config.semantic:
            if self.config.room_inference:
                ext = segmenter.extended
                for rid, reg in regions.items():
                    f = _expected_ratio(reg.weights, cats, lr)
                    self.region_factor[rid] = f
                    if ext is not None:
                        factor[(ext == rid) & cand] = f
            else:
                factor = _local_factor(grid, semantic.tracks, self.knowledge, lr, cand)
        prior = np.where(cand, factor, 0.0).astype(np.float32)
        if self.config.memory and self.memory_hints:
            ii, jj = np.nonzero(cand)
            pts = grid.centre(ii, jj)
            for hint in self.memory_hints:
                d2 = np.sum((pts - np.asarray(hint["xy"], float)[:2]) ** 2, axis=1)
                bump = MEMORY_PEAK * float(hint.get("confidence", 1.0))
                prior[ii, jj] += (bump * np.exp(-d2 / (2 * MEMORY_SIGMA**2))).astype(np.float32)
        self.candidate, self.prior = cand, prior
        self.mass = prior * (1.0 - semantic.coverage)

    # ----- queries -----------------------------------------------------------------------------
    def total(self) -> float:
        return 0.0 if self.mass is None else float(self.mass.sum())

    def region_mass(self, segmenter: RoomSegmenter) -> dict[int, float]:
        """Share of the current mass per region (-1: unexplored/unassigned space)."""
        if self.mass is None or segmenter.extended is None:
            return {}
        total = self.total()
        ext = segmenter.extended
        ids = np.unique(ext[self.mass > 0])
        return {int(r): float(self.mass[ext == r].sum() / total) for r in ids} if total else {}

    def view(self, grid: OccupancyGrid, semantic: SemanticMap, xy) -> ViewValue:
        """Expected value of a full turn in place at ``xy`` (rays over the robot's own map)."""
        cells, rng = _visible_cells(grid, np.asarray(xy, float)[:2])
        if not len(cells) or self.mass is None:
            return ViewValue(np.asarray(xy, float), 0.0, 0.0, 0)
        i, j = cells[:, 0], cells[:, 1]
        q = semantic.coverage[i, j]
        q_new = np.maximum(q, detection_quality(rng))
        gained = np.where(q < 1.0, (q_new - q) / np.maximum(1.0 - q, 1e-6), 0.0)
        detection = float((self.mass[i, j] * gained).sum() / DETECTION_UNIT)
        gain = float(grid.unknown()[i, j].sum() * CELL * CELL)
        return ViewValue(np.asarray(xy, float), detection, gain, len(cells))


def _expected_ratio(weights: dict[str, float], cats, lr: np.ndarray) -> float:
    """E[LR(target, room)] under the region's heuristic weights; 'unknown' counts as 1."""
    f = weights.get(UNKNOWN_ROOM, 0.0)
    for k, c in enumerate(cats):
        f += weights.get(c, 0.0) * float(lr[k])
    return float(f) if weights else 1.0


def _local_factor(grid, tracks: list[Track], knowledge, lr, cand) -> np.ndarray:
    """Ablation of room inference: each cell's factor from the objects within NEAR_OBJECTS of
    it (distinct labels, same evidence rule), on a 0.5 m lattice."""
    factor = np.ones(cand.shape, np.float32)
    if not tracks:
        return factor
    step = int(0.5 / CELL)
    ii, jj = np.nonzero(cand)
    if not len(ii):
        return factor
    keys = np.stack([ii // step, jj // step], 1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    centres = grid.centre(uniq[:, 0] * step + step // 2, uniq[:, 1] * step + step // 2)
    pos = np.array([t.position[:2] for t in tracks])
    vals = np.ones(len(uniq), np.float32)
    for k, c in enumerate(centres):
        near = np.flatnonzero(np.linalg.norm(pos - c, axis=1) <= NEAR_OBJECTS)
        if not len(near):
            continue
        evidence: dict[str, float] = {}
        for n in near:
            tr = tracks[n]
            for label in tr.labels:
                evidence[label] = max(evidence.get(label, 0.0), label_confidence(tr, label))
        vals[k] = _expected_ratio(category_weights(evidence, knowledge), knowledge.categories, lr)
    factor[ii, jj] = vals[inverse.ravel()]
    return factor


def _visible_cells(grid: OccupancyGrid, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique cells a full turn at ``xy`` would see, and their ranges (m)."""
    angles = np.linspace(0, 2 * np.pi, RAYS, endpoint=False)
    radii = np.arange(CELL, VIEW_RANGE + 1e-9, CELL / 2)
    pts = (
        xy[None, None, :]
        + radii[None, :, None] * np.stack([np.cos(angles), np.sin(angles)], -1)[:, None, :]
    )
    i, j = grid.cell(pts)
    inside = grid.inside(i, j)
    i, j = np.clip(i, 0, grid.n - 1), np.clip(j, 0, grid.n - 1)
    occ = grid.occupied()[i, j] & inside
    unk = grid.unknown()[i, j] | ~inside
    # A ray ends at its first occupied cell (inclusive) or after UNKNOWN_SEE_THROUGH of unknown.
    blocked_after = np.cumsum(occ, axis=1) - occ  # occupied cells strictly before this sample
    unknown_run = np.cumsum(unk, axis=1) * (CELL / 2)
    seen = (blocked_after == 0) & (unknown_run <= UNKNOWN_SEE_THROUGH) & inside
    r = np.broadcast_to(radii[None, :], seen.shape)[seen]
    flat = i[seen] * grid.n + j[seen]
    uniq, first = np.unique(flat, return_index=True)
    return np.stack([uniq // grid.n, uniq % grid.n], 1), r[first]


def unobserved_pockets(grid: OccupancyGrid, radius: float = POCKET_RADIUS) -> dict:
    """Unknown space inside the convex hull of the known free floor that could hide an object
    of width 2 * radius (an opening of the unknown mask with a disk of that radius)."""
    from scipy.spatial import Delaunay

    free = np.argwhere(grid.free())
    if len(free) < 4:
        return {"area_m2": 0.0, "pockets": 0}
    try:
        hull = Delaunay(free[:: max(1, len(free) // 4000)].astype(float))
    except Exception:  # degenerate (collinear) free space
        return {"area_m2": 0.0, "pockets": 0}
    unknown = grid.unknown()
    lo, hi = free.min(0), free.max(0) + 1
    sub = unknown[lo[0] : hi[0], lo[1] : hi[1]]
    ii, jj = np.nonzero(sub)
    inside = np.zeros_like(sub)
    if len(ii):
        pts = np.stack([ii + lo[0], jj + lo[1]], 1).astype(float)
        inside[ii, jj] = hull.find_simplex(pts) >= 0
    r = int(np.ceil(radius / CELL))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    disk = yy * yy + xx * xx <= r * r
    opened = ndimage.binary_opening(inside, structure=disk)
    labels, n = ndimage.label(opened)
    return {"area_m2": round(float(opened.sum()) * CELL * CELL, 2), "pockets": int(n)}
