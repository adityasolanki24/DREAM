"""Room regions from the robot's own occupancy map, and uncertain room-category weights.

Segmentation (classical morphological room segmentation, as surveyed by Bormann et al.,
ICRA 2016): free space farther than ``DOOR_CLEARANCE`` from anything not free forms room
"cores"; doorways and narrow gaps are cut because their centre is closer to a wall than that.
Each core of at least ``MIN_CORE_AREA`` seeds a region, and every free cell joins the core it
reaches first through free space (a watershed). Regions follow the map: they appear, grow,
merge or split as exploration proceeds, and keep their ids by overlap with the previous map.

Room categories are never assigned outright. Each region gets HEURISTIC RANKING WEIGHTS over
``knowledge.categories`` plus ``unknown`` from the objects tracked in it:

    score(c) = TEMPERING * sum over distinct labels l of e(l) * log LR(l, c);  score(unknown) = 0
    weights = softmax(score)

where LR is the knowledge's association ratio and e(l) in [0, 1] is the label's best
per-track confidence in the region (view purity times min(1, views/2)). Labels count once
per region however many instances carry them, and TEMPERING < 1 discounts the evidence
because co-located objects are not independent (a sink and a countertop arrive together).
These weights are a ranking with an explicit ``unknown`` option, not calibrated
probabilities; their accuracy is measured against ground truth in evaluation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from dream.perception.occupancy import CELL, OccupancyGrid
from dream.perception.semantic import Track
from dream.semantics.knowledge import UNKNOWN_ROOM, SemanticKnowledge

# A G1 needs about 0.7 m of clear width; doorways in homes are 0.8-1.3 m wide, so their centre
# lies within 0.65 m of a jamb. Rooms are wider than 2.5 m. 0.6 m separates the two.
DOOR_CLEARANCE = 0.6  # m
MIN_CORE_AREA = 0.8  # m^2: smaller cores are clutter gaps, not rooms
REGION_REACH = 1.5  # m: occupied/unknown cells belong to the region of a free cell this close
TEMPERING = 0.5  # dependence discount; a design constant, not fitted (see module docstring)
CLASSIFY_MIN = 0.5  # weight of the leading category before a region is reported as it
CLASSIFY_MARGIN = 0.2  # over the second; otherwise the classification is "ambiguous"


@dataclass
class Region:
    id: int
    area: float  # m^2 of free floor
    centroid: np.ndarray  # (2,) odometry frame
    tracks: list[int] = field(default_factory=list)
    evidence: dict[str, float] = field(default_factory=dict)  # label -> e(l)
    weights: dict[str, float] = field(default_factory=dict)  # category -> heuristic weight

    def leading(self) -> tuple[str, float, float]:
        """(category, weight, margin over the second)."""
        ranked = sorted(self.weights.items(), key=lambda kv: -kv[1])
        if not ranked:
            return UNKNOWN_ROOM, 0.0, 0.0
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return ranked[0][0], ranked[0][1], ranked[0][1] - second

    def classification(self) -> str:
        cat, w, margin = self.leading()
        return cat if w >= CLASSIFY_MIN and margin >= CLASSIFY_MARGIN else "ambiguous"

    def summary(self) -> dict:
        top = sorted(self.weights.items(), key=lambda kv: -kv[1])[:3]
        known = [kv for kv in self.weights.items() if kv[0] != UNKNOWN_ROOM]
        top1 = max(known, key=lambda kv: kv[1])[0] if self.evidence and known else "none"
        return {
            "id": self.id,
            "reference": f"region:{self.id}",
            "area_m2": round(self.area, 2),
            "centroid": np.round(self.centroid, 2).tolist(),
            "classification": self.classification(),
            "top_category": top1,  # leading non-unknown category; "none" without evidence
            "weights_heuristic": {k: round(v, 3) for k, v in top},
            "evidence": {k: round(v, 2) for k, v in sorted(self.evidence.items())},
            "objects": list(self.tracks),
        }


def label_confidence(track: Track, label: str) -> float:
    """e(l) for one track: view purity of ``label`` times min(1, views/2), minus misses."""
    views = sum(track.labels.values())
    if not views:
        return 0.0
    conf = track.purity(label) * min(1.0, views / 2)
    return float(conf * views / (views + track.misses))


def category_weights(
    evidence: dict[str, float], knowledge: SemanticKnowledge, tempering: float = TEMPERING
) -> dict[str, float]:
    """Heuristic weights over ``knowledge.categories`` and ``unknown`` from label evidence."""
    cats = knowledge.categories
    score = np.zeros(len(cats))
    for label, e in evidence.items():
        if e > 0 and knowledge.has(label):
            score += tempering * e * np.log(knowledge.likelihood_ratio(label))
    full = np.append(score, 0.0)  # unknown: no evidence moves it
    w = np.exp(full - full.max())
    w /= w.sum()
    return {**{c: float(v) for c, v in zip(cats, w[:-1], strict=True)}, UNKNOWN_ROOM: float(w[-1])}


class RoomSegmenter:
    """Region map of the robot's occupancy grid, with stable ids across updates."""

    def __init__(self, door_clearance: float = DOOR_CLEARANCE) -> None:
        self.door = door_clearance
        self.labels: np.ndarray | None = None  # (n, n) int32 region id per free cell, -1 none
        self.extended: np.ndarray | None = None  # region id for any cell within REGION_REACH
        self._next = 1

    def update(self, grid: OccupancyGrid) -> np.ndarray:
        free = grid.free()
        known = ~grid.unknown()
        if not known.any():
            self.labels = np.full(free.shape, -1, np.int32)
            self.extended = self.labels.copy()
            return self.labels
        # Work inside the known bounding box (plus a margin): the grid is mostly unexplored.
        ii, jj = np.nonzero(known)
        m = int(REGION_REACH / CELL) + 2
        i0, i1 = max(ii.min() - m, 0), min(ii.max() + m + 1, free.shape[0])
        j0, j1 = max(jj.min() - m, 0), min(jj.max() + m + 1, free.shape[1])
        sub = free[i0:i1, j0:j1]
        clearance = ndimage.distance_transform_edt(sub) * CELL
        cores, n = ndimage.label(clearance > self.door, structure=np.ones((3, 3)))
        sizes = ndimage.sum_labels(np.ones_like(cores), cores, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes * CELL * CELL >= MIN_CORE_AREA) + 1
        lab = np.where(np.isin(cores, keep), cores, 0).astype(np.int32)
        lab = _watershed(lab, sub)
        local = np.full(free.shape, 0, np.int32)
        local[i0:i1, j0:j1] = lab
        self.labels = self._stable_ids(local)
        # Occupied and unknown cells take the region of the nearest labelled free cell.
        has = self.labels[i0:i1, j0:j1] >= 0
        ext = np.full(free.shape, -1, np.int32)
        if has.any():
            d, (ni, nj) = ndimage.distance_transform_edt(~has, return_indices=True)
            near = self.labels[i0:i1, j0:j1][ni, nj]
            ext[i0:i1, j0:j1] = np.where(d * CELL <= REGION_REACH, near, -1)
        self.extended = ext
        return self.labels

    def _stable_ids(self, local: np.ndarray) -> np.ndarray:
        out = np.full(local.shape, -1, np.int32)
        previous = self.labels
        used: set[int] = set()
        for k in np.unique(local[local > 0]):
            cells = local == k
            rid = None
            if previous is not None:
                ids, counts = np.unique(previous[cells & (previous >= 0)], return_counts=True)
                for i in ids[np.argsort(-counts)]:
                    if int(i) not in used:
                        rid = int(i)
                        break
            if rid is None:
                rid = self._next
                self._next += 1
            used.add(rid)
            out[cells] = rid
        return out

    def region_of(self, grid: OccupancyGrid, xy) -> int:
        if self.extended is None:
            return -1
        i, j = grid.cell(np.asarray(xy, float)[:2])
        return int(self.extended[i, j]) if grid.inside(i, j) else -1


def _watershed(seeds: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Grow seed labels through free cells, one cell per round (geodesic nearest seed)."""
    lab = np.where(free, seeds, -1).astype(np.int32)
    lab[~free] = -1
    pending = free & (lab == 0)
    while pending.any():
        grown = lab.copy()
        changed = False
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = _shift(lab, di, dj)
            take = pending & (grown == 0) & (src > 0)
            if take.any():
                grown[take] = src[take]
                changed = True
        lab = grown
        pending = free & (lab == 0)
        if not changed:
            break
    lab[lab == 0] = -1
    return lab


def _shift(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    """``a`` moved by (di, dj) cells, filled with -1 (no wrap-around at the map edge)."""
    out = np.full_like(a, -1)
    n, m = a.shape
    out[max(di, 0) : n + min(di, 0), max(dj, 0) : m + min(dj, 0)] = a[
        max(-di, 0) : n + min(-di, 0), max(-dj, 0) : m + min(-dj, 0)
    ]
    return out


def object_front(track: Track) -> np.ndarray:
    """The point an object faces: its centre stepped toward the camera of its closest view
    (it was seen from that side, so that side is open floor). A centre alone would put a
    refrigerator against a wall into the room behind the wall."""
    cams = [c for views in track.label_views.values() for c in views]
    centre = track.position[:2]
    if cams:
        cam = min(cams, key=lambda c: float(np.linalg.norm(c - centre)))
        d = cam - centre
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            half = float(np.max(track.hi[:2] - track.lo[:2])) / 2
            centre = centre + d / n * min(n, half + 0.3)
    return centre


def object_region(grid: OccupancyGrid, segmenter: RoomSegmenter, track: Track) -> int:
    """The region an object faces: from its centre, step toward the camera of its closest
    view (it was seen from that side, so that side is open floor), then take that region.
    A centre alone would put a refrigerator against a wall into the room behind the wall."""
    return segmenter.region_of(grid, object_front(track))


def build_regions(
    grid: OccupancyGrid,
    segmenter: RoomSegmenter,
    tracks: list[Track],
    knowledge: SemanticKnowledge,
    tempering: float = TEMPERING,
) -> dict[int, Region]:
    labels = segmenter.labels
    regions: dict[int, Region] = {}
    if labels is None:
        return regions
    for rid in np.unique(labels[labels >= 0]):
        cells = np.argwhere(labels == rid)
        regions[int(rid)] = Region(
            int(rid), len(cells) * CELL * CELL, grid.centre(*cells.mean(0).T).astype(float)
        )
    for tr in tracks:
        rid = object_region(grid, segmenter, tr)
        if rid not in regions:
            continue
        reg = regions[rid]
        reg.tracks.append(tr.id)
        for label in tr.labels:
            reg.evidence[label] = max(reg.evidence.get(label, 0.0), label_confidence(tr, label))
    for reg in regions.values():
        reg.weights = category_weights(reg.evidence, knowledge, tempering)
    return regions
