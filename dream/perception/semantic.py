"""Semantic object evidence for a mobile robot: persistent object tracks built from a semantic
detector's per-frame regions, and a coverage layer that says how well each mapped surface
has been looked at.

Built only from the robot's own frames (calibration in its odometry frame). The detector may
be a vision-language model or the simulator's ORACLE-ASSISTED detector; either way a
detection is a fallible observation, never a fact:

- A ``Track`` is one physical thing seen one or more times. Detections join a track by
  footprint overlap, whatever their label, because a detector may call the same refrigerator a
  wardrobe from far away. Each track keeps per-label view counts, the camera positions it was
  labelled from, and how often it should have been detected (in view, unoccluded, close) and
  was not. Identity and label are judged later from that history (dream/semantics/verify.py).
- ``coverage`` holds, per occupancy cell, the best detection quality any frame has achieved
  on that surface: the agent's ASSUMED detector sensitivity at the range it was seen
  (``detection_quality``). A surface covered well without a target detection is evidence the
  target is not there; an unseen or poorly seen one is not.

Per-episode state; persistent memory is dream/memory/home.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dream.perception.geometry import backproject, visibility
from dream.perception.occupancy import CELL, OBSTACLE_MAX, OBSTACLE_MIN, OccupancyGrid
from dream.types import RGBDFrame, SemanticDetection

MIN_PIXELS = 40  # finite-depth pixels for a region to count (a ~10x4 px blob at 320x240)
DEPTH_BAND = 0.5  # m: points farther than this behind a region's nearest depth mode are background
ASSOCIATE_IOU = 0.25  # footprint IoU joining a detection to a track
ASSOCIATE_DISTANCE = 0.35  # m: or centroids this close
CLOSE_RANGE = 2.5  # m: a miss only counts within this range (the coverage model's good band)
COVERAGE_STRIDE = 4  # pixels


def detection_quality(range_m) -> np.ndarray:
    """The agent's ASSUMED probability of detecting a large object at ``range_m`` metres when
    its surface is in view and unoccluded: 0.85 out to 2 m, falling linearly to 0.4 at 3 m,
    nothing beyond the 3 m depth limit. A sensor model, not a calibration of any detector."""
    r = np.asarray(range_m, float)
    q = np.where(r <= 2.0, 0.85, 0.85 - 0.45 * (r - 2.0))
    return np.where((r < 0.3) | (r > 3.0), 0.0, q)


@dataclass
class Track:
    id: int
    position: np.ndarray  # (3,) odometry frame, robust centroid of the visible surface
    lo: np.ndarray  # (3,) footprint/extent box of everything seen
    hi: np.ndarray
    labels: dict[str, int] = field(default_factory=dict)  # views per label
    label_views: dict[str, list] = field(default_factory=dict)  # label -> camera xy per view
    closest: dict[str, float] = field(default_factory=dict)  # label -> nearest range (m)
    last_labelled: dict[str, float] = field(default_factory=dict)  # label -> last time (s)
    weight: float = 0.0
    views: int = 0
    misses: int = 0  # in view, unoccluded, within CLOSE_RANGE, and not detected
    first_seen: float = 0.0
    last_seen: float = 0.0

    def support(self, label: str) -> int:
        return self.labels.get(label, 0)

    def purity(self, label: str) -> float:
        total = sum(self.labels.values())
        return self.labels.get(label, 0) / total if total else 0.0

    def top_label(self) -> str:
        return max(self.labels, key=lambda k: (self.labels[k], k)) if self.labels else ""

    def summary(self) -> dict:
        return {
            "id": self.id,
            "reference": f"object:{self.id}",
            "labels": dict(sorted(self.labels.items(), key=lambda kv: -kv[1])),
            "position": np.round(self.position, 3).tolist(),
            "extent": np.round(self.hi - self.lo, 3).tolist(),
            "views": self.views,
            "misses": self.misses,
            "closest_m": {k: round(v, 2) for k, v in self.closest.items()},
        }


def _iou_xy(lo_a, hi_a, lo_b, hi_b) -> float:
    w = np.minimum(hi_a[:2], hi_b[:2]) - np.maximum(lo_a[:2], lo_b[:2])
    if (w <= 0).any():
        return 0.0
    inter = float(w[0] * w[1])
    area = lambda lo, hi: float(np.prod(np.maximum(hi[:2] - lo[:2], 1e-3)))  # noqa: E731
    return inter / (area(lo_a, hi_a) + area(lo_b, hi_b) - inter)


def region_points(points: np.ndarray, mask: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """3-D points of a detection region, background removed: a bounding-box region from a
    VLM includes whatever is behind the object, so keep points within DEPTH_BAND of the
    nearest well-populated range mode."""
    pts = points[mask]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < MIN_PIXELS:
        return pts[:0]
    r = np.linalg.norm(pts - cam, axis=1)
    hist, edges = np.histogram(r, bins=np.arange(r.min(), r.max() + 0.1, 0.1))
    dense = np.flatnonzero(hist >= max(MIN_PIXELS // 4, 0.1 * hist.max()))
    near = edges[dense[0]] if len(dense) else r.min()
    keep = (r >= near - 0.05) & (r <= near + DEPTH_BAND)
    pts = pts[keep]
    above = pts[:, 2] > OBSTACLE_MIN  # floor inside a box region is not the object
    return pts[above] if above.sum() >= MIN_PIXELS // 2 else pts


class SemanticMap:
    """Object tracks and surface coverage for one episode, on the robot's occupancy grid."""

    def __init__(self, grid: OccupancyGrid) -> None:
        self.grid = grid
        self.tracks: list[Track] = []
        self.coverage = np.zeros_like(grid.logodds, np.float32)
        self.frames = 0
        self._next = 1

    def update(self, frame: RGBDFrame) -> list[int]:
        """Integrate one frame (after the occupancy grid has); returns the track ids detected.

        A frame no detector ran on (``detections is None``) adds neither coverage nor misses:
        not looking for the target is not evidence of its absence."""
        if frame.detections is None:
            return []
        points = backproject(frame)
        cam = frame.calibration.position
        self._cover(points, frame, cam)
        valid = np.isfinite(frame.depth) & ~frame.self_mask
        seen: list[int] = []
        for det in frame.detections:
            pts = region_points(points, det.mask & valid, cam)
            if len(pts) < MIN_PIXELS:
                continue
            seen.append(self._associate(det, pts, cam, frame.time))
        self._count_misses(frame, set(seen))
        self.frames += 1
        return seen

    def _cover(self, points: np.ndarray, frame: RGBDFrame, cam: np.ndarray) -> None:
        s = COVERAGE_STRIDE
        pts = points[::s, ::s].reshape(-1, 3)
        own = frame.self_mask[::s, ::s].reshape(-1)
        ok = np.isfinite(pts).all(axis=1) & ~own
        pts = pts[ok]
        surface = pts[(pts[:, 2] > OBSTACLE_MIN) & (pts[:, 2] < OBSTACLE_MAX)]
        if not len(surface):
            return
        q = detection_quality(np.linalg.norm(surface - cam, axis=1)).astype(np.float32)
        i, j = self.grid.cell(surface[:, :2])
        inside = self.grid.inside(i, j)
        np.maximum.at(self.coverage, (i[inside], j[inside]), q[inside])

    def _associate(self, det: SemanticDetection, pts: np.ndarray, cam, t: float) -> int:
        lo, hi = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
        lo[2], hi[2] = 0.0, float(np.percentile(pts[:, 2], 98))  # standing on the floor
        centre = np.median(pts, axis=0)
        rng = float(np.linalg.norm(centre - cam))
        best, best_score = None, 0.0
        for tr in self.tracks:
            iou = _iou_xy(lo, hi, tr.lo, tr.hi)
            close = float(np.linalg.norm(tr.position[:2] - centre[:2])) < ASSOCIATE_DISTANCE
            score = max(iou, 1.0 if close else 0.0)
            if score >= ASSOCIATE_IOU and score > best_score:
                best, best_score = tr, score
        weight = 1.0 / max(rng, 0.5) ** 2  # depth noise grows with range squared
        if best is None:
            best = Track(self._next, centre, lo, hi, first_seen=t)
            self._next += 1
            self.tracks.append(best)
        else:
            w = weight / (best.weight + weight)
            best.position = (1 - w) * best.position + w * centre
            best.lo, best.hi = np.minimum(best.lo, lo), np.maximum(best.hi, hi)
        best.weight += weight
        best.views += 1
        best.last_seen = t
        best.labels[det.label] = best.labels.get(det.label, 0) + 1
        best.label_views.setdefault(det.label, []).append(np.asarray(cam[:2], float).copy())
        best.closest[det.label] = min(best.closest.get(det.label, np.inf), rng)
        best.last_labelled[det.label] = t
        return best.id

    def _count_misses(self, frame: RGBDFrame, seen: set[int]) -> None:
        """A track whose centre this frame plainly saw at close range, without any detection
        joining it, earns a miss: evidence against it (a hallucination, or a moved object)."""
        for tr in self.tracks:
            if tr.id in seen:
                continue
            centre = tr.position.copy()
            centre[2] = min(max(centre[2], 0.3), 1.0)
            if visibility(frame, centre, max_range=CLOSE_RANGE) == "visible":
                tr.misses += 1

    # ----- queries ---------------------------------------------------------------------------
    def candidates(self, label: str) -> list[Track]:
        return sorted(
            (t for t in self.tracks if t.support(label) > 0),
            key=lambda t: (-t.support(label), t.id),
        )

    def covered_fraction(self, cells: np.ndarray) -> float:
        if not len(cells):
            return 0.0
        return float(self.coverage[cells[:, 0], cells[:, 1]].mean())

    @property
    def cell_area(self) -> float:
        return CELL * CELL
