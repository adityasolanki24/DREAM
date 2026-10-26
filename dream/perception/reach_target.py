"""Where is the handle I was told to touch, and what is around it? From RGB-D detections only.

CLASSICAL BASELINE perception for the reach test, not open-world vision. Input: the detections
of ``dream.perception.detect.detect`` (colour and depth segmentation; a "handle" is a small
component protruding from the side of a box, no higher than the box top). The estimator keeps
the sightings of the handle whose colour was named in the goal, fuses them into a 3D position
in the ODOMETRY frame with a covariance, and turns every other detection into a perceived
obstacle box. The box the handle protrudes from is the fixture: the one surface the hand may
touch.

Covariance model, per sighting: depth variance (a + b z^2)^2 along the viewing ray (the head
camera's depth specification, known like a datasheet value) and a lateral pixel-quantisation
term, plus ONE systematic term common to all sightings (``HANDLE_BIAS_STD``, calibrated on
development scenes: the detector reports the handle's visible top-centre, not its centre).
Sightings are only fused while the robot stands still: odometry drifts while walking, so the
caller ``reset``s the track after moving and re-observes (the plan frame is the odometry frame
of the moment).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from dream.perception.detect import Detection

DEPTH_NOISE = (0.001, 0.0015)  # sigma = a + b z^2, the camera's specification
HANDLE_BIAS_STD = 0.004  # m, systematic term: rms 3D error of the fused estimate at 0.5-0.8 m
# range on development scenes (results/g1_reach_heights_dev/perception_calibration.json)
MIN_OBSTACLE_THICKNESS = 0.02  # m, half thickness floor for slabs seen face-on
MAX_SIGHTINGS = 8
FRAGMENT_GAP = 0.05  # m, same-coloured boxes this close to the fixture are pieces of it
MIN_OCCLUDED_DEPTH = 0.5  # m, assumed depth of the unseen volume behind a visible face


@dataclass(frozen=True)
class ObstacleBox:
    """A perceived box: odometry-frame centre, half extents (along its own x, y, z), yaw."""

    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    yaw: float = 0.0
    hand_may_touch: bool = False
    color: str = ""
    kind: str = "box"


@dataclass(frozen=True)
class TargetEstimate:
    position: np.ndarray  # (3,) handle position, odometry frame
    covariance: np.ndarray  # (3, 3)
    color: str
    n_views: int
    candidates: int  # distinct handles of that colour in view (1 = unambiguous)
    obstacles: tuple[ObstacleBox, ...] = ()
    fixture: ObstacleBox | None = None
    face_normal: tuple[float, float] | None = None  # outward normal of the fixture's front face
    time: float = 0.0

    @property
    def sigma(self) -> float:
        """Largest 1-sigma along any axis (m)."""
        return float(np.sqrt(max(np.linalg.eigvalsh(self.covariance).max(), 0.0)))


def _sighting_cov(det: Detection, cam: np.ndarray, noise: tuple[float, float]) -> np.ndarray:
    ray = det.position - cam
    z = float(np.linalg.norm(ray))
    u = ray / max(z, 1e-6)
    s_depth = noise[0] + noise[1] * z * z
    s_lat = max(det.position_std, 1e-3)
    return s_lat**2 * (np.eye(3) - np.outer(u, u)) + s_depth**2 * np.outer(u, u)


def _box_of(d: Detection, cam, may_touch: bool = False) -> ObstacleBox:
    """A detection as an obstacle box. A depth camera sees only the near surface, so the space
    behind it is unknown, not free: a box whose extent along the viewing direction is below
    ``MIN_OCCLUDED_DEPTH`` is extruded away from the camera to that depth (assumption)."""
    sides = [float(d.extent[0]), float(d.extent[1])]
    centre = np.array([float(d.position[0]), float(d.position[1])])
    c, s = np.cos(d.yaw), np.sin(d.yaw)
    axes = [np.array([c, s]), np.array([-s, c])]
    view = centre - np.asarray(cam, float)[:2]
    view = view / max(float(np.linalg.norm(view)), 1e-6)
    i = int(np.argmax([abs(float(a @ view)) for a in axes]))
    if d.kind == "box" and sides[i] < MIN_OCCLUDED_DEPTH:
        extra = MIN_OCCLUDED_DEPTH - sides[i]
        centre = centre + axes[i] * np.sign(float(axes[i] @ view)) * extra / 2
        sides[i] = MIN_OCCLUDED_DEPTH
    top = float(d.extent[2])
    half = (max(sides[0] / 2, MIN_OBSTACLE_THICKNESS), max(sides[1] / 2, MIN_OBSTACLE_THICKNESS),
            top / 2)  # fmt: skip
    return ObstacleBox(
        (float(centre[0]), float(centre[1]), top / 2), half, float(d.yaw), may_touch, d.color,
        d.kind,
    )  # fmt: skip


def _seat_behind(box: ObstacleBox, handle, normal, depth: float) -> ObstacleBox:
    """Move the fixture box along ``normal`` so its front face lies ``depth`` behind the handle
    centre: a handle protrudes from the face, so the face cannot be in front of it. The
    detector's face position is a depth-edge estimate that can sit a centimetre either way, and
    a planning copy with the face in front of the target would block the hand."""
    n = np.asarray(normal, float)
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    axes = [np.array([c, s]), np.array([-s, c])]
    i = int(np.argmax([abs(float(a @ n)) for a in axes]))
    front = float(np.array(box.center[:2]) @ n) + box.half_extents[i]
    want = float(np.asarray(handle[:2]) @ n) - depth
    shift = n * (want - front)
    return ObstacleBox(
        (box.center[0] + float(shift[0]), box.center[1] + float(shift[1]), box.center[2]),
        box.half_extents, box.yaw, box.hand_may_touch, box.color, box.kind,
    )  # fmt: skip


HANDLE_REACH = 0.07  # m, how far from a box's footprint a handle may lie (as in detect.py)
HANDLE_MAX_SIZE = 0.2  # m, a handle is smaller than this in every dimension


def as_handle(d: Detection, detections: list[Detection]) -> Detection | None:
    """``d`` as a handle, or None. The detector labels a small component "handle" only when its
    top-centre lies OUTSIDE a box's footprint rectangle; a thin face (a slab seen face-on, a
    few centimetres deep) has a footprint rectangle as deep as the depth noise, and a handle
    protruding a centimetre or two can fall inside it and stay an "object". So a small, flush
    object is also a handle: its top-centre within HANDLE_REACH of a box footprint (inside
    counts) and no higher than the box top. Handles the detector labelled are kept as they are."""
    if d.kind == "handle":
        return d
    if d.kind != "object" or d.top_centre is None or float(max(d.extent)) > HANDLE_MAX_SIZE:
        return None
    for b in detections:
        if b.kind != "box":
            continue
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        local = (d.top_centre[:2] - b.position[:2]) @ np.array([[c, -s], [s, c]])
        outside = np.abs(local) - np.asarray(b.extent[:2]) / 2
        if outside.max() < HANDLE_REACH and d.top_centre[2] <= b.extent[2] + 0.01:
            return replace(d, kind="handle", position=d.top_centre.copy())
    return None


def _rect_points(d: Detection, n: int = 12) -> np.ndarray:
    """A grid of points over a detection's footprint rectangle (odometry xy)."""
    c, s = np.cos(d.yaw), np.sin(d.yaw)
    u = np.linspace(-0.5, 0.5, n) * float(d.extent[0])
    v = np.linspace(-0.5, 0.5, n) * float(d.extent[1])
    g = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)
    return g @ np.array([[c, s], [-s, c]]) + d.position[:2]


def _fragment_of(d: Detection, fixture: Detection) -> bool:
    """Another piece of the fixture's face: a box of the same colour within FRAGMENT_GAP of it."""
    if d.kind != "box" or d.color != fixture.color:
        return False
    a, b = _rect_points(d), _rect_points(fixture)
    gap = float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))
    return gap < FRAGMENT_GAP


@dataclass
class HandleEstimator:
    color: str
    noise: tuple[float, float] = DEPTH_NOISE
    bias_std: float = HANDLE_BIAS_STD
    _sightings: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    _last: list[Detection] = field(default_factory=list)
    _cam: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _time: float = 0.0
    _candidates: int = 0
    _fixture_det: Detection | None = None
    _handle_det: Detection | None = None

    def reset(self) -> None:
        self._sightings, self._last, self._candidates = [], [], 0

    def ingest(self, detections: list[Detection], camera_position, time: float = 0.0) -> bool:
        """Add one frame's detections; True if the named handle was seen in it."""
        cam = np.asarray(camera_position, float)
        self._cam, self._time, self._last = cam, time, list(detections)
        mine = [
            d
            for d in (as_handle(x, detections) for x in detections)
            if d is not None and d.color == self.color and not d.occluded
        ]
        self._candidates = max(self._candidates, len(mine))
        if not mine:
            return False
        d = min(mine, key=lambda d: float(np.linalg.norm(d.position - cam)))
        self._handle_det = d
        self._sightings.append((d.position.copy(), _sighting_cov(d, cam, self.noise)))
        self._sightings = self._sightings[-MAX_SIGHTINGS:]
        return True

    @property
    def seen(self) -> bool:
        return bool(self._sightings)

    def estimate(self) -> TargetEstimate | None:
        if not self._sightings:
            return None
        info = sum(np.linalg.inv(c) for _, c in self._sightings)
        cov = np.linalg.inv(info)
        pos = cov @ sum(np.linalg.inv(c) @ p for p, c in self._sightings)
        cov = cov + self.bias_std**2 * np.eye(3)
        self._fixture_det = None
        fixture, normal = self._fixture(pos)
        obstacles = []
        for d in self._last:
            h = as_handle(d, self._last)
            if h is not None and h.color == self.color:
                continue  # the target itself (or another handle of its colour)
            if d is self._fixture_det:
                obstacles.append(fixture)
            elif self._fixture_det is not None and _fragment_of(d, self._fixture_det):
                # the same face split into pieces (shading, the handle's shadow): it carries the
                # target too, and lies behind the handle like the rest of the face
                box = _box_of(d, self._cam, True)
                obstacles.append(_seat_behind(box, pos, normal, self._handle_depth()))
            elif d.kind in ("box", "container", "object", "handle"):
                obstacles.append(_box_of(d, self._cam, False))
        return TargetEstimate(
            pos, cov, self.color, len(self._sightings), self._candidates, tuple(obstacles),
            fixture, normal, self._time,
        )  # fmt: skip

    def _fixture(self, handle: np.ndarray) -> tuple[ObstacleBox | None, tuple[float, float] | None]:
        """The box whose footprint the handle protrudes from (nearest within 0.1 m), and the
        outward normal of that face (the footprint axis most aligned with the offset)."""
        best, best_gap = None, 0.1
        for d in self._last:
            if d.kind != "box":
                continue
            c, s = np.cos(d.yaw), np.sin(d.yaw)
            local = (handle[:2] - d.position[:2]) @ np.array([[c, -s], [s, c]])
            gap = float(np.max(np.abs(local) - np.asarray(d.extent[:2]) / 2))
            if gap < best_gap:
                best, best_gap = d, gap
        if best is None:
            return None, None
        rel = handle[:2] - best.position[:2]
        axes = [np.array([np.cos(best.yaw), np.sin(best.yaw)])]
        axes.append(np.array([-axes[0][1], axes[0][0]]))
        ax = max(axes, key=lambda u: abs(u @ rel))
        n = ax * np.sign(ax @ rel)
        self._fixture_det = best
        box = _seat_behind(_box_of(best, self._cam, True), handle, n, self._handle_depth())
        return box, (float(n[0]), float(n[1]))

    def _handle_depth(self) -> float:
        """Distance from the handle's centre to the face it protrudes from: half the handle's
        smaller visible footprint side (assumption: a bar handle, thinner than wide)."""
        d = self._handle_det
        return 0.5 * float(min(d.extent[:2])) if d is not None else 0.02
