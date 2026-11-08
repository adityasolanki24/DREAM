"""ORACLE-ASSISTED semantic detector: a stand-in for a neural detector, not perception.

It renders the true instance segmentation of the head camera's view (scene geoms only, robot
geoms excluded) and corrupts it with an assumed noise model, so that experiments can study how
a search policy copes with missed, confused and spurious detections before a real detector
(vision-language model) replaces it. Every number below is ASSUMED, not calibrated against any
real detector. Wherever its output is used or reported it must be labelled oracle-assisted.

Per frame, using only pixels with finite depth (<= MAX_RANGE) in the same frame:
- an instance with at least ``min_pixels`` visible is detected with probability
  logistic((visible pixels - a50) / slope);
- a detection is relabelled, with probability rising linearly with the median range of its
  pixels, to a uniformly drawn member of a confusion group containing its class (look-alike
  appliances and furniture, ``CONFUSION_GROUPS``);
- with probability ``false_positive`` per frame, a spurious detection with a random vocabulary
  label is placed on a random connected chunk (>= 300 px) of visible wall or furniture pixels;
- scores are drawn per case from overlapping distributions, so a score does not separate right
  from wrong labels. They are not calibrated probabilities.

``SemanticCamera.detect`` takes the simulator state and the frame's depth; the wrapper
``SemanticG1Base`` adds detections to the frames of a ``G1MobileBase`` without touching
g1_mobile.py. The kinematic surrogate base owns a ``SemanticCamera`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from dream.sim.camera import MAX_RANGE, ROBOT_GEOM_GROUP, CameraSpec
from dream.sim.home import CLASSES, flood
from dream.types import RGBDFrame, SemanticDetection

SOURCE = "oracle_segmentation_noisy"
# Symmetric look-alike groups (a dishwasher would join the stove group but is not in the
# vocabulary). Written by the benchmark author.
CONFUSION_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(g)
    for g in (
        ("refrigerator", "wardrobe", "chest_freezer", "storage_shelf"),
        ("sink", "washbasin"),
        ("stove", "countertop"),
        ("bed", "sofa"),
        ("desk", "dining_table", "workbench"),
        ("toilet", "washbasin"),
        ("kitchen_cabinet", "filing_cabinet", "nightstand"),
        ("bathtub", "shower"),
        ("washing_machine", "dryer", "chest_freezer"),
    )
)


def confusable_with(cls: str) -> tuple[str, ...]:
    """Classes sharing a confusion group with ``cls`` (sorted, excluding ``cls``)."""
    return tuple(sorted({c for g in CONFUSION_GROUPS if cls in g for c in g} - {cls}))


@dataclass(frozen=True)
class SemanticNoise:
    """Assumed (not calibrated) detector error model; see the module docstring."""

    a50: float = 250.0  # visible pixels at 320x240 giving a 50 % detection probability
    slope: float = 60.0  # pixels: logistic scale
    min_pixels: int = 30  # below this nothing is detected
    confusion_near: float = 0.05  # confusion probability at <= near_range
    confusion_far: float = 0.2  # ... rising linearly to this at far_range
    near_range: float = 1.5  # m
    far_range: float = 3.0  # m
    false_positive: float = 0.03  # spurious detections per frame
    fp_pixels: tuple[int, int] = (300, 1500)  # size range of a spurious chunk
    vocabulary: tuple[str, ...] = field(default=CLASSES)

    def detection_probability(self, pixels: float) -> float:
        if pixels < self.min_pixels:
            return 0.0
        return float(1.0 / (1.0 + np.exp(-(pixels - self.a50) / self.slope)))

    def confusion_probability(self, range_m: float) -> float:
        f = np.clip((range_m - self.near_range) / (self.far_range - self.near_range), 0.0, 1.0)
        return float(self.confusion_near + f * (self.confusion_far - self.confusion_near))


NOMINAL = SemanticNoise()
DEGRADED = replace(
    NOMINAL, a50=450.0, slope=100.0, confusion_near=0.1, confusion_far=0.4, false_positive=0.08
)


class SemanticCamera:
    """Segmentation renderer plus the noise model. ``index`` maps geom id to (instance, class)
    from ``dream.sim.home.semantic_index``. Owns its random stream."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_spec: CameraSpec,
        index: dict[int, tuple[str, str]],
        rng: np.random.Generator,
        noise: SemanticNoise = NOMINAL,
    ) -> None:
        self.model, self.spec, self.rng, self.noise = model, camera_spec, rng, noise
        self._renderer: mujoco.Renderer | None = None
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[ROBOT_GEOM_GROUP] = 0  # the robot never labels itself
        self._instances = sorted({iid for iid, _ in index.values()})
        self._classes = {iid: cls for iid, cls in index.values()}
        pos = {iid: k for k, iid in enumerate(self._instances)}
        self._lut = np.full(model.ngeom + 1, -1, dtype=np.int32)  # last slot: "no geom"
        for g, (iid, _) in index.items():
            self._lut[g] = pos[iid]
        self._scene_geom = np.zeros(model.ngeom + 1, bool)
        for g in range(model.ngeom):
            if model.geom(g).name.startswith(("wall:", "sem:")):
                self._scene_geom[g] = True

    def segmentation(self, data: mujoco.MjData) -> np.ndarray:
        """(H, W) int geom ids of the nearest geom per pixel, -1 where there is none."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.spec.height, self.spec.width)
            self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(data, camera=self.spec.name, scene_option=self._opt)
        raw = self._renderer.render()
        return np.where(raw[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, raw[..., 0], -1)

    def detect(self, data: mujoco.MjData, depth: np.ndarray) -> tuple[SemanticDetection, ...]:
        return self.detect_from_arrays(self.segmentation(data), depth)

    def detect_from_arrays(
        self, seg: np.ndarray, depth: np.ndarray
    ) -> tuple[SemanticDetection, ...]:
        """The noise model on given segmentation ids and depth (separate for testing)."""
        n, rng = self.noise, self.rng
        valid = np.isfinite(depth) & (depth <= MAX_RANGE)
        inst = np.where(valid, self._lut[seg], -1)  # seg == -1 indexes the "no geom" slot
        counts = np.bincount(inst[inst >= 0], minlength=len(self._instances))
        out: list[SemanticDetection] = []
        for k in np.flatnonzero(counts):
            if rng.random() >= n.detection_probability(counts[k]):
                continue
            mask = inst == k
            true = self._classes[self._instances[k]]
            label, score = true, float(np.clip(rng.normal(0.75, 0.12), 0.2, 0.99))
            if rng.random() < n.confusion_probability(float(np.median(depth[mask]))):
                mates = confusable_with(true)
                if mates:
                    label = mates[int(rng.integers(len(mates)))]
                    score = float(np.clip(rng.normal(0.6, 0.15), 0.2, 0.99))
            out.append(SemanticDetection(label, score, mask, SOURCE))
        if rng.random() < n.false_positive:
            chunk = self._chunk(valid & self._scene_geom[np.where(seg >= 0, seg, -1)])
            if chunk is not None:
                label = n.vocabulary[int(rng.integers(len(n.vocabulary)))]
                score = float(np.clip(rng.normal(0.5, 0.15), 0.2, 0.99))
                out.append(SemanticDetection(label, score, chunk, SOURCE))
        return tuple(out)

    def _chunk(self, scene: np.ndarray) -> np.ndarray | None:
        """A random connected chunk of scene pixels of 300 to 1500 px, or None."""
        lo, hi = self.noise.fp_pixels
        target = float(self.rng.uniform(lo, hi))
        r = int(np.ceil(0.65 * np.sqrt(target)))  # window half side that holds about target px
        ys, xs = np.nonzero(scene)
        for _ in range(5):
            if len(ys) == 0:
                return None
            i = int(self.rng.integers(len(ys)))
            y0, y1, x0, x1 = max(ys[i] - r, 0), ys[i] + r + 1, max(xs[i] - r, 0), xs[i] + r + 1
            region = flood(scene[y0:y1, x0:x1], (ys[i] - y0, xs[i] - x0))
            if region.sum() >= lo:
                mask = np.zeros_like(scene)
                mask[y0:y1, x0:x1] = region
                return mask
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class SemanticG1Base:
    """A ``G1MobileBase`` whose frames also carry oracle-assisted semantic detections. Rendering
    happens from the environment's own model and data at the capture state; the environment's
    behaviour and random streams are untouched (the detector has its own)."""

    def __init__(self, base, semantic: SemanticCamera) -> None:
        self.base, self.semantic = base, semantic

    @property
    def time(self) -> float:
        return self.base.time

    @property
    def stop_reason(self) -> str | None:
        return self.base.stop_reason

    def odometry(self) -> np.ndarray:
        return self.base.odometry()

    def walk(self, forward: float, lateral: float, turn: float, seconds: float) -> bool:
        return self.base.walk(forward, lateral, turn, seconds)

    def capture(self) -> RGBDFrame:
        frame = self.base.capture()
        detections = self.semantic.detect(self.base.env.data, frame.depth)
        return replace(frame, detections=detections)
