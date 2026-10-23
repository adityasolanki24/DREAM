"""``VisionPerception``: the agent-facing ``Perception`` backend built on camera frames.

Every call that needs fresh information captures a new RGB-D frame from each camera (unless the
robot clock has not advanced since the last capture), runs detection per camera, fuses the
detections and updates the world model. Observations are converted to the same types the
oracle tracker produces, so the runtime, tools and skills do not know which backend is active.
``backend`` names it in logs.

Fusion across cameras: detections of the same class and colour closer than ``FUSE`` are one
entity. The view without occlusion and with more pixels wins, and a receptacle keeps the
largest opening any camera observed (an opening seen by one camera is open space; another
camera not seeing it is occlusion, not evidence of a lid).

Articulated objects: a receptacle with a handle is reported through ``observe_articulated``.
- The handle position is measured from images.
- The handle frame is the receptacle side the handle protrudes from, seen while closed. The
  camera cannot recover a round knob's orientation, so ``orientation_observed`` is False.
- ``handle_visible`` is False when the latest frames do not show the handle; its position is
  then last-known only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from dream.perception.detect import Detection, detect
from dream.perception.world import OPENING_MAX_AGE_S, Track, WorldModel, entity_class
from dream.types import ArticulatedObservation, ObjectObservation, RGBDFrame

STD_SCALE = 1.0  # uncertainty inflation (see the README for the measured calibration)
# Handle position uncertainty, from results/handle_measurement_calibration (development scenes):
# camera error p90 6.7 mm with the knob in clear view, 24.3 mm with the hand on or above it.
HANDLE_STD_CLEAR = 0.007  # m
HANDLE_STD_OCCLUDED = 0.025  # m
FUSE = {"small": 0.03, "receptacle": 0.1}  # m


class VisionPerception:
    backend = "vision"

    def __init__(
        self,
        frame_source: Callable[[], RGBDFrame] | list[Callable[[], RGBDFrame]],
        clock: Callable[[], float],
        rng: np.random.Generator,
        memory: bool = True,
    ) -> None:
        """``frame_source``: one camera, or a list of cameras (fused). ``memory=False`` is the
        single-frame ablation: entities not in the latest frames are forgotten instead of
        being kept as last-known estimates."""
        self._sources = frame_source if isinstance(frame_source, list) else [frame_source]
        self._clock = clock
        self._rng = rng
        self.memory = memory
        self.world = WorldModel()
        self.last_frame: RGBDFrame | None = None  # the first camera's latest frame
        self.last_frames: list[RGBDFrame] = []
        self.last_detections: list[Detection] = []  # fused
        self._planes: dict[int, tuple] = {}
        self._face: dict[int, np.ndarray] = {}  # receptacle track id -> outward face normal
        self.frames = 0

    def refresh(self) -> None:
        """Capture and process new frames if the robot clock has moved since the last ones."""
        if self.last_frame is not None and self._clock() <= self.last_frame.time:
            return
        frames, per_camera = [], []
        for i, source in enumerate(self._sources):
            frame = source()
            dets, plane = detect(frame, self._rng, self._planes.get(i))
            self._planes.setdefault(i, plane)  # the support plane is static: fit it once
            frames.append(frame)
            per_camera.append(dets)
        fused = fuse(per_camera)
        self.world.update(fused, frames[0].time)
        if not self.memory:
            self.world.tracks = [t for t in self.world.tracks if t.visible]
        self.last_frame, self.last_frames, self.last_detections = frames[0], frames, fused
        self.frames += len(frames)

    def observe_objects(self) -> list[ObjectObservation]:
        self.refresh()
        names = self.world.names()
        out = []
        for t in self.world.tracks:
            d = t.detection
            if d.kind != "object":
                continue
            out.append(
                ObjectObservation(
                    name=names[t.id],
                    color=d.color,
                    position=d.position.copy(),
                    yaw=d.yaw,
                    size=d.size,
                    position_std=STD_SCALE * d.position_std,
                    visible=t.visible,
                    last_seen=t.last_seen,
                )
            )
        return out

    def fixtures(self) -> dict[str, tuple[np.ndarray, float]]:
        """Regions things can be put into: open containers, and the observed opening of a
        compartment. A compartment with no observed opening is not a fixture."""
        self.refresh()
        names = self.world.names()
        out = {}
        for t in self.world.tracks:
            d = t.detection
            if entity_class(d.kind) != "receptacle":
                continue
            if t.articulated:
                if (
                    d.opening is not None
                    and t.visible
                    and t.opening_last_seen is not None
                    and self._clock() - t.opening_last_seen <= OPENING_MAX_AGE_S
                ):
                    out[names[t.id]] = (d.opening[0].copy(), float(d.opening[1]))
            elif d.kind == "container":
                out[names[t.id]] = (d.position.copy(), d.inner_half)
        return out

    def observe_articulated(self) -> list[ArticulatedObservation]:
        self.refresh()
        names = self.world.names()
        out = []
        for t in self.world.tracks:
            if not t.articulated:
                continue
            h = self.world.handle_of(t)
            if h is None:
                continue
            normal = self._face_normal(t, h)
            yaw = float(np.arctan2(normal[1], normal[0]))
            d = t.detection
            ca, sa = np.abs(np.cos(d.yaw)), np.abs(np.sin(d.yaw))
            a, b, top = d.extent
            box_half = np.array([(ca * a + sa * b) / 2, (sa * a + ca * b) / 2, top / 2])
            out.append(
                ArticulatedObservation(
                    name=names[t.id],
                    handle_pos=h.detection.position.copy(),
                    handle_quat=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
                    box_centre=np.array([d.position[0], d.position[1], top / 2]),
                    box_half=box_half,
                    handle_half=_handle_half(h.detection, normal),
                    orientation_observed=False,
                    handle_visible=h.visible,
                    position_std=HANDLE_STD_OCCLUDED if h.detection.occluded else HANDLE_STD_CLEAR,
                    handle_id=h.id,
                    last_seen=h.last_seen,
                )
            )
        return out

    def _face_normal(self, receptacle: Track, handle: Track) -> np.ndarray:
        """Outward normal of the receptacle side the handle protrudes from, fixed at first
        sight (later views of an opened box no longer show which side was the front)."""
        n = self._face.get(receptacle.id)
        if n is None:
            d = receptacle.detection
            rel = handle.detection.position[:2] - d.position[:2]
            axes = [np.array([np.cos(d.yaw), np.sin(d.yaw)])]
            axes.append(np.array([-axes[0][1], axes[0][0]]))
            ax = max(axes, key=lambda u: abs(u @ rel))
            n2 = ax * np.sign(ax @ rel)
            n = np.array([n2[0], n2[1], 0.0])
            self._face[receptacle.id] = n
        return n


def _handle_half(h: Detection, normal: np.ndarray) -> np.ndarray:
    """Half extent of the handle in its frame: x along the face normal, y along the face."""
    u = np.array([np.cos(h.yaw), np.sin(h.yaw)])
    a, b, height = h.extent
    along_normal, along_face = (a, b) if abs(u @ normal[:2]) > 0.7 else (b, a)
    return np.array([along_normal, along_face, height]) / 2


def fuse(per_camera: list[list[Detection]]) -> list[Detection]:
    """One detection per entity across cameras (see the module docstring)."""

    def quality(d: Detection):
        return (d.occluded, -d.pixels)

    merged: list[Detection] = []
    for d in sorted((d for dets in per_camera for d in dets), key=quality):
        cls = entity_class(d.kind)
        same = [
            i
            for i, m in enumerate(merged)
            if entity_class(m.kind) == cls
            and m.color == d.color
            and np.linalg.norm(m.position[:2] - d.position[:2]) < FUSE[cls]
        ]
        if not same:
            merged.append(d)
            continue
        m = merged[same[0]]
        if cls == "receptacle" and d.opening is not None:
            if m.opening is None or d.opening[1] > m.opening[1]:
                merged[same[0]] = replace(m, opening=d.opening)
    return merged
