"""``VisionPerception``: the agent-facing ``Perception`` backend built on camera frames.

Every call that needs fresh information captures a new RGB-D frame through ``frame_source``
(unless the robot clock has not advanced since the last frame), runs detection, and updates the
world model. Observations are converted to the same types the oracle tracker produces, so the
runtime, tools and skills do not know which backend is active. ``backend`` names it in logs.

Not provided yet: articulated objects (handles) from vision. ``observe_articulated`` returns an
empty list. It never falls back to the oracle.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dream.perception.detect import Detection, detect
from dream.perception.world import WorldModel
from dream.types import ArticulatedObservation, ObjectObservation, RGBDFrame

# Uncertainty inflation measured on development scenes (see experiments/vision_accuracy.py).
STD_SCALE = 1.0


class VisionPerception:
    backend = "vision"

    def __init__(
        self,
        frame_source: Callable[[], RGBDFrame],
        clock: Callable[[], float],
        rng: np.random.Generator,
        memory: bool = True,
    ) -> None:
        """``memory=False`` is the single-frame ablation: entities not in the latest frame are
        forgotten instead of being kept as last-known estimates."""
        self._source = frame_source
        self._clock = clock
        self._rng = rng
        self.memory = memory
        self.world = WorldModel()
        self.last_frame: RGBDFrame | None = None
        self.last_detections: list[Detection] = []
        self._plane = None
        self.frames = 0

    def refresh(self) -> None:
        """Capture and process a new frame if the robot clock has moved since the last one."""
        if self.last_frame is not None and self._clock() <= self.last_frame.time:
            return
        frame = self._source()
        dets, plane = detect(frame, self._rng, self._plane)
        if self._plane is None:  # the support plane is static: fit it once, from a full view
            self._plane = plane
        self.world.update(dets, frame.time)
        if not self.memory:
            self.world.tracks = [t for t in self.world.tracks if t.visible]
        self.last_frame, self.last_detections = frame, dets
        self.frames += 1

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
        self.refresh()
        names = self.world.names()
        return {
            names[t.id]: (t.detection.position.copy(), t.detection.inner_half)
            for t in self.world.tracks
            if t.detection.kind == "container"
        }

    def observe_articulated(self) -> list[ArticulatedObservation]:
        return []
