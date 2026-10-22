"""A small persistent world model: tracks of detected entities across camera frames.

Each track keeps its latest estimate, its uncertainty, whether it was seen in the latest frame,
and when it was last seen. Frames are not treated as unrelated scenes:
- association: same kind and colour, nearest within a gate (or, if the colour is unique on
  both sides, any distance: the robot may have carried the object far between frames);
- an unoccluded measurement replaces the estimate;
- an occluded measurement only updates the estimate if it disagrees with the track by more
  than the occlusion can explain (the object moved). Otherwise the last full view is kept;
- a track not seen in a frame stays as a last-known estimate with ``visible=False``. Its
  position is never refreshed from anything but a detection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dream.perception.detect import Detection

GATE = 0.05  # m: a same-coloured detection farther than this from a track is a new entity...
MOVED = 0.02  # m: ...and an occluded one closer than this has not moved


@dataclass(frozen=True)
class Track:
    id: int
    detection: Detection  # the estimate currently believed
    last_seen: float  # s
    visible: bool  # seen in the latest frame
    n_seen: int


class WorldModel:
    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self._next = 1
        self.time = -np.inf

    def update(self, detections: list[Detection], time: float) -> None:
        self.time = time
        unmatched = list(range(len(self.tracks)))
        new: list[Track] = []
        for d in sorted(detections, key=lambda d: d.occluded):  # full views claim tracks first
            i = self._match(d, unmatched, detections)
            if i is None:
                new.append(Track(self._next, d, time, True, 1))
                self._next += 1
                continue
            unmatched.remove(i)
            t = self.tracks[i]
            moved = np.linalg.norm(d.position - t.detection.position) > MOVED
            keep_old = d.occluded and not t.detection.occluded and not moved
            est = replace(t.detection, occluded=True) if keep_old else d
            self.tracks[i] = Track(t.id, est, time, True, t.n_seen + 1)
        for i in unmatched:
            self.tracks[i] = replace(self.tracks[i], visible=False)
        self.tracks.extend(new)

    def _match(self, d: Detection, free: list[int], detections: list[Detection]) -> int | None:
        same = [i for i in free if self._compatible(self.tracks[i].detection, d)]
        if not same:
            return None
        dist = [np.linalg.norm(self.tracks[i].detection.position - d.position) for i in same]
        best = same[int(np.argmin(dist))]
        if min(dist) < GATE:
            return best
        unique_track = len([t for t in self.tracks if self._compatible(t.detection, d)]) == 1
        unique_now = len([e for e in detections if self._compatible(e, d)]) == 1
        return best if unique_track and unique_now else None

    @staticmethod
    def _compatible(a: Detection, b: Detection) -> bool:
        return a.kind == b.kind and a.color == b.color

    def names(self) -> dict[int, str]:
        """Track id -> name. Objects are named by colour, numbered when a colour repeats;
        containers are "tray" (numbered if several). Names are labels for goals, not identity
        claims beyond what the image supports."""
        out, counts = {}, {}
        for t in sorted(self.tracks, key=lambda t: t.id):
            base = t.detection.color if t.detection.kind == "object" else "tray"
            counts[base] = counts.get(base, 0) + 1
            out[t.id] = base if counts[base] == 1 else f"{base}_{counts[base]}"
        return out
