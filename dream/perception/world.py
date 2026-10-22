"""A small persistent world model: tracks of detected entities across camera frames.

Each track keeps its latest estimate, its uncertainty, whether it was seen in the latest frame,
and when it was last seen. Frames are not treated as unrelated scenes:
- association: same class (small thing / receptacle) and colour, nearest within a gate (or,
  if the colour is unique on both sides, any distance: the robot may have carried an object
  far between frames);
- an unoccluded measurement replaces the estimate;
- an occluded measurement only updates the estimate if it disagrees with the track by more
  than the occlusion can explain (the object moved). Otherwise the last full view is kept;
- a track not seen in a frame stays as a last-known estimate with ``visible=False``. Its
  position is never refreshed from anything but a detection;
- a receptacle keeps its identity as it changes appearance (a closed box that opens into an
  open container), and is "articulated" once a handle has been seen on it. A handle stays a
  handle while it moves (a knob rising with a lid no longer sits beside the box).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dream.perception.detect import Detection

# m: farther same-coloured detections are new entities. A receptacle's detected centre moves
# when it opens (a lid rising shifted it by more than 0.15 m); two distinct 20 cm boxes cannot
# have centres closer than 0.2 m.
GATE = {"small": 0.05, "receptacle": 0.2}
MOVED = 0.02  # m: an occluded detection closer than this to its track has not moved
HANDLE_ON_RECEPTACLE = 0.25  # m from a receptacle's centre for a handle to belong to it


def entity_class(kind: str) -> str:
    return "receptacle" if kind in ("box", "container") else "small"


@dataclass(frozen=True)
class Track:
    id: int
    detection: Detection  # the estimate currently believed
    last_seen: float  # s
    visible: bool  # seen in the latest frame
    n_seen: int
    articulated: bool = False  # a receptacle on which a handle has been seen


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
            if t.detection.kind == "handle" and est.kind == "object":
                est = replace(est, kind="handle", position=est.top_centre)
            if d.occluded and t.detection.opening is not None:
                # An occluded view hides cells; it is no evidence that an opening shrank or
                # closed. Keep the opening last seen in a clear view.
                est = replace(est, opening=t.detection.opening)
            self.tracks[i] = Track(t.id, est, time, True, t.n_seen + 1, t.articulated)
        for i in unmatched:
            self.tracks[i] = replace(self.tracks[i], visible=False)
        self.tracks.extend(new)
        self._mark_articulated()

    def _mark_articulated(self) -> None:
        handles = [t.detection.position for t in self.tracks if t.detection.kind == "handle"]
        for i, t in enumerate(self.tracks):
            if entity_class(t.detection.kind) != "receptacle" or t.articulated:
                continue
            centre = t.detection.position[:2]
            if any(np.linalg.norm(h[:2] - centre) < HANDLE_ON_RECEPTACLE for h in handles):
                self.tracks[i] = replace(t, articulated=True)

    def _match(self, d: Detection, free: list[int], detections: list[Detection]) -> int | None:
        same = [i for i in free if self._compatible(self.tracks[i].detection, d)]
        if not same:
            return None
        dist = [np.linalg.norm(self.tracks[i].detection.position - d.position) for i in same]
        best = same[int(np.argmin(dist))]
        if min(dist) < GATE[entity_class(d.kind)]:
            return best
        unique_track = len([t for t in self.tracks if self._compatible(t.detection, d)]) == 1
        unique_now = len([e for e in detections if self._compatible(e, d)]) == 1
        return best if unique_track and unique_now else None

    @staticmethod
    def _compatible(a: Detection, b: Detection) -> bool:
        return entity_class(a.kind) == entity_class(b.kind) and a.color == b.color

    def handle_of(self, receptacle: Track) -> Track | None:
        """The handle track nearest to an articulated receptacle, visible or last-known."""
        handles = [t for t in self.tracks if t.detection.kind == "handle"]
        if not handles:
            return None
        centre = receptacle.detection.position[:2]
        return min(handles, key=lambda t: np.linalg.norm(t.detection.position[:2] - centre))

    def names(self) -> dict[int, str]:
        """Track id -> name. Objects are named by colour, numbered when a colour repeats.
        Receptacles with a handle are "compartment", other receptacles "tray" (numbered if
        several); handles are "handle". Names are labels for goals, not identity
        claims beyond what the image supports."""
        out, counts = {}, {}
        for t in sorted(self.tracks, key=lambda t: t.id):
            d = t.detection
            if d.kind == "handle":
                continue
            if entity_class(d.kind) == "receptacle":
                base = "compartment" if t.articulated else "tray"
            else:
                base = d.color
            counts[base] = counts.get(base, 0) + 1
            out[t.id] = base if counts[base] == 1 else f"{base}_{counts[base]}"
        for t in self.tracks:
            if t.detection.kind == "handle":
                out[t.id] = "handle"
        return out
