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
# Conservative, uncalibrated freshness policy: historical open space may bridge brief arm
# occlusions, but cannot justify placement indefinitely. Not a dynamics prediction.
OPENING_MAX_AGE_S = 5.0


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
    opening_last_seen: float | None = None  # when the retained open space was measured
    handle_id: int | None = None  # stable association once observed, not nearest each frame


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
                new.append(
                    Track(
                        self._next,
                        d,
                        time,
                        True,
                        1,
                        opening_last_seen=time if d.opening is not None else None,
                    )
                )
                self._next += 1
                continue
            unmatched.remove(i)
            t = self.tracks[i]
            moved = np.linalg.norm(d.position - t.detection.position) > MOVED
            keep_old = d.occluded and not t.detection.occluded and not moved
            est = replace(t.detection, occluded=True) if keep_old else d
            if t.detection.kind == "handle" and est.kind == "object":
                est = replace(est, kind="handle", position=est.top_centre)
            opening_time = time if d.opening is not None else None
            previous = t.detection.opening
            if d.occluded and previous is not None:
                # A partial view cannot disprove old open space. Newly observed wider space
                # can replace it, but only at a compatible floor location; disjoint openings
                # are conflicting evidence, not proof that the old region is still clear.
                compatible = (
                    d.opening is not None
                    and np.linalg.norm(d.opening[0] - previous[0]) <= previous[1] - d.opening[1]
                )
                if d.opening is None or (d.opening[1] < previous[1] and compatible):
                    est = replace(est, opening=previous)
                    opening_time = t.opening_last_seen
                else:
                    est = replace(est, opening=d.opening)
            else:
                est = replace(est, opening=d.opening)
            self.tracks[i] = replace(
                t,
                detection=est,
                last_seen=time,
                visible=True,
                n_seen=t.n_seen + 1,
                opening_last_seen=opening_time,
            )
        for i in unmatched:
            self.tracks[i] = replace(self.tracks[i], visible=False)
        self.tracks.extend(new)
        self._mark_articulated()

    def _mark_articulated(self) -> None:
        assigned = {t.handle_id for t in self.tracks if t.handle_id is not None}
        handles = [
            t
            for t in self.tracks
            if t.visible and t.detection.kind == "handle" and t.id not in assigned
        ]
        for i, t in enumerate(self.tracks):
            if entity_class(t.detection.kind) != "receptacle" or t.articulated:
                continue
            centre = t.detection.position[:2]
            nearby = sorted(
                handles, key=lambda h: np.linalg.norm(h.detection.position[:2] - centre)
            )
            if not nearby:
                continue
            nearest = nearby[0]
            distance = np.linalg.norm(nearest.detection.position[:2] - centre)
            if distance >= HANDLE_ON_RECEPTACLE:
                continue
            if len(nearby) > 1:
                other_distance = np.linalg.norm(nearby[1].detection.position[:2] - centre)
                # Ambiguous visual association needs another observation, not a guess.
                if other_distance - distance <= 3 * max(
                    nearest.detection.position_std, nearby[1].detection.position_std
                ):
                    continue
            self.tracks[i] = replace(t, articulated=True, handle_id=nearest.id)
            handles.remove(nearest)

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
        """The originally associated handle, even when opening moves another handle closer."""
        return next((t for t in self.tracks if t.id == receptacle.handle_id), None)

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
