"""General semantic knowledge: which kinds of room an object class is usually found in.

Source: the ConceptNet 5.7 extract in third_party/household_knowledge/knowledge.json
(CC BY-SA 4.0; provenance in that folder), never the benchmark's own scene generator. Each
(object, place) pair carries ConceptNet edges with weights (ConceptNet's own, roughly the
square root of the number of contributing sources).

What the numbers are, and are not:
- ``W(o, p)`` sums edge weights times a per-relation factor (``RELATION_FACTOR``: containment
  relations count fully, nearness half, mere relatedness a quarter). Design constants.
- ``association(o)`` normalises ``W + PSEUDO`` over the room categories. It is a HEURISTIC
  ASSOCIATION DISTRIBUTION derived from crowd-sourced edge weights, not a probability that has
  been estimated or calibrated. ``PSEUDO`` keeps every room possible (a refrigerator can be in
  a garage) and keeps one weak edge from dominating.
- An object with no edges gets the uniform distribution (no knowledge), never a hand-written
  guess. ConceptNet's coverage of household pairs is sparse and sometimes odd (its only bed
  edge in the AtLocation extract was "basement"); those gaps are kept, measured and reported.

General knowledge is kept apart from everything the robot observes (dream/perception/semantic),
from inferred room classifications (dream/semantics/rooms) and from search hypotheses
(dream/semantics/belief): none of it is ever stored as an observation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "third_party/household_knowledge/knowledge.json"
# The room categories the robot reasons about (the place vocabulary of the extract).
ROOM_CATEGORIES = (
    "kitchen", "bedroom", "bathroom", "living_room", "dining_room", "office", "garage",
    "workshop", "laundry_room", "storage_room", "pantry", "basement", "hallway", "closet",
)  # fmt: skip
UNKNOWN_ROOM = "unknown"  # a region whose evidence fits no category: neutral for search
PSEUDO = 0.25  # weight added to every (object, category) pair before normalising
RELATION_FACTOR = {"AtLocation": 1.0, "PartOf": 1.0, "HasA": 1.0, "LocatedNear": 0.5,
                   "RelatedTo": 0.25}  # fmt: skip


@dataclass(frozen=True)
class SemanticKnowledge:
    """Object-room associations and object synonyms, with their source."""

    categories: tuple[str, ...]
    strength: dict[str, dict[str, float]]  # object -> category -> summed weighted edges W
    synonyms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source: str = "uniform"

    def association(self, obj: str) -> np.ndarray:
        """Heuristic distribution over ``categories`` (uniform when the object is unknown)."""
        s = self.strength.get(obj)
        if not s:
            return np.full(len(self.categories), 1.0 / len(self.categories))
        a = np.array([s.get(c, 0.0) + PSEUDO for c in self.categories])
        return a / a.sum()

    def likelihood_ratio(self, obj: str) -> np.ndarray:
        """``association / uniform``: >1 for rooms the object points to, <1 against."""
        return self.association(obj) * len(self.categories)

    def has(self, obj: str) -> bool:
        return bool(self.strength.get(obj))

    def typical_rooms(self, obj: str, k: int = 3) -> list[tuple[str, float]]:
        a = self.association(obj)
        order = np.argsort(-a)[:k]
        return [(self.categories[i], round(float(a[i]), 3)) for i in order]


def uniform_knowledge(categories=ROOM_CATEGORIES) -> SemanticKnowledge:
    """No semantic knowledge at all: the ablation baseline."""
    return SemanticKnowledge(tuple(categories), {}, {}, "uniform (no knowledge)")


def load_knowledge(
    path: str | Path = DEFAULT_PATH, categories=ROOM_CATEGORIES
) -> SemanticKnowledge:
    """Read the extract: relations in RELATION_FACTOR to a category count; synonyms only
    where the extract marks them ``grammar_safe`` (the object's own WordNet sense)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    strength: dict[str, dict[str, float]] = {}
    synonyms: dict[str, tuple[str, ...]] = {}
    for obj, entry in data.get("objects", {}).items():
        row = {}
        for place, edges in (entry.get("places") or {}).items():
            if place not in categories:
                continue
            w = 0.0
            for e in edges:
                if e.get("variant_same_sense") is False:  # reached through a wrong sense
                    continue
                rel = str(e.get("relation", "")).rsplit("/", 1)[-1]
                w += RELATION_FACTOR.get(rel, 0.0) * float(e.get("weight", 0.0))
            if w > 0:
                row[place] = w
        if row:
            strength[obj] = row
        synonyms[obj] = tuple(dict.fromkeys(
            s["text"] for s in entry.get("synonyms") or ()
            if isinstance(s.get("text"), str) and s.get("grammar_safe") is True
        ))  # fmt: skip
    sources = ", ".join(str(s.get("name", s)) if isinstance(s, dict) else str(s)
                        for s in data.get("sources", []))  # fmt: skip
    return SemanticKnowledge(tuple(categories), strength, synonyms, sources or str(path))
