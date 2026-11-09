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


# ----- alternative knowledge sources (selectable per evaluation arm) ----------------------------
QWEN_PRIORS_PATH = ROOT / "results/qwen_room_priors/priors.json"
KNOWLEDGE_SOURCES = ("conceptnet", "qwen", "combined", "none")
# Numerical guard only: Qwen answers contain exact zeros, and log(0) is undefined in the room
# inference (dream/semantics/rooms.py takes the log of the association ratio). The value is
# the order of the smallest non-zero score Qwen itself gives (about 0.01); it is not tuned.
QWEN_FLOOR = 0.01


@dataclass(frozen=True)
class TableKnowledge(SemanticKnowledge):
    """Knowledge given directly as per-object distributions over ``categories`` (the Qwen and
    combined sources). Used by the agent exactly as the ConceptNet association is: through
    ``association`` / ``likelihood_ratio`` / ``has``. Objects without a row get the uniform
    distribution and ``has`` is False, the same handling as a ConceptNet gap."""

    table: dict[str, np.ndarray] = field(default_factory=dict)

    def association(self, obj: str) -> np.ndarray:
        row = self.table.get(obj)
        if row is None:
            return np.full(len(self.categories), 1.0 / len(self.categories))
        return row

    def has(self, obj: str) -> bool:
        return obj in self.table


def qwen_knowledge(
    path: str | Path = QWEN_PRIORS_PATH,
    categories=ROOM_CATEGORIES,
    synonyms: dict[str, tuple[str, ...]] | None = None,
) -> TableKnowledge:
    """Room priors from the saved Qwen3-VL-4B text-only answers (results/qwen_room_priors).

    Per object class: the raw scores over ``categories`` (the answer's own "none" mass is
    dropped), floored at ``QWEN_FLOOR`` and normalised to a distribution. A class whose answer
    was invalid, lacks a room category, or scores every category zero gets no knowledge.
    Uncalibrated association weights, used like the ConceptNet ones: no temperature, no
    rescaling."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[str, np.ndarray] = {}
    for obj, row in data.get("objects", {}).items():
        scores = row.get("table") if row.get("valid") else None
        if not scores or any(c not in scores for c in categories):
            continue
        raw = np.array([float(scores[c]) for c in categories])
        if raw.sum() <= 0:
            continue
        raw = np.maximum(raw / raw.sum(), QWEN_FLOOR)
        table[obj] = raw / raw.sum()
    return TableKnowledge(tuple(categories), {}, dict(synonyms or {}), "Qwen3-VL-4B text-only",
                          table)  # fmt: skip


def combined_knowledge(a: SemanticKnowledge, b: SemanticKnowledge) -> TableKnowledge:
    """Per class, the equal-weight mixture of two sources' distributions; one source alone
    where the other lacks the class; no entry where both lack it."""
    if a.categories != b.categories:
        raise ValueError("sources use different room categories")
    objects = set(a.strength) | set(getattr(a, "table", {})) | set(b.strength) | set(
        getattr(b, "table", {}))  # fmt: skip
    table = {}
    for obj in objects:
        rows = [s.association(obj) for s in (a, b) if s.has(obj)]
        if rows:
            table[obj] = np.mean(rows, axis=0)
    return TableKnowledge(a.categories, {}, dict(a.synonyms), f"combined({a.source}; {b.source})",
                          table)  # fmt: skip


def load_source(name: str) -> SemanticKnowledge:
    """The knowledge source an arm selects: conceptnet (default), qwen, combined or none."""
    if name == "conceptnet":
        return load_knowledge()
    if name == "none":
        return uniform_knowledge()
    words = load_knowledge()  # synonyms for reading requests, not search knowledge
    qwen = qwen_knowledge(synonyms=words.synonyms)
    if name == "qwen":
        return qwen
    if name == "combined":
        return combined_knowledge(words, qwen)
    raise ValueError(f"unknown knowledge source {name!r}; choose from {KNOWLEDGE_SOURCES}")
