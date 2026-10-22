"""Structured task goals and their verification from observations.

DREAM has no natural-language understanding yet. Goals are typed structures, written either as
dicts ({"type": "in", "object": "red", "target": "tray"}) or in a strict two-word command
syntax ("in red tray", "on red blue"). That syntax is *not* language understanding, and
anything else is rejected. A pretrained language model can later produce these structures
through the same interface.

Verification uses only perception (noisy tracked poses) and reports a margin, so callers know
how close to the decision boundary a judgement is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dream.types import ObjectObservation


@dataclass(frozen=True)
class InRegion:
    """Put ``object`` inside the named fixture region (e.g. the tray)."""

    object: str
    region: str


@dataclass(frozen=True)
class OnTop:
    """Stack ``object`` on top of ``support``."""

    object: str
    support: str


Goal = InRegion | OnTop


def parse_goal(spec: str | dict) -> Goal:
    """Parse a structured goal. Raises ``ValueError`` for anything that is not one."""
    if isinstance(spec, str):
        words = spec.strip().split()
        if len(words) != 3 or words[0] not in ("in", "on"):
            raise ValueError(
                f"not a structured goal: {spec!r}. Use 'in <object> <region>' or "
                "'on <object> <support>' (natural language is not supported yet)."
            )
        spec = {"type": words[0], "object": words[1], "target": words[2]}
    kind, obj, target = spec.get("type"), spec.get("object"), spec.get("target")
    if not (isinstance(obj, str) and isinstance(target, str)):
        raise ValueError(f"goal needs string 'object' and 'target': {spec!r}")
    if kind == "in":
        return InRegion(obj, target)
    if kind == "on":
        if obj == target:
            raise ValueError("an object cannot be stacked on itself")
        return OnTop(obj, target)
    raise ValueError(f"unknown goal type {kind!r}")


@dataclass(frozen=True)
class Verdict:
    satisfied: bool
    margin: float  # m, distance from the decision boundary (positive = comfortably satisfied)
    uncertain: bool  # the margin is within the perception noise allowance
    reason: str


PERCEPTION_ALLOWANCE = 0.006  # m: ~3 sigma of the tracker's 2 mm position noise


def verify(goal: Goal, objects: list[ObjectObservation], fixtures: dict) -> Verdict:
    """Judge a goal from one set of observations (no physics, no ground truth)."""
    seen = {o.name: o for o in objects}
    if goal.object not in seen:
        return Verdict(False, -np.inf, True, f"'{goal.object}' not observed")
    obj = seen[goal.object]
    if isinstance(goal, InRegion):
        if goal.region not in fixtures:
            return Verdict(False, -np.inf, False, f"unknown region '{goal.region}'")
        centre, half = fixtures[goal.region]
        inside = half - obj.size / 4 - np.max(np.abs(obj.position[:2] - centre[:2]))
        # Resting on the region's floor (centre z is the floor top), not held above it.
        low = centre[2] + obj.size / 2 + 0.026 - obj.position[2]
        margin = float(min(inside, low))
        return Verdict(
            margin > 0,
            margin,
            abs(margin) < PERCEPTION_ALLOWANCE,
            "in region" if margin > 0 else "outside region",
        )
    support = seen.get(goal.support)
    if support is None:
        return Verdict(False, -np.inf, True, f"'{goal.support}' not observed")
    xy = support.size / 2 - float(np.linalg.norm(obj.position[:2] - support.position[:2]))
    dz = 0.01 - abs((obj.position[2] - support.position[2]) - (obj.size + support.size) / 2)
    margin = float(min(xy, dz))
    return Verdict(
        margin > 0,
        margin,
        abs(margin) < PERCEPTION_ALLOWANCE,
        "stacked" if margin > 0 else "not stacked",
    )


def resting_on(name: str, objects: list[ObjectObservation]) -> list[str]:
    """Names of objects observed resting on top of ``name`` (one edge higher, overlapping)."""
    seen = {o.name: o for o in objects}
    base = seen.get(name)
    if base is None:
        return []
    out = []
    for o in objects:
        if o.name == name:
            continue
        dz = o.position[2] - base.position[2]
        overlap = np.linalg.norm(o.position[:2] - base.position[:2]) < (o.size + base.size) / 2
        if overlap and 0.5 * base.size < dz < 1.5 * (o.size + base.size) / 2:
            out.append(o.name)
    return out
