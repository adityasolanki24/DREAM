"""The user's original request as an immutable typed record, and the checks that stop any later
stage from silently replacing it.

Why this exists (README, 7 Nov 2026 closeout): asked to "Walk to the refrigerator", the model
bound the request to ``GoToPlace('entrance')``; the robot walked there, the verifier confirmed
the *bound* goal and the system claimed success. Asked to "Pick up the kettle", it bound
"inspect the cabinet" and did the same. Verification was correct about the wrong thing.

Five things are kept apart here and in ``TaskAgent``:

1. **Original intent**: ``GoalSpecification``. Created once when the request arrives, frozen,
   with a digest, from the request's own words. Nothing downstream may change it.
2. **Grounded task**: the typed ``Goal`` a planner executes (dream/tools/goals.py). It must pass
   ``check_binding(spec, goal)``: a goal that cannot establish the requested outcome on the
   requested target is rejected, whoever proposed it.
3. **Planning subgoals**: tool calls (walk to a frontier, look around, approach a viewpoint).
   They are never goals and never verified as completion.
4. **Physical tool results**: measured ``ToolResult``s.
5. **Final verification**: ``verify_goal`` on the grounded goal, accepted as success only when
   the binding still holds (``TaskAgent`` re-checks it at the end).

``ground_request`` is a SMALL EXPLICIT GRAMMAR (verb phrases and a vocabulary of nouns), not
language understanding. When it cannot ground a request, a configured language model may
propose a specification; ``check_anchoring`` then requires every field to be anchored
in the request's own words, so a model can paraphrase-match but never substitute. A request
nothing can ground is refused, not approximated: a false refusal is recoverable, a false
success is not.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

# Required outcomes, from weakest to strongest within the navigation family. Manipulation
# outcomes are separate: none of them is implied by any amount of looking or walking.
LOCALIZED = "visually_localized"  # the requested object is seen and its identity verified
REACHED = "reached"  # the robot stands at the verified object, room or remembered place
INSPECTED = "inspected"  # examined from a valid close viewpoint (the cabinet task)
GRASPED = "grasped"
DELIVERED = "delivered"  # brought to the user: grasp, carry, hand over
OPENED = "opened"
CLOSED = "closed"
PLACED = "placed"
UNKNOWN = "unknown"
OUTCOMES = (LOCALIZED, REACHED, INSPECTED, GRASPED, DELIVERED, OPENED, CLOSED, PLACED)
# Satisfying a stronger navigation outcome also satisfies a weaker one: standing at a verified
# refrigerator has localized it. Nothing else is ordered.
_IMPLIES = {REACHED: {LOCALIZED}, INSPECTED: {LOCALIZED}}

# Verb phrases of the grammar, longest first within each outcome. Matching is on whole words.
VERBS: dict[str, tuple[str, ...]] = {
    DELIVERED: ("bring me", "bring", "fetch", "get me", "hand me", "carry"),
    GRASPED: ("pick up", "pick", "grab", "grasp", "lift", "hold", "take"),
    OPENED: ("open",),
    CLOSED: ("close", "shut"),
    PLACED: ("put", "place"),
    INSPECTED: ("take a look at", "look at", "inspect", "examine", "check"),
    REACHED: (
        "walk over to", "go over to", "walk up to", "walk to", "go to", "navigate to",
        "move to", "head to", "take me to", "get to", "approach", "goto",
    ),
    LOCALIZED: (
        "look for", "search for", "where is", "where's", "show me", "find", "locate", "spot",
    ),
}  # fmt: skip
TERMINAL_OUTCOMES = (
    "target_found_verified",  # the requested outcome is established by evidence
    "possible_target_unconfirmed",  # a candidate with the target's label, not verified
    "target_not_yet_found",  # budget spent while reachable unexplored space remains
    "search_exhausted",  # the reachable space was covered and the target was not seen
    "target_unreachable",  # verified, but no reachable place to stand at it
    "goal_unsupported",  # the request needs a capability or concept the robot lacks
    "goal_substitution_rejected",  # a proposed goal would have changed the request
    "safety_interruption",
    "cancelled",
)


@dataclass(frozen=True)
class GoalSpecification:
    """What the user asked for, in typed form, fixed when the request arrives.

    ``target_kind`` says what ``target_class`` names: an ``object`` class the robot's
    perception reports, a ``room`` category, or a ``place`` remembered by name. ``anchors``
    are the substrings of the instruction the fields were read from (audit trail).
    """

    original_instruction: str
    target_class: str | None
    target_kind: str | None  # "object" | "room" | "place" | None (not grounded)
    target_instance: str | None  # a stable reference the user named, e.g. "candidate:3"
    required_outcome: str  # one of OUTCOMES, or UNKNOWN
    also_required: tuple[str, ...] = ()  # further outcomes named in the same request
    grounding: str = "grammar"  # "grammar" | "model" | "typed"
    anchors: tuple[str, ...] = ()

    @property
    def outcomes(self) -> tuple[str, ...]:
        return (self.required_outcome, *self.also_required)

    @property
    def digest(self) -> str:
        """Content hash: recorded at intake and compared at verification."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {**asdict(self), "digest": self.digest}


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s':#-]", " ", text.casefold()).split())


def _contains(text: str, phrase: str) -> int:
    """Index of ``phrase`` as whole words in normalised ``text``, or -1."""
    m = re.search(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", text)
    return -1 if m is None else m.start()


def surface_forms(canonical: str, synonyms: Iterable[str] = ()) -> tuple[str, ...]:
    """The ways a class may be written: its name with spaces, a plural, and given synonyms."""
    base = canonical.replace("_", " ")
    forms = {base, base + "s", base + "es"} | {normalise(s) for s in synonyms}
    return tuple(sorted(forms, key=len, reverse=True))


def default_vocabulary(classes: Iterable[str], synonyms: Mapping[str, Iterable[str]] = ()):
    """``{canonical: surface forms}`` for ``classes``; ``synonyms`` adds per-class forms."""
    synonyms = dict(synonyms)
    return {c: surface_forms(c, synonyms.get(c, ())) for c in classes}


def _find_targets(
    text: str, vocabularies: Mapping[str, Mapping[str, Iterable[str]]]
) -> list[tuple[int, str, str, str]]:
    """(position, canonical, surface, kind) for every class written in ``text``, across the
    vocabularies of each kind; a longer match hides a shorter one inside it ("kitchen
    cabinet" is not also "kitchen")."""
    hits = []
    for kind, vocabulary in vocabularies.items():
        for canonical, forms in vocabulary.items():
            for form in forms:
                at = _contains(text, form)
                if at >= 0:
                    hits.append((at, at + len(form), canonical, form, kind))
                    break
    keep = [
        h
        for h in hits
        if not any(o is not h and o[0] <= h[0] and h[1] <= o[1] and o[1] - o[0] > h[1] - h[0]
                   for o in hits)
    ]  # fmt: skip
    return sorted((h[0], h[2], h[3], h[4]) for h in keep)


def _find_outcomes(text: str) -> list[tuple[int, str, str]]:
    hits, taken = [], []
    for outcome, phrases in VERBS.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            for m in re.finditer(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", text):
                span = (m.start(), m.end())
                if any(a < span[1] and span[0] < b for a, b in taken):
                    continue
                taken.append(span)
                hits.append((m.start(), outcome, phrase))
    return sorted(hits)


def ground_request(
    text: str,
    objects: Mapping[str, Iterable[str]],
    rooms: Mapping[str, Iterable[str]] = (),
    places: Iterable[str] = (),
) -> GoalSpecification:
    """Ground a request with the explicit grammar. Fields the grammar cannot read stay
    ungrounded (None / UNKNOWN); this function never guesses and never refuses: support is
    decided separately (``assess_support``)."""
    norm = normalise(text)
    instance = re.search(r"\bcandidate\s*[:#]?\s*(\d+)\b", norm)
    named = {"place": {p: (normalise(p),) for p in places}}
    targets = _find_targets(norm, {"object": objects, "room": dict(rooms), **named})
    verbs = _find_outcomes(norm)
    outcomes = [v[1] for v in verbs]
    anchors = [v[2] for v in verbs]
    target = kind = None
    if targets:
        # The first named target is the request's object ("put the cup on the table" names
        # two; PLACED is unsupported here, so a second target never decides an outcome).
        _, target, surface, kind = targets[0]
        anchors.append(surface)
    if instance is not None:
        anchors.append(instance[0])
    primary = outcomes[0] if outcomes else UNKNOWN
    rest = tuple(dict.fromkeys(o for o in outcomes[1:] if o != primary))
    return GoalSpecification(
        original_instruction=text,
        target_class=target,
        target_kind=kind,
        target_instance=None if instance is None else f"candidate:{int(instance[1])}",
        required_outcome=primary,
        also_required=rest,
        grounding="grammar",
        anchors=tuple(anchors),
    )


@dataclass(frozen=True)
class Capabilities:
    """What this embodiment and its perception can establish and verify."""

    outcomes: frozenset[str]
    object_classes: frozenset[str]  # classes perception can report and the verifier check
    room_categories: frozenset[str] = frozenset()
    places: frozenset[str] = frozenset()
    # Object classes only some outcomes apply to, e.g. "inspected" for the cabinet only.
    outcome_classes: Mapping[str, frozenset[str]] | None = None


def assess_support(spec: GoalSpecification, caps: Capabilities) -> str | None:
    """None when the robot can attempt the request as stated, else why not (a refusal reason
    for the user). Unsupported is decided on the request, never by changing it."""
    for outcome in spec.outcomes:
        if outcome == UNKNOWN:
            return "the request names no action I can recognise"
        if outcome not in caps.outcomes:
            return f"this robot cannot make something {outcome.replace('_', ' ')}"
    if spec.target_class is None and spec.target_instance is None:
        return "the request names nothing I can recognise or remember"
    if spec.target_kind == "object" and spec.target_class not in caps.object_classes:
        return f"my perception cannot recognise a {spec.target_class.replace('_', ' ')}"
    if spec.target_kind == "room" and spec.target_class not in caps.room_categories:
        return f"I cannot recognise a {spec.target_class.replace('_', ' ')}"
    if spec.target_kind == "place" and spec.target_class not in caps.places:
        return f"{spec.target_class} is not a place I remember"
    limits = (caps.outcome_classes or {}).get(spec.required_outcome)
    if limits is not None and spec.target_kind == "object" and spec.target_class not in limits:
        return (
            f"I can only establish '{spec.required_outcome}' for "
            f"{', '.join(sorted(limits))}, not for a {spec.target_class.replace('_', ' ')}"
        )
    return None


def check_anchoring(
    spec: GoalSpecification,
    objects: Mapping[str, Iterable[str]],
    rooms: Mapping[str, Iterable[str]] = (),
    places: Iterable[str] = (),
) -> list[str]:
    """Problems with a (model-proposed) specification against the request's own words.

    Every field must be anchored: the target written in the instruction (as its name or a
    listed synonym), the outcome's verb phrase present, a named instance present. A model may
    ground what the grammar could not read; it may not supply what the user did not say."""
    norm = normalise(spec.original_instruction)
    problems = []
    if spec.required_outcome not in (*OUTCOMES, UNKNOWN):
        problems.append(f"unknown outcome {spec.required_outcome!r}")
    for outcome in spec.outcomes:
        if outcome in VERBS and not any(_contains(norm, p) >= 0 for p in VERBS[outcome]):
            problems.append(f"outcome {outcome!r} is not stated in the request")
    if spec.target_class is not None:
        table = {"object": dict(objects), "room": dict(rooms)}.get(spec.target_kind)
        if spec.target_kind == "place":
            forms = (normalise(spec.target_class),) if spec.target_class in set(places) else ()
        elif table is not None:
            forms = tuple(table.get(spec.target_class, ()))
        else:
            forms = ()
        if not forms:
            problems.append(f"{spec.target_kind} {spec.target_class!r} is not in the vocabulary")
        elif not any(_contains(norm, f) >= 0 for f in forms):
            problems.append(f"target {spec.target_class!r} is not named in the request")
    if spec.target_instance is not None:
        m = re.fullmatch(r"candidate:(\d+)", spec.target_instance)
        named = re.search(r"\bcandidate\s*[:#]?\s*(\d+)\b", norm)
        if m is None or named is None or int(named[1]) != int(m[1]):
            problems.append(f"instance {spec.target_instance!r} is not named in the request")
    return problems


def check_binding(spec: GoalSpecification, goal) -> tuple[bool, str]:
    """Whether completing ``goal`` would establish ``spec``'s required outcome on its target.

    A refusal is always consistent (it claims nothing). Exploration points are never
    consistent: reaching one says nothing about any target."""
    from dream.tools.goals import (
        FindObject,
        GoToObserved,
        GoToPlace,
        InRegion,
        Inspect,
        OnTop,
        UnsupportedGoal,
    )

    if isinstance(goal, UnsupportedGoal):
        return True, "a refusal claims nothing"
    if spec.also_required:
        return False, (
            f"the request also requires {', '.join(spec.also_required)}; "
            f"no single goal here establishes all of it"
        )
    want = spec.required_outcome

    def satisfies(outcome: str) -> bool:
        return outcome == want or want in _IMPLIES.get(outcome, set())

    if isinstance(goal, FindObject):
        if spec.target_kind != "object" or goal.target != spec.target_class:
            return False, f"the request is about {spec.target_class!r}, not {goal.target!r}"
        if spec.target_instance is not None:
            return False, "a category search cannot establish a named instance"
        if not satisfies(goal.outcome):
            return False, f"the request requires {want!r}; the goal establishes {goal.outcome!r}"
        return True, "same target class and outcome"
    if isinstance(goal, Inspect):
        # "Inspect candidate 3" names an instance, not a class: the instance fixes the target.
        named_only = spec.target_class is None and spec.target_instance is not None
        if (not named_only and spec.target_class != goal.target) or not satisfies(INSPECTED):
            return False, (
                f"the request is '{want}' of {spec.target_class!r}; "
                f"the goal inspects {goal.target!r}"
            )
        if spec.target_instance is not None and spec.target_instance != (
            None if goal.candidate is None else f"candidate:{goal.candidate}"
        ):
            return False, "the goal does not keep the requested instance"
        return True, "same target and instance"
    if isinstance(goal, GoToPlace):
        if spec.target_kind == "place" and goal.place == spec.target_class and satisfies(REACHED):
            return True, "the requested remembered place"
        return False, (
            f"walking to the remembered place {goal.place!r} does not establish "
            f"'{want}' of {spec.target_class!r}"
        )
    if isinstance(goal, GoToObserved):
        if spec.target_instance is not None and goal.reference == spec.target_instance:
            return True, "the requested observed reference"
        return False, (
            f"reaching {goal.reference} (an observed position) does not establish "
            f"'{want}' of {spec.target_class!r}"
        )
    if isinstance(goal, (InRegion, OnTop)):
        support = goal.region if isinstance(goal, InRegion) else goal.support
        if want == PLACED and spec.target_class == goal.object:
            return True, f"placing {goal.object} on/in {support}"
        return False, f"the request is not to place {goal.object!r}"
    return False, f"no binding rule for {type(goal).__name__}"


def bind_request(spec: GoalSpecification, caps: Capabilities):
    """The typed goal that establishes ``spec`` with this robot's capabilities, or a refusal.

    A deterministic mapping: the request's target and outcome decide the goal type. No model
    chooses among goal types, so there is nothing to substitute."""
    from dream.tools.goals import FindObject, GoToPlace, Inspect, UnsupportedGoal

    reason = assess_support(spec, caps)
    if reason is not None:
        return UnsupportedGoal(reason)
    if spec.also_required:
        return UnsupportedGoal(
            f"the request combines {', '.join(spec.outcomes)}; this robot runs one at a time"
        )
    if spec.required_outcome == INSPECTED:
        cid = None if spec.target_instance is None else int(spec.target_instance.split(":")[1])
        return Inspect(spec.target_class or "cabinet", cid)
    if spec.target_instance is not None:
        return UnsupportedGoal("a named instance can only be inspected in this session")
    if spec.target_kind == "object" and spec.required_outcome in (LOCALIZED, REACHED):
        return FindObject(spec.target_class, spec.required_outcome)
    if spec.target_kind == "place" and spec.required_outcome == REACHED:
        return GoToPlace(spec.target_class)
    return UnsupportedGoal(
        f"no capability establishes '{spec.required_outcome}' for a {spec.target_kind}"
    )
