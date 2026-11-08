"""Wiring of the original-request architecture into the live Isaac G1 session (semantic mode),
and the independent original-request scoring used for both modes.

Kept apart from ``experiments/isaac_g1_live.py`` so that it runs on a CPU with fake backends
(tests/test_isaac_semantic.py); the live process only calls these functions.

Decision side (nothing here reads simulator state):
- ``ground_and_bind``: the request is grounded once by the explicit grammar
  (dream/tools/intent.ground_request); if target or outcome stay ungrounded the Qwen backend
  may propose a grounding, validated against the request's own words (propose_grounding);
  the typed goal is then ``bind_request(spec, capabilities)``, deterministic.
- ``attach_semantic_search``: the semantic world, the VLM detector (detect_when="stationary":
  Qwen is too slow to run while walking) and the planner for ``FindObject`` goals.

Evaluation side (ground truth enters here, never the agent):
- ``true_instances``: labelled Isaac prims -> ``TrueInstance`` list for the independent scorer.
- ``claim_position`` / ``typed_completion`` / ``score_original_request``: the ORIGINAL request
  is scored whatever goal was bound, for the 7 Nov 2026 arm as well as the new one.
"""

from __future__ import annotations

import numpy as np

from dream.eval.semantic_scoring import TrueInstance, odometry_to_world, score_episode
from dream.semantics.knowledge import SemanticKnowledge
from dream.tools.goals import FindObject, UnsupportedGoal
from dream.tools.intent import (
    INSPECTED,
    LOCALIZED,
    REACHED,
    UNKNOWN,
    Capabilities,
    GoalSpecification,
    bind_request,
    default_vocabulary,
    ground_request,
)

# Canonical detector classes (task statement of the 8 Nov 2026 Isaac semantic evaluation). Only
# the first six exist in the NVIDIA kitchens; the rest are distractors the model may hallucinate
# and the absent targets of "Find the bed".
DETECTOR_VOCABULARY = (
    "refrigerator", "sink", "stove", "countertop", "kitchen_cabinet", "microwave", "dishwasher",
    "wardrobe", "bed", "toilet", "sofa", "washing_machine", "bathtub", "kettle",
)  # fmt: skip
# "Inspect the cabinet" uses the box-and-handle cabinet perception, not the VLM detector, so
# "cabinet" is a recognised class for the inspected outcome only.
INSPECT_CLASS = "cabinet"
# Scene semantic labels (Isaac replicator kitchens) -> scorer classes. Others are ignored.
SCENE_CLASSES = {
    "refrigerator": "refrigerator",
    "sink": "sink",
    "sink_basin": "sink",
    "sink_rim": "sink",
    "oven": "stove",
    "countertop": "countertop",
}
# Planner weights of the frozen MuJoCo held-out evaluation (experiments/configs/
# semantic_search_heldout.json, commit 9d08c27): kappa chosen there by the pre-registered dev3 rule.
FROZEN_WEIGHTS = {"kappa": 3.0, "ig": 1.0, "gamma": 1.0}
MERGE_MARGIN_M = 0.02  # AABBs of one merged class closer than this are one physical instance
MERGE_CLASSES = frozenset({"sink"})  # a sink is a rim, a basin and a bowl; countertops are not
# The sink must exist in the scene for the scorer to mean anything. The refrigerator and the
# oven are DECLARED by the NVIDIA kitchens but not present: their SimReady asset references
# (../../SimReady/Residential/Kitchen/Appliances/...) return 404 on NVIDIA's server, so the prims
# carry no geometry and no collider (results/isaac_semantic_eval/scene_probe.json). They are
# therefore absent targets, exactly as the bed is.
REQUIRED_CLASSES = ("sink",)


def vocabularies(knowledge: SemanticKnowledge) -> tuple[dict, dict]:
    """(object vocabulary, room vocabulary) for the grammar: names plus the knowledge
    extract's grammar-safe synonyms."""
    objects = default_vocabulary((*DETECTOR_VOCABULARY, INSPECT_CLASS), knowledge.synonyms)
    rooms = default_vocabulary(knowledge.categories, knowledge.synonyms)
    return objects, rooms


def detector_vocabulary(knowledge: SemanticKnowledge) -> dict:
    return default_vocabulary(DETECTOR_VOCABULARY, knowledge.synonyms)


def capabilities(places) -> Capabilities:
    """What this robot can establish: seeing and standing at a detector class, and inspecting
    the cabinet. Nothing manipulative."""
    detector = frozenset(DETECTOR_VOCABULARY)
    return Capabilities(
        outcomes=frozenset({LOCALIZED, REACHED, INSPECTED}),
        object_classes=detector | {INSPECT_CLASS},
        places=frozenset(places),
        outcome_classes={
            INSPECTED: frozenset({INSPECT_CLASS}),
            LOCALIZED: detector,
            REACHED: detector,
        },
    )


def _ungrounded(spec: GoalSpecification) -> bool:
    return (spec.target_class is None and spec.target_instance is None) or (
        spec.required_outcome == UNKNOWN
    )


def ground_and_bind(text: str, backend, knowledge: SemanticKnowledge, places) -> dict:
    """The original request as a ``GoalSpecification`` and the goal that establishes it.

    Returns ``{"spec", "goal", "record"}``; ``record`` keeps the grammar's spec, whether the
    model was consulted and every rejected model proposal, for the evaluation log."""
    from dream.runtime.mobile_model import propose_grounding

    objects, rooms = vocabularies(knowledge)
    places = sorted(places)
    spec = ground_request(text, objects, rooms, places)
    record: dict = {"grammar_spec": spec.to_dict(), "model_consulted": False}
    if _ungrounded(spec):
        record["model_consulted"] = True
        try:
            proposed, rejected = propose_grounding(backend, text, objects, rooms, places)
        except Exception as error:  # a failed call leaves the grammar's (ungrounded) spec
            proposed, rejected = None, []
            record["model_error"] = f"{type(error).__name__}: {error}"
        record["model_rejected"] = rejected
        record["model_spec"] = None if proposed is None else proposed.to_dict()
        if proposed is not None:
            spec = proposed
    goal = bind_request(spec, capabilities(places))
    record["final_spec"] = spec.to_dict()
    record["goal"] = repr(goal)
    return {"spec": spec, "goal": goal, "record": record}


def attach_semantic_search(toolbox, memory, detector_backend, knowledge: SemanticKnowledge):
    """Give ``toolbox`` the semantic world and the VLM detector, and return the planner.

    Applied only to FindObject goals: Inspect and refusal paths keep the closeout toolbox
    (their prompts and tool behaviour unchanged). The planner's memory hints are empty because
    a live Isaac session has no home memory."""
    from dream.perception.vlm_detect import VLMDetector
    from dream.runtime.semantic_search import SemanticSearchPlanner, UtilityWeights
    from dream.semantics.world import SemanticWorld

    toolbox.semantic = SemanticWorld(memory.grid, knowledge)
    toolbox.detector = VLMDetector(detector_backend, detector_vocabulary(knowledge))
    toolbox.detect_when = "stationary"
    return SemanticSearchPlanner(toolbox, "utility", UtilityWeights(**FROZEN_WEIGHTS))


# ---------------------------------------------------------------------------------------------
# Evaluation only: ground truth and independent scoring
# ---------------------------------------------------------------------------------------------


def true_instances(labelled, room_type: str = "kitchen") -> list[TrueInstance]:
    """Scorer instances from labelled prims (``.path``, ``.label``, ``.lo``, ``.hi``).

    The axis-aligned box of a prim gives centre and half extents (yaw 0). Prims of one scorer
    class whose boxes touch are one physical instance (a sink is a rim, a basin and a bowl):
    their union box. Labels outside ``SCENE_CLASSES`` are ignored."""
    members: dict[str, list] = {}
    for obj in labelled:
        cls = SCENE_CLASSES.get(obj.label)
        lo, hi = np.asarray(obj.lo, float), np.asarray(obj.hi, float)
        if cls is None or not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            continue
        if np.abs(np.concatenate([lo, hi])).max() > 1e8 or np.any(hi < lo):
            continue  # USD's empty-range sentinel: no usable geometry
        members.setdefault(cls, []).append(obj)
    out: list[TrueInstance] = []
    for cls in sorted(members):
        boxes = [(np.asarray(o.lo, float)[:2], np.asarray(o.hi, float)[:2]) for o in members[cls]]
        groups: list[list[int]] = []
        for i, (lo, hi) in enumerate(boxes):
            touching = []
            if cls in MERGE_CLASSES:
                touching = [
                    g
                    for g in groups
                    if any(
                        np.all(lo <= boxes[j][1] + MERGE_MARGIN_M)
                        and np.all(boxes[j][0] <= hi + MERGE_MARGIN_M)
                        for j in g
                    )
                ]
            merged = [i]
            for g in touching:
                merged += g
                groups.remove(g)
            groups.append(merged)
        for k, g in enumerate(sorted(groups, key=min)):
            lo = np.min([boxes[j][0] for j in g], axis=0)
            hi = np.max([boxes[j][1] for j in g], axis=0)
            out.append(
                TrueInstance(f"{cls}:{k}", cls, (lo + hi) / 2, (hi - lo) / 2, 0.0, room_type)
            )
    return out


def missing_truth(instances: list[TrueInstance]) -> list[str]:
    """Required scorer classes with no usable instance (the scorer would call a real object
    absent): the live process refuses to run rather than score against nothing."""
    return [c for c in REQUIRED_CLASSES if not any(i.cls == c for i in instances)]


def declared_absent(labelled) -> list[dict]:
    """Prims the scene labels as a scorer class but that have no usable geometry (declared in
    the scene description, not present in the world): reported with the scoring instances."""
    out = []
    for obj in labelled:
        cls = SCENE_CLASSES.get(obj.label)
        lo, hi = np.asarray(obj.lo, float), np.asarray(obj.hi, float)
        usable = (
            np.isfinite(lo).all()
            and np.isfinite(hi).all()
            and np.abs(np.concatenate([lo, hi])).max() <= 1e8
            and not np.any(hi < lo)
        )
        if cls is not None and not usable:
            out.append({"path": obj.path, "label": obj.label, "class": cls})
    return out


def claim_position(goal, verdict: dict, odometry_end) -> list[float] | None:
    """Where the agent claims to have found/reached the target, in odometry coordinates.

    FindObject: the verifier's own object position. Other goals (the 7 Nov path: GoToPlace,
    GoToObserved, Inspect): the robot's final odometry, which is where it stands."""
    if isinstance(goal, FindObject):
        position = (verdict or {}).get("position")
        return None if position is None else [float(position[0]), float(position[1])]
    return [float(odometry_end[0]), float(odometry_end[1])]


def typed_completion(goal, episode_completion: str | None, claimed: bool, reason: str, toolbox):
    """Typed outcome of an episode. A refusal records no completion of its own, so an
    ``UnsupportedGoal`` is typed ``goal_unsupported`` here (source recorded by the caller);
    other stops come from ``stop_outcome`` on the robot's own state."""
    if isinstance(goal, UnsupportedGoal):
        return "goal_unsupported", "driver:UnsupportedGoal"
    if episode_completion:
        return episode_completion, "episode"
    if claimed:
        return None, "claimed"
    if isinstance(goal, FindObject):
        from dream.runtime.semantic_search import stop_outcome

        return stop_outcome(toolbox, goal.target, None, reason), "stop_outcome"
    return None, "none"


def score_original_request(
    task: dict,
    *,
    goal,
    claimed: bool,
    completion: str | None,
    verdict: dict,
    odometry_end,
    true_pose_end,
    instances: list[TrueInstance],
    cabinet_success: bool | None = None,
) -> dict:
    """Score the request as the user wrote it. ``task`` is the frozen config entry
    (``target``, ``outcome``, ``supported``, optionally ``scorer``). To world coordinates: the
    true pose against the odometry at the same instant (drift compensation for scoring only).

    ``scorer == "cabinet"`` (the cabinet-inspection request) has no instance in the semantic
    scorer; it is scored by the existing handle/viewpoint scorer (``cabinet_success``), and
    only when the bound goal really was an inspection."""
    if task.get("scorer") == "cabinet":
        ok = bool(cabinet_success) and goal.__class__.__name__ == "Inspect"
        result = ("success" if ok else "false_success") if claimed else "missed"
        return {
            "request": task,
            "scorer": "cabinet handle and viewpoint scorer (experiments/isaac_g1_kitchen.score)",
            "claimed": claimed,
            "completion": completion,
            "result": result,
            "original_goal_success": result == "success",
            "false_success_claim": result == "false_success",
        }
    to_world = odometry_to_world(true_pose_end, odometry_end)
    score = score_episode(
        target=task["target"],
        outcome=task["outcome"],
        supported_expected=bool(task["supported"]),
        claimed=claimed,
        completion=completion,
        claim_xy_odom=claim_position(goal, verdict, odometry_end) if claimed else None,
        instances=instances,
        to_world=to_world,
        true_pose_end=true_pose_end,
    )
    return {"request": task, "instances": [i.id for i in instances], **score}
