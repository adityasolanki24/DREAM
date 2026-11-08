"""Original-request preservation (dream/tools/intent.py) and its regression on the 7 Nov 2026
false claims: the frozen Qwen outputs that bound "Walk to the refrigerator" to the entrance
and "Pick up the kettle" to a cabinet inspection are replayed through the new path."""

import dataclasses
import json
import pathlib
import threading

import pytest

from dream.runtime.agent import Finish, TaskAgent, ToolCall
from dream.runtime.mobile_model import interpret_mobile_goal, propose_grounding
from dream.tools.goals import FindObject, GoToObserved, GoToPlace, Inspect, UnsupportedGoal
from dream.tools.intent import (
    DELIVERED,
    GRASPED,
    INSPECTED,
    LOCALIZED,
    OPENED,
    REACHED,
    UNKNOWN,
    Capabilities,
    GoalSpecification,
    bind_request,
    check_anchoring,
    check_binding,
    default_vocabulary,
    ground_request,
)
from dream.tools.toolbox import ToolResult

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/isaac_qwen_closeout_20261107/attempt1/attempt1"
OBJECTS = default_vocabulary(
    ["refrigerator", "kettle", "cabinet", "kitchen_cabinet", "sink", "bed", "washing_machine"],
    {"refrigerator": ["fridge"]},
)
ROOMS = default_vocabulary(["kitchen", "bedroom", "bathroom"])
SEARCH_CAPS = Capabilities(
    outcomes=frozenset({LOCALIZED, REACHED, INSPECTED}),
    object_classes=frozenset({"refrigerator", "cabinet", "sink", "bed", "washing_machine"}),
    places=frozenset({"entrance"}),
    outcome_classes={INSPECTED: frozenset({"cabinet"})},
)
NOV7_CAPS = dataclasses.replace(SEARCH_CAPS, object_classes=frozenset({"cabinet"}))


@pytest.mark.parametrize(
    "text, target, kind, outcome, also",
    [
        ("Find the refrigerator", "refrigerator", "object", LOCALIZED, ()),
        ("Walk to the refrigerator", "refrigerator", "object", REACHED, ()),
        ("where's the fridge?", "refrigerator", "object", LOCALIZED, ()),
        ("Pick up the kettle", "kettle", "object", GRASPED, ()),
        ("Bring me the kettle", "kettle", "object", DELIVERED, ()),
        ("Inspect the cabinet", "cabinet", "object", INSPECTED, ()),
        ("Inspect the kitchen cabinet", "kitchen_cabinet", "object", INSPECTED, ()),
        ("Find the refrigerator and open it", "refrigerator", "object", LOCALIZED, (OPENED,)),
        ("Go to the kitchen", "kitchen", "room", REACHED, ()),
        ("goto entrance", "entrance", "place", REACHED, ()),
        ("Walk to the far end of the room and come back", None, None, REACHED, ()),
        ("the refrigerator", "refrigerator", "object", UNKNOWN, ()),
    ],
)
def test_grammar_reads_target_and_outcome_and_never_guesses(text, target, kind, outcome, also):
    spec = ground_request(text, OBJECTS, ROOMS, places=["entrance"])
    assert (spec.target_class, spec.target_kind, spec.required_outcome) == (target, kind, outcome)
    assert spec.also_required == also
    assert spec.original_instruction == text and spec.grounding == "grammar"


def test_named_instance_is_kept():
    spec = ground_request("Inspect candidate 6", OBJECTS)
    assert spec.target_instance == "candidate:6" and spec.required_outcome == INSPECTED
    assert bind_request(spec, SEARCH_CAPS) == Inspect("cabinet", 6)


def test_specification_is_immutable_and_digest_is_stable():
    spec = ground_request("Find the refrigerator", OBJECTS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.target_class = "entrance"
    assert spec.digest == ground_request("Find the refrigerator", OBJECTS).digest
    assert spec.digest != ground_request("Find the sink", OBJECTS).digest


@pytest.mark.parametrize(
    "text, caps, expected",
    [
        ("Find the refrigerator", SEARCH_CAPS, FindObject("refrigerator", LOCALIZED)),
        ("Walk to the fridge", SEARCH_CAPS, FindObject("refrigerator", REACHED)),
        ("Inspect the cabinet", SEARCH_CAPS, Inspect("cabinet")),
        ("goto entrance", SEARCH_CAPS, GoToPlace("entrance")),
    ],
)
def test_binding_is_deterministic(text, caps, expected):
    spec = ground_request(text, OBJECTS, ROOMS, places=caps.places)
    assert bind_request(spec, caps) == expected
    assert check_binding(spec, expected)[0]


@pytest.mark.parametrize(
    "text, caps, why",
    [
        ("Pick up the kettle", SEARCH_CAPS, "cannot make something grasped"),
        ("Bring me the kettle", SEARCH_CAPS, "cannot make something delivered"),
        ("Find the refrigerator and open it", SEARCH_CAPS, "cannot make something opened"),
        ("Walk to the refrigerator", NOV7_CAPS, "cannot recognise a refrigerator"),
        ("Inspect the refrigerator", SEARCH_CAPS, "only establish 'inspected'"),
        ("Walk to the far end of the room and come back", SEARCH_CAPS, "names nothing"),
        ("the refrigerator", SEARCH_CAPS, "no action"),
    ],
)
def test_unsupported_requests_are_refused_not_substituted(text, caps, why):
    spec = ground_request(text, OBJECTS, ROOMS, places=caps.places)
    goal = bind_request(spec, caps)
    assert isinstance(goal, UnsupportedGoal) and why in goal.reason


@pytest.mark.parametrize(
    "text, goal",
    [
        ("Walk to the refrigerator", GoToPlace("entrance")),  # s7_T3, s8_T3: false claims
        ("Walk to the refrigerator", GoToObserved(1.0, 2.0, "frontier:0")),  # s9_T3
        ("Walk to the refrigerator", Inspect("cabinet")),
        ("Pick up the kettle", Inspect("cabinet", 1)),  # s9_T4: false claim
        ("Find the refrigerator", FindObject("kitchen_cabinet")),
        ("Walk to the refrigerator", FindObject("refrigerator", LOCALIZED)),  # weaker outcome
        ("Inspect candidate 6", Inspect("cabinet", 5)),  # another instance
        ("Find the refrigerator and open it", FindObject("refrigerator", LOCALIZED)),
    ],
)
def test_substitutions_never_bind(text, goal):
    spec = ground_request(text, OBJECTS, ROOMS, places=["entrance"])
    ok, why = check_binding(spec, goal)
    assert not ok and why


def test_reaching_implies_localizing_but_not_the_reverse():
    find = ground_request("Find the refrigerator", OBJECTS)
    walk = ground_request("Walk to the refrigerator", OBJECTS)
    assert check_binding(find, FindObject("refrigerator", REACHED))[0]
    assert not check_binding(walk, FindObject("refrigerator", LOCALIZED))[0]


def test_refusal_is_always_consistent():
    spec = ground_request("Find the refrigerator", OBJECTS)
    assert check_binding(spec, UnsupportedGoal("no"))[0]


# ----- model proposals must be anchored in the request ------------------------------------
class Scripted:
    """A backend that returns fixed tool-call outputs in order (repeating the last)."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def generate(self, messages, tools):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return out


def call(name, **args):
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args})}\n</tool_call>"


@pytest.mark.parametrize(
    "proposal, problem",
    [
        (dict(target="entrance", target_kind="place", outcome=REACHED), "not in the vocabulary"),
        (dict(target="kitchen_cabinet", target_kind="object", outcome=REACHED), "not named"),
        (dict(target="refrigerator", target_kind="object", outcome=LOCALIZED), "not stated"),
    ],
)
def test_model_substitutions_are_rejected(proposal, problem):
    backend = Scripted([call("specify_request", **proposal)])
    spec, rejected = propose_grounding(backend, "Walk to the refrigerator", OBJECTS)
    assert spec is None and len(rejected) == 2
    assert any(problem in p for p in rejected[0]["problems"])


def test_model_may_ground_a_listed_synonym():
    backend = Scripted(
        [call("specify_request", target="refrigerator", target_kind="object", outcome=REACHED)]
    )
    spec, rejected = propose_grounding(backend, "please walk to the fridge", OBJECTS)
    assert spec is not None and not rejected and spec.grounding == "model"
    assert (spec.target_class, spec.required_outcome) == ("refrigerator", REACHED)


def test_validation_rejects_an_unnamed_instance():
    spec = GoalSpecification("Inspect the cabinet", "cabinet", "object", "candidate:2", INSPECTED)
    assert any("instance" in p for p in check_anchoring(spec, OBJECTS))


# ----- replay of the frozen 7 Nov 2026 interpretations --------------------------------------
def _frozen(case: str):
    d = json.loads((FROZEN / case / "live/qwen_decisions/0000_decision.json").read_text())
    request = json.loads(d["messages"][1]["text"])
    return d["output"], request


@pytest.mark.parametrize(
    "case, original",
    [
        ("s7_T3", GoToPlace("entrance")),
        ("s8_T3", GoToPlace("entrance")),
        ("s9_T4", Inspect("cabinet", 1)),
        ("s7_T4", Inspect("cabinet", 1)),
    ],
)
def test_frozen_substitutions_are_reproduced_without_spec_and_refused_with_it(case, original):
    if not (FROZEN / case).is_dir():
        pytest.skip("frozen Qwen evidence not present")
    output, request = _frozen(case)
    scene, places = request["observed_scene"], set(request["available_places"])
    old = interpret_mobile_goal(Scripted([output]), request["request"], places, scene, lambda: None)
    assert old == original  # the 7 Nov behaviour, unchanged without a spec
    spec = ground_request(request["request"], OBJECTS, ROOMS, places=places)
    new = interpret_mobile_goal(
        Scripted([output]), request["request"], places, scene, lambda: None, spec=spec
    )
    assert isinstance(new, UnsupportedGoal) and "without changing it" in new.reason


# ----- TaskAgent's completion gate -------------------------------------------------------
class FakeToolbox:
    """Verifies whatever goal it is given as satisfied: the 7 Nov verifier's blind spot."""

    def __init__(self, satisfied=True):
        self.cancel, self.satisfied, self.calls = threading.Event(), satisfied, []

    def call(self, name, args=None):
        self.calls.append(name)
        data = {
            "observe_scene": {},
            "verify_goal": {"satisfied": self.satisfied, "reason": "bound goal verified"},
            "get_robot_state": {"protective_stop": False},
        }.get(name, {})
        return ToolResult(name, "succeeded", "", data)


class ClaimsSuccess:
    def next_action(self, goal, scene, verdict, history):
        return Finish(True, "done") if history else ToolCall("look_around")


def test_agent_refuses_an_inconsistent_binding_before_acting():
    spec = ground_request("Walk to the refrigerator", OBJECTS, places=["entrance"])
    tools = FakeToolbox()
    ep = TaskAgent(tools, ClaimsSuccess()).run(GoToPlace("entrance"), spec=spec)
    assert not ep.success_claimed and ep.completion == "goal_substitution_rejected"
    assert "look_around" not in tools.calls and ep.spec is spec


def test_agent_without_spec_keeps_the_old_behaviour():
    ep = TaskAgent(FakeToolbox(), ClaimsSuccess()).run(GoToPlace("entrance"))
    assert ep.success_claimed


def test_agent_rejects_a_claim_the_verifier_does_not_support():
    spec = ground_request("Find the refrigerator", OBJECTS)
    ep = TaskAgent(FakeToolbox(satisfied=False), ClaimsSuccess()).run(
        FindObject("refrigerator"), spec=spec
    )
    assert not ep.success_claimed and "not verified" in ep.reason


def test_agent_accepts_a_consistent_verified_claim():
    spec = ground_request("Find the refrigerator", OBJECTS)
    ep = TaskAgent(FakeToolbox(), ClaimsSuccess()).run(FindObject("refrigerator"), spec=spec)
    assert ep.success_claimed and ep.to_dict()["spec"]["digest"] == spec.digest
