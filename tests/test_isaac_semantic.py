"""CPU tests of the Isaac semantic-mode wiring and the original-request scoring of both arms.

Backends and labelled prims are fakes; nothing here needs Isaac, a GPU or a network."""

import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from dream.semantics.knowledge import load_knowledge
from dream.tools.goals import FindObject, GoToObserved, GoToPlace, Inspect, UnsupportedGoal
from experiments import isaac_semantic as sem
from experiments import isaac_semantic_eval as ev

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/isaac_semantic_eval.json"
KNOWLEDGE = load_knowledge()
PLACES = {"entrance": np.zeros(2)}


class FakeBackend:
    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), 0

    def generate(self, messages, tools):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def specify(**arguments):
    call = {"name": "specify_request", "arguments": arguments}
    return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"


@pytest.mark.parametrize(
    "text, goal",
    [
        ("Find the refrigerator", FindObject("refrigerator", "visually_localized")),
        ("Walk to the refrigerator", FindObject("refrigerator", "reached")),
        ("Find the sink", FindObject("sink", "visually_localized")),
        ("Find the bed", FindObject("bed", "visually_localized")),
        ("Inspect the cabinet", Inspect("cabinet", None)),
    ],
)
def test_supported_requests_bind_without_consulting_the_model(text, goal):
    backend = FakeBackend()
    bound = sem.ground_and_bind(text, backend, KNOWLEDGE, PLACES)
    assert bound["goal"] == goal
    assert backend.calls == 0 and not bound["record"]["model_consulted"]
    assert bound["spec"].original_instruction == text


@pytest.mark.parametrize("text", ["Pick up the kettle", "Bring me the kettle"])
def test_manipulation_is_refused_by_capability_not_approximated(text):
    backend = FakeBackend()
    goal = sem.ground_and_bind(text, backend, KNOWLEDGE, PLACES)["goal"]
    assert isinstance(goal, UnsupportedGoal) and backend.calls == 0


def test_the_closeout_substitution_requests_cannot_bind_to_the_entrance_or_a_cabinet():
    for text in ("Walk to the refrigerator", "Pick up the kettle"):
        goal = sem.ground_and_bind(text, FakeBackend(), KNOWLEDGE, PLACES)["goal"]
        assert not isinstance(goal, (GoToPlace, GoToObserved, Inspect))


def test_ungrounded_request_consults_the_model_and_is_still_refused():
    text = "Walk to the far end of the room and come back"
    backend = FakeBackend([specify(target="", target_kind="none", outcome="reached")])
    bound = sem.ground_and_bind(text, backend, KNOWLEDGE, PLACES)
    assert backend.calls == 1 and bound["record"]["model_consulted"]
    assert isinstance(bound["goal"], UnsupportedGoal)


def test_a_model_cannot_ground_a_target_the_user_did_not_name():
    text = "Walk to the far end of the room and come back"
    lie = specify(target="refrigerator", target_kind="object", outcome="reached")
    backend = FakeBackend([lie, lie])
    bound = sem.ground_and_bind(text, backend, KNOWLEDGE, PLACES)
    assert isinstance(bound["goal"], UnsupportedGoal)
    assert bound["record"]["model_rejected"] and bound["record"]["model_spec"] is None


def test_a_failing_model_call_is_recorded_and_the_request_is_refused():
    backend = FakeBackend([RuntimeError("server down")])
    bound = sem.ground_and_bind("Walk to the far end", backend, KNOWLEDGE, PLACES)
    assert isinstance(bound["goal"], UnsupportedGoal)
    assert "server down" in bound["record"]["model_error"]


def test_undetectable_class_is_not_searched_for():
    goal = sem.ground_and_bind("Find the cabinet", FakeBackend(), KNOWLEDGE, PLACES)["goal"]
    assert isinstance(goal, UnsupportedGoal)


def test_attach_semantic_search_wires_the_toolbox_for_stationary_detection():
    toolbox = SimpleNamespace(semantic=None, detector=None, detect_when="always")
    memory = SimpleNamespace(
        grid=__import__("dream.perception.occupancy").perception.occupancy.OccupancyGrid()
    )
    planner = sem.attach_semantic_search(toolbox, memory, FakeBackend(), KNOWLEDGE)
    assert toolbox.semantic is not None and toolbox.detect_when == "stationary"
    assert toolbox.detector.classes == sorted(c.replace("_", " ") for c in sem.DETECTOR_VOCABULARY)
    assert planner.policy == "utility" and planner.toolbox is toolbox and not planner.memory_hints
    assert (planner.weights.kappa, planner.weights.ig, planner.weights.gamma) == (3.0, 1.0, 1.0)


# -- scorer mapping -------------------------------------------------------------------------


def prim(path, label, lo, hi):
    return SimpleNamespace(path=path, label=label, lo=np.array(lo, float), hi=np.array(hi, float))


KITCHEN = [
    prim("/World/Fridge", "refrigerator", [2.0, 1.0, 0.0], [3.0, 2.0, 1.8]),
    prim("/World/Sink/rim", "sink_rim", [-1.0, 1.0, 0.8], [0.0, 1.6, 0.9]),
    prim("/World/Sink/basin", "sink_basin", [-0.8, 1.1, 0.7], [-0.2, 1.5, 0.8]),
    prim("/World/Sink", "sink", [-1.0, 1.0, 0.7], [0.0, 1.6, 0.9]),
    prim("/World/Oven", "oven", [0.5, 1.0, 0.0], [1.3, 1.7, 0.9]),
    prim("/World/Top", "countertop", [-3.0, 1.0, 0.85], [-1.02, 1.7, 0.9]),
    prim("/World/Cabinet", "kitchen_cabinet", [0.0, -2.0, 0.0], [1.0, -1.0, 0.9]),
    prim("/World/Bagel", "bagel", [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
]


def test_scene_labels_map_to_scorer_instances():
    instances = {i.id: i for i in sem.true_instances(KITCHEN)}
    assert sorted(instances) == [
        "countertop:0", "refrigerator:0", "sink:0", "stove:0",
    ]  # fmt: skip
    sink = instances["sink:0"]
    assert np.allclose(sink.centre, [-0.5, 1.3]) and np.allclose(sink.half, [0.5, 0.3])
    assert instances["stove:0"].cls == "stove" and instances["stove:0"].yaw == 0.0
    assert instances["refrigerator:0"].room_type == "kitchen"


def test_separate_sinks_stay_separate_instances():
    far = prim("/World/Sink2/rim", "sink_rim", [4.0, 1.0, 0.8], [5.0, 1.6, 0.9])
    sinks = [i for i in sem.true_instances([*KITCHEN, far]) if i.cls == "sink"]
    assert len(sinks) == 2


def instances():
    return sem.true_instances(KITCHEN)


FRIDGE_REQUEST = {"target": "refrigerator", "outcome": "reached", "supported": True}
KETTLE_REQUEST = {"target": "kettle", "outcome": "grasped", "supported": False}
BED_REQUEST = {"target": "bed", "outcome": "visually_localized", "supported": True}


def score(task, goal, **kw):
    base = dict(
        goal=goal, claimed=False, completion=None, verdict={},
        odometry_end=[0.0, 0.0, 0.0], true_pose_end=[0.0, 0.0, 0.0], instances=instances(),
    )  # fmt: skip
    return sem.score_original_request(task, **{**base, **kw})


def test_the_nov7_failures_are_false_successes_on_the_original_request():
    # "Walk to the refrigerator" bound to the entrance, robot standing at it, claimed.
    walk = score(FRIDGE_REQUEST, GoToPlace("entrance"), claimed=True)
    assert walk["result"] == "false_success" and walk["false_success_claim"]
    # "Pick up the kettle" bound to a cabinet inspection and claimed.
    kettle = score(KETTLE_REQUEST, Inspect("cabinet"), claimed=True, odometry_end=[0.5, -1.2, 0])
    assert kettle["result"] == "false_success"


def test_a_verified_find_claim_at_the_refrigerator_succeeds():
    verdict = {"position": [2.5, 1.5, 0.9]}
    find = dict(FRIDGE_REQUEST, outcome="visually_localized")
    out = score(find, FindObject("refrigerator"), claimed=True, verdict=verdict)
    assert out["result"] == "success"
    # The same claim made at the stove is a false success (wrong object).
    wrong = score(find, FindObject("refrigerator"), claimed=True, verdict={"position": [0.9, 1.3]})
    assert wrong["result"] == "false_success"


def test_reached_requires_to_stand_at_the_object_in_the_world_frame():
    verdict = {"position": [2.5, 1.5, 0.9]}
    reached = FindObject("refrigerator", "reached")
    near, far = (
        score(FRIDGE_REQUEST, reached, claimed=True, verdict=verdict, true_pose_end=pose,
              odometry_end=pose)
        for pose in ([1.4, 1.5, 0.0], [-2.0, -1.0, 0.0])
    )  # fmt: skip
    assert near["result"] == "success" and far["result"] == "false_success"


def test_odometry_drift_is_compensated_with_the_true_pose():
    # Odometry claims (0, 0) but the robot truly stands 2.4 m from the origin, at the fridge.
    out = score(
        dict(FRIDGE_REQUEST, outcome="visually_localized"), GoToPlace("entrance"), claimed=True,
        odometry_end=[0.0, 0.0, 0.0], true_pose_end=[1.8, 1.5, 0.0],
    )  # fmt: skip
    assert out["result"] == "success"


def test_absent_target_and_refusals_are_scored_correctly():
    searched = score(BED_REQUEST, FindObject("bed"), completion="search_exhausted")
    assert searched["result"] == "correct_refusal"
    claimed = score(BED_REQUEST, FindObject("bed"), claimed=True, verdict={"position": [2.5, 1.5]})
    assert claimed["result"] == "false_success"
    refused = score(KETTLE_REQUEST, UnsupportedGoal("no"), completion="goal_unsupported")
    assert refused["result"] == "correct_refusal"


def test_cabinet_request_uses_the_handle_scorer_and_requires_an_inspection_goal():
    task = {"target": "cabinet", "outcome": "inspected", "supported": True, "scorer": "cabinet"}
    ok = score(task, Inspect("cabinet"), claimed=True, cabinet_success=True)
    assert ok["result"] == "success"
    bad = score(task, Inspect("cabinet"), claimed=True, cabinet_success=False)
    assert bad["result"] == "false_success"
    substituted = score(task, GoToPlace("entrance"), claimed=True, cabinet_success=True)
    assert substituted["result"] == "false_success"
    assert score(task, Inspect("cabinet"), cabinet_success=True)["result"] == "missed"


def test_claim_position_and_typed_completion():
    assert sem.claim_position(FindObject("sink"), {"position": [1, 2, 3]}, [9, 9]) == [1.0, 2.0]
    assert sem.claim_position(FindObject("sink"), {}, [9, 9]) is None
    assert sem.claim_position(GoToPlace("entrance"), {}, [3.0, 4.0, 0.5]) == [3.0, 4.0]
    toolbox = SimpleNamespace(
        base=SimpleNamespace(stop_reason=None),
        semantic=SimpleNamespace(map=SimpleNamespace(candidates=lambda target: [])),
    )
    assert sem.typed_completion(UnsupportedGoal("x"), None, False, "", toolbox) == (
        "goal_unsupported",
        "driver:UnsupportedGoal",
    )
    assert sem.typed_completion(FindObject("sink"), "search_exhausted", False, "", toolbox) == (
        "search_exhausted",
        "episode",
    )
    assert sem.typed_completion(FindObject("sink"), None, False, "time budget", toolbox) == (
        "target_not_yet_found",
        "stop_outcome",
    )


# -- config, case plan and gates -------------------------------------------------------------


def test_case_plan_matches_the_task_statement():
    cfg = ev.load_config(CONFIG, allow_draft=True)
    cases = ev.build_cases(cfg)
    assert len(cases) == 5 * 2 * (7 + 3)
    assert {c["split"] for c in cases} == {"development", "held_out"}
    assert {c["kitchen"] for c in cases if c["split"] == "held_out"} == {
        "g_shape", "l_island", "peninsula", "u_shape",
    }  # fmt: skip
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert [c["text"] for c in cases if c["arm"] == "nov7"][:3] == [
        "Walk to the refrigerator", "Pick up the kettle", "Find the refrigerator",
    ]  # fmt: skip
    kettle = [c for c in cases if c["text"] == "Pick up the kettle"]
    assert all(c["unsupported_expected"] for c in kettle)
    with pytest.raises(ValueError, match="frozen"):
        ev.load_config(CONFIG)  # a draft cannot be run by accident


def test_scoring_table_entries_agree_across_arms():
    cfg = ev.load_config(CONFIG, allow_draft=True)
    seen: dict[str, dict] = {}
    for arm in cfg["arms"].values():
        for entry in arm["tasks"]:
            entry = {k: v for k, v in entry.items() if k not in ("id", "note")}
            assert seen.setdefault(entry["text"], entry) == entry


def case(
    arm, task, result, claimed=False, completion=None, split="held_out", fell=False, ev_ok=True
):
    return {
        "case_id": f"{split}_{arm}_{task}_{result}",
        "split": split, "arm": arm, "task": task, "evaluation_present": ev_ok,
        "original_result": result, "success_claimed": claimed, "completion": completion,
        "false_success": result == "false_success", "fell": fell, "text": task,
    }  # fmt: skip


def passing_cases():
    return [
        case("semantic", "S1", "success", True),
        case("semantic", "S4", "correct_refusal", completion="search_exhausted"),
        case("semantic", "S5", "correct_refusal", completion="goal_unsupported"),
        case("semantic", "S6", "correct_refusal", completion="goal_unsupported"),
        case("nov7", "N1", "false_success", True),
    ]


def test_gates_pass_and_nov7_false_claims_do_not_fail_the_semantic_gate():
    cfg = ev.load_config(CONFIG, allow_draft=True)
    gates = ev.evaluate_gates(cfg, passing_cases())["pooled"]
    assert {g["result"] for g in gates.values()} == {"PASS"}
    tables = ev.report_tables(passing_cases())
    assert tables["nov7_false_claims"]["total"] == 1 and tables["nov7_false_claims"]["of"] == 1


@pytest.mark.parametrize(
    "mutation, gate",
    [
        (
            lambda c: c.append(case("semantic", "S2", "false_success", True)),
            "semantic_false_success_zero",
        ),
        (
            lambda c: c.append(case("semantic", "S5", "missed", completion=None)),
            "unsupported_refused_every_case",
        ),
        (
            lambda c: c.append(case("semantic", "S4", "false_success", True)),
            "find_bed_never_claimed",
        ),
        (lambda c: c.append(case("nov7", "N3", "missed", fell=True)), "falls_zero"),
    ],
)
def test_each_gate_can_fail(mutation, gate):
    cfg = ev.load_config(CONFIG, allow_draft=True)
    cases = passing_cases()
    mutation(cases)
    assert ev.evaluate_gates(cfg, cases)["pooled"][gate]["result"] == "FAIL"


def test_a_case_without_evaluation_makes_gates_incomplete_never_pass():
    cfg = ev.load_config(CONFIG, allow_draft=True)
    cases = [*passing_cases(), case("semantic", "S1", "unscored", ev_ok=False, fell=None)]
    gates = ev.evaluate_gates(cfg, cases)["pooled"]
    assert {g["result"] for g in gates.values()} == {"INCOMPLETE"}


def test_deferred_cases_are_not_counted_but_flagged():
    cfg = ev.load_config(CONFIG, allow_draft=True)
    cases = [*passing_cases(), {**ev.build_cases(cfg)[0], "deferred": True}]
    assert ev.evaluate_gates(cfg, cases)["pooled"]["falls_zero"]["result"] == "INCOMPLETE"


def test_build_case_prefers_the_original_request_score_over_the_bound_goal_score():
    spec = {"case_id": "c", "arm": "semantic", "task": "S1", "text": "Find the refrigerator"}
    evaluation = {
        "episode": {"goal": "FindObject(...)", "reason": "ok", "tool_calls": 3},
        "independent_verification": {
            "false_claim": True,  # score_task flags every claim without ground truth
            "goal_kind": "FindObject",
            "physical_safety": {"fell": False},
        },
        "original_request": {"claimed": True, "result": "success", "false_success_claim": False},
        "completion_typed": {"value": None, "source": "claimed"},
    }
    outcome = {"submitted": True, "task_id": "t", "row": {"state": "succeeded"}}
    built = ev.build_case(spec, outcome, evaluation, None)
    assert built["original_result"] == "success" and not built["false_success"]
    lost = ev.build_case(spec, outcome, None, None)
    assert lost["original_result"] == "unscored" and lost["success_claimed"]
    assert not lost["evaluation_present"]


def test_decision_recorder_folder_is_configurable_and_default_unchanged(tmp_path):
    from experiments.isaac_g1_live import DecisionRecorder

    assert DecisionRecorder(FakeBackend(), tmp_path, lambda: 0.0).folder.name == "qwen_decisions"
    detector = DecisionRecorder(FakeBackend(), tmp_path, lambda: 0.0, folder_name="qwen_detector")
    assert detector.folder.name == "qwen_detector" and detector.folder.is_dir()


def test_usd_empty_bounds_are_not_geometry_and_absent_appliances_are_reported():
    """Regression (session 877a82 smoke): the refrigerator and oven prims returned float32-max
    bounds, became instances with a huge half extent, and any claim would have scored as a hit.
    They are declared by the scene but have no geometry (their asset references are 404)."""
    sentinel = 3.4028234663852886e38
    empty = prim("/World/Fridge", "refrigerator", [sentinel] * 3, [-sentinel] * 3)
    broken = [empty if p.label == "refrigerator" else p for p in KITCHEN]
    instances = sem.true_instances(broken)
    assert not [i for i in instances if i.cls == "refrigerator"]
    assert [d["label"] for d in sem.declared_absent(broken)] == ["refrigerator"]
    assert sem.missing_truth(instances) == []
    no_sink = [p for p in broken if not p.label.startswith("sink")]
    assert sem.missing_truth(sem.true_instances(no_sink)) == ["sink"]
    assert all(np.all(i.half < 100) for i in sem.true_instances(KITCHEN))


def test_adjacent_countertops_stay_separate_instances():
    tops = [
        prim("/World/T0", "countertop", [0, 0, 0.8], [1.0, 0.6, 0.9]),
        prim("/World/T1", "countertop", [1.0, 0, 0.8], [2.0, 0.6, 0.9]),
    ]
    assert len([i for i in sem.true_instances(tops) if i.cls == "countertop"]) == 2
