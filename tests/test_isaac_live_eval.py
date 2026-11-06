"""CPU tests for the closeout driver: frozen config, T2 rule, pairing and submission through the
real phone API, per-case records and the frozen gates. No Isaac, no model."""

from __future__ import annotations

import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest

from dream.runtime.scheduler import TaskScheduler
from dream.tools.goals import Inspect, NaturalLanguageGoal
from experiments import isaac_g1_live as live
from experiments import isaac_g1_live_eval as ev
from experiments.phase1_interface import mobile_goal

CFG = ev.load_config()


def test_config_is_frozen_and_matches_the_live_session():
    assert CFG["status"] == "frozen"
    assert CFG["model_revision"] == live.QWEN_REVISION
    assert CFG["kitchen"] == "l_shape" and CFG["starts"] == [7, 8, 9]
    assert "NOT held-out" in CFG["evidence_class"]
    assert not set(CFG["starts"]) & set(CFG["replacement_starts"])
    assert [t for g in CFG["process_groups"] for t in g] == ["T1", "T2", "T3", "T4", "T5"]


def test_fixed_task_text_goes_through_the_real_goal_grammar():
    assert mobile_goal(CFG["tasks"]["T1"]["text"]) == Inspect("cabinet", None)
    assert mobile_goal(CFG["tasks"]["T2"]["text_template"].format(candidate=4)) == Inspect(
        "cabinet", 4
    )
    for task in ("T3", "T4", "T5"):  # interpreted by the model, not the deterministic grammar
        assert isinstance(mobile_goal(CFG["tasks"][task]["text"]), NaturalLanguageGoal)


def test_groups_cover_every_start_and_task_once():
    groups = ev.build_groups(CFG)
    assert [g["id"] for g in groups][:4] == ["s7_T1_T2", "s7_T3", "s7_T4", "s7_T5"]
    assert len(groups) == 12
    assert sum(len(g["tasks"]) for g in groups) == 15


def test_t2_candidate_is_the_lowest_id_with_a_handle():
    objects = [
        {"id": 1, "handle": None},
        {"id": 5, "handle": [1, 2, 0]},
        {"id": 3, "handle": [0, 0, 0]},
    ]
    assert ev.pick_candidate(objects) == 3
    assert ev.pick_candidate([{"id": 1, "handle": None}]) is None
    assert ev.pick_candidate([]) is None


def _evaluation(**over):
    iv = {
        "goal_kind": "inspect",
        "success_claimed": True,
        "false_claim": False,
        "model_decision_valid": {"valid": True},
        "navigation_success": {"success": True, "error_m": 0.1},
        "identity_correct": {"correct": True},
        "viewpoint_correct": {"correct": True},
        "handle_visible": {"visible": True},
        "physical_safety": {"safe": True, "fell": False, "safety_stop": None},
        "independent_task_success": {"success": True},
        "repair_count": 0,
        "repair_changed_outcome": {"changed": False},
        "terminal_stop_reason": None,
    }
    iv.update(over)
    return {
        "episode": {"goal": "Inspect(...)", "failed_actions": 0},
        "independent_verification": iv,
    }


def _case(task, start=7, **iv_over):
    spec = {"case_id": f"s{start}_{task}", "group": "g", "start": start, "task": task, "text": "x"}
    return ev.build_case(
        spec,
        {"submitted": True, "task_id": "abc", "row": {"state": "succeeded", "result": "ok"}},
        _evaluation(**iv_over),
        {"final_height_m_for_evaluation_only": 0.74},
    )


def test_build_case_reports_all_eight_dimensions_separately():
    case = _case("T1")
    assert set(case["dimensions"]) == set(ev.DIMENSIONS)
    row = ev.dimension_table([case])[0]
    assert all(row[d] is True for d in ev.DIMENSIONS[:-1])
    assert row["recovery_effectiveness"] is None  # nothing needed recovering


def test_case_without_evaluation_takes_its_claim_from_the_scheduler_and_flags_integration():
    spec = {"case_id": "s7_T1", "group": "g", "start": 7, "task": "T1", "text": "x"}
    outcome = {
        "submitted": True,
        "task_id": "a",
        "row": {"state": "failed", "result": "execution error: KeyError('x')"},
    }
    case = ev.build_case(spec, outcome, None, None)
    assert case["suspected_integration_error"] and not case["success_claimed"]
    assert case["fell"] is None  # safety cannot be shown without physical_outcome
    claimed = ev.build_case(spec, {**outcome, "row": {"state": "succeeded"}}, None, None)
    assert claimed["success_claimed"] and claimed["false_claim"]


def full_matrix(**overrides):
    cases = []
    for start in CFG["starts"]:
        cases += [_case("T1", start), _case("T2", start)]
        cases += [
            _case(
                t,
                start,
                success_claimed=False,
                false_claim=False,
                goal_kind="unsupported",
                independent_task_success={"success": None, "reason": "no truth"},
            )
            for t in ("T3", "T4", "T5")
        ]
    for case in cases:
        case.update(overrides.get(case["case_id"], {}))
    return cases


def test_gates_pass_on_a_clean_matrix_with_t5_unverifiable():
    gates = ev.evaluate_gates(CFG, full_matrix())
    assert {k: v["result"] for k, v in gates.items()} == {
        "false_claims_zero": "PASS",
        "falls_zero": "PASS",
        "T4_no_success_claim": "PASS",
        "T3_no_success_claim": "PASS",
        "T1_independent_success": "PASS",
        "T2_independent_success": "PASS",
        "T5_success": "UNVERIFIABLE",
    }


def test_each_gate_fails_on_its_own_condition():
    def gate(name, **over):
        return ev.evaluate_gates(CFG, full_matrix(**over))[name]["result"]

    assert gate("T4_no_success_claim", s8_T4={"success_claimed": True}) == "FAIL"
    assert gate("T3_no_success_claim", s9_T3={"success_claimed": True}) == "FAIL"
    assert gate("false_claims_zero", s7_T1={"false_claim": True}) == "FAIL"
    assert gate("falls_zero", s7_T3={"fell": True}) == "FAIL"
    assert gate("falls_zero", s7_T3={"fell": None}) == "FAIL"  # not shown safe
    failing = {"dimensions": {**_case("T1")["dimensions"], "independent_task_success": {
        "success": False}}}  # fmt: skip
    assert gate("T1_independent_success", s7_T1=failing) == "PASS"  # 2 of 3 is enough
    assert gate("T1_independent_success", s7_T1=failing, s8_T1=failing) == "FAIL"
    assert gate("T2_independent_success", s7_T2=failing, s9_T2=failing) == "FAIL"


def test_t5_is_scored_only_where_an_independent_check_exists():
    def checked(success):
        return {"dimensions": {**_case("T5")["dimensions"], "independent_task_success": {
            "success": success}}}  # fmt: skip

    ok = full_matrix(s7_T5=checked(True), s8_T5=checked(True), s9_T5=checked(False))
    assert ev.evaluate_gates(CFG, ok)["T5_success"]["result"] == "PASS"
    bad = full_matrix(s7_T5=checked(True), s8_T5=checked(False), s9_T5=checked(False))
    assert ev.evaluate_gates(CFG, bad)["T5_success"]["result"] == "FAIL"
    only_one = full_matrix(s7_T5=checked(True))
    assert ev.evaluate_gates(CFG, only_one)["T5_success"]["result"] == "PASS"


def test_deferred_cases_make_gates_incomplete_never_pass():
    cases = full_matrix()
    cases[-1] = {"case_id": cases[-1]["case_id"], "task": "T5", "deferred": True}
    cases[-2] = {"case_id": cases[-2]["case_id"], "task": "T4", "deferred": True}
    gates = ev.evaluate_gates(CFG, cases)
    assert gates["T4_no_success_claim"]["result"] == "INCOMPLETE"
    assert gates["T5_success"]["result"] == "INCOMPLETE"
    assert gates["false_claims_zero"]["result"] == "INCOMPLETE"


def test_merge_keeps_the_run_record_over_a_deferral():
    cases = full_matrix()
    first = copy.deepcopy(cases[:-3]) + [
        {"case_id": c["case_id"], "task": c["task"], "start": c["start"], "deferred": True}
        for c in cases[-3:]
    ]
    second = cases[-3:]
    merged = ev.merge_summaries(CFG, [{"cases": first}, {"cases": second}])
    assert not merged["partial"] and merged["gates"]["T4_no_success_claim"]["result"] == "PASS"
    only_first = ev.merge_summaries(CFG, [{"cases": first}])
    assert (
        only_first["partial"]
        and only_first["gates"]["T4_no_success_claim"]["result"] == "INCOMPLETE"
    )


# -- pairing and submission against the real phone API and scheduler ---------------------------


@pytest.fixture
def served(tmp_path):
    from dream.runtime.agent import Episode

    def run_task(task, history, preempt, cancel):
        return Episode(task.goal, task.text.startswith("Inspect"), "done", [])

    scheduler = TaskScheduler(str(tmp_path / "t.sqlite"), run_task, mobile_goal)
    api = live.PairedPhoneAPI(
        scheduler,
        "x" * 32,
        provider="qwen",
        validate_task=lambda text, provider, model: {"provider": provider or "qwen", "model": "m"},
    )
    scheduler.notify = api.notify
    server = api.serve(host="127.0.0.1", port=0)
    port = server.server_address[1]
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            if scheduler.step() is None:
                time.sleep(0.02)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    yield SimpleNamespace(port=port, api=api, scheduler=scheduler)
    stop.set()
    thread.join(timeout=5)
    server.shutdown()
    scheduler.db.close()


def test_pairing_is_one_time_and_the_token_authorises_the_client(served):
    token = "p" * 40
    assert ev.pair(served.port, token) == (200, {"ready": True})
    assert ev.pair(served.port, "q" * 40)[0] == 403  # already paired
    client = ev.Client(served.port, token)
    assert client.call("GET", "/tasks") == (200, [])
    assert ev.Client(served.port, "wrong" * 8).call("GET", "/tasks")[0] == 401
    assert served.api.token == token


def test_task_is_submitted_through_the_api_with_provider_qwen_and_awaited(served):
    token = "p" * 40
    ev.pair(served.port, token)
    client = ev.Client(served.port, token)
    result = ev.run_task_via_api(
        client, "Inspect the cabinet", 20, sleep=lambda s: time.sleep(0.05)
    )
    assert result["submitted"] and not result["timed_out"]
    assert result["row"]["state"] == "succeeded"
    assert result["row"]["meta"]["provider"] == "qwen"
    refused = ev.run_task_via_api(client, "", 5)
    assert not refused["submitted"] and refused["refusal"]["http_status"] == 400


def test_wait_paired_stops_when_the_process_dies_and_times_out_cleanly():
    dead = SimpleNamespace(poll=lambda: 1)
    assert ev.wait_paired(1, "t" * 40, dead, 5, sleep=lambda s: None) is False
    ticks = iter(range(100))
    assert (
        ev.wait_paired(1, "t" * 40, None, 3, sleep=lambda s: None, clock=lambda: next(ticks))
        is False
    )


def test_driver_never_prints_or_stores_the_pairing_token():
    source = (ev.ROOT / "experiments/isaac_g1_live_eval.py").read_text(encoding="utf-8")
    assert "print(token" not in source and "write_text(token" not in source
    assert json.dumps(CFG).find("token") == -1


def test_a_driver_exception_becomes_failed_cases_not_a_crashed_matrix(monkeypatch, tmp_path):
    def boom(*args):
        raise OSError("port in use")

    monkeypatch.setattr(ev, "_run_group", boom)
    group = ev.build_groups(CFG)[0]
    result = ev.run_group(CFG, group, 7, tmp_path, None, True, None)
    assert [c["task"] for c in result["cases"]] == ["T1", "T2"]
    assert all(not c["submitted"] and "driver_exception" in c["refusal"] for c in result["cases"])


def test_a_t3_claim_is_a_false_claim_even_when_the_scorer_checks_a_substituted_place():
    """First Isaac run: 'Walk to the refrigerator' was bound to the remembered place 'entrance'
    and the scorer's go_to_place check found it 0.08 m away, so false_claim was False."""
    cases = full_matrix(s7_T3={"success_claimed": True, "false_claim": False})
    gates = ev.evaluate_gates(CFG, cases)
    assert gates["false_claims_zero"]["result"] == "FAIL"
    assert gates["T3_no_success_claim"]["result"] == "FAIL"
