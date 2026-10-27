from experiments.g1_reach_attribute import cause


def row(verdict, reason, *, success=False, false_claim=False, stops=None):
    return {
        "agent": {"verdict": verdict, "reason": reason},
        "evaluator": {"physical_success": success, "false_success_claim": false_claim,
                      "protective_stops": []},
        "stop_details": stops or [],
    }


def test_recorded_stops_take_precedence_over_refusal_heuristic():
    r = row("refused", "unsafe", stops=[{"reason": "torque", "joint_name": "shoulder"}])
    assert cause(r) == "stop: torque (shoulder)"


def test_unexplained_refusal_is_unresolved_not_planner():
    assert cause(row("refused", "unsupported instruction")) == "unresolved refusal"


def test_physical_success_and_false_claim_are_distinct():
    assert cause(row("failed", "no claim", success=True)) == "physical success without claim"
    assert cause(row("reached", "", false_claim=True)) == "false claim"
