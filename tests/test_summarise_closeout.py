"""The closeout summary script and the frozen closeout configs."""

import json
import pathlib

from experiments.summarise_closeout import compartment_summary, room_summary

CFGS = pathlib.Path(__file__).resolve().parents[1] / "experiments/configs"


def _row(kind, seed, method, success=True, claimed=True, sim_s=100.0, div=0, **kw):
    r = {
        "kind": kind, "seed": seed, "method": method, "success": success, "claimed": claimed,
        "false_claim": claimed and not success, "fell": False, "reason": "x", "sim_s": sim_s,
        "correct_refusal": (not claimed) if kind == "no_cabinet" else None,
        "phantom_handles": 0, "refusal_category": None if claimed else "exhausted_exploration",
        "contacts": [], "contact_classes": [], "strict_success": success,
        "frontier_decisions": [{}] * 3, "frontier_divergences": div,
    }  # fmt: skip
    return r | kw


def test_room_summary_gates_comparison_and_strict_refusal():
    cfg = json.loads((CFGS / "g1_explore_room_closeout.json").read_text())
    rows = [_row("nominal", s, "gain") for s in range(10)]
    for s in range(10):
        rows += [_row("shifted", s, "gain", sim_s=100), _row("shifted", s, "nearest", sim_s=130)]
        rows += [
            _row("no_cabinet", s, "gain", success=False, claimed=False, div=1),
            _row("no_cabinet", s, "nearest", success=False, claimed=False),
        ]
    rows[-2]["refusal_category"] = "phantom_or_unreachable_candidate"
    rows[-2]["phantom_handles"] = 1
    rows[-2]["seed"] = 9
    out = room_summary(rows, cfg)
    assert out["gates"]["nominal_success"]["pass"] and out["all_gates_pass"]
    cmp = out["frontier_comparison"]
    assert cmp["n_pairs"] == 20 and cmp["divergent_decisions_gain_arm"] == 10
    assert cmp["mean_sim_s_diff_gain_minus_nearest"] < 0 or cmp["by_kind"]["shifted"]
    assert cmp["conclusion"].startswith("gain faster") or "no claim" in cmp["conclusion"]
    # The phantom-driven refusal counts as correct under the original scorer, not strictly.
    nc = out["no_cabinet_strict"]
    assert nc["correct_refusals_original"] == 10
    assert nc["strict_correct_refusals_excluding_phantom_driven"] == 9
    # Too little divergence: no claim, whatever the times say.
    for r in rows:
        r["frontier_divergences"] = 0
    out = room_summary(rows, cfg)
    assert out["frontier_comparison"]["conclusion"] == "insufficient divergence, no claim"


def test_room_gate_fails_on_false_claim_and_low_success():
    cfg = json.loads((CFGS / "g1_explore_room_closeout.json").read_text())
    rows = [_row("nominal", s, "gain", success=s < 8) for s in range(10)]
    rows[9]["claimed"] = True  # a claim without success is a false claim
    out = room_summary(rows, cfg)
    assert not out["gates"]["nominal_success"]["pass"]
    assert not out["gates"]["false_claims_all_rows"]["pass"]


def test_compartment_summary_gates_and_failure_categories():
    cfg = json.loads((CFGS / "compartment_vision_closeout.json").read_text())

    def r(kind, seed, success, **kw):
        base = {
            "kind": kind, "seed": seed, "success": success, "claimed": success,
            "false_claim": False, "correct_rejection": kind == "fixed" and not success,
            "reason": "x", "open_calls": 0, "tool_calls": [], "failed_actions": 0,
            "investigation": {"status": "identified", "reason": "r"},
            "identified_correctly": True,
        }  # fmt: skip
        return base | kw

    rows = [r("drawer", i, i < 9) for i in range(10)] + [r("lid", i, i < 4) for i in range(10)]
    rows += [r("fixed", i, False) for i in range(10)]
    out = compartment_summary(rows, cfg)
    assert out["gates"]["dream_openable_success"]["value"] == 13 / 20
    assert not out["gates"]["dream_openable_success"]["pass"]
    assert out["gates"]["dream_fixed_declined"]["pass"] and out["gates"]["false_claims"]["pass"]
    assert out["failure_categories"] == {"identified_but_not_opened_or_placed": 7}


def test_closeout_configs_are_frozen_and_gates_not_weaker_than_requested():
    room = json.loads((CFGS / "g1_explore_room_closeout.json").read_text())
    comp = json.loads((CFGS / "compartment_vision_closeout.json").read_text())
    assert room["status"] == comp["status"] == "frozen"
    old = json.loads((CFGS / "compartment.json").read_text())["gates"]
    assert comp["gates"]["dream_openable_success_min"] >= max(
        0.7, old["dream_openable_success_min"]
    )
    # Fresh seeds: disjoint from every other config's seed ranges.
    used = set()
    for f in CFGS.glob("g1_explore*.json"):
        if f.name == "g1_explore_room_closeout.json":
            continue
        s = json.loads(f.read_text()).get("seeds")
        if isinstance(s, dict):
            for r in s.values():
                used |= {r[0]} if len(r) == 1 else set(range(*r))  # single-seed lists exist
    for a, b in room["seeds"].values():
        assert not used & set(range(a, b))
    assert comp["seed0"] >= 5030  # the seen compartment set is 5000-5029
