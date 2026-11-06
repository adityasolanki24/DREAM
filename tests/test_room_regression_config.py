"""The regression-diagnostic config: arms overlay seeds and methods; other configs are untouched."""

import json
import pathlib

import pytest

from experiments.g1_explore_room import select_arm

CFGS = pathlib.Path(__file__).resolve().parents[1] / "experiments/configs"


def test_arm_overlays_seeds_and_methods_only():
    cfg = json.loads((CFGS / "g1_explore_room_regression_closeout.json").read_text())
    a2 = select_arm(cfg, "A2")
    assert a2["seeds"]["nominal"] == [1300, 1320]
    assert a2["seeds"]["shifted"] == [2300, 2310]
    assert a2["seeds"]["no_cabinet"] == [3100, 3110]
    assert a2["methods"] == ["gain"] and a2["arm"] == "A2"
    assert a2["time_budget_s"] == 240 and a2["max_steps"] == 60
    assert select_arm(cfg, "A1")["seeds"]["nominal"] == [1000, 1020]
    assert select_arm(cfg, "A3")["seeds"]["no_cabinet"] == [3000, 3006]


def test_arm_required_when_config_has_arms_and_rejected_otherwise():
    cfg = json.loads((CFGS / "g1_explore_room_regression_closeout.json").read_text())
    with pytest.raises(SystemExit):
        select_arm(cfg, None)
    plain = json.loads((CFGS / "g1_explore_room_closeout.json").read_text())
    assert select_arm(plain, None) is plain
    with pytest.raises(SystemExit):
        select_arm(plain, "A1")


def test_compare_flags_pass_flips_per_seed():
    from experiments.compare_room_regression import compare

    def row(kind, seed, ok, **kw):
        base = {"kind": kind, "seed": seed, "method": "gain", "claimed": ok, "false_claim": False,
                "fell": False, "contact_steps": 0, "reason": "x", "sim_s": 1.0,
                "success": ok,
                "correct_refusal": (not ok) if kind == "no_cabinet" else None}  # fmt: skip
        return base | kw

    base = {("nominal", 1): row("nominal", 1, False), ("nominal", 2): row("nominal", 2, True)}
    arm = {("nominal", 1): row("nominal", 1, True), ("nominal", 2): row("nominal", 2, True)}
    out = compare(arm, base)
    assert out["by_kind"]["nominal"]["fixed"] == [1] and out["seeds_changed_total"] == 1
    assert out["by_kind"]["nominal"]["broken"] == []
