"""Exclusion list of the Isaac composer: omit, log, refuse (owner decision of 9 Nov 2026)."""

import numpy as np
import pytest

from dream.sim.home import TARGETS, sample_home
from experiments import isaac_home_compose as comp
from tests.test_isaac_home_compose import fake_manifest

GONE = ("bathtub", "dryer", "chest_freezer", "tv_stand", "tool_rack", "shower")


def excl(manifest, classes=GONE):
    manifest["excluded"] = {"version": 1, "classes": list(classes)}
    return manifest


def home(seed, scenario, target):
    return sample_home(np.random.default_rng(seed), scenario, target, "nominal", seed)


def test_excluded_classes_are_omitted_logged_and_need_no_asset():
    m = excl(fake_manifest(skip=GONE))
    h = home(100, "B", "refrigerator")
    assert any(o.cls == "tool_rack" for o in h.objects)
    _, placements, truth = comp.compose_plan("nominal", 100, "B", "refrigerator", m)
    assert not any(p.cls in GONE for p in placements)
    om = truth["omitted"]
    assert om and all(o["class"] in GONE for o in om)
    assert len(placements) + len(om) == len(h.objects)
    assert truth["exclusion_list"]["version"] == 1 and "simplification" in truth["rule3_class"]


def test_missing_non_excluded_class_still_raises():
    m = excl(fake_manifest(skip=("tool_rack", "refrigerator")))
    with pytest.raises(comp.MissingAssetError) as e:
        comp.compose_plan("nominal", 100, "B", "refrigerator", m)
    assert "refrigerator" in e.value.classes and "tool_rack" not in e.value.classes


def test_excluded_target_is_refused():
    with pytest.raises(comp.CaseRefused):
        comp.check_case(home(103, "B", "bed"), excl(fake_manifest(), classes=("bed",)))


def test_look_alike_scenario_without_a_remaining_look_alike_is_refused():
    h = home(103, "F", "stove")
    with pytest.raises(comp.CaseRefused) as e:
        comp.check_case(h, excl(fake_manifest(), classes=tuple(TARGETS["stove"].confusers)))
    assert "look-alike" in str(e.value)
    comp.check_case(h, excl(fake_manifest(), classes=("tool_rack",)))  # other omissions are fine
