"""Pure-Python parts of the Isaac home composer: placement maths, refusal on a missing asset,
and the agent view carrying no scene truth."""

import json

import numpy as np
import pytest

from dream.sim.home import CLASSES, DIMS, sample_home
from experiments import isaac_home_compose as comp


def fake_manifest(skip=(), failed=()):
    classes = {}
    for cls in CLASSES:
        if cls in skip:
            continue
        d, w, h = DIMS[cls]
        classes[cls] = {
            "status": "failed" if cls in failed else "ok",
            "failure": "404" if cls in failed else None,
            "url": f"file:///assets/{cls}_asset.usd",
            "size_m": [d * 1.1, w * 0.9, h],
            "base_centre_native": [0.1, 0.2, -0.3],
            "yaw_offset_deg": 90.0,
        }
    return {"validated_utc": "2026-11-09", "classes": classes}


def home(seed=100, scenario="B", target="refrigerator"):
    return sample_home(np.random.default_rng(seed), scenario, target, "nominal", seed)


def test_matrix_puts_base_centre_at_pose_and_faces_yaw():
    base = np.array([0.1, 0.2, -0.3])
    m = comp.placement_matrix(base, 90.0, 1.2, (2.0, 3.0), 0.5, np.pi / 2)
    p = m @ [*base, 1.0]
    assert np.allclose(p[:3], [2.0, 3.0, 0.5])
    # native +x is rotated by offset (+90) and yaw (+90): points along world -x
    q = m @ [*(base + [1.0, 0, 0]), 1.0]
    assert np.allclose(q[:3] - p[:3], [-1.2, 0.0, 0.0], atol=1e-9)


def test_every_instance_gets_a_placement_inside_its_footprint():
    h = home()
    plan = comp.plan_placements(h, fake_manifest())
    assert len(plan) == len(h.objects)
    by_id = {o.instance_id: o for o in h.objects}
    for p in plan:
        o = by_id[p.instance_id]
        assert p.mode == "fit"  # the fake assets are near the generator's sizes
        base = p.scale[2]
        assert comp.SCALE_MIN - 1e-9 <= base <= comp.SCALE_MAX + 1e-9
        assert all(abs(k / base - 1) <= comp.AXIS_DEV + 1e-9 for k in p.scale)
        assert p.size_m[0] <= 2 * o.half[0] + 1e-9 and p.size_m[1] <= 2 * o.half[1] + 1e-9
        assert p.cls == o.cls and p.xy == o.xy


def test_target_present_scenario_contains_the_target_asset():
    _, placements, truth = comp.compose_plan("nominal", 100, "B", "refrigerator", fake_manifest())
    assert truth["target_present"] is True and truth["target_instances"]
    assert any(p.cls == "refrigerator" for p in placements)


def test_absent_scenario_keeps_lookalikes_and_no_target():
    _, placements, truth = comp.compose_plan("nominal", 100, "E", "refrigerator", fake_manifest())
    assert truth["target_present"] is False and truth["target_instances"] == []
    assert not any(p.cls == "refrigerator" for p in placements)


def test_missing_required_class_refuses_naming_it_never_dropping():
    h = home(scenario="B")
    with pytest.raises(comp.MissingAssetError) as e:
        comp.plan_placements(h, fake_manifest(skip=("refrigerator",)))
    assert "refrigerator" in e.value.classes


def test_failed_asset_is_a_refusal_with_its_recorded_reason():
    h = home(scenario="B")
    with pytest.raises(comp.MissingAssetError) as e:
        comp.plan_placements(h, fake_manifest(failed=("refrigerator",)))
    assert e.value.classes["refrigerator"] == "404"


def test_only_classes_in_the_home_matter():
    # a class absent from the home may lack an asset without blocking composition
    h = home(scenario="E")
    present = {o.cls for o in h.objects}
    unused = next(c for c in CLASSES if c not in present)
    assert comp.plan_placements(h, fake_manifest(skip=(unused,)))


def test_oversized_asset_is_refused_not_shrunk_below_the_limit():
    m = fake_manifest()
    m["classes"]["refrigerator"]["size_m"] = [5.0, 5.0, 2.0]
    with pytest.raises(comp.MissingAssetError):
        comp.plan_placements(home(), m)


def test_agent_view_carries_no_scene_truth():
    h, placements, truth = comp.compose_plan("nominal", 101, "D", "bed", fake_manifest())
    view = comp.agent_view(h, "/work/stage.usda")
    assert set(view) == {"stage", "start", "request"}
    text = json.dumps(view)
    for p in placements:
        assert p.instance_id not in text and p.url not in text
    other = {p.cls for p in placements if p.cls != "bed"}
    assert not any(c in text for c in other)
    assert all("asset" not in k for k in view)


def test_prim_paths_do_not_name_classes():
    _, placements, _ = comp.compose_plan("nominal", 102, "A", "bed", fake_manifest())
    for p in placements:
        leaf = p.prim_path.rsplit("/", 1)[1]
        assert leaf.startswith("o") and leaf[1:].isdigit()


def test_truth_manifest_is_complete_and_json_serialisable():
    _, placements, truth = comp.compose_plan("nominal", 103, "B", "sofa", fake_manifest())
    json.dumps(truth)
    assert len(truth["instances"]) == len(placements)
    assert {i["class"] for i in truth["instances"]} >= {"sofa"}
    assert truth["rooms"] and truth["generator"] == "nominal"


def test_composer_is_not_imported_by_decision_packages():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "dream"
    for path in root.rglob("*.py"):
        assert "isaac_home_compose" not in path.read_text(encoding="utf-8")
