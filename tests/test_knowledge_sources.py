"""Selectable knowledge sources: conceptnet (default), qwen, combined, none."""

import json
from pathlib import Path

import numpy as np
import pytest

from dream.semantics.knowledge import (
    KNOWLEDGE_SOURCES,
    QWEN_FLOOR,
    ROOM_CATEGORIES,
    combined_knowledge,
    load_knowledge,
    load_source,
    qwen_knowledge,
    uniform_knowledge,
)

N = len(ROOM_CATEGORIES)
ROOT = Path(__file__).resolve().parents[1]


def _priors(tmp_path, objects):
    rows = {}
    for name, scores in objects.items():
        if scores is None:
            rows[name] = {"valid": False, "problem": "malformed"}
        else:
            table = {c: scores.get(c, 0.0) for c in (*ROOM_CATEGORIES, "none")}
            rows[name] = {"valid": True, "table": table}
    path = tmp_path / "priors.json"
    path.write_text(json.dumps({"objects": rows}))
    return path


def test_default_source_is_the_existing_conceptnet_knowledge():
    base, chosen = load_knowledge(), load_source("conceptnet")
    assert chosen.strength == base.strength and chosen.source == base.source
    for obj in ("refrigerator", "bed", "stove", "washing_machine", "not_an_object"):
        assert np.array_equal(base.association(obj), chosen.association(obj))
        assert base.has(obj) == chosen.has(obj)


def test_none_is_uniform_for_every_class():
    k = load_source("none")
    for obj in ("refrigerator", "bed", "unknown_thing"):
        assert np.allclose(k.association(obj), 1.0 / N)
        assert np.allclose(k.likelihood_ratio(obj), 1.0)
        assert not k.has(obj)


def test_qwen_is_normalised_over_rooms_and_drops_none(tmp_path):
    path = _priors(tmp_path, {"stove": {"kitchen": 0.6, "bedroom": 0.1, "none": 0.3}})
    k = qwen_knowledge(path)
    a = k.association("stove")
    assert a.shape == (N,) and np.isclose(a.sum(), 1.0)
    i, j = ROOM_CATEGORIES.index("kitchen"), ROOM_CATEGORIES.index("bedroom")
    assert a[i] / a[j] == pytest.approx(6.0, rel=0.1)  # 'none' mass removed, ratios kept
    assert a.min() >= QWEN_FLOOR / 2  # exact zeros are floored, never log(0)
    assert k.has("stove") and np.all(np.isfinite(np.log(k.likelihood_ratio("stove"))))


def test_qwen_missing_or_invalid_classes_get_no_knowledge(tmp_path):
    path = _priors(tmp_path, {"bad": None, "empty": {"none": 1.0}, "ok": {"kitchen": 1.0}})
    k = qwen_knowledge(path)
    for obj in ("bad", "empty", "never_asked"):
        assert not k.has(obj)
        assert np.allclose(k.association(obj), 1.0 / N)
    assert k.has("ok")


def test_combined_is_the_equal_weight_mixture_with_fallback(tmp_path):
    path = _priors(tmp_path, {"stove": {"kitchen": 1.0}, "qwen_only": {"bedroom": 1.0}})
    cn = load_knowledge()
    qw = qwen_knowledge(path)
    both = combined_knowledge(cn, qw)
    assert cn.has("stove") and qw.has("stove")
    mixture = (cn.association("stove") + qw.association("stove")) / 2
    assert np.allclose(both.association("stove"), mixture)
    assert not cn.has("qwen_only")
    assert np.allclose(both.association("qwen_only"), qw.association("qwen_only"))
    cn_only = next(o for o in cn.strength if not qw.has(o))
    assert np.allclose(both.association(cn_only), cn.association(cn_only))
    assert not both.has("neither") and np.allclose(both.association("neither"), 1.0 / N)
    assert np.isclose(both.association("stove").sum(), 1.0)


def test_real_qwen_priors_and_source_names():
    k = load_source("qwen")
    assert k.has("refrigerator") and k.typical_rooms("washing_machine", 1)[0][0] == "laundry_room"
    assert k.synonyms  # request words come from the shared vocabulary
    for name in KNOWLEDGE_SOURCES:
        assert load_source(name).categories == ROOM_CATEGORIES
    with pytest.raises(ValueError):
        load_source("oracle")
    assert uniform_knowledge().source.startswith("uniform")


def test_the_elicitation_prompt_holds_only_the_object_and_the_place_vocabulary():
    """The Qwen answers behind ``qwen`` must not have seen the generator's furnishing tables."""
    from experiments.qwen_room_priors import SYSTEM, USER, build_messages

    data = json.loads((ROOT / "results/qwen_room_priors/priors.json").read_text())
    assert build_messages("kettle", data["rooms"]) == data["example_messages"]
    text = (SYSTEM + USER).lower()
    for word in ("furnish", "benchmark", "generator", "scenario", "table"):
        assert word not in text


def test_k1_config_is_the_registered_design():
    from experiments.semantic_search_eval import jobs_for

    cfg = json.loads((ROOT / "experiments/configs/semantic_knowledge_xl.json").read_text())
    split = cfg["splits"]["K1"]
    jobs = jobs_for(cfg, "K1", split["arms"], split["base"])
    assert len(jobs) == 576 and len(split["arms"]) == 6
    assert {j["generator"] for j in jobs} == {"xl"} and {j["noise"] for j in jobs} == {"nominal"}
    assert {j["seed"] for j in jobs} == {9300, 9301, 9302, 9303}
    assert {j["scenario"] for j in jobs} == {"B", "D", "E"} and len(
        {j["target"] for j in jobs}
    ) == 8
    for name in split["arms"]:
        source = cfg["arms"][name].get("knowledge")
        assert source is None or source in KNOWLEDGE_SOURCES
    assert cfg["weights"]["kappa"] == 3.0 and cfg["time_budget_s"] == 600
