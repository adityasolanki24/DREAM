"""Procedural multi-room homes: structure, reachability, and every scenario's recipe.

``verify_scenario`` is the generator's own final gate, so the tests also attack it: homes that
are deliberately broken must be flagged (a wrong generator would fail them).
"""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

import dream.sim.home as H
from dream.sim.home import (
    CLASSES,
    SCENARIOS,
    TARGETS,
    GroundTruthView,
    HomeGenerationError,
    footprints,
    instances_of,
    problems,
    room_at,
    sample_home,
    semantic_index,
    verify_scenario,
)

GENERATORS = ("nominal", "shifted")


def _home(scenario, seed, generator="nominal", target="refrigerator"):
    return sample_home(np.random.default_rng(seed), scenario, target, generator)


@pytest.mark.parametrize("scenario", ["A", "B", "E", "H", "K"])
def test_fast_scenarios_hold_on_a_few_seeds(scenario):
    for seed in range(3):
        home = _home(scenario, seed)
        assert verify_scenario(home) == [], (scenario, seed)


@pytest.mark.slow
@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_invariants_and_reachability_over_30_seeds(scenario, generator):
    for seed in range(30):
        home = _home(scenario, 1000 + seed, generator)
        assert verify_scenario(home) == [], (scenario, generator, seed)
        assert problems(home) == []


def test_layout_contract():
    for generator in GENERATORS:
        for seed in range(6):
            home = _home("B", seed, generator)
            assert len(home.rooms) >= 4
            for r in home.rooms:
                assert min(np.subtract(r.hi, r.lo)) >= H.MIN_ROOM - 1e-9
            assert all(H.DOOR_WIDTH[0] <= d.width <= H.DOOR_WIDTH[1] for d in home.doorways)
            assert all(r.type in H.ROOM_TYPES + ("studio",) for r in home.rooms)
            assert {o.cls for o in home.objects} <= set(CLASSES)
            # doors connect all rooms (a spanning tree at least)
            reach, frontier = {0}, [0]
            while frontier:
                a = frontier.pop()
                for d in home.doorways:
                    for u, v in ((d.a, d.b), (d.b, d.a)):
                        if u == a and v not in reach:
                            reach.add(v)
                            frontier.append(v)
            assert len(reach) == len(home.rooms)
            assert home.width <= (12.0 if generator == "shifted" else 10.0) + 1e-9
            assert home.width / 2 + abs(home.start[0]) <= H.EXTENT_LIMIT


def test_shifted_is_a_different_distribution():
    nom = [_home("E", s) for s in range(12)]
    shf = [_home("E", s, "shifted") for s in range(12)]
    assert np.mean([len(h.rooms) for h in shf]) > np.mean([len(h.rooms) for h in nom])
    assert np.mean([len(h.objects) for h in shf]) > np.mean([len(h.objects) for h in nom])


def test_same_seed_same_home_and_seed_is_recorded():
    a = sample_home(np.random.default_rng(5), "D")
    b = sample_home(np.random.default_rng(0), "D", seed=a.seed)
    assert a == b


def test_geom_names_and_semantic_index():
    home = _home("F", 3)
    spec = mujoco.MjSpec()
    H.add_home(spec, home)
    model = spec.compile()
    names = [model.geom(g).name for g in range(model.ngeom)]
    assert all(n.startswith(("wall:", "sem:")) for n in names)
    index = semantic_index(model, home)
    classes = {o.instance_id: o.cls for o in home.objects}
    assert set(index.values()) == {(k, v) for k, v in classes.items()}
    for g, (iid, _) in index.items():
        assert names[g].startswith(f"sem:{iid}:")


@pytest.mark.slow
def test_scene_builds_with_the_g1_model():
    from dream.sim.g1_mobile import build_mobile_g1

    home = _home("A", 2)
    model = build_mobile_g1(scene=lambda s: H.add_home(s, home))
    assert model.geom("floor").id >= 0
    assert any(model.geom(g).name.startswith("sem:refrigerator") for g in range(model.ngeom))


def test_room_helpers_and_footprints():
    home = _home("B", 1)
    fr = instances_of(home, "refrigerator")
    assert len(fr) == 1
    assert room_at(home, fr[0].xy).id == fr[0].room_id
    poly = footprints(home)[fr[0].instance_id]
    assert poly.shape == (4, 2)
    area = np.linalg.norm(poly[1] - poly[0]) * np.linalg.norm(poly[3] - poly[0])
    assert area == pytest.approx(4 * fr[0].half[0] * fr[0].half[1])


def test_furnishing_probabilities_are_respected():
    """A kitchen need not have every item: the table is probabilistic, not a template."""
    sinks = [
        any(o.cls == "sink" and o.room_id == r.id for o in h.objects)
        for h in (_home("E", s) for s in range(40))
        for r in h.rooms
        if r.type == "kitchen"
    ]
    assert 0.6 < np.mean(sinks) < 1.0


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_other_targets_follow_the_recipes(target):
    for scenario in ("A", "D", "E", "H"):
        home = _home(scenario, 7, target=target)
        assert home.target_class == target
        assert verify_scenario(home) == []
        assert home.request == f"Find the {target.replace('_', ' ')}"


def test_manipulation_requests_and_memory_variants():
    reqs = {_home("L", s).request for s in range(10)}
    assert reqs <= {"Pick up the refrigerator", "Open the refrigerator", "Bring me the kettle"}
    variants = {_home("I", s).scenario for s in range(12)}
    assert variants == {"I_moved", "I_removed", "I_valid"}
    for s in range(6):
        home = _home("I_moved", s)
        (cls, xyz, age) = home.memory_prior[0]
        assert room_at(home, xyz[:2]).id != instances_of(home, cls)[0].room_id
        assert 86400 <= age <= 7 * 86400


# -- the checks must be able to fail ----------------------------------------------------------
def test_verify_flags_a_wrong_start_for_scenario_a():
    home = _home("A", 1)
    far = replace(home, start=(home.start[0], home.start[1], home.start[2] + np.pi))
    assert any("px" in p or "400" in p for p in verify_scenario(far))


def test_verify_flags_blocked_doorway_and_unreachable_object():
    home = _home("B", 2)
    d = home.doorways[0]
    half, parts = H.make_object(np.random.default_rng(0), "wardrobe", "x")
    plug = H.ObjectSpec("wardrobe_99", "wardrobe", d.a, d.centre, 0.0, (0.5, 1.0, 1.0), parts)
    blocked = replace(home, objects=(*home.objects, plug))
    assert problems(blocked)


def test_verify_flags_visible_target_in_scenario_b():
    home = _home("A", 1)  # in A the target is visible at the start, which B forbids
    assert verify_scenario(replace(home, scenario="B")) != []


def test_generation_gives_up_instead_of_looping(monkeypatch):
    monkeypatch.setattr(H, "MAX_ATTEMPTS", 0)
    with pytest.raises(HomeGenerationError):
        sample_home(np.random.default_rng(0), "A")


def test_bad_arguments():
    with pytest.raises(ValueError):
        sample_home(np.random.default_rng(0), "Z")
    with pytest.raises(ValueError):
        sample_home(np.random.default_rng(0), "A", target="piano")
    with pytest.raises(ValueError):
        sample_home(np.random.default_rng(0), "A", generator="other")


def test_ray_pixel_counts_agree_between_strides():
    home = _home("A", 4)
    view = GroundTruthView(home)
    ids = [o.instance_id for o in instances_of(home, "refrigerator")]
    full = view.pixels(ids, *home.start, stride=1)
    assert full >= 400
    assert abs(view.pixels(ids, *home.start, stride=2) - full) < 0.1 * full
