"""Persistent home memory: identities, freshness, absence, moves, restarts (CPU, no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from dream.memory.home import FRESH_S, HALF_LIFE_S, HomeMemory

MAP = "home-0"


@pytest.fixture
def mem(tmp_path):
    m = HomeMemory(tmp_path / "home.sqlite", clock=lambda: 0.0)
    kitchen = m.add_place("kitchen", "room", MAP, (0.0, 0.0))
    bedroom = m.add_place("bedroom", "room", MAP, (6.0, 0.0))
    m.connect_places(kitchen, bedroom)
    yield m
    m.close()


def test_repeated_sightings_keep_one_identity_and_a_second_bottle_is_new(mem):
    a = mem.observe("bottle", (1.0, 0.5, 0.9), MAP, time=0)
    assert mem.observe("bottle", (1.05, 0.52, 0.9), MAP, time=5) == a
    b = mem.observe("bottle", (2.5, 0.5, 0.9), MAP, time=6)  # a different bottle nearby
    assert b != a
    assert {x.object_id for x in mem.where_is("bottle", MAP, now=6)} == {a, b}


def test_observed_vs_remembered_vs_decayed(mem):
    a = mem.observe("bottle", (1.0, 0.5, 0.9), MAP, time=0)
    (now,) = mem.where_is("bottle", MAP, now=FRESH_S / 2)
    assert now.object_id == a and now.state == "observed"
    (later,) = mem.where_is("bottle", MAP, now=HALF_LIFE_S)
    assert later.state == "remembered" and later.confidence == pytest.approx(0.5)


def test_absence_marks_missing_and_a_later_sighting_elsewhere_is_a_move(mem):
    a = mem.observe("bottle", (1.0, 0.5, 0.9), MAP, time=0, descriptor=[1, 0, 0])
    assert mem.record_absence((1.0, 0.5), 0.6, MAP, time=100) == [a]
    (b,) = mem.where_is("bottle", MAP, now=101)
    assert b.state == "missing" and b.confidence == 0
    moved = mem.observe("bottle", (6.2, 0.3, 0.8), MAP, time=200, descriptor=[0.99, 0.05, 0])
    assert moved == a  # the same identity, now in the bedroom
    (b,) = mem.where_is("bottle", MAP, now=201)
    assert b.state == "observed" and b.place_id == "room:bedroom"
    assert [h["place_id"] for h in mem.history(a)] == ["room:kitchen", "room:bedroom"]


def test_a_dissimilar_object_is_not_merged_into_a_missing_one(mem):
    a = mem.observe("bottle", (1.0, 0.5, 0.9), MAP, time=0, descriptor=[1, 0, 0])
    mem.record_absence((1.0, 0.5), 0.6, MAP, time=100)
    other = mem.observe("bottle", (6.2, 0.3, 0.8), MAP, time=200, descriptor=[0, 1, 0])
    assert other != a
    states = {b.object_id: b.state for b in mem.where_is("bottle", MAP, now=201)}
    assert states == {a: "missing", other: "observed"}


def test_absence_outside_the_looked_at_region_changes_nothing(mem):
    mem.observe("bottle", (1.0, 0.5, 0.9), MAP, time=0)
    assert mem.record_absence((4.0, 0.0), 0.5, MAP, time=10) == []


def test_memory_survives_a_restart(tmp_path):
    path = tmp_path / "home.sqlite"
    m = HomeMemory(path)
    m.add_place("kitchen", "room", MAP, (0, 0))
    a = m.observe("bottle", (1.0, 0.5, 0.9), MAP, time=10, source="head_rgbd")
    m.record_affordance(a, "grasp", {"approach": "side"}, "succeeded", time=11)
    m.record_episode("bring the bottle", 0, 12, "succeeded", {"steps": 3})
    m.close()
    m = HomeMemory(path)
    (b,) = m.where_is("bottle", MAP, now=20)
    assert b.object_id == a and b.place_id == "room:kitchen" and b.sightings == 1
    np.testing.assert_allclose(b.position, [1.0, 0.5, 0.9])
    assert m.known_procedure(a, "grasp")["model"] == {"approach": "side"}
    assert m.episodes()[0]["goal"] == "bring the bottle"
    assert m.neighbours("room:kitchen") == []
    m.close()


def test_a_later_failure_withdraws_a_known_procedure(mem):
    a = mem.observe("drawer", (0.5, 1.0, 0.4), MAP, time=0)
    mem.record_affordance(a, "open", {"axis": [1, 0, 0]}, "succeeded", time=1)
    assert mem.known_procedure(a, "open") is not None
    mem.record_affordance(a, "open", {"axis": [1, 0, 0]}, "failed", time=2)
    assert mem.known_procedure(a, "open") is None


def test_maps_are_not_mixed(mem):
    mem.observe("bottle", (1.0, 0.5, 0.9), MAP, time=0)
    assert mem.where_is("bottle", "other-map", now=1) == []
    assert mem.neighbours("room:kitchen") == ["room:bedroom"]
    assert [b.label for b in mem.objects_in("room:kitchen", now=1)] == ["bottle"]


def test_invalid_observations_are_rejected(mem):
    with pytest.raises(ValueError):
        mem.observe("bottle", (np.nan, 0, 0), MAP, time=0)
    with pytest.raises(ValueError):
        mem.observe("bottle", (0, 0, 0), MAP, sigma=0, time=0)
