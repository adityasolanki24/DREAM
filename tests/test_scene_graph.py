"""Home memory as a Spark-DSG (Hydra) scene graph: structure, queries, file round trip."""

from __future__ import annotations

import pytest

spark_dsg = pytest.importorskip("spark_dsg")

from dream.memory.home import HomeMemory  # noqa: E402
from dream.memory.scene_graph import SceneGraphView, build_scene_graph  # noqa: E402

MAP = "home"


@pytest.fixture
def mem(tmp_path):
    m = HomeMemory(tmp_path / "h.sqlite")
    k = m.add_place("kitchen", "room", MAP, (0, 0))
    b = m.add_place("bedroom", "room", MAP, (6, 0))
    h = m.add_place("hall", "room", MAP, (3, 3))
    m.connect_places(k, h)
    m.connect_places(h, b)
    m.add_place("counter", "place", MAP, (0.5, 0.8), parent=k)
    m.observe("bottle", (0.5, 0.8, 0.9), MAP, time=0)
    m.observe("charger", (6.2, 0.4, 0.5), MAP, time=0)
    gone = m.observe("mug", (0.2, -0.5, 0.9), MAP, time=0)
    m.record_absence((0.2, -0.5), 0.4, MAP, time=10)
    yield m, gone
    m.close()


def test_graph_has_hydra_layers_and_room_connectivity(mem):
    m, _ = mem
    view = SceneGraphView(build_scene_graph(m, MAP, now=20))
    assert sorted(view.rooms()) == ["bedroom", "hall", "kitchen"]
    assert view.connected_rooms("hall") == ["bedroom", "kitchen"]
    assert view.connected_rooms("kitchen") == ["hall"]
    assert view.room_of((5.5, 0.2)) == "bedroom"


def test_objects_in_rooms_keep_evidence_state(mem):
    m, gone = mem
    view = SceneGraphView(build_scene_graph(m, MAP, now=20))
    assert [o["label"] for o in view.objects_in("kitchen")] == ["bottle"]
    missing = view.objects_in("kitchen", include_missing=True)
    assert {o["id"]: o["state"] for o in missing}[gone] == "missing"
    (bottle,) = view.last_seen("bottle")
    assert bottle["state"] == "observed" and bottle["place_id"] == "room:kitchen"


def test_scene_graph_file_round_trip(mem, tmp_path):
    m, _ = mem
    path = str(tmp_path / "home_dsg.json")
    SceneGraphView(build_scene_graph(m, MAP, now=20)).save(path)
    view = SceneGraphView.load(path)
    assert view.graph.metadata.get()["map_id"] == MAP
    assert [o["label"] for o in view.objects_in("bedroom")] == ["charger"]
    assert view.connected_rooms("hall") == ["bedroom", "kitchen"]
