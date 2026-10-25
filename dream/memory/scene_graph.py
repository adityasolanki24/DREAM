"""Hierarchical scene graph of home memory, in Hydra's own data structure (Spark-DSG).

``build_scene_graph`` writes rooms, places and remembered objects from ``HomeMemory`` into a
``spark_dsg.DynamicSceneGraph`` (MIT-SPARK/Spark-DSG 1.1.3, BSD; the graph library Hydra
produces and consumes): ROOMS and PLACES layers from registered places, room connectivity
as intra-layer edges, OBJECTS with their last-seen position, time and evidence state, each
parented to its room. The file it saves is a standard Spark-DSG graph, so a Hydra-built
graph and DREAM's can be loaded by the same ``SceneGraphView``, and alternative backends
(Hydra online, ConceptGraphs/HOV-SG enrichment) can be compared without changing the agent.

The view answers the questions the agent asks: which room is this, which rooms connect,
what is in a room, where was something last seen. Evidence states (observed / remembered /
missing) come from ``HomeMemory`` and are kept as node metadata, never flattened into
"present". Optional dependency: the ``scenegraph`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dream.memory.home import HomeMemory

# Node ids: explicit disjoint integer ranges per layer. In spark_dsg 1.1.3's Python binding
# NodeSymbol("R", 2).value == NodeSymbol("O", 2).value == 2, so symbols of different
# categories collide (checked 26 Oct 2026); plain integers avoid that.
ROOM_ID0, PLACE_ID0, OBJECT_ID0 = 1_000_000, 2_000_000, 3_000_000


def _dsg():
    import spark_dsg

    return spark_dsg


def build_scene_graph(memory: HomeMemory, map_id: str, now: float):
    d = _dsg()
    g = d.DynamicSceneGraph()
    g.metadata.add({"map_id": map_id, "built_at": float(now), "source": "dream.memory.home"})
    ids: dict[str, int] = {}
    rows = memory.db.execute(
        "SELECT id, name, kind, x, y FROM places WHERE map_id=? ORDER BY id", (map_id,)
    ).fetchall()
    for k, (pid, name, kind, x, y) in enumerate(rows):
        room = kind == "room"
        attrs = d.RoomNodeAttributes() if room else d.PlaceNodeAttributes()
        attrs.position = np.array([x, y, 0.0])
        attrs.name = name
        attrs.metadata.add({"place_id": pid, "kind": kind})
        sym = (ROOM_ID0 if room else PLACE_ID0) + k
        if not g.add_node(d.DsgLayers.ROOMS if room else d.DsgLayers.PLACES, sym, attrs):
            raise RuntimeError(f"could not add {pid} to the scene graph")
        ids[pid] = sym
    for pid in ids:
        for other in memory.neighbours(pid):
            if other in ids and not g.has_edge(ids[pid], ids[other]):
                g.insert_edge(ids[pid], ids[other])
    labels = sorted({r[0] for r in memory.db.execute(
        "SELECT DISTINCT label FROM objects WHERE map_id=?", (map_id,))})  # fmt: skip
    k = 0
    for label in labels:
        for b in memory.where_is(label, map_id, now=now):
            attrs = d.ObjectNodeAttributes()
            attrs.position = np.asarray(b.position, float)
            attrs.name = b.object_id
            attrs.semantic_label = labels.index(label)
            attrs.last_update_time_ns = int(b.last_seen * 1e9)
            attrs.is_active = b.state != "missing"
            attrs.metadata.add(
                {"label": label, "state": b.state, "confidence": b.confidence,
                 "sightings": b.sightings, "place_id": b.place_id or ""}
            )  # fmt: skip
            sym = OBJECT_ID0 + k
            k += 1
            if not g.add_node(d.DsgLayers.OBJECTS, sym, attrs):
                raise RuntimeError(f"could not add {b.object_id} to the scene graph")
            if b.place_id in ids:
                g.insert_edge(ids[b.place_id], sym)
    return g


@dataclass
class SceneGraphView:
    """Queries over a Spark-DSG graph (built here or loaded from a file)."""

    graph: object

    @classmethod
    def load(cls, path: str) -> SceneGraphView:
        return cls(_dsg().DynamicSceneGraph.load(path))

    def save(self, path: str) -> None:
        self.graph.save(path)

    def _layer(self, name):
        d = _dsg()
        return list(self.graph.get_layer(getattr(d.DsgLayers, name)).nodes)

    def rooms(self) -> list[str]:
        return [n.attributes.name for n in self._layer("ROOMS")]

    def room_of(self, xy) -> str | None:
        """Nearest room centre (rooms are registered by their centres)."""
        rooms = self._layer("ROOMS")
        if not rooms:
            return None
        xy = np.asarray(xy, float)[:2]
        return min(
            rooms, key=lambda n: np.linalg.norm(n.attributes.position[:2] - xy)
        ).attributes.name

    def connected_rooms(self, room: str) -> list[str]:
        nodes = {n.id.value: n for n in self._layer("ROOMS")}
        node = next((n for n in nodes.values() if n.attributes.name == room), None)
        if node is None:
            return []
        return sorted(nodes[s].attributes.name for s in node.siblings() if s in nodes)

    def objects_in(self, room: str, label: str | None = None, include_missing=False) -> list[dict]:
        rooms = {n.id.value: n.attributes.name for n in self._layer("ROOMS")}
        out = []
        for n in self._layer("OBJECTS"):
            meta = n.attributes.metadata.get()
            parent = n.get_parent()
            if parent is None or rooms.get(parent) != room:
                continue
            if (label and meta["label"] != label) or (meta["state"] == "missing" and not
                                                       include_missing):  # fmt: skip
                continue
            out.append(self._object(n, meta))
        return out

    def last_seen(self, label: str) -> list[dict]:
        """Every instance of ``label`` with its evidence state, most confident first."""
        out = [self._object(n, n.attributes.metadata.get()) for n in self._layer("OBJECTS")
               if n.attributes.metadata.get()["label"] == label]  # fmt: skip
        return sorted(out, key=lambda o: -o["confidence"])

    @staticmethod
    def _object(n, meta) -> dict:
        return {"id": n.attributes.name, "label": meta["label"], "state": meta["state"],
                "confidence": meta["confidence"], "position": n.attributes.position.tolist(),
                "last_seen": n.attributes.last_update_time_ns / 1e9,
                "place_id": meta["place_id"] or None}  # fmt: skip
