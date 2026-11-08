"""The robot's semantic scene graph: regions, observed objects, room categories and object
classes, with every relation labelled by what kind of claim it is.

Edge ``status`` keeps the kinds of statement structurally apart:

- ``observed``: an object track exists in a region (perception; evidence = track views);
- ``verified``: an object's identity passed the verifier (dream/semantics/verify.py);
- ``inferred``: a region's heuristic room-category weight (dream/semantics/rooms.py);
- ``hypothesis``: a region (or unexplored space) may contain the searched class, scored by
  its share of the current search mass (dream/semantics/belief.py);
- ``knowledge``: a class is typically found in a room category (the ConceptNet extract);
- ``remembered``: persistent memory placed an instance there earlier (dream/memory/home.py).

Hypotheses point at class nodes, never at object nodes: an object node exists only for a
physical track the robot observed (``validate`` enforces it). Physical positions are in the
named ``frame`` (the robot's odometry frame of one episode); no display/layout coordinates are
stored here. The planner reads region rankings and candidates from this graph, the frontend
draws it, and it is saved with each episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATUSES = ("observed", "verified", "inferred", "hypothesis", "knowledge", "remembered")
NODE_KINDS = ("region", "object", "category", "class", "unexplored", "memory")


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    status: str
    provenance: str
    evidence: tuple = ()
    time: float | None = None
    score: float | None = None
    score_kind: str | None = None  # what ``score`` is: never presented as a probability
    frame: str | None = None


class SemanticGraph:
    def __init__(self, frame: str) -> None:
        self.frame = frame
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def node(self, id: str, kind: str, **attrs) -> str:
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {kind!r}")
        if id not in self.nodes:
            self.nodes[id] = Node(id, kind, attrs)
        return id

    def edge(self, source: str, relation: str, target: str, status: str, provenance: str,
             **kw) -> None:  # fmt: skip
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("edge endpoints must be nodes")
        self.edges.append(Edge(source, relation, target, status, provenance, **kw))

    # ----- invariants ----------------------------------------------------------------------
    def validate(self) -> list[str]:
        problems = []
        observed_objects = {e.target for e in self.edges if e.status == "observed"}
        for n in self.nodes.values():
            if n.kind == "object" and n.id not in observed_objects:
                problems.append(f"{n.id} has no observation")
        for e in self.edges:
            if e.status == "hypothesis" and self.nodes[e.target].kind == "object":
                problems.append(f"hypothesis {e.source}->{e.target} points at an object")
            if e.status in ("inferred", "hypothesis") and e.score_kind == "probability":
                problems.append(f"{e.relation} presents a heuristic score as a probability")
        return problems

    # ----- queries used for decisions -------------------------------------------------------
    def facts(self) -> list[Edge]:
        return [e for e in self.edges if e.status in ("observed", "verified")]

    def hypotheses(self, cls: str) -> list[tuple[str, float]]:
        """(region or unexplored node, share of search mass), largest first."""
        out = [(e.source, e.score or 0.0) for e in self.edges
               if e.status == "hypothesis" and e.target == f"class:{cls}"]  # fmt: skip
        return sorted(out, key=lambda x: -x[1])

    def objects_labelled(self, cls: str) -> list[str]:
        return sorted(
            n.id for n in self.nodes.values()
            if n.kind == "object" and cls in n.attrs.get("labels", {})
        )  # fmt: skip

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
            "statuses": list(STATUSES),
        }
