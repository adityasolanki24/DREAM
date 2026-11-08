"""One episode's semantic state for the mobile robot: object tracks, room regions, the target
belief and the scene graph, kept up to date from the robot's own frames.

``MobileToolbox`` owns one (when semantic search is enabled), feeds it every frame, refreshes
regions and the belief on ``observe_scene``, and asks it to verify ``FindObject`` goals.
Per-episode state; what persists is written to dream/memory/home.py by the caller.
"""

from __future__ import annotations

import numpy as np

from dream.perception.occupancy import OccupancyGrid
from dream.perception.semantic import SemanticMap
from dream.semantics.belief import BeliefConfig, TargetBelief
from dream.semantics.graph import SemanticGraph
from dream.semantics.knowledge import SemanticKnowledge
from dream.semantics.rooms import RoomSegmenter, build_regions, object_region
from dream.semantics.verify import VerifierThresholds, check_track, verify_find
from dream.types import RGBDFrame


class SemanticWorld:
    def __init__(
        self,
        grid: OccupancyGrid,
        knowledge: SemanticKnowledge,
        belief: BeliefConfig | None = None,
        thresholds: VerifierThresholds | None = None,
        frame_name: str = "odometry",
    ) -> None:
        self.grid, self.knowledge = grid, knowledge
        self.thresholds = thresholds or VerifierThresholds()
        self.belief_config = belief or BeliefConfig()
        self.map = SemanticMap(grid)
        self.segmenter = RoomSegmenter()
        self.regions: dict = {}
        self.target: TargetBelief | None = None
        self.frame_name = frame_name
        self.last_refresh = -1

    def set_target(self, target: str, memory_hints=()) -> None:
        if self.target is None or self.target.target != target:
            self.target = TargetBelief(
                target, self.knowledge, self.belief_config, list(memory_hints)
            )

    def observe(self, frame: RGBDFrame) -> None:
        self.map.update(frame)

    def refresh(self) -> None:
        """Recompute regions, room weights and the target belief from the current map."""
        self.segmenter.update(self.grid)
        self.regions = build_regions(self.grid, self.segmenter, self.map.tracks, self.knowledge)
        if self.target is not None:
            self.target.update(self.grid, self.map, self.segmenter, self.regions)
        self.last_refresh = self.map.frames

    def verify(self, target: str, outcome: str, pose, now: float) -> dict:
        return verify_find(self.map, target, outcome, np.asarray(pose, float), now, self.thresholds)

    # ----- reporting -------------------------------------------------------------------------
    def summary(self, limit: int = 12) -> dict:
        tracks = sorted(self.map.tracks, key=lambda t: -t.views)[:limit]
        out = {
            "objects": [
                {**t.summary(), "region": object_region(self.grid, self.segmenter, t)}
                for t in tracks
            ],
            "regions": [r.summary() for r in self.regions.values()],
            "knowledge_source": self.knowledge.source,
        }
        if self.target is not None:
            mass = self.target.region_mass(self.segmenter)
            out["search"] = {
                "target": self.target.target,
                "typical_rooms_knowledge": self.knowledge.typical_rooms(self.target.target),
                "mass_share_by_region": {
                    ("unexplored" if k < 0 else f"region:{k}"): round(v, 3)
                    for k, v in sorted(mass.items(), key=lambda kv: -kv[1])[:6]
                },
                "note": "search hypotheses under an assumed detector model; not observations",
            }
        return out

    def graph(self, now: float, memory_hints=()) -> SemanticGraph:
        g = SemanticGraph(self.frame_name)
        regions = self.regions
        for rid, reg in regions.items():
            g.node(f"region:{rid}", "region", area_m2=round(reg.area, 2),
                   centroid=np.round(reg.centroid, 2).tolist(), frame=self.frame_name)  # fmt: skip
        for t in self.map.tracks:
            oid = f"object:{t.id}"
            g.node(oid, "object", labels=dict(t.labels), position=np.round(t.position, 3).tolist(),
                   views=t.views, misses=t.misses, frame=self.frame_name)  # fmt: skip
            rid = object_region(self.grid, self.segmenter, t)
            parent = f"region:{rid}" if rid in regions else g.node("unexplored", "unexplored")
            g.edge(parent, "contains", oid, "observed", "perception:semantic_tracker",
                   evidence=(f"views:{t.views}",), time=t.last_seen,
                   score=float(t.views), score_kind="view_count",
                   frame=self.frame_name)  # fmt: skip
            if self.target is not None:
                report = check_track(t, self.target.target, self.thresholds)
                if report["verified"]:
                    cls = g.node(f"class:{self.target.target}", "class")
                    g.edge(oid, "identified_as", cls, "verified", "verifier:multi_view",
                           evidence=tuple(f"{k}:{v}" for k, v in report["checks"].items()),
                           time=t.last_seen)  # fmt: skip
        for rid, reg in regions.items():
            cat, w, margin = reg.leading()
            cnode = g.node(f"category:{cat}", "category")
            g.edge(f"region:{rid}", "classified_as", cnode, "inferred", "inference:room_weights",
                   evidence=tuple(sorted(reg.evidence)), time=now, score=round(w, 3),
                   score_kind="heuristic_weight")  # fmt: skip
        if self.target is not None and self.target.mass is not None:
            cls = g.node(f"class:{self.target.target}", "class")
            for rid, share in self.target.region_mass(self.segmenter).items():
                src = f"region:{rid}" if rid in regions else g.node("unexplored", "unexplored")
                g.edge(src, "may_contain", cls, "hypothesis", "belief:target_search",
                       time=now, score=round(share, 4),
                       score_kind="share_of_search_mass")  # fmt: skip
            for cat, a in self.knowledge.typical_rooms(self.target.target):
                cnode = g.node(f"category:{cat}", "category")
                g.edge(cls, "typically_in", cnode, "knowledge", self.knowledge.source,
                       score=a, score_kind="heuristic_association")  # fmt: skip
            for hint in memory_hints:
                mid = g.node(f"memory:{hint.get('id', 'unknown')}", "memory",
                             position=list(hint["xy"]), frame=self.frame_name)  # fmt: skip
                g.edge(mid, "remembered_as", cls, "remembered", "memory:home",
                       time=hint.get("last_seen"), score=hint.get("confidence"),
                       score_kind="memory_confidence")  # fmt: skip
        return g
