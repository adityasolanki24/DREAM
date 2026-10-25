"""Reuse what physical discovery learned about an object, and stop trusting it when it fails.

``remember_investigation`` stores a successful ``Investigation`` (including its particle
belief, which opening along the learned constraint needs) as a checksummed artifact, and
records it in ``HomeMemory`` as the object's "open" affordance at the handle's observed
position. ``ExperienceToolbox`` is the Panda ``Toolbox`` with one addition: before the task,
``recall`` looks for a remembered procedure for the handle it now observes (same map, within
``RECALL_RADIUS``); if found, the investigation is supplied from experience and the planner
proceeds straight to opening, with no probes.

Remembered is not assumed true. While opening, the existing closed-loop check still compares
the handle with the remembered constraint; if the opening fails, leaves the constraint, or
makes no progress, the procedure is withdrawn in memory (a failure record) and dropped from
the session, so the planner investigates the object afresh. Nothing reads simulator state.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from dream.memory.home import HomeMemory
from dream.tools.toolbox import Toolbox, ToolResult

RECALL_RADIUS = 0.02  # m: handle re-observed this close to where it was (tracker ~2 mm noise)
# Progress that counts as "the remembered mechanism moved": a third of the smallest opening
# segment (dream/tools/discovery.SEGMENT: 0.04 m, 0.25 rad).
MIN_PROGRESS = {"prismatic": 0.013, "revolute": 0.08}
LABEL = "{}_handle"


def remember_investigation(memory: HomeMemory, artifacts: Path, name: str, inv, handle_xyz,
                           map_id: str, time: float, outcome: dict) -> str:  # fmt: skip
    """Store ``inv`` after it opened ``name`` successfully; returns the object identity."""
    artifacts.mkdir(parents=True, exist_ok=True)
    oid = memory.observe(LABEL.format(name), handle_xyz, map_id, sigma=0.005,
                         source="handle tracker", time=time)  # fmt: skip
    blob = pickle.dumps(inv)
    digest = hashlib.sha256(blob).hexdigest()
    path = artifacts / f"{oid}_{digest[:12]}.pkl"
    path.write_bytes(blob)
    memory.record_affordance(
        oid, "open",
        {"artifact": path.name, "sha256": digest, "mechanism": inv.mechanism,
         "constraint": inv.constraint, "interactions": inv.interactions},
        "succeeded", evidence=outcome, time=time,
    )  # fmt: skip
    return oid


class ExperienceToolbox(Toolbox):
    """``Toolbox`` that can start from remembered investigations and withdraws them on failure."""

    def __init__(self, *args, memory: HomeMemory, artifacts: Path, map_id: str, clock, **kw):
        super().__init__(*args, **kw)
        self.home, self.artifacts, self.map_id, self.clock = memory, artifacts, map_id, clock
        self.recalled: dict[str, str] = {}  # object name -> memory identity
        self.experience_log: list[dict] = []

    def recall(self, name: str) -> bool:
        """Supply the investigation of ``name`` from memory if its handle is where it was."""
        from dream.skills import articulation as art

        obs = art.find_articulated(self.perception, name)
        if obs is None:
            return False
        handle = np.asarray(obs.handle_pos, float)
        for b in self.home.where_is(LABEL.format(name), self.map_id, now=self.clock()):
            if b.state == "missing" or np.linalg.norm(b.position - handle) > RECALL_RADIUS:
                continue
            proc = self.home.known_procedure(b.object_id, "open")
            if proc is None:
                continue
            blob = (self.artifacts / proc["model"]["artifact"]).read_bytes()
            if hashlib.sha256(blob).hexdigest() != proc["model"]["sha256"]:
                self.experience_log.append({"event": "artifact_checksum_mismatch"})
                continue
            inv = pickle.loads(blob)  # our own artifact, checksum verified
            self.investigations[name] = inv
            self.recalled[name] = b.object_id
            gap = float(np.linalg.norm(b.position - handle))
            self.experience_log.append({"event": "recalled", "object": b.object_id,
                                        "mechanism": inv.mechanism, "distance_m": gap})  # fmt: skip
            return True
        return False

    def call(self, name: str, args: dict | None = None) -> ToolResult:
        result = super().call(name, args)
        target = (args or {}).get("object")
        if name == "open_articulated" and target in self.recalled:
            self._check_recalled(target, result)
        return result

    def _check_recalled(self, target: str, result: ToolResult) -> None:
        inv = self.investigations.get(target)
        kind = getattr(inv, "mechanism", None)
        reached = float(result.data.get("q_reached", 0.0))
        progressed = kind in MIN_PROGRESS and reached >= MIN_PROGRESS[kind]
        if result.ok and progressed:
            self.experience_log.append({"event": "recalled_procedure_worked", "q": reached})
            return
        oid = self.recalled.pop(target)
        self.home.record_affordance(oid, "open", {"mechanism": kind}, "failed",
                                    evidence={"tool_status": result.status,
                                              "message": result.message, "q_reached": reached},
                                    time=self.clock())  # fmt: skip
        self.investigations.pop(target, None)  # the planner will investigate afresh
        self.experience_log.append({"event": "recalled_procedure_withdrawn", "q": reached,
                                    "message": result.message})  # fmt: skip
