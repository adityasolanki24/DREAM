"""Persistent home memory: object identities, time-indexed observations, places, affordances
and task episodes in one SQLite file that survives restarts.

The canonical object store that scene-graph and temporal-map backends (Hydra, ConceptGraphs,
HOV-SG, SuperMap) will feed or be compared against; it holds *evidence*, never ground truth.

What it keeps apart, because acting on the wrong one is a real failure:
- ``observed``: seen within ``fresh_s`` of the query time;
- ``remembered``: last seen earlier, nothing contradicts it, confidence decays with age;
- ``missing``: the robot looked where it was last seen and did not see it;
- ``moved``: the same identity was later observed elsewhere (its old place is history).
A new sighting of the same label far from a known instance becomes a *new* instance unless
the known one is missing and appearance descriptors (when given) agree: a second bottle is
not silently merged into the first.

Positions are in a named persistent map frame (``map_id``). The robot's per-session odometry
frame is not a map frame: until a localisation system aligns sessions, callers must not
store odometry coordinates under a shared map id. Times are seconds on one declared clock.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time as _time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
GATE_MIN = 0.3  # m: smallest association gate (detector position noise ~0.1 m, 3 sigma)
FRESH_S = 30.0  # s: an observation this recent describes the present
HALF_LIFE_S = 6 * 3600.0  # s: confidence half-life of an unconfirmed last-known location
SAME_APPEARANCE = 0.85  # cosine similarity for "the same object", with descriptors

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS places (id TEXT PRIMARY KEY, name TEXT, kind TEXT, map_id TEXT,
    x REAL, y REAL, parent TEXT, created REAL);
CREATE TABLE IF NOT EXISTS place_edges (a TEXT, b TEXT, kind TEXT, PRIMARY KEY (a, b, kind));
CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, label TEXT, descriptor TEXT,
    created REAL, map_id TEXT);
CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id TEXT, time REAL, map_id TEXT, x REAL, y REAL, z REAL, sigma REAL,
    confidence REAL, source TEXT, place_id TEXT);
CREATE TABLE IF NOT EXISTS absences (id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL,
    map_id TEXT, x REAL, y REAL, radius REAL, source TEXT);
CREATE TABLE IF NOT EXISTS affordances (id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT,
    skill TEXT, model TEXT, outcome TEXT, time REAL, evidence TEXT);
CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, goal TEXT, started REAL,
    ended REAL, outcome TEXT, record TEXT);
CREATE INDEX IF NOT EXISTS obs_object ON observations (object_id, time);
"""


@dataclass(frozen=True)
class Belief:
    """Where an object is believed to be, and how far that belief can be trusted."""

    object_id: str
    label: str
    position: np.ndarray  # map frame
    map_id: str
    last_seen: float
    age_s: float
    state: str  # observed | remembered | missing
    confidence: float  # detector confidence x age decay; 0 when missing
    place_id: str | None
    sightings: int


class HomeMemory:
    def __init__(self, path: str | Path, clock=_time.time) -> None:
        self.path, self.clock = str(path), clock
        self.db = sqlite3.connect(self.path)
        self.db.executescript(_SCHEMA)
        row = self.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row is None:
            self.db.execute("INSERT INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
            self.db.commit()
        elif int(row[0]) != SCHEMA_VERSION:
            raise ValueError(f"memory schema {row[0]} is not {SCHEMA_VERSION}")

    def close(self) -> None:
        self.db.close()

    # -- places -------------------------------------------------------------------------------
    def add_place(self, name: str, kind: str, map_id: str, xy, parent: str | None = None) -> str:
        pid = f"{kind}:{name}"
        self.db.execute(
            "INSERT OR REPLACE INTO places VALUES (?,?,?,?,?,?,?,?)",
            (pid, name, kind, map_id, float(xy[0]), float(xy[1]), parent, self.clock()),
        )
        self.db.commit()
        return pid

    def connect_places(self, a: str, b: str) -> None:
        for u, v in ((a, b), (b, a)):
            self.db.execute("INSERT OR IGNORE INTO place_edges VALUES (?,?,'connects')", (u, v))
        self.db.commit()

    def neighbours(self, place: str) -> list[str]:
        rows = self.db.execute("SELECT b FROM place_edges WHERE a=? ORDER BY b", (place,))
        return [r[0] for r in rows]

    def place_of(self, map_id: str, xy, kind: str = "room") -> str | None:
        """The nearest registered place of ``kind`` (rooms are given by their centres)."""
        rows = self.db.execute(
            "SELECT id, x, y FROM places WHERE map_id=? AND kind=?", (map_id, kind)
        ).fetchall()
        if not rows:
            return None
        return min(rows, key=lambda r: math.hypot(r[1] - xy[0], r[2] - xy[1]))[0]

    # -- observations -------------------------------------------------------------------------
    def observe(
        self,
        label: str,
        position,
        map_id: str,
        *,
        sigma: float = 0.1,
        confidence: float = 1.0,
        source: str = "",
        descriptor=None,
        time: float | None = None,
    ) -> str:
        """Record a detection; returns the object identity it was associated with."""
        t = self.clock() if time is None else float(time)
        p = np.asarray(position, float)
        if p.shape != (3,) or not np.isfinite(p).all() or not sigma > 0:
            raise ValueError("observation needs a finite 3D position and positive sigma")
        oid = self._associate(label, p, map_id, sigma, descriptor, t)
        if oid is None:
            oid = f"{label}-{uuid.uuid4().hex[:8]}"
            desc = None if descriptor is None else json.dumps(np.asarray(descriptor).tolist())
            self.db.execute("INSERT INTO objects VALUES (?,?,?,?,?)", (oid, label, desc, t, map_id))
        self.db.execute(
            "INSERT INTO observations (object_id, time, map_id, x, y, z, sigma, confidence,"
            " source, place_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (oid, t, map_id, *map(float, p), sigma, confidence, source,
             self.place_of(map_id, p[:2])),
        )  # fmt: skip
        self.db.commit()
        return oid

    def _last(self, oid: str):
        return self.db.execute(
            "SELECT time, x, y, z, sigma, confidence, place_id, map_id FROM observations"
            " WHERE object_id=? ORDER BY time DESC LIMIT 1",
            (oid,),
        ).fetchone()

    def _associate(self, label, p, map_id, sigma, descriptor, t) -> str | None:
        same = [r[0] for r in self.db.execute(
            "SELECT id FROM objects WHERE label=? AND map_id=?", (label, map_id))]  # fmt: skip
        best, best_d = None, math.inf
        for oid in same:
            last = self._last(oid)
            d = float(np.linalg.norm(np.array(last[1:4]) - p))
            gate = max(GATE_MIN, 3 * math.hypot(last[4], sigma))
            if d <= gate and d < best_d and self._appearance_ok(oid, descriptor, None):
                best, best_d = oid, d
        if best is not None:
            return best
        # Far from every known instance: the same object only if a known one is missing
        # (looked for and not seen) and the appearance, when measured, agrees.
        missing = [o for o in same if self._state(o, t) == "missing"]
        if len(missing) == 1 and self._appearance_ok(missing[0], descriptor, SAME_APPEARANCE):
            return missing[0]
        return None

    def _appearance_ok(self, oid: str, descriptor, threshold) -> bool:
        row = self.db.execute("SELECT descriptor FROM objects WHERE id=?", (oid,)).fetchone()
        if descriptor is None or row[0] is None:
            return threshold is None  # without appearance, only proximity can associate
        a, b = np.asarray(json.loads(row[0]), float), np.asarray(descriptor, float)
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        return cos >= (threshold if threshold is not None else 0.5)

    def record_absence(self, center, radius: float, map_id: str, *, source: str = "",
                       time: float | None = None) -> list[str]:  # fmt: skip
        """The robot looked at this region and saw none of what was remembered there.

        ``radius`` must describe space actually inside the sensor's view, unoccluded; the
        caller (perception) owns that judgement. Returns identities now considered missing."""
        t = self.clock() if time is None else float(time)
        c = np.asarray(center, float)[:2]
        self.db.execute(
            "INSERT INTO absences (time, map_id, x, y, radius, source) VALUES (?,?,?,?,?,?)",
            (t, map_id, float(c[0]), float(c[1]), float(radius), source),
        )
        self.db.commit()
        ids = [r[0] for r in self.db.execute("SELECT id FROM objects WHERE map_id=?", (map_id,))]
        return [o for o in ids if self._state(o, t) == "missing"]

    def _state(self, oid: str, now: float) -> str:
        last = self._last(oid)
        t_last, x, y = last[0], last[1], last[2]
        hit = self.db.execute(
            "SELECT COUNT(*) FROM absences WHERE map_id=? AND time>? AND time<=?"
            " AND (x-?)*(x-?)+(y-?)*(y-?) <= radius*radius",
            (last[7], t_last, now, x, x, y, y),
        ).fetchone()[0]
        if hit:
            return "missing"
        return "observed" if now - t_last <= FRESH_S else "remembered"

    # -- queries ------------------------------------------------------------------------------
    def where_is(self, label: str, map_id: str, now: float | None = None) -> list[Belief]:
        """Every known instance of ``label``, most trustworthy first."""
        t = self.clock() if now is None else float(now)
        out = []
        for oid, lab in self.db.execute(
            "SELECT id, label FROM objects WHERE label=? AND map_id=?", (label, map_id)
        ).fetchall():
            last = self._last(oid)
            n = self.db.execute(
                "SELECT COUNT(*) FROM observations WHERE object_id=?", (oid,)
            ).fetchone()[0]
            state = self._state(oid, t)
            age = t - last[0]
            conf = 0.0 if state == "missing" else last[5] * 0.5 ** (max(age, 0) / HALF_LIFE_S)
            out.append(Belief(oid, lab, np.array(last[1:4]), last[7], last[0], age, state,
                              conf, last[6], n))  # fmt: skip
        return sorted(out, key=lambda b: -b.confidence)

    def history(self, oid: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT time, x, y, z, sigma, confidence, source, place_id FROM observations"
            " WHERE object_id=? ORDER BY time",
            (oid,),
        ).fetchall()
        keys = ("time", "x", "y", "z", "sigma", "confidence", "source", "place_id")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def objects_in(self, place: str, now: float | None = None) -> list[Belief]:
        t = self.clock() if now is None else float(now)
        labels = self.db.execute("SELECT DISTINCT label, map_id FROM objects").fetchall()
        return [b for lab, m in labels for b in self.where_is(lab, m, t)
                if b.place_id == place and b.state != "missing"]  # fmt: skip

    # -- affordances and episodes ---------------------------------------------------------------
    def record_affordance(
        self,
        oid: str,
        skill: str,
        model: dict,
        outcome: str,
        evidence: dict | None = None,
        time: float | None = None,
    ) -> None:
        if outcome not in ("succeeded", "failed", "unsafe"):
            raise ValueError("affordance outcome must be succeeded, failed or unsafe")
        self.db.execute(
            "INSERT INTO affordances (object_id, skill, model, outcome, time, evidence)"
            " VALUES (?,?,?,?,?,?)",
            (oid, skill, json.dumps(model), outcome, self.clock() if time is None else time,
             json.dumps(evidence or {})),
        )  # fmt: skip
        self.db.commit()

    def known_procedure(self, oid: str, skill: str) -> dict | None:
        """The model of the most recent successful use, unless a later attempt failed."""
        row = self.db.execute(
            "SELECT model, outcome, time FROM affordances WHERE object_id=? AND skill=?"
            " ORDER BY time DESC LIMIT 1",
            (oid, skill),
        ).fetchone()
        if row is None or row[1] != "succeeded":
            return None
        return {"model": json.loads(row[0]), "time": row[2]}

    def record_episode(self, goal: str, started: float, ended: float, outcome: str,
                       record: dict) -> str:  # fmt: skip
        eid = uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?)",
            (eid, goal, started, ended, outcome, json.dumps(record, default=str)),
        )
        self.db.commit()
        return eid

    def episodes(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, goal, started, ended, outcome, record FROM episodes ORDER BY started"
        ).fetchall()
        return [{"id": r[0], "goal": r[1], "started": r[2], "ended": r[3], "outcome": r[4],
                 "record": json.loads(r[5])} for r in rows]  # fmt: skip
