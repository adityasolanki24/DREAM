"""A small scripted home-memory database for developing the interface scene-graph view.

NOT robot data. Every row is written through the real ``HomeMemory`` API so the SQLite
schema and association behaviour are exactly what the runtime produces, but the scenario
itself is authored here: every ``source`` string is prefixed ``demo:`` so nothing in this
database can be mistaken for a recorded run (the interface shows sources verbatim).
Sessions with real data are served by pointing the interface backend at a real
``--memory`` database instead; this file only exists so the graph, details and filters
can be developed against a fully populated memory without simulating a robot.

    python experiments/demo_home_memory.py [--out path] [--now unix-time]

The default path is under the machine's temp directory, next to the other interface
dev stores.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.memory.home import HomeMemory  # noqa: E402

DAY, HOUR = 86400.0, 3600.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, help="where to write the SQLite home memory")
    ap.add_argument("--now", type=float, default=time.time(), help="reference unix time (s)")
    args = ap.parse_args()
    out = args.out or (pathlib.Path(os.environ.get("TEMP", "/tmp")) / "dream-demo" / "home.sqlite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    now = float(args.now)
    mem = HomeMemory(out)
    map_id = "demo-home"

    # Rooms and connectivity, spelled the way the Spark-DSG contract test does; the
    # connectivity is honest stored data, not something the graph infers later.
    kitchen = mem.add_place("kitchen", "room", map_id, (0.0, 0.0))
    hall = mem.add_place("hall", "room", map_id, (3.0, 3.0))
    bedroom = mem.add_place("bedroom", "room", map_id, (6.0, 0.0))
    mem.connect_places(kitchen, hall)
    mem.connect_places(hall, bedroom)
    mem.add_place("counter", "place", map_id, (0.5, 0.8), parent=kitchen)

    # The kitchen cabinet handle: inspected three times, affordance models stored.
    handle = mem.observe(
        "cabinet_handle",
        (0.5, 0.8, 0.9),
        map_id,
        sigma=0.08,
        confidence=0.92,
        source="demo: head_rgbd inspection",
        time=now - 2 * HOUR,
    )
    mem.observe(
        "cabinet_handle",
        (0.49, 0.82, 0.9),
        map_id,
        sigma=0.1,
        confidence=0.9,
        source="demo: head_rgbd inspection",
        time=now - 26 * HOUR,
    )
    mem.observe(
        "cabinet_handle",
        (0.52, 0.79, 0.92),
        map_id,
        sigma=0.12,
        confidence=0.88,
        source="demo: head_rgbd inspection",
        time=now - 3 * DAY,
    )
    mem.record_affordance(
        handle,
        "reveal_close",
        {"use": "grip", "pull_hint": False},
        "succeeded",
        {"evidence": "demo: 3 probes, closed-loop pull, 0.12 m"},
        time=now - 2 * HOUR,
    )
    mem.record_affordance(
        handle,
        "reveal_close",
        {"use": "grip"},
        "failed",
        {"evidence": "demo: grasp slipped before load"},
        time=now - 5 * HOUR,
    )

    # A second remembered handle in the hall, never operated.
    mem.observe(
        "door_handle",
        (3.1, 3.2, 0.95),
        map_id,
        sigma=0.1,
        confidence=0.85,
        source="demo: head_rgbd inspection",
        time=now - 7 * HOUR,
    )

    # The bedroom charger: remembered, seen once yesterday.
    mem.observe(
        "charger",
        (6.2, 0.4, 0.5),
        map_id,
        sigma=0.1,
        confidence=0.8,
        source="demo: head_rgbd",
        time=now - 30 * HOUR,
    )

    # The kit crate in the hall, and a mug the robot then looked for and did not find:
    # the absence is recorded like the runtime records it (MemoryGuidedPlanner).
    crate = mem.observe(
        "crate",
        (2.4, 2.6, 0.2),
        map_id,
        sigma=0.15,
        confidence=0.7,
        source="demo: head_rgbd",
        time=now - 8 * HOUR,
    )
    mem.record_affordance(
        crate,
        "carry",
        {"use": "grip_top"},
        "succeeded",
        {"evidence": "demo: carried 1.4 m and set down"},
        time=now - 8 * HOUR,
    )
    mug = mem.observe(
        "mug",
        (0.2, -0.5, 0.9),
        map_id,
        sigma=0.1,
        confidence=0.75,
        source="demo: head_rgbd",
        time=now - 40 * HOUR,
    )
    mem.record_absence((0.2, -0.5), 0.4, map_id, source="demo: head_rgbd", time=now - 1 * HOUR)
    mem.record_episode(
        "inspect cabinet",
        now - 2 * HOUR,
        now - 2 * HOUR + 95,
        "succeeded",
        {
            "note": "demo: scripted episode attached to the handle sighting",
            "object": handle,
            "mug": mug,
        },
    )
    mem.close()
    print(out)


if __name__ == "__main__":
    main()
