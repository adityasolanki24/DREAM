"""Render top-down maps of sampled homes and summarise the scenario distributions.

Ground truth for inspection only: rooms coloured by type, objects labelled, start pose,
doorways. Writes one PNG per (generator, scenario, seed) and ``summary.json`` with room-type
counts, object counts per class and the target-room distribution per scenario.

    python experiments/render_homes.py --seeds 3 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

import dream  # noqa: E402,F401  (selects the offscreen GL backend before MuJoCo)
from dream.sim.home import (  # noqa: E402
    SCENARIOS,
    HomeGenerationError,
    HomeSpec,
    footprint,
    instances_of,
    room_at,
    sample_home,
)

COLOURS = {
    "kitchen": "#f6d7a7", "bedroom": "#c9d9f2", "bathroom": "#bfe6e3", "living_room": "#d9ecc4",
    "office": "#e6d4ee", "garage": "#d3d3d3", "laundry": "#f2c9d5", "storage": "#e2dcc4",
    "studio": "#f2e3b3",
}  # fmt: skip


def draw(home: HomeSpec, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9 * home.depth / home.width))
    for r in home.rooms:
        ax.add_patch(
            Rectangle(r.lo, r.hi[0] - r.lo[0], r.hi[1] - r.lo[1], fc=COLOURS[r.type], ec="none")
        )
        ax.text(
            (r.lo[0] + r.hi[0]) / 2,
            r.hi[1] - 0.2,
            f"{r.id}:{r.type}",
            ha="center",
            va="top",
            fontsize=8,
        )
    for w in home.walls:
        ax.add_patch(
            Rectangle(
                (w.centre[0] - w.half[0], w.centre[1] - w.half[1]),
                2 * w.half[0],
                2 * w.half[1],
                fc="k",
            )
        )
    for d in home.doorways:
        ax.plot(*d.centre, "s", color="white", ms=4, mec="g")
    for o in home.objects:
        hit = o.cls == home.target_class
        ax.add_patch(
            Polygon(
                footprint(o).corners(),
                fc="#ff8a80" if hit else "#fafafa",
                ec="r" if hit else "#444",
                lw=1.5 if hit else 0.7,
                alpha=0.9,
            )
        )
        ax.text(*o.xy, o.cls.replace("_", "\n"), ha="center", va="center", fontsize=5)
    sx, sy, yaw = home.start
    ax.arrow(sx, sy, 0.8 * np.cos(yaw), 0.8 * np.sin(yaw), head_width=0.25, color="b", zorder=5)
    ax.plot(sx, sy, "bo", zorder=5)
    for cls, xyz, age in home.memory_prior:
        ax.plot(xyz[0], xyz[1], "mX", ms=10, zorder=5)
        ax.text(xyz[0], xyz[1] + 0.3, f"memory {cls} {age / 86400:.1f} d", color="m", fontsize=6)
    ax.set_xlim(-home.width / 2 - 0.3, home.width / 2 + 0.3)
    ax.set_ylim(-home.depth / 2 - 0.3, home.depth / 2 + 0.3)
    ax.set_aspect("equal")
    ax.set_title(f"{home.scenario} {home.generator} seed {home.seed}: {home.request}", fontsize=9)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(Path(os.environ.get("TEMP", tempfile.gettempdir())) / "dream_homes")
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--target", default="refrigerator")
    ap.add_argument("--scenarios", default="".join(SCENARIOS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for generator in ("nominal", "shifted"):
        for sc in args.scenarios:
            rooms, objs, where, failures = Counter(), Counter(), Counter(), 0
            for seed in range(args.seeds):
                try:
                    home = sample_home(np.random.default_rng(seed), sc, args.target, generator)
                except HomeGenerationError:
                    failures += 1
                    continue
                draw(home, out / f"{generator}_{home.scenario}_{seed}.png")
                rooms.update(r.type for r in home.rooms)
                objs.update(o.cls for o in home.objects)
                found = instances_of(home, args.target)
                where[room_at(home, found[0].xy).type if found else "none"] += 1
            summary[f"{generator}/{sc}"] = {
                "room_types": dict(rooms), "objects": dict(objs), "target_room": dict(where),
                "failures": failures,
            }  # fmt: skip
            print(generator, sc, dict(where), "failures", failures, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
