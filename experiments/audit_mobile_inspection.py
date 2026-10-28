"""Replay saved mobile observations and audit estimated inspection reachability on CPU.

The categorical occupancy snapshot preserves the planner's free/occupied/unknown tests.
It cannot establish physical reachability in Isaac or recover missing original RGB-D.
No simulator truth, candidate IDs or scenario answers enter decision-making code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dream.navigation.planning import path_to
from dream.perception.occupancy import CELL
from dream.perception.spatial import Candidate, SpatialMemory
from dream.runtime.agent import Step, ToolCall
from dream.runtime.mobile_model import MobileModelPlanner
from dream.tools.goals import Inspect
from dream.tools.mobile import MobileToolbox
from dream.tools.toolbox import ToolResult


def load_saved(path):
    saved = json.loads(Path(path).read_text())
    scene = saved["decision"]["observation"]
    memory = SpatialMemory(np.random.default_rng(0))
    grid = memory.grid
    cropped = saved["map"]
    assert cropped["resolution_m"] == CELL and cropped["axes"] == "cells[x][y]"
    lo = np.rint((np.asarray(cropped["origin_m"]) + grid.size / 2) / CELL).astype(int)
    cells = np.asarray(cropped["cells"])
    grid.logodds[lo[0] : lo[0] + cells.shape[0], lo[1] : lo[1] + cells.shape[1]] = np.select(
        [cells == 1, cells == 2], [-2.0, 2.0], default=0.0
    )
    for c in scene["candidates"]:
        memory.candidates.append(
            Candidate(
                c["id"],
                c["color"],
                np.array(c["position"]),
                0.0,
                np.array(c["extent"]),
                views=c["views"],
                handle=None if c["handle"] is None else np.array(c["handle"]),
                handle_views=c["handle_views"],
                sides_close=set(c["sides_seen_close"]),
            )
        )
    pose = np.array(scene["pose"], float)
    base = SimpleNamespace(
        odometry=lambda: pose.copy(), stop_reason=None, time=saved["robot_time_s"]
    )
    toolbox = MobileToolbox(base, memory, navigator="rpp")
    toolbox.inspections = {int(k): v for k, v in scene["inspected"].items()}
    history = [
        Step(ToolCall(s["tool"], s["args"]), ToolResult(s["tool"], s["status"], s["message"]), 0)
        for s in saved["decision"]["history"]
    ]
    return saved, toolbox, pose, history


def audit(path):
    saved, toolbox, pose, history = load_saved(path)

    class Replay:
        def generate(self, messages, tools):
            return saved["model_output"]

    planner = MobileModelPlanner(Replay(), saved["decision"]["instruction"], lambda: None)
    result = planner.next_action(
        Inspect("cabinet"), saved["decision"]["observation"], saved["decision"]["verdict"], history
    )
    passable, start, dist = toolbox._passable()
    grid = toolbox.memory.grid
    records = []
    # Audit every detected handle, without selecting targets by their saved IDs.
    for c in toolbox.memory.candidates:
        if c.handle is None:
            continue
        options = []
        for v in saved["decision"]["observation"]["viewpoints"]:
            if v["candidate"] != c.id:
                continue
            pose[:] = [*v["xy"], v["heading"]]
            valid, why = toolbox.at_inspection_pose(c)
            cell = tuple(int(k) for k in grid.cell(v["xy"]))
            route = path_to(dist, cell) if grid.inside(*cell) else []
            options.append(
                {
                    **v,
                    "valid_inspection_pose": valid,
                    "pose_reason": why,
                    "reachable_on_saved_map": bool(route) or cell == start,
                    "path": [grid.centre(*p).tolist() for p in route],
                }
            )
        # Exhaustive cell audit distinguishes a missing sampled viewpoint from no
        # reachable inspection pose anywhere in the represented observed free space.
        valid_cells = []
        for cell in np.argwhere(passable & np.isfinite(dist)):
            xy = grid.centre(*cell)
            delta = c.handle[:2] - xy
            pose[:] = [*xy, float(np.arctan2(delta[1], delta[0]))]
            if toolbox.at_inspection_pose(c)[0]:
                valid_cells.append((float(dist[tuple(cell)]), cell))
        records.append(
            {
                "candidate": c.id,
                "handle": c.handle.tolist(),
                "handle_side": c.handle_side(),
                "offered_viewpoints": options,
                "reachable_inspection_cells": len(valid_cells),
                "best_reachable_inspection_xy": None
                if not valid_cells
                else grid.centre(*min(valid_cells, key=lambda v: v[0])[1]).tolist(),
            }
        )
    return {
        "source_git_sha": saved["source_git_sha"],
        "replayed_action": repr(result),
        "scope": "saved estimated map and geometry only; physical reachability unverified",
        "candidates": records,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(audit(args.saved), indent=2) + "\n")


if __name__ == "__main__":
    main()
