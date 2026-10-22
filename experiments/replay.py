"""Render a saved live session (results/sessions/<time>/trajectory.json) to a GIF.

    python experiments/replay.py results/sessions/<time> [--fps 10]

The replay sets logged joint and object states (qpos) and renders them. It shows exactly what
happened in the physics simulation, at the 10 Hz logging rate.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.sim.tabletop import build_model, sample_tabletop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--every", type=int, default=2, help="use every n-th logged state")
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args()
    d = pathlib.Path(args.session_dir)
    log = json.loads((d / "trajectory.json").read_text())
    model = build_model(sample_tabletop(np.random.default_rng(log["seed"])))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.width * 3 // 4, args.width)
    frames = []
    for state in log["states"][:: args.every]:
        data.qpos[:] = state["qpos"]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="overview")
        frames.append(Image.fromarray(renderer.render()))
    frames[0].save(
        d / "replay.gif", save_all=True, append_images=frames[1:], duration=1000 // args.fps, loop=0
    )
    print(f"wrote {d / 'replay.gif'} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
