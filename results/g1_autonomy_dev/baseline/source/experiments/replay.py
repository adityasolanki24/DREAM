"""Render a saved live session (results/sessions/<time>/trajectory.json) to a GIF.

    python experiments/replay.py results/sessions/<time> [--fps 10]

The replay sets logged joint and object states (qpos) and renders them. It shows exactly what
happened in the physics simulation, at the 10 Hz logging rate. A text panel shows what the
session displayed at that instant: goal, current tool call, last result and, during
investigate_physics, the mechanism probabilities, the latest probe, its predicted and observed
handle motion, and the model-adequacy check.
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
from PIL import Image, ImageDraw  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.session import make_scene  # noqa: E402
from dream.sim.camera import CAMERA_NAME, CameraSpec  # noqa: E402
from dream.sim.tabletop import build_model  # noqa: E402

PANEL_WIDTH = 330


def panel_lines(overlay: dict | None, t: float) -> list[str]:
    if not overlay:
        return [f"t = {t:.1f} s"]
    lines = [
        f"t = {t:.1f} s   state: {overlay.get('state')}",
        f"goal: {overlay.get('goal')}",
        f"tool: {overlay.get('action') or '-'}",
        f"last: {overlay.get('last_result') or '-'}",
    ]
    d = overlay.get("discovery")
    if d:
        p = d["type_probs"]
        lines += [
            "",
            "mechanism hypotheses:",
            *(f"  {k:9s} {v:5.2f} " + "#" * int(round(20 * v)) for k, v in p.items()),
            f"probe {d['probe']}: dir {d['direction']}  cap {d['max_force_n']:.0f} N",
            f"  predicted {d['predicted_mm']:.1f} mm  observed {d['observed_mm']:.1f} mm",
            f"  prediction error {d['prediction_error_mm']:.1f} mm",
            f"  chi2/dof {d['chi2_per_dof']:.2f}  adequate: {d['model_adequate']}",
        ]
    if overlay.get("message"):
        lines += ["", f"result: {overlay['message']}"]
    return lines


def _camera_view(renderer, data, overlay: dict | None, size: tuple[int, int]) -> Image.Image:
    """The RGB-D camera's view with the logged detections of the latest perception frame:
    boxes, world-model labels, position and 1-sigma uncertainty. (The image is re-rendered from
    the logged state; the boxes are what perception reported.)"""
    renderer.update_scene(data, camera=CAMERA_NAME)
    img = Image.fromarray(renderer.render())
    p = (overlay or {}).get("perception")
    draw = ImageDraw.Draw(img)
    if p:
        for d in p["detections"]:
            colour = (255, 40, 40) if d["occluded"] else (40, 220, 40)
            draw.rectangle(d["bbox"], outline=colour)
            x, y = d["bbox"][0], max(0, d["bbox"][1] - 11)
            draw.text((x, y), f"{d['label']} +-{d['std_mm']:.0f}mm", fill=colour)
        draw.text((4, 4), f"camera frame t={p['frame_time']:.1f}s", fill=(255, 255, 255))
        if p["hidden"]:
            draw.text((4, 16), f"not visible: {', '.join(p['hidden'])}", fill=(255, 200, 0))
    return img.resize(size)


def _wrap(line: str, width: int = 48) -> list[str]:
    return [line[i : i + width] for i in range(0, max(len(line), 1), width)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--every", type=int, default=2, help="use every n-th logged state")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--camera", default="overview")
    args = ap.parse_args()
    d = pathlib.Path(args.session_dir)
    log = json.loads((d / "trajectory.json").read_text())
    model = build_model(make_scene(log.get("scene", "tray"), log["seed"]))
    data = mujoco.MjData(model)
    h = args.width * 3 // 4
    renderer = mujoco.Renderer(model, h, args.width)
    vision = log.get("perception") == "vision"
    cam_w, cam_h = CameraSpec().width, CameraSpec().height
    cam_r = mujoco.Renderer(model, cam_h, cam_w) if vision else None
    extra = int(cam_w * h / cam_h) if vision else 0  # camera view scaled to the panel height
    frames = []
    for state in log["states"][:: args.every]:
        data.qpos[:] = state["qpos"]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=args.camera)
        frame = Image.new("RGB", (args.width + extra + PANEL_WIDTH, h), (250, 250, 248))
        frame.paste(Image.fromarray(renderer.render()), (0, 0))
        if vision:
            frame.paste(
                _camera_view(cam_r, data, state.get("overlay"), (extra, h)), (args.width, 0)
            )
        draw = ImageDraw.Draw(frame)
        y = 6
        for line in panel_lines(state.get("overlay"), state["t"]):
            for part in _wrap(line):
                draw.text((args.width + extra + 8, y), part, fill=(20, 20, 20))
                y += 13
        frames.append(frame)
    renderer.close()
    if cam_r is not None:
        cam_r.close()
    frames[0].save(
        d / "replay.gif", save_all=True, append_images=frames[1:], duration=1000 // args.fps, loop=0
    )
    print(f"wrote {d / 'replay.gif'} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
