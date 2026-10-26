"""Render a chase-camera video of a recorded Isaac Sim G1 run (Isaac Sim Python).

    python experiments/isaac_replay_video.py --run <dir of isaac_g1_kitchen.py output> \
        --robot-dir <dir> --out <dir> [--width 960 --height 540]

The run's ``trajectory.jsonl`` holds the robot's full state (root pose and every joint
angle) at 10 Hz, written while the policy walked it under physics. Here the same kitchen is
reloaded and, for each recorded row, the articulation is set to that state with zero velocity
and one 5 ms physics step propagates the transforms before rendering (the only motion the
replay adds is that step's settling, well under a millimetre). Nothing is re-planned,
interpolated or re-timed: one JPEG per recorded row, 10 per second of robot time.
``experiments/compose_demo_video.py`` adds the head-camera frames and captions locally.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def look_at(eye, target):
    from pxr import Gf

    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    return view.GetInverse()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--robot-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        render(app, args)
    finally:
        app.close()


def render(app, args) -> None:
    import numpy as np
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage
    from PIL import Image
    from pxr import UsdGeom

    from dream.sim.g1_mobile import DEFAULT_ODOMETRY_NOISE, HEAD_CAMERA
    from dream.sim.isaac_g1 import IsaacG1Env
    from experiments.isaac_g1_kitchen import FLOOR

    truth = json.loads((args.run / "scene_truth.json").read_text())
    rows = [json.loads(line) for line in (args.run / "trajectory.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "q" in r]
    if not rows:
        raise SystemExit("trajectory has no full-state rows (recorded before 26 Oct 2026)")
    if not open_stage(truth["url"]):
        raise RuntimeError(f"could not open {truth['url']}")
    app.update()
    world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
    receipt = json.loads((args.robot_dir / "receipt.json").read_text())
    robot_usd = str(args.robot_dir / "usd" / receipt["usda"])
    x, y, yaw = truth["start"]
    env = IsaacG1Env(world, robot_usd, np.random.default_rng(0), DEFAULT_ODOMETRY_NOISE,
                     HEAD_CAMERA, (FLOOR,))  # fmt: skip
    env.reset(x, y, yaw, floor_z=float(truth["floor_hi"][2]))
    cam_path = "/World/ReplayChaseCamera"
    cam = UsdGeom.Camera.Define(world.stage, cam_path)
    cam.CreateFocalLengthAttr(14.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * args.height / args.width)
    cam.CreateClippingRangeAttr((0.05, 100.0))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp()
    rp = rep.create.render_product(cam_path, (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    frames = args.out / "chase"
    frames.mkdir(parents=True, exist_ok=True)
    full = np.asarray(env.art.get_joint_positions(), float)
    heading = yaw
    t0 = time.perf_counter()
    with open(args.out / "chase_frames.jsonl", "w") as log:
        for k, r in enumerate(rows):
            full[env.index] = r["q"]
            env.art.set_world_pose(position=np.asarray(r["root_pos"]),
                                   orientation=np.asarray(r["root_quat"]))  # fmt: skip
            env.art.set_joint_positions(full)
            env.art.set_joint_velocities(np.zeros_like(full))
            env.art.set_linear_velocity(np.zeros(3))
            env.art.set_angular_velocity(np.zeros(3))
            world.step(render=False)
            px, py, pyaw = r["true_pose"]
            heading += 0.15 * math.atan2(math.sin(pyaw - heading), math.cos(pyaw - heading))
            eye = (px - 2.4 * math.cos(heading), py - 2.4 * math.sin(heading), 1.9)
            op.Set(look_at(eye, (px, py, 0.8)))
            world.render()
            world.render()
            img = np.asarray(rgb.get_data())[..., :3]
            Image.fromarray(img).save(frames / f"{k:05d}.jpg", quality=90)
            log.write(json.dumps({"index": k, "time_s": r["time_s"]}) + "\n")
    (args.out / "chase_meta.json").write_text(json.dumps({
        "run": str(args.run), "frames": len(rows), "fps": 10, "kitchen": truth["kitchen"],
        "render_wall_s": time.perf_counter() - t0,
        "method": "recorded full states set kinematically, one 5 ms physics step, rendered",
    }, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
