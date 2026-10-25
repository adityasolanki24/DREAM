"""The walking G1 in NVIDIA's own Isaac Sim kitchens: locomotion bring-up and frozen-agent
transfer episodes, scored from the scene author's ground truth.

    python experiments/isaac_g1_kitchen.py --config experiments/configs/isaac_g1_kitchen.json \
        --mode bringup --kitchen l_shape --robot-dir <prepared USD> --out <dir>
    python experiments/isaac_g1_kitchen.py ... --mode explore --kitchen g_shape --start 1 ...

Scenes: Isaac Sim 6.2's replicator_kitchen layouts (NVIDIA-authored, with real hinge and
drawer joints, colliders and semantic labels). DREAM did not author or tune them. The robot
is DREAM's G1 model under NVIDIA's pretrained WBC policy (dream/sim/isaac_g1.py).

bringup: stand, walk forward, turn, stop from a deterministic start, with the Milestone A
gates; head RGB-D validity; a hinge test proving the scene's joints are simulated.
explore: the Milestone B agent (TaskAgent + RoomSearchPlanner "gain" + MobileToolbox),
UNCHANGED, on goal Inspect("cabinet"). Scoring mirrors Milestone B against every labelled
cabinet handle: claim AND robot 0.5-1.5 m from the handle it identified, facing it within
40 degrees, on the cabinet's front side, AND the reported handle within 0.3 m of that true
handle relative to the robot (drift-free).

Evidence (camera frames, decisions, trajectory, contacts, failures) is written as it
happens and before the simulator closes. ODOMETRY IS ORACLE-ASSISTED (as Milestone B).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
import urllib.request
from math import atan2, cos, pi, sin

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLOOR = "/World/KitchenRoom/Floor"
HANDLE_LABELS = ("cabinet_pull", "cabinet_knob")


class Evidence:
    """Append-only, fsynced evidence files; nothing is buffered until shutdown."""

    def __init__(self, out: pathlib.Path) -> None:
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()

    def line(self, name: str, row: dict) -> None:
        with open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def stage(self, name: str, **data) -> None:
        self.line("stages.jsonl", {"stage": name, "wall_s": time.perf_counter() - self.t0, **data})
        print(f"[stage] {name}", flush=True)

    def write(self, name: str, obj) -> None:
        (self.out / name).write_text(json.dumps(obj, indent=2, default=_json), encoding="utf-8")


def _json(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "__dict__"):
        return vars(o)
    return str(o)


def save_frame(folder: pathlib.Path, index: int, frame) -> dict:
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{index:05d}"
    try:
        from PIL import Image

        Image.fromarray(frame.rgb).save(str(stem) + ".png")
    except ImportError:
        np.save(str(stem) + "_rgb.npy", frame.rgb)
    np.savez_compressed(str(stem) + "_depth.npz", depth=frame.depth, self_mask=frame.self_mask)
    c = frame.calibration
    finite = float(np.isfinite(frame.depth).mean())
    return {"index": index, "time_s": frame.time, "finite_depth": finite,
            "self_mask": float(frame.self_mask.mean()), "rgb_std": float(frame.rgb.std()),
            "calibration_odom": {"position": c.position, "rotation": c.rotation,
                                 "fovy_deg": c.fovy_deg, "width": c.width,
                                 "height": c.height}}  # fmt: skip


def kitchen_url(root: str, name: str) -> str:
    return f"{root}/Isaac/Environments/replicator_kitchen/kitchen_{name}.usda"


def layer_sha256(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return hashlib.sha256(r.read()).hexdigest()
    except Exception:
        return None


def free_space(stage, cfg):
    """Free start cells by the config's frozen rule (scene ground truth, setup only)."""
    import numpy as np
    from pxr import Usd, UsdGeom

    from dream.eval.isaac_scene import free_starts, obstacle_boxes

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render"])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(FLOOR)).ComputeAlignedRange()
    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
    boxes = obstacle_boxes(stage, floors=(FLOOR,))
    cells = free_starts(boxes, lo[:2], hi[:2], cfg["start_clearance_m"])
    return cells, boxes, lo, hi


def clear_run(xy, yaw, boxes, step=0.05, limit=3.0) -> float:
    """Distance along a heading before entering any obstacle footprint (setup only)."""
    import numpy as np

    d = 0.0
    while d < limit:
        p = np.asarray(xy) + d * np.array([cos(yaw), sin(yaw)])
        for lo, hi in boxes:
            if lo[0] - 0.35 <= p[0] <= hi[0] + 0.35 and lo[1] - 0.35 <= p[1] <= hi[1] + 0.35:
                return d
        d += step
    return d


def choose_start(cells, boxes, mode: str, seed: int):
    import numpy as np

    if not len(cells):
        raise RuntimeError("no free start cell at the configured clearance")
    if mode == "bringup":  # deepest free cell, heading with the longest clear run
        best = None
        for xy in cells[:: max(1, len(cells) // 400)]:
            for yaw in np.radians(np.arange(-180, 180, 10)):
                run = clear_run(xy, yaw, boxes)
                if best is None or run > best[2]:
                    best = (xy, yaw, run)
        return float(best[0][0]), float(best[0][1]), float(best[1]), float(best[2])
    rng = np.random.default_rng([seed, 7])
    xy = cells[rng.integers(len(cells))]
    yaw = float(rng.uniform(-pi, pi))
    return float(xy[0]), float(xy[1]), yaw, clear_run(xy, yaw, boxes)


def handle_truth(objects):
    """Each labelled handle with its cabinet (nearest labelled-cabinet ancestor)."""
    cabinets = {o.path: o for o in objects if o.label == "kitchen_cabinet"}
    out = []
    for o in objects:
        if o.label not in HANDLE_LABELS:
            continue
        parent = max((p for p in cabinets if o.path.startswith(p + "/")), key=len, default=None)
        c = cabinets[parent].centre if parent else None
        out.append({"path": o.path, "label": o.label, "centre": o.centre, "cabinet": parent,
                    "cabinet_centre": c})  # fmt: skip
    return out


def score(env, tb, claimed: bool, handles: list[dict]) -> dict:
    import numpy as np

    from dream.navigation.follow import wrap

    x, y, yaw = env.true_pose()
    found = [c for c in tb.memory.candidates if c.handle is not None]
    out = {"candidates": len(tb.memory.candidates), "candidates_with_handle": len(found)}

    def robot_frame(v, px, py, pyaw):
        r = np.asarray(v)[:2] - [px, py]
        return np.array([cos(-pyaw) * r[0] - sin(-pyaw) * r[1],
                         sin(-pyaw) * r[0] + cos(-pyaw) * r[1]])  # fmt: skip

    # Phantom handles: reported handles not within 0.3 m of any labelled handle.
    ox, oy, oyaw = env.odometry()
    phantom = 0
    for c in found:
        est = robot_frame(c.handle, ox, oy, oyaw)
        errs = [np.linalg.norm(est - robot_frame(h["centre"], x, y, yaw)) for h in handles]
        phantom += not errs or min(errs) >= 0.3
    out["phantom_handles"] = phantom
    if not found or not handles:
        return {**out, "success": False, "false_claim": claimed, "identity_error_m": None}
    best = max(found, key=lambda c: c.handle_views)
    est = robot_frame(best.handle, ox, oy, oyaw)
    errs = [float(np.linalg.norm(est - robot_frame(h["centre"], x, y, yaw))) for h in handles]
    k = int(np.argmin(errs))
    h = handles[k]
    hc = np.asarray(h["centre"])[:2]
    d = float(np.hypot(hc[0] - x, hc[1] - y))
    facing = abs(wrap(atan2(hc[1] - y, hc[0] - x) - yaw))
    front = None
    if h["cabinet_centre"] is not None:
        n = hc - np.asarray(h["cabinet_centre"])[:2]
        if np.linalg.norm(n) > 1e-6:
            front = bool((np.array([x, y]) - np.asarray(h["cabinet_centre"])[:2]) @ n > 0)
    at_pose = 0.5 <= d <= 1.5 and facing < np.radians(40) and bool(front)
    ok = at_pose and errs[k] < 0.3
    return {**out, "success": claimed and ok, "false_claim": claimed and not ok,
            "identity_error_m": errs[k], "identified_handle": h["path"],
            "true_handle_distance_m": d, "facing_error_deg": float(np.degrees(facing)),
            "on_front_side": front}  # fmt: skip


def run(args, cfg, ev: Evidence) -> dict:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    ev.stage("app_started")
    try:
        result = _run(app, args, cfg, ev)
        ev.write("episode.json", result)  # before close: Kit's fast shutdown may end Python
        return result
    except BaseException as error:
        ev.write("failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        ev.stage("closing")
        # Evidence is already fsynced; a full Kit teardown took ~8.5 min (session 4, job 1).
        try:
            app.close(wait_for_replicator=False, skip_cleanup=True)
        except TypeError:  # older SimulationApp.close signature
            app.close()


def _run(app, args, cfg, ev: Evidence) -> dict:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.storage.native import get_assets_root_path

    from dream.eval.isaac_scene import labelled_objects, physics_audit
    from dream.robot.g1_wbc import WBCLocomotion
    from dream.sim.g1_mobile import DEFAULT_ODOMETRY_NOISE, HEAD_CAMERA, G1MobileBase
    from dream.sim.isaac_g1 import IsaacG1Env

    assets = get_assets_root_path()
    url = kitchen_url(assets, args.kitchen)
    ev.stage("assets_root", root=assets, kitchen=url)
    if not open_stage(url):
        raise RuntimeError(f"could not open {url}")
    app.update()
    world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
    stage = world.stage
    objects = labelled_objects(stage)
    scene = {"kitchen": args.kitchen, "url": url, "root_layer_sha256": layer_sha256(url),
             "physics": physics_audit(stage), "labels": {}}  # fmt: skip
    for o in objects:
        scene["labels"][o.label] = scene["labels"].get(o.label, 0) + 1
    handles = handle_truth(objects)
    cells, boxes, lo, hi = free_space(stage, cfg)
    x, y, yaw, run_m = choose_start(cells, boxes, args.mode, args.start)
    scene.update(floor_lo=lo, floor_hi=hi, free_cells=len(cells), start=[x, y, yaw],
                 start_clear_run_m=run_m)  # fmt: skip
    ev.write("scene_truth.json", {**scene, "handles": handles})  # evaluator-only file
    ev.stage("scene_loaded", free_cells=len(cells), start=[x, y, yaw])
    rng = np.random.default_rng([args.start, 1])
    env = IsaacG1Env(world, args.robot_usd, rng, DEFAULT_ODOMETRY_NOISE, HEAD_CAMERA,
                     (FLOOR,), checkpoint=ev.stage)  # fmt: skip
    env.reset(x, y, yaw, floor_z=float(hi[2]))
    ev.write(
        "parity.json",
        {
            **env.parity_report(),
            "dof_names": env.dof_names,
            "onnxruntime": __import__("onnxruntime").__version__,
            "robot_receipt": json.loads(
                (pathlib.Path(args.robot_dir) / "receipt.json").read_text()
            ),
        },
    )
    ev.stage("robot_reset")
    frames = {"n": 0}

    def capture_logged():
        frame = env.capture_head()
        ev.line("frames.jsonl", save_frame(ev.out / "frames", frames["n"], frame))
        frames["n"] += 1
        return frame

    tick = {"n": 0, "wall": time.perf_counter()}

    def on_tick():
        tick["n"] += 1
        if tick["n"] % 5 == 0:  # 10 Hz telemetry: true pose for scoring, odometry as estimated
            p = env.true_pose()
            ev.line(
                "trajectory.jsonl",
                {
                    "time_s": env.time,
                    "true_pose": p,
                    "height_m": env.true_height(),
                    "odom": env.odometry(),
                    "command": base.controller.command,
                    "active": base.controller.active,
                    "stopped": base.controller.stopped,
                    "contact_steps": env.contact_steps,
                    "wall_s": time.perf_counter() - tick["wall"],
                },
            )

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick)
    base.capture = capture_logged
    if args.mode == "bringup":
        result = bringup(env, base, cfg, ev, capture_logged)
        result["hinge_test"] = hinge_test(world, env, stage, ev)
    else:
        result = explore(env, base, cfg, ev, handles)
    wall = time.perf_counter() - tick["wall"]
    contacts = [vars(c) for c in env.contacts]
    ev.write("contacts.json", contacts)
    result.update(sim_s=env.time, wall_s=wall, real_time_factor=env.time / wall,
                  fell=base.stop_reason is not None, stop_reason=base.stop_reason,
                  contact_steps=env.contact_steps, frames=frames["n"],
                  touched=sorted({c["other"] for c in contacts}),
                  physics_steps=env.physics_steps)  # fmt: skip
    return result


def bringup(env, base, cfg, ev, capture) -> dict:
    import numpy as np

    g = cfg["bringup"]
    first = capture()
    p0 = env.true_pose()
    base.walk(0, 0, 0, g["stand_s"])
    p1 = env.true_pose()
    drift = float(np.hypot(*(p1[:2] - p0[:2])))
    capture()
    v, seconds = g["forward_m_s"], g["forward_s"]
    base.walk(v, 0, 0, 1.0)  # acceleration, excluded from the tracking ratio as Milestone A
    a = env.true_pose()
    base.walk(v, 0, 0, seconds - 1.0)
    b = env.true_pose()
    fwd = float((b[:2] - a[:2]) @ [cos(a[2]), sin(a[2])]) / (v * (seconds - 1.0))
    capture()
    w, seconds = g["turn_rad_s"], g["turn_s"]
    base.walk(0, 0, w, 1.0)
    a = env.true_pose()
    base.walk(0, 0, w, seconds - 1.0)
    b = env.true_pose()
    from dream.navigation.follow import wrap

    yaw_ratio = float(wrap(b[2] - a[2])) / (w * (seconds - 1.0))
    capture()
    base.walk(0, 0, 0, g["stop_s"])
    last = capture()
    frame_ok = all(
        np.isfinite(f.depth).mean() >= g["min_finite_depth"]
        and f.self_mask.mean() <= g["max_self_mask"]
        and f.rgb.std() > 1.0
        for f in (first, last)
    )
    gates = {
        "no_fall": base.stop_reason is None,
        "standing_drift": drift <= g["max_stand_drift_m"],
        "forward_tracking": g["tracking"][0] <= fwd <= g["tracking"][1],
        "yaw_tracking": g["tracking"][0] <= yaw_ratio <= g["tracking"][1],
        "head_rgbd_valid": bool(frame_ok),
    }
    out = {"mode": "bringup", "stand_drift_m": drift, "forward_ratio": fwd,
           "yaw_ratio": yaw_ratio, "gates": gates, "gate_pass": all(gates.values())}  # fmt: skip
    ev.stage("bringup_done", **gates)
    return out


def hinge_test(world, env, stage, ev) -> dict:
    """Push one cabinet door at its free edge and measure the hinge rotation (scene audit)."""
    import numpy as np
    from omni.physx import get_physx_interface
    from pxr import UsdPhysics

    hinges = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)
              and not str(p.GetPath()).startswith("/World/G1")]  # fmt: skip
    if not hinges:
        return {"tested": False, "reason": "no revolute joints"}
    joint = UsdPhysics.RevoluteJoint(hinges[0])
    door = str(joint.GetBody1Rel().GetTargets()[0])
    physx = get_physx_interface()

    def pose():
        t = physx.get_rigidbody_transformation(door)
        return np.asarray(t["position"], float), np.asarray(t["rotation"], float)

    p0, q0 = pose()
    for _ in range(100):  # 0.5 s of a 15 N push along the door's world x/y, then release
        physx.apply_force_at_pos(door, (15.0, 15.0, 0.0), tuple(p0 + [0, 0, 0.1]))
        world.step(render=False)
    for _ in range(100):
        world.step(render=False)
    p1, q1 = pose()
    angle = 2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(q0, q1))))))
    out = {"tested": True, "joint": str(hinges[0].GetPath()), "door": door,
           "rotation_deg": angle, "door_translation_m": float(np.linalg.norm(p1 - p0)),
           "note": "scene audit after the robot episode; not an agent action"}  # fmt: skip
    ev.stage("hinge_test", rotation_deg=angle)
    return out


def explore(env, base, cfg, ev, handles) -> dict:
    import numpy as np

    from dream.perception.spatial import SpatialMemory
    from dream.runtime.agent import TaskAgent
    from dream.runtime.search import RoomSearchPlanner
    from dream.tools.goals import Inspect
    from dream.tools.mobile import MobileToolbox
    from experiments.g1_explore_room import Budgeted

    e = cfg["explore"]

    def log(kind, text, **data):
        ev.line("events.jsonl", {"time_s": env.time, "kind": kind, "text": text})

    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([e["memory_seed"]])), on_event=log)

    def on_event(kind, data):
        ev.line("decisions.jsonl", {"time_s": env.time, "kind": kind, "data": data})

    planner = Budgeted(RoomSearchPlanner(e["method"]), base, e["time_budget_s"])
    ep = TaskAgent(tb, planner, max_steps=e["max_steps"]).run(Inspect("cabinet"), on_event)
    out = {"mode": "explore", "claimed": ep.success_claimed, "reason": ep.reason,
           "tool_calls": [s.call.name for s in ep.steps],
           "explored_m2": float(tb.memory.grid.free().sum()) * 0.05**2}  # fmt: skip
    out.update(score(env, tb, ep.success_claimed, handles))
    ev.write("memory.json", [vars(c) for c in tb.memory.candidates])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--mode", choices=["bringup", "explore"], required=True)
    ap.add_argument("--kitchen", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--robot-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    allowed = cfg["development_kitchens"] + cfg["evaluation_kitchens"]
    if args.kitchen not in allowed:
        raise SystemExit(f"kitchen {args.kitchen} is not in the frozen config")
    receipt = json.loads((args.robot_dir / "receipt.json").read_text())
    args.robot_usd = str(args.robot_dir / "usd" / receipt["usd"])
    ev = Evidence(args.out)
    sha = (
        os.environ.get("DREAM_CODE_SHA")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    )
    ev.write("meta.json", {"argv": sys.argv, "config": cfg, "git_sha": sha,
                           "kitchen_split": "development" if args.kitchen in
                           cfg["development_kitchens"] else "evaluation"})  # fmt: skip
    result = run(args, cfg, ev)
    print(json.dumps(result, indent=2, default=_json), flush=True)


if __name__ == "__main__":
    main()
