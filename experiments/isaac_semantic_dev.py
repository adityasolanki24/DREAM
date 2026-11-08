"""Coordinate-convention check for the Qwen3-VL detector on the development kitchen.

The detector (dream/perception/vlm_detect.py) reads boxes in either the relative 0..1000 or the
absolute-pixel convention. This measures which one the model uses, against projected ground
truth, before any result is reported (docs of the module: "checked, never assumed").

Evaluation only: the true refrigerator and sink positions are used here to place the robot
where each is in view, and to project their axis-aligned boxes into the head camera. Nothing
reaches an agent. The robot is reset to each pose (the scene-setup mechanism), stands for a few
seconds, and the unmodified head camera and unmodified ``VLMDetector`` prompt are used.

    python experiments/isaac_semantic_dev.py --out <dir> --robot-dir <dir> [--kitchen l_shape]

Pure helpers (projection, box IoU, aggregation) are importable without Isaac for tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONVENTIONS = ("relative1000", "absolute")
# Pre-registered decision rule (before the measurement): the detector's convention changes to
# "absolute" only if absolute boxes both match more targets at IoU 0.5 and have a mean best IoU
# at least this much higher than relative1000.
SWITCH_MARGIN = 0.1
MATCH_IOU = 0.5
FRAMES_PER_TARGET = 6
VISIBILITY_SAMPLES = (0.2, 0.35, 0.5, 0.65, 0.8)  # heights within the box, as a fraction
MIN_VISIBLE_FRACTION = 0.2  # of the projected box inside the image
STAND_S = 3.0


def project_box(camera: dict, lo, hi, width: int, height: int) -> dict | None:
    """Pixel box of a world axis-aligned box seen by a pinhole camera (``camera``: position,
    forward, up, fovy_deg, aspect, as dream.eval.task_scoring.head_camera_world). None when
    any corner is behind the image plane (the box is only partly in front of the camera)."""
    p = np.asarray(camera["position"], float)
    f = np.asarray(camera["forward"], float)
    f = f / np.linalg.norm(f)
    right = np.cross(f, np.asarray(camera["up"], float))
    right = right / np.linalg.norm(right)
    up = np.cross(right, f)
    focal = (height / 2) / math.tan(math.radians(camera["fovy_deg"]) / 2)
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float
    )
    v = corners - p
    depth = v @ f
    if np.any(depth <= 0.05):
        return None
    u = width / 2 + focal * (v @ right) / depth
    w = height / 2 - focal * (v @ up) / depth
    full = [float(u.min()), float(w.min()), float(u.max()), float(w.max())]
    clipped = [
        max(0.0, full[0]),
        max(0.0, full[1]),
        min(float(width), full[2]),
        min(float(height), full[3]),
    ]
    area = max(0.0, full[2] - full[0]) * max(0.0, full[3] - full[1])
    inside = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    return {
        "box": clipped,
        "full_box": full,
        "visible_fraction": 0.0 if area <= 0 else inside / area,
        "depth_m": float(depth.min()),
    }


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def mask_box(mask) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def score_frame(answer: str, width: int, height: int, labels, truths: dict) -> dict:
    """For each convention: every parsed detection with its IoU against the projected true box
    of its own class, and the best IoU per true class. ``truths``: class -> pixel box."""
    from dream.perception.vlm_detect import parse_grounding

    out = {}
    for convention in CONVENTIONS:
        detections, problems = parse_grounding(answer, width, height, labels, convention)
        rows = []
        for d in detections:
            box = mask_box(d.mask)
            rows.append(
                {
                    "label": d.label,
                    "box": box,
                    # a detection with no pixel inside the image overlaps nothing
                    "iou_with_true_same_class": (
                        (iou(box, truths[d.label]) if box else 0.0) if d.label in truths else None
                    ),
                }
            )
        best = {
            cls: max(
                (r["iou_with_true_same_class"] for r in rows if r["label"] == cls), default=0.0
            )
            for cls in truths
        }
        out[convention] = {"detections": rows, "problems": problems, "best_iou_by_class": best}
    return out


def summarise(frames: list[dict]) -> dict:
    """Aggregate best IoUs over frames and apply the pre-registered switch rule."""
    table = {}
    for convention in CONVENTIONS:
        scores = [
            iou_value
            for fr in frames
            for iou_value in fr["scores"][convention]["best_iou_by_class"].values()
        ]
        table[convention] = {
            "targets_scored": len(scores),
            "mean_best_iou": float(np.mean(scores)) if scores else 0.0,
            "matched_at_0.5": int(sum(s >= MATCH_IOU for s in scores)),
            "parse_problems": sum(len(fr["scores"][convention]["problems"]) for fr in frames),
        }
    rel, ab = table["relative1000"], table["absolute"]
    switch = (
        ab["matched_at_0.5"] > rel["matched_at_0.5"]
        and ab["mean_best_iou"] >= rel["mean_best_iou"] + SWITCH_MARGIN
    )
    return {
        "per_convention": table,
        "decision_rule": f"absolute only if matched@{MATCH_IOU} is higher and mean best IoU is at "
        f"least {SWITCH_MARGIN} higher than relative1000",
        "evidence_says": "absolute" if switch else "relative1000",
        "frames": len(frames),
    }


def candidate_poses(lo_xy, hi_xy, cells, limit: int = 40, near: float = 0.8, far: float = 3.5):
    """Standing poses (x, y, yaw): free cells whose distance to the object's footprint is in
    [near, far] m, spread evenly in bearing around it (``limit`` of them), each turned to face
    the footprint centre. Deterministic; the first attempt's ring of fixed distances found no
    refrigerator frame because few free cells lie exactly on a ring."""
    lo, hi = np.asarray(lo_xy, float), np.asarray(hi_xy, float)
    cells = np.asarray(cells, float).reshape(-1, 2)
    gap = np.maximum(np.maximum(lo - cells, cells - hi), 0.0)
    distance = np.linalg.norm(gap, axis=1)
    keep = np.flatnonzero((distance >= near) & (distance <= far))
    if not len(keep):
        return []
    centre = (lo + hi) / 2
    bearing = np.arctan2(cells[keep, 1] - centre[1], cells[keep, 0] - centre[0])
    ordered = keep[np.argsort(bearing, kind="stable")]
    picks = ordered[
        np.unique(np.linspace(0, len(ordered) - 1, min(limit, len(ordered))).astype(int))
    ]
    return [
        (float(cells[k, 0]), float(cells[k, 1]),
         math.atan2(centre[1] - cells[k, 1], centre[0] - cells[k, 0]))
        for k in picks
    ]  # fmt: skip


def main() -> None:  # pragma: no cover - needs Isaac Sim
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--robot-dir", type=pathlib.Path, default=pathlib.Path("/work/g1_usd"))
    parser.add_argument("--kitchen", default="l_shape")
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument("--qwen-port", type=int, default=8767)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    cfg = json.loads(
        (args.config or ROOT / "experiments/configs/isaac_g1_kitchen.json").read_text()
    )
    if args.kitchen not in cfg["development_kitchens"]:
        parser.error("the coordinate check uses the development kitchen only")
    args.out.mkdir(parents=True, exist_ok=True)
    measure(args, cfg)


def measure(args, cfg) -> None:  # pragma: no cover - needs Isaac Sim
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.storage.native import get_assets_root_path
    from PIL import Image

    from dream.eval.isaac_scene import labelled_objects, obstacle_boxes
    from dream.eval.task_scoring import handle_visibility, head_camera_world
    from dream.perception.vlm_detect import PROMPT, VLMDetector, canonical_labels
    from dream.robot.g1_wbc import WBCLocomotion
    from dream.runtime.qwen import OpenAICompatibleBackend
    from dream.semantics.knowledge import load_knowledge
    from dream.sim.g1_mobile import DEFAULT_ODOMETRY_NOISE, HEAD_CAMERA, G1MobileBase
    from dream.sim.isaac_g1 import IsaacG1Env
    from experiments import isaac_semantic as sem
    from experiments.isaac_g1_kitchen import FLOOR, Evidence, free_space, kitchen_url
    from experiments.isaac_g1_live import QWEN_REVISION
    from experiments.isaac_g1_live_eval import start_model_server, stop_process

    evidence = Evidence(args.out)
    url = kitchen_url(get_assets_root_path(), args.kitchen)
    if not open_stage(url):
        raise RuntimeError(f"could not open {url}")
    app.update()
    world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
    cells, _boxes, _lo, hi = free_space(world.stage, cfg)
    labelled = labelled_objects(world.stage)
    scene_boxes = obstacle_boxes(world.stage, floors=(FLOOR,))
    rng = np.random.default_rng([1, 1])
    receipt = json.loads((args.robot_dir / "receipt.json").read_text(encoding="utf-8"))
    robot_usd = str(args.robot_dir / "usd" / (receipt.get("usda") or receipt["usd"]))
    env = IsaacG1Env(
        world,
        robot_usd,
        rng,
        DEFAULT_ODOMETRY_NOISE,
        HEAD_CAMERA,
        (FLOOR,),
        checkpoint=evidence.stage,
    )
    knowledge = load_knowledge()
    model_cfg = {
        "model_revision": QWEN_REVISION,
        "timeouts_s": {"model_start": 1500},
        "qwen_max_new_tokens": args.qwen_max_new_tokens,
    }
    server, server_log = start_model_server(model_cfg, args.out, args.qwen_port)
    backend = OpenAICompatibleBackend(f"http://127.0.0.1:{args.qwen_port}", timeout_s=120)
    detector = VLMDetector(backend, sem.detector_vocabulary(knowledge))
    labels = canonical_labels(sem.detector_vocabulary(knowledge))
    frames_dir = args.out / "frames"
    frames_dir.mkdir(exist_ok=True)
    targets = {
        "refrigerator": [o for o in labelled if o.label == "refrigerator"],
        "sink": [o for o in labelled if o.label in ("sink", "sink_rim", "sink_basin")],
    }
    records = []
    truth_log: dict = {}
    failure = None
    try:
        for cls, prims in targets.items():
            lo = np.min([p.lo for p in prims], axis=0)
            hi3 = np.max([p.hi for p in prims], axis=0)
            centre = ((lo + hi3) / 2)[:2]
            accepted = 0
            poses = candidate_poses(lo[:2], hi3[:2], cells)
            truth_log[cls] = {"lo": lo.tolist(), "hi": hi3.tolist(), "candidates": len(poses),
                              "rejected": []}  # fmt: skip
            for i, (x, y, yaw) in enumerate(poses):
                if accepted >= FRAMES_PER_TARGET:
                    break
                env.reset(x, y, yaw, floor_z=float(hi[2]))
                base = G1MobileBase(env, WBCLocomotion.from_cache())
                base.walk(0.0, 0.0, 0.0, STAND_S)
                if base.stop_reason is not None or env.true_height() < 0.6:
                    truth_log[cls]["rejected"].append({"i": i, "reason": "not standing"})
                    continue
                position, rotation = env._torso_pose()
                camera = head_camera_world(position, rotation, HEAD_CAMERA)
                # The head camera looks 35 degrees down, so a tall object is only partly in the
                # image: accept a frame when a point inside the box is unoccluded and in the
                # frustum, and score against the box clipped to the image.
                visible = {"visible": False}
                for fraction in VISIBILITY_SAMPLES:
                    point = np.array([centre[0], centre[1], lo[2] + fraction * (hi3[2] - lo[2])])
                    visible = handle_visibility(camera, point.tolist(), scene_boxes)
                    if visible.get("visible"):
                        break
                projected = project_box(camera, lo, hi3, HEAD_CAMERA.width, HEAD_CAMERA.height)
                if not visible.get("visible") or projected is None:
                    truth_log[cls]["rejected"].append(
                        {"i": i, "reason": "not visible", "pose": [x, y, yaw],
                         "projected": projected, "visible": visible}
                    )  # fmt: skip
                    continue
                if projected["visible_fraction"] < MIN_VISIBLE_FRACTION:
                    truth_log[cls]["rejected"].append(
                        {"i": i, "reason": "too little in the image", "projected": projected}
                    )
                    continue
                frame = base.capture()
                name = f"{cls}_{i:02d}"
                Image.fromarray(frame.rgb).save(frames_dir / f"{name}.png")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame.rgb},
                            {
                                "type": "text",
                                "text": PROMPT.format(classes=", ".join(detector.classes)),
                            },
                        ],
                    }
                ]
                answer = backend.generate(messages, [])
                usage = backend.last_usage
                h, w = frame.depth.shape
                truths = {cls: projected["box"]}
                # The other class in view is a second true box for the same frame.
                records.append(
                    {
                        "frame": name,
                        "pose": [x, y, yaw],
                        "class": cls,
                        "visibility": visible,
                        "projected": projected,
                        "answer": answer,
                        "usage": usage,
                        "scores": score_frame(answer, w, h, labels, truths),
                    }
                )
                accepted += 1
                (args.out / "progress.json").write_text(json.dumps(records, indent=2, default=str))
        result = {
            "kitchen": args.kitchen,
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "model_revision": QWEN_REVISION,
            "max_new_tokens": args.qwen_max_new_tokens,
            "prompt": PROMPT,
            "vocabulary": detector.classes,
            "image_size": [HEAD_CAMERA.width, HEAD_CAMERA.height],
            "method": "robot reset to ring poses around the true object, head camera frame, "
            "VLMDetector prompt, true AABB projected with the true camera pose (occlusion by "
            "other furniture not modelled; unoccluded in-frustum sample point required). "
            "Evaluation only.",
            "summary": summarise(records),
            "scene_truth_used": truth_log,
            "frames": records,
            "max_output_tokens": max(
                ((r["usage"] or {}).get("output_tokens", 0) for r in records), default=0
            ),
        }
        args.out.joinpath("coordinate_check.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(result["summary"], indent=2))
    except BaseException:
        # SimulationApp.close(skip_cleanup=True) ends the process with status 0 and hides the
        # traceback (the first attempt of this job exited 0 with no result), so record it first.
        failure = traceback.format_exc()
        print(failure, flush=True)
        (args.out / "error.txt").write_text(failure, encoding="utf-8")
    finally:
        stop_process(server, 60, term=True)
        if server_log is not None:
            server_log.close()
        if failure is not None:
            sys.stdout.flush()
            os._exit(1)
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
