"""Development runs in composed Isaac homes: validation, recognisability, perception capture.

Simulator/evaluation side. For each home in a development config (generator variant, seed,
scenario, target; development seeds only) this job

1. composes the home (experiments/isaac_home_compose.py), loads it in Isaac Sim, and checks that
   every placed instance has a finite, non-empty semantic bound (the l_island failure mode);
2. spawns the G1 under the pretrained WBC at the generator's start pose and checks it stands;
3. explores with DREAM's existing mobile tools (frontier ``observe_scene`` -> ``go_to``, rpp
   navigator, camera-built occupancy map; the agent side sees only its own sensors), recording
   per frame RGB, depth, intrinsics, the agent's odometry, the TRUE camera pose and the Isaac
   instance-segmentation ground truth (boxes, instance ids, classes), and every ``qwen_every``-th
   stationary frame also Qwen3-VL's raw answer and parsed boxes;
4. per placed class, resets the robot at up to ``views`` standing poses 1.5 to 3.0 m from an
   instance (line of sight on the generator geometry, facing it) and asks the unmodified
   ``VLMDetector`` prompt for that class among all classes: per-class recognisability.

Raw frames go under ``--raw-dir`` (outside Git); ``--out`` gets compact per-home JSON. Evidence is
written as it happens; the job stops admitting homes when the remaining budget is short.

    python experiments/isaac_home_runs.py --config experiments/configs/isaac_home_dev.json \
        --out <dir> --raw-dir <dir> --robot-dir <dir> [--kitchen l_shape] [--budget-s 3000]

Pure helpers (viewpoints, box extraction) are importable and tested without Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEAR_M, FAR_M = 1.5, 3.0  # viewing distance band of the recognisability test (owner brief)
VIEW_CLEARANCE_M = 0.45  # standing pose at least this far from furniture and walls
MATCH_IOU = 0.5
STAND_S = 3.0
FALL_HEIGHT_M = 0.45  # dream/eval/task_scoring.FALL_HEIGHT_M


# -- pure helpers --------------------------------------------------------------------------------
def _segment_hits(p, q, rects, step=0.05) -> bool:
    n = max(2, int(np.linalg.norm(np.asarray(q) - np.asarray(p)) / step))
    pts = np.linspace(p, q, n)
    return any(bool(r.inside(pts).any()) for r in rects)


def home_blockers(home):
    """(rects of furniture with their instance ids, rects of walls) in the generator frame."""
    from dream.sim.home import Rect, footprint

    furniture = [(o.instance_id, footprint(o)) for o in home.objects]
    walls = [Rect(w.centre[0], w.centre[1], w.half[0], w.half[1]) for w in home.walls]
    return furniture, walls


def viewpoints_for(home, obj, k: int = 3, spacing: float = 0.25):
    """Up to ``k`` standing poses (x, y, yaw), 1.5 to 3.0 m from the object's footprint, in its
    room, clear of furniture and walls by VIEW_CLEARANCE_M, with an unobstructed sight line to
    its centre on the generator geometry, spread in bearing and facing the object. Deterministic.
    """
    from dream.sim.home import footprint, room_at

    furniture, walls = home_blockers(home)
    own = footprint(obj)
    others = [r for i, r in furniture if i != obj.instance_id]
    room = room_at(home, obj.xy)
    lo, hi = np.array(room.lo), np.array(room.hi)
    xs = np.arange(lo[0] + 0.3, hi[0] - 0.3, spacing)
    ys = np.arange(lo[1] + 0.3, hi[1] - 0.3, spacing)
    cands = []
    centre = np.array(obj.xy)
    for x in xs:
        for y in ys:
            pt = np.array([x, y])
            d = float(own.distance(pt[None])[0])
            if not NEAR_M <= d <= FAR_M:
                continue
            if any(float(r.distance(pt[None])[0]) < VIEW_CLEARANCE_M for r in [*others, *walls]):
                continue
            if _segment_hits(pt, centre, others + walls):
                continue
            cands.append((x, y, math.atan2(centre[1] - y, centre[0] - x)))
    if not cands:
        return []
    cands.sort(key=lambda c: math.atan2(c[1] - centre[1], c[0] - centre[0]))
    picks = np.unique(np.linspace(0, len(cands) - 1, min(k, len(cands))).astype(int))
    return [cands[i] for i in picks]


def mask_box(mask) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def instance_boxes(ids, id_to_path: dict, instance_paths: dict[str, str]) -> dict:
    """Per placed instance: pixel count and tight 2D box from the instance segmentation.
    ``instance_paths``: instance_id -> its prim path. Evaluation side."""
    by_instance: dict[str, list[int]] = {}
    for key, path in id_to_path.items():
        path = str(path)
        for iid, prefix in instance_paths.items():
            if path == prefix or path.startswith(prefix + "/"):
                by_instance.setdefault(iid, []).append(int(key))
    out = {}
    for iid, keys in by_instance.items():
        mask = np.isin(np.asarray(ids), keys)
        n = int(mask.sum())
        if n:
            out[iid] = {"pixels": n, "box": mask_box(mask)}
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def recognisability(rows: list[dict]) -> dict:
    """Per class over view rows ``{class, detections: [{label, box}], truth_box}``: views, views
    with a same-class detection, and views with one at IoU >= MATCH_IOU against the true box;
    plus the wrong-class detections that overlapped the true box at IoU >= MATCH_IOU."""
    out: dict[str, dict] = {}
    for r in rows:
        c = out.setdefault(
            r["class"], {"views": 0, "label_hit": 0, "iou_hit": 0, "confused_as": {}}
        )
        c["views"] += 1
        same = [d for d in r["detections"] if d["label"] == r["class"] and d["box"]]
        c["label_hit"] += bool(same)
        c["iou_hit"] += any(iou(d["box"], r["truth_box"]) >= MATCH_IOU for d in same)
        for d in r["detections"]:
            if d["label"] != r["class"] and d["box"] and iou(d["box"], r["truth_box"]) >= MATCH_IOU:
                c["confused_as"][d["label"]] = c["confused_as"].get(d["label"], 0) + 1
    return out


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- Isaac side ----------------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - needs Isaac Sim
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--raw-dir", type=pathlib.Path, required=True)
    ap.add_argument("--robot-dir", type=pathlib.Path, default=pathlib.Path("/work/g1_usd"))
    ap.add_argument("--kitchen", default="", help="also capture in this development kitchen")
    ap.add_argument("--homes", default="", help="comma list of seeds to run (default all)")
    ap.add_argument("--budget-s", type=float, default=3000.0)
    ap.add_argument("--qwen-port", type=int, default=8767)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg.get("status") != "development":
        raise SystemExit("this runner is for development configs only")
    args.out.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    failure = None
    server = server_log = None
    try:
        from experiments.isaac_g1_live import QWEN_REVISION
        from experiments.isaac_g1_live_eval import start_model_server

        model_cfg = {"model_revision": QWEN_REVISION, "timeouts_s": {"model_start": 1500},
                     "qwen_max_new_tokens": cfg.get("qwen_max_new_tokens", 512)}  # fmt: skip
        server, server_log = start_model_server(model_cfg, args.out, args.qwen_port)
        run_all(app, args, cfg)
    except BaseException:
        failure = traceback.format_exc()
        print(failure, flush=True)
        (args.out / "error.txt").write_text(failure, encoding="utf-8")
    finally:
        try:
            from experiments.isaac_g1_live_eval import stop_process

            stop_process(server, 60, term=True)
            if server_log is not None:
                server_log.close()
        except Exception:
            pass
        if failure is not None:
            sys.stdout.flush()
            os._exit(1)
        app.close(wait_for_replicator=False, skip_cleanup=True)


def run_all(app, args, cfg) -> None:  # pragma: no cover - needs Isaac Sim
    t0 = time.monotonic()
    done = []
    wanted = {int(s) for s in args.homes.split(",") if s}
    items = [("home", h) for h in cfg["homes"] if not wanted or h["seed"] in wanted]
    if args.kitchen:
        items.append(("kitchen", {"kitchen": args.kitchen, "seed": 1}))
    for kind, item in items:
        left = args.budget_s - (time.monotonic() - t0)
        if left < cfg["min_item_budget_s"]:
            print(f"[budget] stopping before {item}: {left:.0f} s left", flush=True)
            break
        name = (f"{cfg['variant']}_{item['seed']}_{item['scenario']}_{item['target']}"
                if kind == "home" else f"kitchen_{item['kitchen']}")  # fmt: skip
        try:
            result = run_item(app, args, cfg, kind, item, name, min(left, cfg["max_item_budget_s"]))
        except BaseException:
            result = {"name": name, "crashed": traceback.format_exc()}
            print(result["crashed"], flush=True)
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=1, default=str))
        done.append(name)
        print(f"[item done] {name}", flush=True)
        if "crashed" in result and "Floating point" in result["crashed"]:
            break


def run_item(app, args, cfg, kind, item, name, budget_s):  # noqa: C901  # pragma: no cover
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.storage.native import get_assets_root_path
    from PIL import Image

    from dream.eval.isaac_scene import labelled_objects
    from dream.eval.task_scoring import head_camera_world
    from dream.perception.spatial import SpatialMemory
    from dream.perception.vlm_detect import PROMPT, VLMDetector, canonical_labels
    from dream.robot.g1_wbc import WBCLocomotion
    from dream.runtime.qwen import OpenAICompatibleBackend
    from dream.semantics.knowledge import load_knowledge
    from dream.sim.g1_mobile import DEFAULT_ODOMETRY_NOISE, HEAD_CAMERA, G1MobileBase
    from dream.sim.home import CLASSES
    from dream.sim.isaac_g1 import IsaacG1Env
    from dream.tools.intent import default_vocabulary
    from dream.tools.mobile import MobileToolbox
    from experiments import isaac_home_compose as comp
    from experiments.isaac_g1_kitchen import Evidence, choose_start, free_space, kitchen_url

    t_start = time.monotonic()
    raw = args.raw_dir / name
    raw.mkdir(parents=True, exist_ok=True)
    ev = Evidence(args.out / "evidence" / name)
    result: dict = {"name": name, "kind": kind, "item": item}
    knowledge = load_knowledge()
    vocab = default_vocabulary(CLASSES, knowledge.synonyms)
    labels = canonical_labels(vocab)

    # -- scene ------------------------------------------------------------------------------
    if kind == "home":
        home, placements, truth = comp.compose_plan(
            cfg["variant"], item["seed"], item["scenario"], item["target"]
        )
        view = comp.write_composition(home, placements, truth, raw / "scene")
        stage_url = view["stage"]
        (args.out / f"{name}_scene_truth.json").write_text(json.dumps(truth, indent=1))
        floors = tuple(f"{comp.ROOT_PRIM}/Floor{r.id}" for r in home.rooms)
        start = tuple(home.start)
        inst_paths = {p.instance_id: p.prim_path for p in placements}
        inst_class = {p.instance_id: p.cls for p in placements}
        floor_z = 0.0
    else:
        stage_url = kitchen_url(get_assets_root_path(), item["kitchen"])
        home = None
        floors = ("/World/KitchenRoom/Floor",)
    if not open_stage(stage_url):
        raise RuntimeError(f"could not open {stage_url}")
    for _ in range(60):
        app.update()
    world = World(physics_dt=0.005, rendering_dt=0.005, stage_units_in_meters=1.0)
    stage = world.stage
    objects = labelled_objects(stage)
    if kind == "home":
        by_path = {o.path: o for o in objects}
        missing = [i for i, p in inst_paths.items() if p not in by_path]
        result["load_check"] = {"labelled": len(objects), "instances": len(inst_paths),
                                "instances_without_finite_bound": missing}  # fmt: skip
        if missing:
            result["failed"] = "placed instances without a finite semantic bound"
            return result
        obstacle_start = start
    else:
        cells, boxes, flo, fhi = free_space(
            stage, json.loads((ROOT / "experiments/configs/isaac_g1_kitchen.json").read_text())
        )
        x, y, yaw, _ = choose_start(cells, boxes, "explore", item["seed"])
        start, floor_z = (x, y, yaw), float(fhi[2])
        inst_paths = {f"k{n:03d}": o.path for n, o in enumerate(objects)}
        inst_class = {f"k{n:03d}": o.label for n, o in enumerate(objects)}
        by_path = {o.path: o for o in objects}
        obstacle_start = start
    # -- robot --------------------------------------------------------------------------------
    rng = np.random.default_rng([item["seed"], 1])
    receipt = json.loads((args.robot_dir / "receipt.json").read_text(encoding="utf-8"))
    robot_usd = str(args.robot_dir / "usd" / (receipt.get("usda") or receipt["usd"]))
    env = IsaacG1Env(world, robot_usd, rng, DEFAULT_ODOMETRY_NOISE, HEAD_CAMERA, floors,
                     checkpoint=ev.stage)  # fmt: skip
    env.reset(*obstacle_start, floor_z=floor_z)
    frames_dir = raw / "frames"
    frames_dir.mkdir(exist_ok=True)
    backend = OpenAICompatibleBackend(f"http://127.0.0.1:{args.qwen_port}", timeout_s=180)
    detector = VLMDetector(backend, vocab)
    prompt = PROMPT.format(classes=", ".join(detector.classes))
    fx = (HEAD_CAMERA.height / 2) / math.tan(math.radians(HEAD_CAMERA.fovy_deg) / 2)
    intrinsics = {"fx": fx, "fy": fx, "cx": HEAD_CAMERA.width / 2, "cy": HEAD_CAMERA.height / 2,
                  "width": HEAD_CAMERA.width, "height": HEAD_CAMERA.height}  # fmt: skip
    result.update(prompt=prompt, intrinsics=intrinsics, start=list(start))
    base = G1MobileBase(env, WBCLocomotion.from_cache())
    base.walk(0.0, 0.0, 0.0, STAND_S)
    result["stands"] = bool(base.stop_reason is None and env.true_height() > 0.6)
    result["stand_height_m"] = env.true_height()
    if not result["stands"]:
        result["failed"] = f"did not stand: {base.stop_reason}"
        return result

    # -- exploration with per-frame capture ---------------------------------------------------
    counters = {"frames": 0, "qwen": 0}
    rows_path = raw / "frames.jsonl"

    def qwen_on(frame):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": frame.rgb}, {"type": "text", "text": prompt}]}]  # fmt: skip
        t1 = time.perf_counter()
        text = backend.generate(messages, [])
        from dream.perception.vlm_detect import parse_grounding

        dets, problems = parse_grounding(text, frame.depth.shape[1], frame.depth.shape[0], labels,
                                         detector.coordinates)  # fmt: skip
        return {
            "raw": text,
            "problems": problems,
            "latency_s": time.perf_counter() - t1,
            "detections": [{"label": d.label, "box": mask_box(d.mask)} for d in dets],
        }

    def capture_logged(with_qwen: bool = False):
        frame = env.capture_head()
        i = counters["frames"]
        counters["frames"] += 1
        stem = frames_dir / f"{i:05d}"
        Image.fromarray(frame.rgb).save(str(stem) + ".png")
        ids, info = env.last_segmentation
        np.savez_compressed(str(stem) + ".npz", depth=frame.depth.astype(np.float16),
                            instance_ids=np.asarray(ids).astype(np.int32))  # fmt: skip
        pos, rot = env._torso_pose()
        cam = head_camera_world(pos, rot, HEAD_CAMERA)
        gt = instance_boxes(ids, info.get("idToLabels", info) if isinstance(info, dict) else {},
                            inst_paths)  # fmt: skip
        row = {"i": i, "sim_time_s": env.time, "odometry_pose": env.odometry().tolist(),
               "true_pose": env.true_pose().tolist(), "true_camera": cam,
               "gt": {iid: {"class": inst_class[iid], **v} for iid, v in gt.items()},
               "png_sha256": sha256_of(pathlib.Path(str(stem) + ".png"))}  # fmt: skip
        if with_qwen:
            row["qwen"] = qwen_on(frame)
            counters["qwen"] += 1
        with open(rows_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
        return frame

    base.capture = lambda: capture_logged(False)

    toolbox = MobileToolbox(base, SpatialMemory(np.random.default_rng([item["seed"], 2])),
                            navigator="rpp")  # fmt: skip
    explore_budget = budget_s * cfg["explore_fraction"]
    rooms_seen, path_m, last = set(), 0.0, env.true_pose()[:2]
    calls = []
    while time.monotonic() - t_start < explore_budget and counters["frames"] < cfg["max_frames"]:
        obs = toolbox.call("observe_scene")
        fronts = obs.data.get("frontiers", [])
        if not fronts:
            calls.append({"call": "observe_scene", "note": "no frontier left"})
            break
        target = fronts[0]["xy"]
        res = toolbox.call("go_to", {"x": float(target[0]), "y": float(target[1])})
        calls.append(
            {"call": "go_to", "xy": target, "status": res.status, "text": res.message[:120]}
        )
        capture_logged(with_qwen=True)  # a stationary frame with Qwen's answer at every stop
        if res.status == "stopped" or base.stop_reason:
            break
        p = env.true_pose()[:2]
        path_m += float(np.linalg.norm(p - last))
        last = p
        if home is not None:
            from dream.sim.home import room_at

            r = room_at(home, p)
            if r is not None:
                rooms_seen.add(r.id)
    result["exploration"] = {"calls": calls, "path_length_m": path_m, "frames": counters["frames"],
                             "qwen_calls": counters["qwen"],
        "rooms_visited_true": sorted(rooms_seen),
                             "explored_m2": float(toolbox.memory.grid.free().sum()) * 0.05**2,
                             "fell": env.true_height() < FALL_HEIGHT_M,
                             "stop_reason": base.stop_reason, "contact_steps": env.contact_steps,
                             "wall_s": time.monotonic() - t_start}  # fmt: skip
    # -- per-class recognisability at reset poses ---------------------------------------------
    rows = []
    if home is not None and not base.stop_reason:
        from experiments.isaac_semantic_dev import project_box

        seen_classes: set[str] = set()
        for obj in home.objects:
            if obj.cls in seen_classes:
                continue
            seen_classes.add(obj.cls)
            if time.monotonic() - t_start > budget_s:
                result["recognisability_truncated_at"] = obj.cls
                break
            o_truth = by_path[inst_paths[obj.instance_id]]
            for k, (x, y, yaw) in enumerate(viewpoints_for(home, obj, cfg["views"])):
                env.reset(x, y, yaw, floor_z=0.0)
                b2 = G1MobileBase(env, WBCLocomotion.from_cache())
                b2.walk(0.0, 0.0, 0.0, STAND_S)
                if b2.stop_reason or env.true_height() < 0.6:
                    rows.append({"class": obj.cls, "view": k, "skipped": "not standing"})
                    continue
                pos, rot = env._torso_pose()
                cam = head_camera_world(pos, rot, HEAD_CAMERA)
                proj = project_box(cam, o_truth.lo, o_truth.hi, HEAD_CAMERA.width,
                                   HEAD_CAMERA.height)  # fmt: skip
                frame = env.capture_head()
                ids, info = env.last_segmentation
                gt = instance_boxes(
                    ids,
                    info.get("idToLabels", info),
                    {obj.instance_id: inst_paths[obj.instance_id]},
                )  # noqa: E501
                q = qwen_on(frame)
                Image.fromarray(frame.rgb).save(raw / f"view_{obj.cls}_{k}.png")
                rows.append({"class": obj.cls, "instance": obj.instance_id, "view": k,
                             "pose": [x, y, yaw], "projected": proj,
                             "seg_pixels": gt.get(obj.instance_id, {}).get("pixels", 0),
                             "seg_box": gt.get(obj.instance_id, {}).get("box"),
                             "truth_box": (gt.get(obj.instance_id) or {}).get("box")
                             or (proj or {}).get("box"),
                             "qwen_raw": q["raw"], "detections": q["detections"]})  # fmt: skip
        scored = [r for r in rows if "qwen_raw" in r and r["truth_box"]
                  and r["seg_pixels"] >= cfg["min_visible_pixels"]]  # fmt: skip
        result["recognisability"] = recognisability(scored)
        result["recognisability_rows"] = rows
    result["wall_s"] = time.monotonic() - t_start
    result["frames_dir"] = str(frames_dir)
    return result


if __name__ == "__main__":
    main()
