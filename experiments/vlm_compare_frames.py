"""Render the fixed observation set for the VLM comparison (local MuJoCo G1 head camera).

Writes results/phase1_vlm_compare/frames/*.png and decisions.json (pre-registered expected
answers, derived from simulator ground truth only; the VLM never sees them). Run once, commit
the output BEFORE any model inference.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from math import atan2, cos, degrees, pi, radians, sin

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402

OUT = ROOT / "results" / "phase1_vlm_compare"
INSTR = "Find the cabinet and inspect it."


def wrap(a):
    return (a + pi) % (2 * pi) - pi


def render(seed, with_cabinet, yaw_fn, tag):
    rng = np.random.default_rng(seed)
    room = sample_room(rng, with_cabinet=with_cabinet)
    model = build_mobile_g1(scene=lambda spec: add_room(spec, room))
    env = G1MobileEnv(model, np.random.default_rng(seed))
    x0, y0, _ = room.start
    yaw = yaw_fn(room, x0, y0)
    env.reset(x0, y0, yaw)
    rgb = env.capture_head().rgb
    Image.fromarray(rgb).save(OUT / "frames" / f"{tag}.png")

    def rel(xy):
        d = atan2(xy[1] - y0, xy[0] - x0)
        return degrees(wrap(d - yaw)), float(np.hypot(xy[0] - x0, xy[1] - y0))

    return room, rel, rgb.shape


def cab_yaw(offset_deg):
    return lambda room, x, y: atan2(room.cabinet.xy[1] - y, room.cabinet.xy[0] - x) - radians(
        offset_deg)


def frontiers():
    return [{"index": i, "xy": [round(2.5 * cos(radians(b)), 2), round(2.5 * sin(radians(b)), 2)],
             "bearing_deg_left_positive": b} for i, b in enumerate((-45, 0, 45))]


def entry(tag, cat, scene, verdict, accept, note, history=None, image=None, **extra):
    return {"id": tag, "category": cat, "image": image or f"frames/{tag}.png", "scene": scene,
            "verdict": verdict, "history": history or [], "accept": accept, "note": note, **extra}


def cand(i, b, r, **kw):
    return {"id": i, "bearing_deg_left_positive": round(b, 1), "range_m": round(r, 2), **kw}


def main():
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    d = []
    anyfront = [{"tool": "explore_frontier", "args": {"index": i}} for i in range(3)]
    for seed, off in ((3, 38), (4, -38), (5, 30)):
        tag = f"nav_{seed}"
        room, rel, shape = render(seed, True, cab_yaw(off), tag)
        b, r = rel(room.cabinet.xy)
        best = int(np.argmin([abs(b - f) for f in (-45, 0, 45)]))
        d.append(entry(tag, "navigation", {"frontiers": frontiers(), "candidates": [],
                                           "viewpoints": []}, {"satisfied": False},
                       [{"tool": "explore_frontier", "args": {"index": best}}],
                       "cabinet visible; pick the frontier on its side",
                       true_cabinet_bearing_deg=round(b, 1), true_cabinet_range_m=round(r, 2)))
    tag = "nav_away"
    render(3, True, cab_yaw(180), tag)
    d.append(entry(tag, "navigation", {"frontiers": frontiers(), "candidates": [],
                                       "viewpoints": []}, {"satisfied": False},
                   [{"tool": "look_around", "args": {}}] + anyfront,
                   "cabinet behind the robot, none visible; search, never inspect/finish-success"))
    for seed, off in ((3, 15), (4, -20), (6, 20)):
        tag = f"obj_{seed}"
        room, rel, shape = render(seed, True, cab_yaw(off), tag)
        cb, cr = rel(room.cabinet.xy)
        cands = [cand(0, cb, cr)]
        if room.furniture:
            fb, fr = rel(room.furniture[0].xy)
            cands.append(cand(1, fb, fr))
        if seed == 4:
            cands = cands[::-1]
        d.append(entry(tag, "object_identification",
                       {"frontiers": [], "candidates": cands, "viewpoints": []},
                       {"satisfied": False},
                       [{"tool": "inspect", "args": {"candidate": 0 if seed != 4 else (
                           cands[0]["id"] if abs(cands[0]["bearing_deg_left_positive"] - cb) < .11
                           else cands[1]["id"])}}],
                       "candidates are unlabeled detections; the image shows which is the cabinet"))
    tag = "obj_nocabinet"
    render(7, False, lambda r, x, y: 0.0, tag)
    d.append(entry(tag, "object_identification",
                   {"frontiers": frontiers(), "candidates": [], "viewpoints": []},
                   {"satisfied": False},
                   [{"tool": "look_around", "args": {}}] + anyfront
                   + [{"tool": "finish", "args": {"success": False}}],
                   "no cabinet in view; finish success=true is wrong"))
    tag = "info_unverified"
    room, rel, shape = render(3, True, cab_yaw(10), tag)
    cb, cr = rel(room.cabinet.xy)
    d.append(entry(tag, "information_gathering",
                   {"frontiers": [], "viewpoints": [{"candidate": 0, "side": 0}],
                    "candidates": [cand(0, cb, cr)]},
                   {"satisfied": False, "reason": "candidate 0 not yet inspected"},
                   [{"tool": "inspect", "args": {"candidate": 0}},
                    {"tool": "approach", "args": {"candidate": 0, "side": 0}}],
                   "cabinet candidate seen but unverified; inspect/approach, not finish success"))
    d.append(entry("info_satisfied", "information_gathering",
                   {"frontiers": [], "viewpoints": [],
                    "candidates": [cand(0, cb, cr, inspected=True, handle_found=True)]},
                   {"satisfied": True, "reason": "cabinet inspected, handle located"},
                   [{"tool": "finish", "args": {"success": True}}],
                   "verdict satisfied: finish success",
                   history=[{"tool": "inspect", "args": {"candidate": 0}, "status": "ok",
                             "message": "cabinet handle located"}],
                   image="frames/info_unverified.png"))
    tag = "info_blind"
    render(3, True, cab_yaw(180), tag)
    d.append(entry(tag, "information_gathering",
                   {"frontiers": [], "candidates": [], "viewpoints": []}, {"satisfied": False},
                   [{"tool": "look_around", "args": {}}],
                   "nothing observed, no frontier: the only informative tool is look_around"))
    cfg = {"instruction": INSTR, "image_shape": list(shape), "decisions": d}
    (OUT / "decisions.json").write_text(json.dumps(cfg, indent=1))
    print(len(d), "decisions")


if __name__ == "__main__":
    main()
