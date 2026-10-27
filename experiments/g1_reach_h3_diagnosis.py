"""H3 diagnosis (W6 step 1): why does the planner return zero candidates on reachable targets?

H3: zero-candidate planning is caused by the planner's INPUTS (target covariance, approach axis,
perceived boxes), not by its stance grid.

For each episode, the planner is rerun offline on the episode's saved perceived inputs (the first
``initial`` plan stage; the saved target is the one the plan was made from for refused episodes),
then with ONE input at a time replaced by truth or removed:

    base        perceived target, perceived boxes, covariance, approach axis (as run)
    true_target the true handle centre
    true_fixture the perceived fixture box replaced by the true fixture box
    true_boxes  every perceived box replaced by the true boxes (fixture + distractors)
    no_axis     approach axis removed
    no_cov      covariance inflation removed (covariance None)
    all_true    true target + true boxes, axis and covariance as run

Per run: candidates found, failure histogram, and for the best failing candidate the hand
residual, the violated constraints and the colliding geom pairs.

    python experiments/g1_reach_h3_diagnosis.py s5001 s5008 --out scratch/h3.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.reach import true_boxes  # noqa: E402
from dream.robot.g1_posture import (  # noqa: E402
    PerceivedBox,
    PlannerConfig,
    PosturePlanner,
)
from dream.robot.wholebody import ControllerCapabilities  # noqa: E402
from dream.sim.reach_scenes import sample_reach_scene  # noqa: E402


def face_axis(fx: dict, p) -> tuple[float, float, float]:
    """approach_axis(): into the fixture face the handle protrudes from (as ReachSkill does)."""
    c, h, y = fx["center"], fx["half_extents"], fx["yaw"]
    cs, sn = math.cos(y), math.sin(y)
    x, yy = p[0] - c[0], p[1] - c[1]
    local = np.array([cs * x + sn * yy, -sn * x + cs * yy])
    i = int(np.argmax(np.abs(local / np.array(h[:2]))))
    n = np.zeros(2)
    n[i] = np.sign(local[i])
    w = np.array([[cs, -sn], [sn, cs]]) @ n
    return (-float(w[0]), -float(w[1]), 0.0)


def name(m, g: int) -> str:
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    return n or f"{m.body(int(m.geom_bodyid[g])).name}:{g}"


def collisions(pl: PosturePlanner, q, regular, touch) -> list[str]:
    """Geom pairs closer than -1 mm in the planned configuration `q`."""
    pm, m = pl.pm, pl.pm.model
    d = pm.data_at(q)
    hands = {g for s in ("left", "right") for g in pm.hand_geoms[s]}
    fromto, out = np.zeros(6), []

    def chk(a, b, tag):
        dist = mujoco.mj_geomDistance(m, d, a, b, 0.1, fromto)
        if dist < -1e-3:
            out.append(f"{tag}:{m.body(int(m.geom_bodyid[a])).name}|{name(m, b)}={dist:.3f}")

    for a_set, b_set in pl._self_pairs:
        for a in a_set:
            for b in b_set:
                chk(a, b, "self")
    for g in pm.no_foot:
        chk(g, pm.floor_geom, "floor")
        for b in regular:
            chk(g, b, "box")
        if g not in hands:
            for b in touch:
                chk(g, b, "fixture")
    return sorted(set(out))


def run(pl: PosturePlanner, label, target, cov, boxes, axis, pose) -> dict:
    res = pl.plan(target, cov, pose, boxes, approach_axis=axis)
    ids = pl.pm.load_obstacles(boxes)
    regular = [g for g, b in zip(ids, boxes, strict=True) if not b.hand_may_touch]
    touch = [g for g, b in zip(ids, boxes, strict=True) if b.hand_may_touch]
    top = 0.0 if cov is None else np.linalg.eigvalsh(np.asarray(cov)).max()
    sigma = float(np.sqrt(max(top, 0.0)))
    limits = pl._limits(regular, touch, sigma)
    fails = [f for f in res.failures if f.metrics]
    best = sorted(fails, key=lambda f: f.metrics["hand_residual"])[:3]
    detail = []
    for f in best:
        q = pl._ik(pl._q_start(f.stance), f.stance, f.arm, target, axis, limits)
        detail.append({
            "stance": [round(v, 3) for v in f.stance], "arm": f.arm, "reasons": list(f.reasons),
            "hand_residual": round(f.metrics["hand_residual"], 4),
            "clearance": round(f.metrics["clearance"], 4),
            "com_margin": round(f.metrics["com_margin"], 4),
            "pairs": [] if q is None else collisions(pl, q, regular, touch)[:6],
        })  # fmt: skip
    out = {"label": label, "feasible": len(res.plans), "failed": len(res.failures),
           "reasons": res.reasons(), "best_failures": detail,
           "best_plan": None if not res.plans else {
               "stance": [round(v, 3) for v in res.plans[0].stance], "arm": res.plans[0].arm,
               "score": round(res.plans[0].score, 3)}}  # fmt: skip
    print(f"  {label:14s} feasible={out['feasible']:2d} failed={out['failed']:2d} {out['reasons']}",
          flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="e.g. s5001")
    ap.add_argument("--set", default="g1_reach_heights_heldout")
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "experiments/configs/g1_reach_heights.json").read_text())
    caps = ControllerCapabilities.from_envelope_v2(ROOT / cfg["envelope"])
    pc = cfg["planner"]
    pl = PosturePlanner(caps, PlannerConfig(
        radii=tuple(pc["radii"]), n_angles=pc["n_angles"], max_stances=pc["max_stances"]))
    report = {}
    for i in args.ids:
        d = json.loads((ROOT / "results" / args.set / "agent" / f"heldout_{i}.json").read_text())
        sp, tg = d["spec"], d["agent"]["target"]
        scene = sample_reach_scene(sp["seed"], sp["height"], sp["distance"], sp["lateral"],
                                   sp["yaw_jitter"], sp["n_crates"], sp["second_fixture"])
        boxes = [PerceivedBox(tuple(o["center"]), tuple(o["half_extents"]), o["yaw"],
                              bool(o["hand_may_touch"])) for o in tg["obstacles"]]
        tb = true_boxes(scene)
        ptar, ttar = np.array(tg["position"]), np.asarray(scene.handle_center)
        cov, axis = np.array(tg["covariance"]), face_axis(tg["fixture"], ptar)
        fx_true = tb[0]
        swapped = [fx_true if b.hand_may_touch else b for b in boxes]
        pose = (0.0, 0.0, 0.0)
        print(f"{i}: h={sp['height']:.2f} perr={np.linalg.norm(ptar - ttar) * 1000:.0f} mm "
              f"sigma={tg['sigma']}", flush=True)
        report[i] = [
            run(pl, "base", ptar, cov, boxes, axis, pose),
            run(pl, "true_target", ttar, cov, boxes, axis, pose),
            run(pl, "true_fixture", ptar, cov, swapped, axis, pose),
            run(pl, "true_boxes", ptar, cov, tb, axis, pose),
            run(pl, "no_axis", ptar, cov, boxes, None, pose),
            run(pl, "no_cov", ptar, None, boxes, axis, pose),
            run(pl, "all_true", ttar, cov, tb, axis, pose),
        ]
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
