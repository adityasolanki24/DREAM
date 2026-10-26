"""Calibration of the handle estimate on DEVELOPMENT scenes (seeds 100-102): how far is the fused
estimate of ``HandleEstimator`` from the true handle centre, by range to the handle and by axis of
the handle's face (along the outward normal, along the face, vertical)?

    python experiments/g1_reach_perception_calibration.py [--out results/g1_reach_heights_dev]

Ground truth is read here, as the evaluator. The robot is placed in front of the face (a
measurement setup, not an agent action); the head camera looks with the gaze the skill would
use. The result sets ``HANDLE_BIAS_STD`` (dream/perception/reach_target.py): the rms error at the
ranges the skill re-observes from (0.5-0.8 m). Held-out seeds are never used.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.perception.detect import detect  # noqa: E402
from dream.perception.reach_target import DEPTH_NOISE, HandleEstimator  # noqa: E402
from dream.perception.spatial import FLOOR_PLANE  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.reach_scenes import add_reach_scene, sample_reach_scene  # noqa: E402

HEIGHTS = (0.2, 0.35, 0.55, 0.75, 0.95, 1.1, 1.3)
SEEDS = (100, 101, 102)
RANGES = (0.5, 0.8, 1.5, 2.0)  # m from the camera to the face
LOOK_UP = -0.3  # rad, the gaze used for high handles (the measured look-up pitch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/g1_reach_heights_dev")
    args = ap.parse_args()
    rows = []
    for h in HEIGHTS:
        for seed in SEEDS:
            for rng_m in RANGES:
                s = sample_reach_scene(seed, h, 1.6, 0.2, 0.2, 0, False, "red")
                env = G1MobileEnv(
                    build_mobile_g1(
                        scene=lambda spec, s=s: add_reach_scene(spec, spec_s(s)), hands=True
                    ),
                    np.random.default_rng(seed),
                )
                n = np.array(s.face_normal)
                pos = np.array(s.handle_center[:2]) + n * rng_m
                yaw = math.atan2(-n[1], -n[0])
                env.reset(pos[0], pos[1], yaw)
                ctrl = WBCLocomotion.from_cache()
                ctrl.set_posture(0.74, (0, LOOK_UP if h > 1.0 else 0.0, 0))
                for _ in range(120):
                    env.step(ctrl)
                est = HandleEstimator("red")
                for k in range(3):
                    f = env.capture_head()
                    dets, _ = detect(f, np.random.default_rng(k), FLOOR_PLANE, DEPTH_NOISE)
                    est.ingest(dets, f.calibration.position, f.time)
                    for _ in range(5):
                        env.step(ctrl)
                e = est.estimate()
                row = {"height": h, "seed": seed, "range_m": rng_m, "seen": e is not None}
                if e is not None:
                    c, sn = math.cos(yaw), math.sin(yaw)
                    p = e.position  # odometry frame: origin at the reset pose
                    w = np.array(
                        [pos[0] + c * p[0] - sn * p[1], pos[1] + sn * p[0] + c * p[1], p[2]]
                    )
                    err = w - np.array(s.handle_center)
                    t = np.array([-n[1], n[0]])
                    row["error_m"] = [float(err[:2] @ n), float(err[:2] @ t), float(err[2])]
                    row["sigma_m"] = e.sigma
                rows.append(row)
                print(row, flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for lo, hi in ((0.0, 1.0), (1.0, 3.0)):
        sel = [r for r in rows if r["seen"] and lo < r["range_m"] <= hi]
        err = np.array([r["error_m"] for r in sel])
        summary[f"range_{lo}-{hi}m"] = {
            "n": len(sel),
            "mean_mm_normal_tangent_vertical": (err.mean(0) * 1000).round(2).tolist(),
            "std_mm": (err.std(0) * 1000).round(2).tolist(),
            "rms_3d_mm": round(float(np.sqrt((err**2).sum(1).mean()) * 1000), 2),
        }
    seen = sum(r["seen"] for r in rows)
    result = {"rows": rows, "summary": summary, "seen": seen, "n": len(rows),
              "note": "handle not seen at long range or out of the camera's view is counted "
                      "as not seen, not as an error"}  # fmt: skip
    (out / "perception_calibration.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1))


def spec_s(s):
    return s


if __name__ == "__main__":
    main()
