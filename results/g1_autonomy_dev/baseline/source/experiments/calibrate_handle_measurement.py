"""Measure the camera's handle-position error on development scenes (calibration data).

    python experiments/calibrate_handle_measurement.py [--workers 4]

For development compartment scenes (seeds 0-9, drawer/lid/fixed; never evaluation seeds), the
vision backend's handle estimate is compared with the true handle position (evaluation code
reads the truth here) in three states:
- closed: before any contact;
- grasped: the hand holds the knob, partly hiding it;
- probed: after a 6 cm compliant pull out of the visible front, still held.
The rigid-grasp alternative is also measured: the change in the hand-handle offset between
grasping and after the probe (the TCP moves with the knob unless the fingers comply or slip).
These numbers set the uncertainty that perception reports for handles, and which source the
investigation uses (results/handle_measurement_calibration).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

if "DISPLAY" not in os.environ:  # offscreen GL for the cameras, chosen when MuJoCo is imported
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dream.skills.articulation as art  # noqa: E402
from dream.perception.vision import VisionPerception  # noqa: E402
from dream.sim.camera import SIDE_CAMERA, CameraSpec  # noqa: E402
from dream.sim.tabletop import sample_compartment_scene  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402

SEEDS = range(0, 10)


def _job(args):
    kind, seed = args
    spec = sample_compartment_scene(kind, np.random.default_rng([seed, 0]))
    env = TabletopEnv(
        spec,
        np.random.default_rng([seed, 1]),
        camera_seed=seed,
        extra_cameras=(CameraSpec(**SIDE_CAMERA),),
    )
    vis = VisionPerception(env.frame_sources(), lambda: env.time, np.random.default_rng(seed))
    rb = env.embodiment()
    out = {"kind": kind, "seed": seed}

    def measure(tag):
        obs = art.find_articulated(vis, "compartment")
        true = env.data.site("handle").xpos.copy()
        if obs is None:
            out[tag] = None
            return None
        out[tag] = {
            "error": (obs.handle_pos - true).tolist(),
            "visible": obs.handle_visible,
            "tcp_minus_true": (rb.state().tcp_pos - true).tolist(),
        }
        return obs

    if measure("closed") is None:
        return out
    # Grasp with the oracle tracker's handle pose so the measurement of vision under a grasp is
    # not confounded by vision's own grasp misses (those are measured separately: "grasp_ok").
    g = art.grasp_handle(rb, env, "compartment")
    out["grasp_ok_with_oracle_pose"] = g.ok
    if not g.ok:
        return out
    env.step(50)
    measure("grasped")
    front = spec.compartment.front_dir if kind != "lid" else np.array([0, 0, 1.0])
    art.probe(rb, env, "compartment", front, 0.06, 12.0)
    measure("probed")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [(k, s) for k in ("drawer", "lid", "fixed") for s in SEEDS]
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(_job, jobs))
    summary = {}
    for tag in ("closed", "grasped", "probed"):
        e = np.array([r[tag]["error"] for r in runs if r.get(tag)])
        g = np.array([r[tag]["tcp_minus_true"] for r in runs if r.get(tag)])
        vis = [r[tag]["visible"] for r in runs if r.get(tag)]
        if len(e) == 0:
            continue
        summary[tag] = {
            "n": len(e),
            "visible": int(sum(vis)),
            "camera_error_mean_xyz_mm": (1000 * e.mean(0)).round(1).tolist(),
            "camera_error_std_xyz_mm": (1000 * e.std(0)).round(1).tolist(),
            "camera_error_norm_mm_median": round(
                1000 * float(np.median(np.linalg.norm(e, axis=1))), 1
            ),
            "camera_error_norm_mm_p90": round(
                1000 * float(np.percentile(np.linalg.norm(e, axis=1), 90)), 1
            ),
            "tcp_minus_true_norm_mm_median": round(
                1000 * float(np.median(np.linalg.norm(g, axis=1))), 1
            ),
        }
    # Rigid-grasp measurement: change of the hand-handle offset from grasped to probed.
    d = [
        np.subtract(r["probed"]["tcp_minus_true"], r["grasped"]["tcp_minus_true"])
        for r in runs
        if r.get("probed") and r.get("grasped")
    ]
    if d:
        d = np.array(d)
        summary["grasp_offset_change"] = {
            "n": len(d),
            "norm_mm_median": round(1000 * float(np.median(np.linalg.norm(d, axis=1))), 1),
            "norm_mm_p90": round(1000 * float(np.percentile(np.linalg.norm(d, axis=1), 90)), 1),
            "norm_mm_max": round(1000 * float(np.linalg.norm(d, axis=1).max()), 1),
        }
    summary["grasp_ok_with_oracle_pose"] = sum(
        r.get("grasp_ok_with_oracle_pose", False) for r in runs
    )
    out = ROOT / "results" / "handle_measurement_calibration"
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (out / "summary.json").write_text(
        json.dumps(
            {"git_sha": sha, "seeds": list(SEEDS), "summary": summary, "runs": runs}, indent=2
        )
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
