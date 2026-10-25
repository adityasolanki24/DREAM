"""Milestone A: does the pretrained G1 whole-body locomotion policy walk, turn, stop, resist
pushes and reach commanded positions in MuJoCo physics?

    python experiments/g1_locomotion.py experiments/configs/g1_locomotion.json [--record]

Everything moves through physics under NVIDIA's released Decoupled WBC policies; nothing is
scripted at the joint level. Velocity commands come from fixed profiles (tracking), or from
dream/navigation/follow.py using the robot's ORACLE-ASSISTED odometry (go-to). Ground truth
(pelvis pose, falls) is read only here, as the evaluator. Results go to results/<name>/ with
the git SHA read at the start.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from math import cos, sin

if "DISPLAY" not in os.environ:  # offscreen GL, chosen when MuJoCo is imported
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.navigation.follow import PathFollower, wrap  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1  # noqa: E402

TICK = 0.02  # s, one policy step


def _env(seed: int, camera: bool = False):
    rng = np.random.default_rng(seed)
    kw = {} if camera else {"head_camera": None}
    env = G1MobileEnv(build_mobile_g1(**kw), rng, **kw)
    env.reset(0.0, 0.0, float(rng.uniform(-np.pi, np.pi)))
    env.data.qpos[7:] += rng.normal(scale=0.02, size=env.data.qpos[7:].shape)  # scene setup
    mujoco.mj_forward(env.model, env.data)
    return env, WBCLocomotion.from_cache(), rng


def _run(env, ctrl, seconds: float, on_tick=None) -> bool:
    """Step for ``seconds``; False if the robot fell (evaluator: tilt or pelvis height)."""
    for _ in range(round(seconds / TICK)):
        env.step(ctrl)
        if on_tick:
            on_tick()
        if ctrl.stopped or env.data.qpos[2] < 0.5:
            return False
    return True


def tracking(seed: int, cfg: dict) -> list[dict]:
    out = []
    for cmd in cfg["profiles"]:
        env, ctrl, _ = _env(seed)
        ok = _run(env, ctrl, cfg["stand_s"])
        ctrl.set_command(*cmd)
        ok = ok and _run(env, ctrl, 1.0)  # gait onset, not scored
        p0, t0 = env.true_pose(), env.time
        turned = [0.0, p0[2]]  # accumulated yaw (rad), last yaw: turns may exceed pi

        def accumulate(env=env, turned=turned):
            y = env.true_pose()[2]
            turned[0] += wrap(y - turned[1])
            turned[1] = y

        ok = ok and _run(env, ctrl, cfg["profile_s"], accumulate)
        p1, dt = env.true_pose(), env.time - t0
        c, s = cos(p0[2]), sin(p0[2])
        d = p1[:2] - p0[:2]
        fwd, left = (c * d[0] + s * d[1]) / dt, (-s * d[0] + c * d[1]) / dt
        yaw_rate = turned[0] / dt
        out.append(
            {
                "seed": seed, "command": cmd, "fell": not ok,
                "forward_m_s": fwd, "lateral_m_s": left, "yaw_rad_s": yaw_rate,
                "pelvis_height_m": float(env.data.qpos[2]),
            }
        )  # fmt: skip
    return out


def standing(seed: int, cfg: dict) -> dict:
    env, ctrl, _ = _env(seed)
    ok = _run(env, ctrl, 1.0)
    p0 = env.true_pose()
    ok = ok and _run(env, ctrl, cfg["standing_s"])
    p1 = env.true_pose()
    return {"seed": seed, "fell": not ok, "drift_m": float(np.linalg.norm(p1[:2] - p0[:2]))}


def push(seed: int, cfg: dict, force: float) -> dict:
    env, ctrl, rng = _env(seed)
    ok = _run(env, ctrl, cfg["stand_s"])
    ctrl.set_command(cfg["push_walk_speed"], 0, 0)
    ok = ok and _run(env, ctrl, 2.0)
    yaw = env.true_pose()[2] + rng.choice([np.pi / 2, -np.pi / 2, 0.0, np.pi])  # side/front/back
    pelvis = env.model.body("pelvis").id
    env.data.xfrc_applied[pelvis, :2] = force * np.array([cos(yaw), sin(yaw)])  # evaluator push
    ok = ok and _run(env, ctrl, cfg["push_duration_s"])
    env.data.xfrc_applied[pelvis] = 0
    ok = ok and _run(env, ctrl, 3.0)
    return {"seed": seed, "force_n": force, "direction_rad": float(yaw), "recovered": ok}


def goto(seed: int, cfg: dict, recorder=None) -> list[dict]:
    env, ctrl, rng = _env(seed, camera=recorder is not None)
    if recorder is not None:
        recorder.env = env
    tick = recorder.sample if recorder else None
    out = []
    ok = _run(env, ctrl, cfg["stand_s"], tick)
    follower = PathFollower()
    for k in range(cfg["goto_per_seed"]):
        # Goal relative to where the robot believes it is (its odometry frame).
        r = rng.uniform(*cfg["goto_distance_m"])
        bearing = rng.uniform(-np.pi, np.pi)
        odom = env.odometry()
        goal_odom = odom[:2] + r * np.array([cos(odom[2] + bearing), sin(odom[2] + bearing)])
        # The same goal in the world, for scoring: the odometry frame is the start frame.
        true0, odom0 = env.true_pose(), env.odometry()
        follower.set_path([goal_odom])
        if recorder:
            recorder.event("goal", f"go to {r:.1f} m at {np.degrees(bearing):+.0f} deg")
        t0, status = env.time, "walking"
        while env.time - t0 < cfg["goto_timeout_s"] and status != "arrived" and ok:
            vx, wz, status = follower.update(env.odometry())
            ctrl.set_command(vx, 0.0, wz)
            ok = _run(env, ctrl, TICK, tick)
        ctrl.set_command(0, 0, 0)
        ok = ok and _run(env, ctrl, 1.0, tick)
        # Score: where the robot really is against where the goal really is.
        c, s = cos(true0[2] - odom0[2]), sin(true0[2] - odom0[2])
        rel = goal_odom - odom0[:2]
        goal_true = true0[:2] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        err = float(np.linalg.norm(env.true_pose()[:2] - goal_true))
        if recorder:
            recorder.event("outcome", f"{status}; true error {err:.2f} m")
        out.append(
            {
                "seed": seed, "k": k, "distance_m": r, "status": status, "fell": not ok,
                "true_error_m": err, "time_s": env.time - t0,
                "odometry_error_m": float(np.linalg.norm(env.odometry()[:2] - goal_odom))
                if status == "arrived" else None,
            }
        )  # fmt: skip
        if not ok:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--record", action="store_true", help="record one go-to episode")
    ap.add_argument("--record-only", action="store_true", help="skip the evaluation")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.record_only:
        return record(cfg, sha)
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "dream"], cwd=ROOT))
    t_start = time.perf_counter()
    rows = {"tracking": [], "standing": [], "push": [], "goto": []}
    for seed in cfg["seeds"]:
        rows["tracking"] += tracking(seed, cfg)
        rows["standing"].append(standing(seed, cfg))
        rows["push"] += [push(seed, cfg, f) for f in cfg["pushes_n"]]
        rows["goto"] += goto(seed, cfg)
        print(f"seed {seed} done ({time.perf_counter() - t_start:.0f} s)", flush=True)
    summary = summarise(rows, cfg)
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    meta = {
        "config": cfg,
        "git_sha": sha,
        "dream_dirty_at_start": dirty,
        "mujoco": mujoco.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "machine": platform.platform(),
        "wall_s": time.perf_counter() - t_start,
        "odometry": "oracle-assisted (true increments + noise), dream/sim/g1_mobile.py",
    }
    (out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    print(json.dumps(summary, indent=2))
    if args.record:
        record(cfg, sha)


def record(cfg: dict, sha: str) -> None:
    """Record one go-to episode (the first seed) and render it from the saved states."""
    from experiments.mobile_recording import MobileRecorder, render

    root = pathlib.Path(os.environ.get("DREAM_RECORDINGS_ROOT", "/tmp/DREAM/recordings"))
    rec = MobileRecorder(
        _env(cfg["seeds"][0], camera=True)[0],
        root / f"g1_locomotion_goto_{time.strftime('%Y%m%dT%H%M%S')}",
        {"experiment": "g1_locomotion go-to", "git_sha": sha, "seed": cfg["seeds"][0]},
    )
    result = goto(cfg["seeds"][0], cfg, recorder=rec)
    rec.save({"goto": result})
    print(json.dumps(render(rec.root), indent=2))


def summarise(rows: dict, cfg: dict) -> dict:
    tr = rows["tracking"]
    fwd = [r["forward_m_s"] / r["command"][0] for r in tr if r["command"][0] and not r["fell"]]
    yaw = [r["yaw_rad_s"] / r["command"][2] for r in tr if r["command"][2] and not r["fell"]]
    per_profile = {}
    for r in tr:
        k = str(r["command"])
        per_profile.setdefault(k, []).append(r)
    profile = {
        k: {
            "falls": sum(r["fell"] for r in v),
            "forward_m_s_median": float(np.median([r["forward_m_s"] for r in v])),
            "lateral_m_s_median": float(np.median([r["lateral_m_s"] for r in v])),
            "yaw_rad_s_median": float(np.median([r["yaw_rad_s"] for r in v])),
        }
        for k, v in per_profile.items()
    }
    gt = rows["goto"]
    g = cfg["gates"]
    goto_ok = [
        r for r in gt if r["status"] == "arrived" and r["true_error_m"] < g["goto_error_m_max"]
    ]
    pushes = {
        f: [r["recovered"] for r in rows["push"] if r["force_n"] == f] for f in cfg["pushes_n"]
    }
    falls = sum(r["fell"] for r in tr) + sum(r["fell"] for r in rows["standing"])
    falls += sum(r["fell"] for r in gt)
    drift = max(r["drift_m"] for r in rows["standing"])
    measured = {
        "falls_nominal": falls,
        "forward_tracking_ratio_median": float(np.median(fwd)),
        "yaw_tracking_ratio_median": float(np.median(yaw)),
        "standing_drift_m_max": drift,
        "goto_success": len(goto_ok) / len(gt),
        "goto_true_error_m_median": float(np.median([r["true_error_m"] for r in gt])),
        "push_recovery": {str(f): sum(v) / len(v) for f, v in pushes.items()},
    }
    lo_f, hi_f = g["forward_tracking_ratio_median"]
    lo_y, hi_y = g["yaw_tracking_ratio_median"]
    gates = {
        "falls_nominal": measured["falls_nominal"] <= g["falls_nominal_max"],
        "forward_tracking": lo_f <= measured["forward_tracking_ratio_median"] <= hi_f,
        "yaw_tracking": lo_y <= measured["yaw_tracking_ratio_median"] <= hi_y,
        "standing_drift": drift <= g["standing_drift_m_max"],
        "goto_success": measured["goto_success"] >= g["goto_success_min"],
        "push_40n": measured["push_recovery"]["40"] >= g["push_40n_recovery_min"],
    }
    return {"measured": measured, "per_profile": profile, "gates": gates}


if __name__ == "__main__":
    main()
