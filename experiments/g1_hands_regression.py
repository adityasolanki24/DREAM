"""Does adding the 14 Dex3 finger servos change the validated G1 locomotion?

    python experiments/g1_hands_regression.py experiments/configs/g1_hands_regression.json

Reruns the acceptance protocol of experiments/g1_locomotion.py (its own functions; only the
environment factory is swapped) on the 29-actuator model (``no_hands``), on the 43-actuator
model with fingers held open and with fingers cycling over their full range while the robot
walks, on identical seeds. The body inertias are identical by construction (tests/test_g1_hands.py),
so a difference can only come from finger dynamics and contacts. Gates are in the config.
Ground truth (pelvis pose, falls) is read only here, as the evaluator.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import g1_locomotion as loco  # noqa: E402

from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.sim import g1_mobile  # noqa: E402
from dream.sim.g1_mobile import G1MobileEnv, build_mobile_g1  # noqa: E402

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
# One record per environment: policy ticks, and ticks with a non-foot robot geom touching the
# floor (evaluator-side, owner correction C3).
CONTACT_LOG: list[dict] = []


def undesired_floor_contact(model: mujoco.MjModel, data: mujoco.MjData, feet: set[int]) -> bool:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for c in data.contact[: data.ncon]:
        pair = {int(c.geom1), int(c.geom2)}
        if floor in pair and not (pair - {floor}) <= feet:
            return True
    return False


def make_env_factory(hands: bool, cycle_hz: float):
    """Same start state, joint offsets and RNG draws as ``g1_locomotion._env``; fingers added."""

    def factory(seed: int, camera: bool = False):
        rng = np.random.default_rng(seed)
        kw = {} if camera else {"head_camera": None}
        env = G1MobileEnv(build_mobile_g1(hands=hands, **kw), rng, **kw)
        env.reset(0.0, 0.0, float(rng.uniform(-np.pi, np.pi)))
        env.data.qpos[env.qadr] += rng.normal(scale=0.02, size=len(env.qadr))  # scene setup
        mujoco.mj_forward(env.model, env.data)
        body_ids = [env.model.body(b).id for b in FOOT_BODIES]
        feet = {int(g) for b in body_ids for g in np.flatnonzero(env.model.geom_bodyid == b)}
        log = {"ticks": 0, "undesired": 0}
        CONTACT_LOG.append(log)
        plain_step = env.step

        def logged(controller, env=env, log=log, feet=feet):
            plain_step(controller)
            log["ticks"] += 1
            log["undesired"] += undesired_floor_contact(env.model, env.data, feet)

        env.step = logged
        if hands and cycle_hz > 0:
            lo, hi = env.hand_range.T
            phase = np.linspace(0, 2 * np.pi, len(lo), endpoint=False)  # fingers out of step
            step = env.step

            def cycling(controller):
                env.set_hand(
                    lo + (hi - lo) * (0.5 - 0.5 * np.cos(2 * np.pi * cycle_hz * env.time + phase))
                )
                step(controller)

            env.step = cycling
        return env, WBCLocomotion.from_cache(), rng

    return factory


def run_protocol(cfg: dict, variant: dict, seeds: list[int]) -> dict:
    loco._env = make_env_factory(variant["hands"], variant.get("cycle_hz", 0.0))
    rows = {"tracking": [], "standing": [], "push": [], "goto": []}
    CONTACT_LOG.clear()
    t0 = time.perf_counter()
    for seed in seeds:
        rows["tracking"] += loco.tracking(seed, cfg)
        rows["standing"].append(loco.standing(seed, cfg))
        rows["push"] += [loco.push(seed, cfg, f) for f in cfg["pushes_n"]]
        rows["goto"] += loco.goto(seed, cfg)
        print(f"  seed {seed} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    rows["undesired_contact"] = {
        "environments": len(CONTACT_LOG),
        "environments_with_contact": sum(r["undesired"] > 0 for r in CONTACT_LOG),
        "ticks": sum(r["ticks"] for r in CONTACT_LOG),
        "undesired_ticks": sum(r["undesired"] for r in CONTACT_LOG),
    }
    return rows


SPEED_KEYS = ("forward_m_s", "lateral_m_s", "yaw_rad_s")


def commanded_axes(profile: str) -> list[int]:
    """Axes (forward, lateral, yaw) the profile commands; a profile like [0.3, 0, 0.3] has two."""
    return [i for i, c in enumerate(json.loads(profile)) if c]


def _inside(value: float, ref: list[float]) -> bool:
    return min(ref) <= value <= max(ref)


def compare(base: dict, other: dict, cfg: dict) -> dict:
    """Gates of one finger variant against the 29-actuator run."""
    g = cfg["locomotion"]["gates"]
    out: dict = {"tracking": {}, "gates": {}}
    profiles = sorted({str(r["command"]) for r in base["tracking"]})
    ok_tracking = True
    for p in profiles:
        for ax in commanded_axes(p):
            key = SPEED_KEYS[ax]
            b = [r[key] for r in base["tracking"] if str(r["command"]) == p and not r["fell"]]
            o = [r[key] for r in other["tracking"] if str(r["command"]) == p and not r["fell"]]
            inside = bool(o) and _inside(float(np.median(o)), b) and len(o) == len(b)
            ok_tracking &= inside
            out["tracking"][f"{p} {key}"] = {
                "base_median": float(np.median(b)), "base_min": min(b), "base_max": max(b),
                "median": float(np.median(o)) if o else None, "inside_spread": inside,
            }  # fmt: skip
    drift_b = [r["drift_m"] for r in base["standing"]]
    drift_o = [r["drift_m"] for r in other["standing"]]
    out["standing"] = {"base": drift_b, "variant": drift_o}
    falls = sum(r["fell"] for k in ("tracking", "standing", "goto") for r in other[k])
    base_falls = sum(r["fell"] for k in ("tracking", "standing", "goto") for r in base[k])
    push = {}
    for f in cfg["locomotion"]["pushes_n"]:
        push[str(f)] = {
            "base": sum(r["recovered"] for r in base["push"] if r["force_n"] == f),
            "variant": sum(r["recovered"] for r in other["push"] if r["force_n"] == f),
        }
    err_b = [r["true_error_m"] for r in base["goto"]]
    err_o = [r["true_error_m"] for r in other["goto"]]
    ok_goto = [r for r in other["goto"] if r["status"] == "arrived"]
    success = sum(r["true_error_m"] < g["goto_error_m_max"] for r in ok_goto) / len(other["goto"])
    out["undesired_contact"] = {
        "base": base["undesired_contact"],
        "variant": other["undesired_contact"],
    }
    out["falls"] = {"base": base_falls, "variant": falls}
    out["push"] = push
    out["goto"] = {
        "success": success, "error_median": float(np.median(err_o)),
        "base_error_min": min(err_b), "base_error_max": max(err_b),
    }  # fmt: skip
    out["gates"] = {
        "no_new_falls": falls <= base_falls + cfg["gates"]["new_falls_max"],
        "tracking_inside_no_hands_spread": ok_tracking,
        "standing_drift": _inside(float(np.median(drift_o)), drift_b)
        and max(drift_o) <= g["standing_drift_m_max"],
        "no_new_undesired_floor_contact": other["undesired_contact"]["environments_with_contact"]
        <= base["undesired_contact"]["environments_with_contact"],
        "push_recovery_not_lower": all(v["variant"] >= v["base"] for v in push.values()),
        "goto": success >= g["goto_success_min"]
        and _inside(float(np.median(err_o)), err_b),
    }  # fmt: skip
    out["pass"] = all(out["gates"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = json.loads(pathlib.Path(args.config).read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "dream", "tests"], cwd=ROOT)
    t_start = time.perf_counter()
    rows, summaries = {}, {}
    for name, variant in cfg["variants"].items():
        print(f"variant {name}", flush=True)
        rows[name] = run_protocol(cfg["locomotion"], variant, cfg["locomotion"]["seeds"])
        summaries[name] = loco.summarise(rows[name], cfg["locomotion"])
    comparison = {
        name: compare(rows["no_hands"], rows[name], cfg) for name in rows if name != "no_hands"
    }
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    meta = {
        "config": cfg,
        "git_sha": sha,
        "dream_dirty_at_start": bool(dirty),
        "mujoco": mujoco.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "machine": platform.platform(),
        "wall_s": time.perf_counter() - t_start,
        "finger_model": {
            "kp": g1_mobile.FINGER_KP,
            "kd": g1_mobile.FINGER_KD,
            "armature": g1_mobile.FINGER_ARMATURE,
            "frictionloss": g1_mobile.FINGER_FRICTIONLOSS,
        },  # fmt: skip
        "odometry": "oracle-assisted (true increments + noise), dream/sim/g1_mobile.py",
    }
    summary = {"meta": meta, "per_variant": summaries, "comparison": comparison}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item()))
    print(json.dumps({"comparison": comparison}, indent=2, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
