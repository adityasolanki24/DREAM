"""Hypothesis test for the shoulder-singularity stall and the armature change that removed it.

    python experiments/stress_shoulder_singularity.py [--n 60] [--workers 4]

Observed failure (development scene, drawer seed 2): with joint2 ~ 0, joints 1 and 3 are
coaxial. Their opposing rotation moves only links 2-3, and it fell into a limit cycle: both
actuators saturated at +-87 N m in opposite directions at ~0.7 rad/s, and the hand stalled
0.46 m short of its target. Raising the armature of joints 1-4 from Menagerie's 0.1 to
0.3 kg m^2 removed it in that case.

Hypotheses tested here, on motions *not* used to make the change:
  H1  the stall is reproducible beyond the one observed case with armature 0.1, i.e. it is a
      property of the model, not of one trajectory;
  H2  with 0.3 it does not occur on the same motions (the change is not a one-case patch);
  H3  the result is not knife-edge: 0.2 and 0.5 behave like 0.3, or the dependence is reported.
Motions start folded near the base (where pulled-out drawers leave the arm) and travel through
the travel height to targets across the workspace, the path on which the stall was seen.
A motion "stalls" if the move times out more than 2 cm from its target while joints 1 and 3
are saturated in opposite directions. Results: results/shoulder_singularity_stress/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dream.skills.manipulation as M  # noqa: E402
from dream.sim.tabletop import TabletopSpec  # noqa: E402
from dream.sim.tabletop_env import TabletopEnv  # noqa: E402

ARMATURES = (0.1, 0.2, 0.3, 0.5)
SEED0 = 300  # disjoint from development (0-39), held-out pick/place (1000+) and task (2000+) seeds


def _polar(rng, r, z, bearing=0.6):
    a = rng.uniform(-bearing, bearing)
    rr = rng.uniform(*r)
    return np.array([rr * np.cos(a), rr * np.sin(a), rng.uniform(*z)])


def motion(args: tuple) -> dict:
    i, armature = args
    rng = np.random.default_rng([SEED0, i])
    start = _polar(rng, (0.30, 0.42), (0.05, 0.16))
    goal = _polar(rng, (0.45, 0.70), (0.04, 0.25))
    yaw0, yaw1 = rng.uniform(-1.5, 1.5, size=2)
    spec = TabletopSpec(objects=(), tray=None)  # empty table: only the arm's own dynamics
    env = TabletopEnv(spec, np.random.default_rng([SEED0, i, 2]), shoulder_armature=armature)
    rb = env.embodiment()
    first = M.move_to(rb, start, float(yaw0))
    r = M.move_via_travel_height(rb, goal, float(yaw1))
    f = env.data.qfrc_actuator[env.robot.dadr]
    opposed = bool(abs(f[0]) > 80 and abs(f[2]) > 80 and np.sign(f[0]) != np.sign(f[2]))
    err = float(np.linalg.norm(env.robot_state().tcp_pos - goal))
    return {
        "motion": i,
        "armature": armature,
        "reached_start": first.ok,
        "start_q2": float(env.robot_state().joint_pos[1]),
        "ok": r.ok,
        "reason": r.reason,
        "final_error_m": err,
        "stall": (not r.ok) and err > 0.02 and opposed,
        "protective_stop": env.robot.protective_stop,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    jobs = [(i, a) for a in ARMATURES for i in range(args.n)]
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(motion, jobs))
    summary = {}
    for a in ARMATURES:
        rs = [r for r in runs if r["armature"] == a]
        summary[str(a)] = {
            "n": len(rs),
            "reached": sum(r["ok"] for r in rs),
            "stalls": sum(r["stall"] for r in rs),
            "other_failures": sum((not r["ok"]) and not r["stall"] for r in rs),
            "protective_stops": sum(r["protective_stop"] for r in rs),
            "failed_motions": [r["motion"] for r in rs if not r["ok"]],
        }
    out = ROOT / "results" / "shoulder_singularity_stress"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"git_sha": sha, "mujoco": mujoco.__version__, "n": args.n, "seed0": SEED0}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    with open(out / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
