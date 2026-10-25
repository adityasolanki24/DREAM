"""Recorded live demo: "in red compartment" with a hidden drawer, lid or fixed box.

    python experiments/demo_compartment.py [--seed 7000] [--kinds drawer,lid,fixed]

Each kind runs in its own live session (background physics thread plus agent thread, exactly
as in experiments/session.py). After the second probe the demo pauses the session: the
current tool is interrupted and the robot holds its pose with the knob in hand. It checks the
hold, then resumes, and the agent replans from the observed state. Logs and a replay GIF with
the hypothesis panel are written to results/demo_compartment/<kind>/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.manipulation import compartment_opening, object_in_compartment  # noqa: E402
from dream.session import LiveSession  # noqa: E402

PAUSE_AFTER_PROBE = 2
HOLD_CHECK_S = 1.0


def run(kind: str, seed: int, out: pathlib.Path) -> dict:
    s = LiveSession(seed=seed, rate=4.0, log_dir=out, scene=kind)
    s.start()
    assert s.submit("in red compartment") == ""
    deadline = time.time() + 600
    while time.time() < deadline and s.busy:
        d = s.status.discovery
        if d is not None and d["probe"] >= PAUSE_AFTER_PROBE:
            break
        time.sleep(0.05)
    pause = {}
    if s.busy:
        s.pause()
        q0 = s.env.robot_state().joint_pos
        time.sleep(HOLD_CHECK_S)
        drift = float(np.max(np.abs(s.env.robot_state().joint_pos - q0)))
        pause = {"paused_after_probe": PAUSE_AFTER_PROBE, "joint_drift_while_paused_rad": drift}
        assert s.resume() == ""
    s.wait(900)
    time.sleep(0.5)
    result = {
        "kind": kind,
        "seed": seed,
        "state": s.status.state,
        "message": s.status.message,
        "cube_in_compartment_ground_truth": object_in_compartment(s.env, "red"),
        "opening_true": compartment_opening(s.env),
        "tool_calls": [e for e in s.status.events if e["kind"] in ("action", "result")],
        **pause,
    }
    s.close()
    (out / "result.json").write_text(json.dumps(result, indent=2))
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "replay.py"), str(out), "--every", "3"],
        check=True,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--kinds", default="drawer,lid,fixed")
    args = ap.parse_args()
    for kind in args.kinds.split(","):
        r = run(kind, args.seed, ROOT / "results" / "demo_compartment" / kind)
        shown = {k: v for k, v in r.items() if k != "tool_calls"}
        print(json.dumps(shown))


if __name__ == "__main__":
    main()
