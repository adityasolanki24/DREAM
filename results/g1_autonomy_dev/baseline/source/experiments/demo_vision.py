"""Recorded live demo of camera-based manipulation: "in red tray", then "on blue green".

    python experiments/demo_vision.py [--seed 9100]

The live session runs with ``perception="vision"``: every object and the tray are located from
RGB-D frames, and goals are verified from new frames. The demo pauses during the first goal,
checks that the robot holds still, and resumes. Logs and a replay GIF (overview, camera view
with the logged detections, status panel) go to results/demo_vision/.
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

from dream.eval.manipulation import object_in_tray, object_on_top  # noqa: E402
from dream.session import LiveSession  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9100)
    args = ap.parse_args()
    out = ROOT / "results" / "demo_vision"
    s = LiveSession(seed=args.seed, rate=4.0, log_dir=out, perception="vision")
    s.start()
    results = []
    for i, goal in enumerate(("in red tray", "on blue green")):
        assert s.submit(goal) == "", goal
        if i == 0:
            time.sleep(3.0)
            s.pause()
            q = s.env.robot_state().joint_pos
            time.sleep(1.0)
            drift = float(np.max(np.abs(s.env.robot_state().joint_pos - q)))
            assert s.resume() == ""
        s.wait(900)
        time.sleep(0.5)
        truth = object_in_tray(s.env, "red") if i == 0 else object_on_top(s.env, "blue", "green")
        results.append(
            {
                "goal": goal,
                "state": s.status.state,
                "message": s.status.message,
                "true_in_world": truth,
                "camera_frames": s.perception.frames,
            }
        )
    s.close()
    summary = {
        "seed": args.seed,
        "perception": "vision",
        "pause_joint_drift_rad": drift,
        "goals": results,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "replay.py"), str(out), "--every", "4"],
        check=True,
    )


if __name__ == "__main__":
    main()
