"""Interactive DREAM session: a live Panda tabletop that accepts goals while it runs.

    python experiments/session.py [--scene tray|drawer|lid|fixed] [--perception oracle|vision]
                                  [--seed 0] [--viewer] [--rate 1.0]

Commands (goals are structured; natural language is not supported yet):
    in <object> <region>     e.g. "in red tray", or "in red compartment" in a compartment scene
    on <object> <support>    e.g. "on red blue"
    status                   goal, current tool call, last result, robot state
    scene                    current observations (tracked object poses, tray, held object)
    pause | resume | cancel  interrupt the current goal (the robot holds still) / continue / drop
    reset-safety             clear a latched protective stop
    quit                     stop, and write logs to results/sessions/<time>/

--viewer opens MuJoCo's live viewer (needs a display). Without it the session is headless.
Replay a saved session with experiments/replay.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.session import LiveSession  # noqa: E402


def _status(s: LiveSession) -> str:
    st = s.env.robot_state()
    held = s.toolbox.held_object()
    return (
        f"goal: {s.status.goal}  state: {s.status.state}  {s.status.message}\n"
        f"current action: {s.status.current_action}\nlast result: {s.status.last_result}\n"
        f"robot: tcp {st.tcp_pos.round(3).tolist()} gripper {st.gripper_width * 1000:.0f} mm "
        f"holding {held} protective_stop {st.protective_stop} {st.stop_reason}\n"
        f"sim time {s.env.time:.1f} s"
        + (f"\ndiscovery: {s.status.discovery}" if s.status.discovery else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="tray", choices=["tray", "drawer", "lid", "fixed"])
    ap.add_argument("--perception", default="oracle", choices=["oracle", "vision"])
    ap.add_argument(
        "--rate", type=float, default=1.0, help="simulated seconds per wall second (0 = max)"
    )
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    log_dir = ROOT / "results" / "sessions" / time.strftime("%Y%m%d-%H%M%S")
    s = LiveSession(
        seed=args.seed,
        rate=args.rate,
        viewer=args.viewer,
        log_dir=log_dir,
        scene=args.scene,
        perception=args.perception,
    )
    s.start()
    print(__doc__.split("Commands")[0].strip().splitlines()[0])
    print("Objects:", [o["name"] for o in s.scene()["objects"]], "| type 'help' for commands")
    last_events = 0
    try:
        for line in sys.stdin:
            cmd = line.strip()
            if s.status.events[last_events:]:
                for e in s.status.events[last_events:]:
                    print("  [event]", e)
                last_events = len(s.status.events)
            if not cmd:
                continue
            if cmd in ("quit", "exit"):
                break
            if cmd == "help":
                print(__doc__)
            elif cmd == "status":
                print(_status(s))
            elif cmd == "scene":
                print(json.dumps(s.scene(), indent=1))
            elif cmd == "pause":
                s.pause()
                print("paused; robot holding.")
            elif cmd == "resume":
                print(s.resume() or "resumed")
            elif cmd == "cancel":
                s.cancel()
                print("cancelled; robot holding.")
            elif cmd == "reset-safety":
                s.reset_safety()
                print("protective stop cleared")
            else:
                err = s.submit(cmd)
                if not err:
                    last_events = 0  # a new goal starts a new event list
                print(err or f"accepted: {s.status.goal}")
    finally:
        s.close()
        print(f"logs written to {log_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
