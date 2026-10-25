"""Live session: the simulated walking G1 takes tasks from the phone page.

    python experiments/phone_session.py --seed 4100 [--host 127.0.0.1] [--port 8765]

Open http://<host>:<port>/ and paste the token from the file this prints (the token itself
is never printed or logged). Requests are structured goals for now ("inspect cabinet",
"goto entrance"); anything else is rejected with the reason. Higher-priority requests
preempt the running task at a safe boundary; the preempted task resumes after it. Binding
to anything but localhost needs --allow-lan and must stay on a private network or VPN.
ODOMETRY IS ORACLE-ASSISTED (MuJoCo, procedural room).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

if os.name != "nt" and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.perception.spatial import SpatialMemory  # noqa: E402
from dream.robot.g1_wbc import WBCLocomotion  # noqa: E402
from dream.runtime.agent import TaskAgent  # noqa: E402
from dream.runtime.phone_api import PhoneAPI, load_or_create_token, token_file  # noqa: E402
from dream.runtime.scheduler import TaskScheduler  # noqa: E402
from dream.runtime.search import PlacePlanner, RoomSearchPlanner  # noqa: E402
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1  # noqa: E402
from dream.sim.room import add_room, sample_room  # noqa: E402
from dream.tools.goals import GoToPlace, parse_goal  # noqa: E402
from dream.tools.mobile import MobileToolbox  # noqa: E402
from experiments.g1_explore_room import Budgeted  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-lan", action="store_true")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost") and not args.allow_lan:
        raise SystemExit("refusing a non-local bind without --allow-lan (use a private network)")
    room = sample_room(np.random.default_rng([args.seed, 0]))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([args.seed, 1])
    )
    env.reset(*room.start)
    base = G1MobileBase(env, WBCLocomotion.from_cache())
    tb = MobileToolbox(base, SpatialMemory(np.random.default_rng([args.seed, 2])), navigator="rpp")
    tb.places = {"entrance": np.zeros(2)}

    def run_task(task, history, preempt, cancel):
        tb.preempt, tb.cancel = preempt, cancel
        planner = (PlacePlanner() if isinstance(task.goal, GoToPlace)
                   else Budgeted(RoomSearchPlanner("gain"), base, base.time + 240))  # fmt: skip
        return TaskAgent(tb, planner, max_steps=60).run(task.goal, history=history, preempt=preempt)

    store = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home())) / "DREAM/phone"
    store.mkdir(parents=True, exist_ok=True)
    sched = TaskScheduler(str(store / "tasks.sqlite"), run_task, parse_goal)
    api = PhoneAPI(sched, load_or_create_token(), status=lambda: {
        "robot_time_s": round(base.time, 1), "odometry": base.odometry().round(2).tolist(),
        "stop_reason": base.stop_reason})  # fmt: skip
    sched.notify = api.notify
    api.serve(args.host, args.port)
    print(f"serving http://{args.host}:{args.port}/ ; token file: {token_file()}", flush=True)
    while True:
        if sched.step() is None:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
