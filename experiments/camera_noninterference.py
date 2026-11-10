"""Does camera delivery disturb the real MuJoCo G1 control loop?

Same seed, same room, same walk script, same duration, three arms:
  off      no head-camera capture, no stream
  capture  render at 10 Hz of sim time and submit to CameraStream (encode, no client)
  on       capture, PhoneAPI and headless Chrome on the app's MJPEG stream
We time every 50 Hz policy tick from G1MobileBase's on_tick hook. Without real-time pacing,
wall time includes the control tick and any time taken by camera work.

Run: python -m experiments.camera_noninterference --out results/camera_delivery_closeout_20261106
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
import time
from pathlib import Path

import numpy as np

from dream.robot.g1_wbc import WBCLocomotion
from dream.runtime.camera_stream import CameraStream
from dream.sim.g1_mobile import G1MobileBase, G1MobileEnv, build_mobile_g1
from dream.sim.room import add_room, sample_room
from experiments import camera_latency_measure as m

SEED = 4100
CAPTURE_EVERY = 5  # ticks: 50 Hz / 5 = 10 Hz


def pct(a, q):
    return float(np.percentile(a, q))


def run_arm(arm, seed, seconds, args):
    room = sample_room(np.random.default_rng([seed, 0]))
    env = G1MobileEnv(
        build_mobile_g1(scene=lambda s: add_room(s, room)), np.random.default_rng([seed, 1])
    )
    env.reset(*room.start)
    ticks = []
    incl, caps = [], []
    state = {"last": None, "pre": None, "n": 0, "stream": None, "rig": None}
    if arm != "off":
        if arm == "on":
            rig = m.Rig(
                args.token, args.raw_root, 10
            )  # source thread stays idle: the sim is the source
            state["rig"], state["stream"] = rig, rig.stream
        else:
            state["stream"] = CameraStream()

    def on_tick():
        now = time.perf_counter()
        if state["last"] is not None:
            ticks.append(
                now - state["last"]
            )  # control work only (policy + physics), capture excluded
            incl.append(now - state["pre"])  # what the loop experiences, capture stall included
        state["n"] += 1
        state["pre"] = now
        if arm != "off" and state["n"] % CAPTURE_EVERY == 0:
            c0 = time.perf_counter()
            frame = env.capture_head()  # render on the sim thread, as the real session does
            caps.append(time.perf_counter() - c0)
            state["stream"].submit(frame)
        state["last"] = time.perf_counter()

    base = G1MobileBase(env, WBCLocomotion.from_cache(), on_tick=on_tick)

    proc = None
    if arm == "on":
        import subprocess

        rig = state["rig"]
        cfg = {
            "chrome": m.CHROME,
            "token": rig.token,
            "base": f"http://127.0.0.1:{rig.port}",
            "ctl": f"http://127.0.0.1:{rig.ctl_port}",
            "out": str(args.out / "_ni_on.raw.json"),
            "client": "raw",
            "mode": "push",
            "scenario": "latency",
            "duration_s": 135,  # allow for the walk with capture at roughly 1 fps wall time
            "first_paint_timeout_ms": 60000,
        }
        cfg_path = args.out / "_ni_on.cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        proc = subprocess.Popen(
            ["node", str(m.ROOT / "experiments/camera_latency_driver.cjs"), str(cfg_path)],
            env={**os.environ, "DREAM_PLAYWRIGHT_CORE": m.PLAYWRIGHT},
            stdout=(args.out / "_ni_on.driver.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        # Prime the stream with frames so the browser can attach.
        t0 = time.time()
        while time.time() - t0 < 60 and rig.api._media_slots._value == 2:
            state["stream"].submit(env.capture_head()) if env._camera else None
            time.sleep(0.2)
        time.sleep(2)
    cpu0, w0 = time.process_time(), time.perf_counter()
    t_sim = 0.0
    ok = True
    while t_sim < seconds and ok:
        ok = base.walk(0.3, 0.0, 0.15, 1.0)
        t_sim += 1.0
    wall = time.perf_counter() - w0
    client_attached_at_end = (
        None if state["rig"] is None else state["rig"].api._media_slots._value == 1
    )
    cpu = time.process_time() - cpu0
    a = np.array(ticks) * 1000
    result = {
        "arm": arm,
        "seed": seed,
        "client_attached_at_end": client_attached_at_end,
        "sim_seconds": t_sim,
        "completed_without_fall": ok,
        "stop_reason": base.stop_reason,
        "ticks": len(a),
        "wall_s": round(wall, 2),
        "process_cpu_s": round(cpu, 2),
        "tick_ms": {
            "median": round(pct(a, 50), 3),
            "p95": round(pct(a, 95), 3),
            "p99": round(pct(a, 99), 3),
            "max": round(float(a.max()), 3),
            "mean": round(float(a.mean()), 3),
            "stdev": round(float(a.std()), 3),
        },
        "final_true_pose": [round(float(v), 4) for v in env.true_pose()],
        "ticks_over_20ms_pct": round(100 * float((a > 20).mean()), 2),
    }
    b = np.array(incl) * 1000
    result["tick_interval_including_capture_stall_ms"] = {
        "median": round(pct(b, 50), 3),
        "p95": round(pct(b, 95), 3),
        "p99": round(pct(b, 99), 3),
        "max": round(float(b.max()), 3),
        "mean": round(float(b.mean()), 3),
    }
    result["tick_intervals_over_20ms_including_capture_pct"] = round(
        100 * float((b > 20).mean()), 2
    )
    if caps:
        c = np.array(caps) * 1000
        result["head_capture_ms"] = {
            "n": len(c),
            "median": round(pct(c, 50), 1),
            "p95": round(pct(c, 95), 1),
            "max": round(float(c.max()), 1),
        }
    if arm != "off":
        result["frames_submitted"] = state["stream"].metadata()["submitted_frames"]
        result["frames_encoded"] = state["stream"].metadata()["encoded_frames"]
        result["frames_dropped_at_encoder"] = state["stream"].metadata()["dropped_frames"]
    np.save(args.out / f"noninterference_ticks_{arm}.npy", a.astype(np.float32))
    if proc is not None:
        try:
            proc.wait(timeout=200)
        except Exception:
            proc.kill()
        raw_path = args.out / "_ni_on.raw.json"
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
            result["browser_frames_painted"] = len(raw.get("paints", []))
        state["rig"].close()
    elif arm == "capture":
        state["stream"].close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    args.token = secrets.token_urlsafe(32)
    args.raw_root = Path(tempfile.mkdtemp(prefix="dream_camera_raw_"))
    parser_src = (m.ROOT / "web/app/camera-client.mjs").read_text().replace("export class", "class")
    (args.raw_root / "index.html").write_text(
        (m.ROOT / "experiments/camera_latency_raw.html")
        .read_text()
        .replace("/*PARSER*/", parser_src),
        encoding="utf-8",
    )
    results = []
    # interleave arms so thermal / background-load drift does not line up with one arm
    for rep in range(args.repeats):
        for arm in ("off", "capture", "on"):
            r = run_arm(arm, SEED, args.seconds, args)
            r["repeat"] = rep
            results.append(r)
            print(json.dumps(r), flush=True)
    (args.out / "noninterference.json").write_text(
        json.dumps({"seed": SEED, "capture_every_ticks": CAPTURE_EVERY, "runs": results}, indent=2)
    )


if __name__ == "__main__":
    main()
