"""Compose an Isaac run's replay frames, head-camera frames and event log into one MP4.

    python experiments/compose_demo_video.py --run <run dir> --replay <replay dir> --out x.mp4

Left: the chase camera from ``experiments/isaac_replay_video.py`` (10 fps of robot time).
Right: the most recent head RGB frame the robot actually captured (``frames/``; it captures
only when it looks, so this panel updates at those moments), and the last events with their
robot time: phone requests, scheduler transitions, agent actions, memory, and the
independent verification. Real speed; nothing is cut or re-timed.
"""

from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw

COLOURS = {"phone": (255, 210, 90), "verify": (120, 230, 140), "memory": (140, 200, 255),
           "task": (230, 150, 230)}  # fmt: skip


def colour(kind: str):
    for k, c in COLOURS.items():
        if kind.startswith(k):
            return c
    return (200, 200, 200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--replay", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--title", default="DREAM G1 in an NVIDIA Isaac Sim kitchen")
    args = ap.parse_args()
    chase = [json.loads(x) for x in (args.replay / "chase_frames.jsonl").read_text().splitlines()]
    heads = [json.loads(x) for x in (args.run / "frames.jsonl").read_text().splitlines()]
    head_t = [h["time_s"] for h in heads]
    events = [json.loads(x) for x in (args.run / "events.jsonl").read_text().splitlines()]
    events.sort(key=lambda e: e["time_s"])
    first = Image.open(args.replay / "chase" / f"{chase[0]['index']:05d}.jpg")
    w, h = first.size
    side = 420
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w + side}x{h}", "-r", "10", "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "22", str(args.out)]  # fmt: skip
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cache = {}
    for c in chase:
        t = c["time_s"]
        canvas = Image.new("RGB", (w + side, h), (14, 14, 18))
        canvas.paste(Image.open(args.replay / "chase" / f"{c['index']:05d}.jpg"), (0, 0))
        head_h = (side - 20) * 3 // 4
        k = bisect.bisect_right(head_t, t) - 1
        if k >= 0:
            if k not in cache:
                img = Image.open(args.run / "frames" / f"{heads[k]['index']:05d}.png")
                cache = {k: img.convert("RGB").resize((side - 20, head_h))}
            canvas.paste(cache[k], (w + 10, 26))
        d = ImageDraw.Draw(canvas)
        d.text((10, 8), f"{args.title}   t = {t:6.1f} s robot time (real speed)",
               fill=(255, 255, 255))  # fmt: skip
        d.text((w + 10, 8), "robot head camera (last capture)", fill=(220, 220, 220))
        y0 = 36 + head_h
        recent = [e for e in events if e["time_s"] <= t][-14:]
        for i, e in enumerate(recent):
            line = f"{e['time_s']:6.1f} {e['kind']}: {e['text']}"[:60]
            d.text((w + 10, y0 + 18 * i), line, fill=colour(e["kind"]))
        enc.stdin.write(np.asarray(canvas).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(json.dumps({"video": str(args.out), "frames": len(chase), "seconds": len(chase) / 10}))


if __name__ == "__main__":
    main()
