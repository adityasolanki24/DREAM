"""Record a mobile-G1 episode as it runs; render video from the saved states afterwards.

The same rules as experiments/g1_recording.py: integration states are copied from the
running simulation at 25 Hz together with timestamped decisions, and footage is rendered
offline on a separate MjData with mj_setState + mj_forward. Nothing is re-simulated, edited,
interpolated or sped up. Large files go outside the repository (external_recording_root).

    python experiments/mobile_recording.py --render <recording dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.g1_recording import digest, external_recording_root  # noqa: E402

FPS = 25  # 40 ms: exactly two 50 Hz policy ticks, so frames are evenly spaced in time
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


class MobileRecorder:
    def __init__(self, env, root: str | Path, metadata: dict) -> None:
        self.env = env
        self.root = external_recording_root(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.period = round(1 / (FPS * env.model.opt.timestep))
        self.states: list[np.ndarray] = []
        self.times: list[float] = []
        self.events: list[dict] = []
        (self.root / "metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "fps": FPS,
                    "mujoco": mujoco.__version__,
                    "method": "actual episode states, offline rendering only, original speed",
                },
                indent=2,
            )
        )
        mujoco.mj_saveModel(env.model, str(self.root / "scene.mjb"))
        self._last_step = -1

    def sample(self) -> None:
        step = round(self.env.data.time / self.env.model.opt.timestep)
        if step % self.period or step == self._last_step:
            return
        if self._last_step >= 0 and step - self._last_step != self.period:
            raise ValueError("recording skipped a frame: sample() must be called every tick")
        state = np.empty(mujoco.mj_stateSize(self.env.model, STATE))
        mujoco.mj_getState(self.env.model, self.env.data, state, STATE)
        self.states.append(state)
        self.times.append(float(self.env.data.time))
        self._last_step = step

    def event(self, kind: str, text: str, **data) -> None:
        """A decision, action or outcome at the current simulation time."""
        t = float(self.env.data.time)
        self.events.append({"time_s": t, "kind": kind, "text": text, **data})

    def save(self, summary: dict) -> Path:
        np.savez_compressed(
            self.root / "states.npz", integration_state=np.array(self.states), time_s=self.times
        )
        with open(self.root / "events.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e, default=_jsonable) + "\n")
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
        return self.root


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x))


def render(root: str | Path, width: int = 960, height: int = 540) -> dict:
    """Spectator view and head-camera view side by side, with the decision log as captions."""
    root = external_recording_root(root)
    model = mujoco.MjModel.from_binary_path(str(root / "scene.mjb"))
    data = mujoco.MjData(model)  # rendering only: never stepped
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    with np.load(root / "states.npz") as a:
        states, times = a["integration_state"], a["time_s"]
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    wide = mujoco.Renderer(model, height, width)
    head = mujoco.Renderer(model, 240, 320) if _has_camera(model, "head_rgbd") else None
    out = root / "episode.mp4"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    cmd += ["-s", f"{width + 320}x{height}", "-r", str(FPS), "-i", "-"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for state, t in zip(states, times, strict=True):
        mujoco.mj_setState(model, data, state, STATE)
        mujoco.mj_forward(model, data)
        wide.update_scene(data, camera="spectator")
        canvas = Image.new("RGB", (width + 320, height), (16, 16, 20))
        canvas.paste(Image.fromarray(wide.render()), (0, 0))
        if head is not None:
            head.update_scene(data, camera="head_rgbd")
            canvas.paste(Image.fromarray(head.render()), (width, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((width + 8, 248), "robot head camera (RGB)", fill=(220, 220, 220))
        recent = [e for e in events if e["time_s"] <= t][-12:]
        for i, e in enumerate(recent):
            line = f"{e['time_s']:6.1f}s {e['kind']}: {e['text']}"[:52]
            draw.text((width + 8, 272 + 20 * i), line, fill=(200, 230, 200))
        draw.text((8, 8), f"t = {t:6.2f} s (simulation time, real speed)", fill=(255, 255, 255))
        enc.stdin.write(np.asarray(canvas).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    result = {"video": str(out), "sha256": digest(out), "frames": len(states)}
    (root / "recording.json").write_text(json.dumps(result, indent=2))
    return result


def _has_camera(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    print(json.dumps(render(ap.parse_args().render), indent=2))
