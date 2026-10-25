"""Spectator evidence and offline encoding; never imported by robot decision code.

Original PNGs and timestamped score telemetry remain alongside the annotated MP4.
Encoding is separate from physics so an encoder failure cannot erase a robot outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class FrameRecorder:
    """Save unedited RGB pixels with strictly increasing simulation timestamps."""

    def __init__(self, out: Path):
        self.out = out
        (out / "originals").mkdir(parents=True, exist_ok=False)
        self.count, self.last_time = 0, -float("inf")

    def capture(self, rgba: np.ndarray, time_s: float, telemetry: dict) -> None:
        if (
            rgba.ndim != 3
            or rgba.shape[2] not in (3, 4)
            or min(rgba.shape[:2]) < 2
            or rgba.dtype != np.uint8
            or not np.isfinite(time_s)
            or time_s <= self.last_time
        ):
            raise ValueError("invalid spectator frame or timestamp")
        path = self.out / "originals" / f"{self.count:06}.png"
        Image.fromarray(rgba[..., :3]).save(path)
        record = {
            **telemetry,
            "index": self.count,
            "time_s": time_s,
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with (self.out / "frames.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
        self.count += 1
        self.last_time = time_s


def encode(folder: Path, *, mock: bool = False) -> dict:
    """Encode measured, uniformly sampled footage; no invented or padded motion."""
    records = [json.loads(s) for s in (folder / "frames.jsonl").read_text().splitlines()]
    if len(records) < 2:
        raise ValueError("at least two recorded frames required")
    dt = np.diff([r["time_s"] for r in records])
    if not np.isfinite(dt).all() or np.any(dt <= 0) or not np.allclose(dt, dt[0], atol=1e-6):
        raise ValueError("nonuniform footage requires an explicit timestamp edit")
    episode = json.loads((folder / "episode.json").read_text()) if not mock else {}
    source = json.loads((folder / "source.json").read_text()) if not mock else {}
    if not mock and (folder.parent / "failure.json").exists():
        raise ValueError("failed execution must retain its failure report, not a success video")
    if not mock and (not episode.get("physics_executed") or not episode.get("real_actor_executed")):
        raise ValueError("real physics and actor evidence required")
    label = (
        "MOCK - ENCODER TEST, NO ROBOT PHYSICS"
        if mock
        else (
            "DREAM | Isaac Sim 5.1 | PPO locomotion | "
            + ("GATES PASSED" if episode.get("gate_pass") else "FAILED / DEVELOPMENT")
        )
    )
    edited = folder / "annotated"
    edited.mkdir(exist_ok=False)
    size = None
    for i, row in enumerate(records):
        frame_path = folder / "originals" / row["file"]
        if (
            row["index"] != i
            or hashlib.sha256(frame_path.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError("frame evidence changed")
        with Image.open(frame_path) as image:
            image = image.convert("RGB")
            if size is not None and image.size != size:
                raise ValueError("frame resolution changed")
            size = image.size
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, size[0], min(95, size[1])), fill=(12, 18, 28))
            text = (
                f"{label}\nt={row['time_s']:.2f}s | command={row.get('command')}"
                f"\nposition={row.get('position_m')} | speed={row.get('speed_m_s')} m/s"
                f" | {row.get('phase', '')}"
            )
            draw.text((12, 8), text, fill="white", font_size=max(10, round(size[0] / 80)))
            image.save(edited / f"{i:06}.png")
    movie = folder / ("mock_encoder_test.mp4" if mock else "dream_t1_demo.mp4")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-n",
        "-framerate",
        str(1 / dt[0]),
        "-i",
        str(edited / "%06d.png"),
        "-vf",
        "fps=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie),
    ]
    subprocess.run(command, check=True, timeout=180)
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(movie)],
            text=True,
            timeout=30,
        )
    )["streams"][0]
    if probe["codec_name"] != "h264" or int(probe["nb_read_frames"]) < 2:
        raise RuntimeError("encoded video validation failed")
    metadata = {
        "mock": mock,
        "original_frame_count": len(records),
        "output_fps": 30,
        "source_sample_period_s": float(dt[0]),
        "speed_change": 1.0,
        "physical_gate_pass": episode.get("gate_pass", False),
        "source": source,
        "stream": probe,
        "movie_sha256": hashlib.sha256(movie.read_bytes()).hexdigest(),
    }
    (folder / "demo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def spectator_camera():
    """Fixed external USD camera; its pixels and score state never reach the actor."""
    from isaacsim.sensors.camera import Camera
    from pxr import Gf

    position = (3.0, -4.0, 2.2)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(0, 0, 0.7), Gf.Vec3d(0, 0, 1))
    quat = view.GetInverse().ExtractRotationQuat()
    camera = Camera(prim_path="/World/dream_spectator", resolution=(1920, 1080))
    camera.initialize()
    camera.set_world_pose(
        position=np.array(position),
        orientation=np.array([quat.GetReal(), *quat.GetImaginary()]),
        camera_axes="usd",
    )
    return camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    encode(args.folder, mock=args.mock)
