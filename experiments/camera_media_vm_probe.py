"""VM-side media diagnosis: NVENC throughput, MediaMTX publish, and the frozen-simulation stall.

Runs on the Isaac VM with no simulator and no model. It uses the production MediaRuntime
(dream/runtime/media_runtime.py) fed synthetic 320x240 frames, because the 31 Oct 2026 session
showed WHEP unavailable with 3 frames written and 434 dropped, and the cause has to be separated
from the simulator, the model and the network.

    python experiments/camera_media_vm_probe.py --out <dir>

Scenarios (each measured, nothing assumed):
  nvenc_throughput  ffmpeg h264_nvenc on 320x240 frames, as fast as it will encode;
  steady_10fps      MediaRuntime fed at 10 Hz wall clock for 30 s, MediaMTX status and log;
  stall_then_resume the live session's real pattern: a short burst, then no frames for 15 s (the
                    simulation is synchronous and stops while the model thinks), then frames again.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.runtime.media_runtime import MediaRuntime, encoder_command, install_binary  # noqa: E402

H, W = 240, 320


def sh(cmd, timeout=60):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {
        "cmd": cmd,
        "code": p.returncode,
        "stdout": p.stdout[-4000:],
        "stderr": p.stderr[-4000:],
    }


def frame_rgb(i):
    rng = np.random.default_rng(i % 8)
    base = np.linspace(0, 255, W, dtype=np.uint8)[None, :, None].repeat(H, 0).repeat(3, 2)
    return np.clip(base + rng.integers(-3, 3, (H, W, 3)), 0, 255).astype(np.uint8)


def nvenc_throughput(out: pathlib.Path) -> dict:
    result = {"nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version,utilization.encoder",
                                "--format=csv"]),
              "ffmpeg_encoders": sh(["bash", "-c", "ffmpeg -hide_banner -encoders | grep -i nvenc"]),  # noqa: E501
              }  # fmt: skip
    frames = 3000
    t0 = time.perf_counter()
    run = sh(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "lavfi", "-i",
            f"testsrc2=size={W}x{H}:rate=10", "-frames:v", str(frames), "-c:v", "h264_nvenc",
            "-preset", "p1", "-tune", "ull", "-rc", "cbr", "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-bf", "0", "-g", "10", "-b:v", "1000k", "-f", "null", "-",
        ],
        timeout=300,
    )  # fmt: skip
    elapsed = time.perf_counter() - t0
    result.update(
        run=run,
        frames=frames,
        wall_s=round(elapsed, 3),
        encode_fps_unthrottled=round(frames / elapsed, 1),
        note="includes ffmpeg start-up and lavfi source generation; an upper bound on "
        "per-frame cost "
        "(ms/frame below), not a latency measurement",
        ms_per_frame_upper_bound=round(elapsed / frames * 1000, 3),
    )
    return result


def drive(media: MediaRuntime, schedule, out_samples: list) -> None:
    """schedule: iterable of (wall_offset_s, index); submits at those wall offsets."""
    t0 = time.monotonic()
    for offset, i in schedule:
        wait = t0 + offset - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        now = time.time()
        media.submit(
            frame_rgb(i), SimpleNamespace(id=f"p-{i}", sim_time_s=offset, captured_unix_s=now)
        )
        if i % 10 == 0:
            out_samples.append({"t": round(offset, 2), "status": media.status()})


def scenario(out: pathlib.Path, name: str, schedule, tail_s: float, binary) -> dict:
    folder = out / name
    media = MediaRuntime(folder, binary)
    samples: list = []
    try:
        drive(media, schedule, samples)
        time.sleep(tail_s)
        final = media.status()
    finally:
        media.close()

    def log(n):
        path = folder / n
        return path.read_text(errors="replace")[-3000:] if path.exists() else ""

    return {
        "final_status": final,
        "status_samples": samples,
        "mediamtx_log": log("mediamtx.log"),
        "nvenc_log": log("nvenc.log"),
        "recorded_bytes": (folder / "head-camera.mkv").stat().st_size
        if (folder / "head-camera.mkv").exists()
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary, lock = install_binary(
        ROOT / "experiments/configs/media_runtime.json", pathlib.Path("/work/cache/mediamtx")
    )
    report = {"mediamtx": lock, "ffmpeg": sh(["ffmpeg", "-version"])["stdout"][:200]}
    report["nvenc_throughput"] = nvenc_throughput(args.out)
    report["steady_10fps"] = scenario(
        args.out, "steady_10fps", [(i * 0.1, i) for i in range(300)], 2.0, binary
    )
    burst = [(i * 0.1, i) for i in range(5)]
    resume = [(15.0 + i * 0.1, 5 + i) for i in range(50)]
    report["stall_then_resume"] = scenario(
        args.out, "stall_then_resume", burst + resume, 2.0, binary
    )
    report["encoder_command"] = encoder_command(W, H, "rec.mkv")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k]["final_status"] for k in ("steady_10fps", "stall_then_resume")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
