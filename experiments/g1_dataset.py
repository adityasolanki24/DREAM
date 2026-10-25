"""Export recorded physical evaluations to GR00T LeRobot v2 without relabeling failures.

Action at t is the existing controller's absolute, slew-limited reference at t+50 ms.
State is measured joint position at t. No EEF, base or unknown training slots are used.
Videos show this same trajectory from a fixed documented external camera, rendered
offline without sensor noise. This is a NEW_EMBODIMENT dataset, not a REAL_G1 repair.
Failed episodes remain diagnostics and are explicitly marked unready for imitation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from experiments.g1_dexterous import write_json
from experiments.g1_recording import FPS, digest, external_recording_root


def normalization_statistics(values: np.ndarray) -> dict:
    """Unclipped per-coordinate statistics; caller supplies training episodes only.

    Zero spread is preserved for constant supported joints. The upstream processor,
    rather than an invented physical range, handles its normalization denominator.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("finite nonempty rows required for normalization")
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def aligned_samples(recording: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Use named upper-body hinges only; validate chronology, dimensions and ranges."""
    metadata = json.loads((recording / "metadata.json").read_text())
    contract = json.loads((recording / "robot_contract.json").read_text())
    names = metadata["joint_names"]
    if names != contract["joint_names"] or len(set(names)) != 43:
        raise ValueError("recorded robot joint identity mismatch")
    ids = [
        i
        for i, name in enumerate(names)
        if name.startswith(
            (
                "waist_",
                "left_shoulder_",
                "right_shoulder_",
                "left_elbow_",
                "right_elbow_",
                "left_wrist_",
                "right_wrist_",
                "left_hand_",
                "right_hand_",
            )
        )
    ]
    if len(ids) != 31:
        raise ValueError("expected 31 supported upper-body joint coordinates")
    with np.load(recording / "snapshots.npz", allow_pickle=False) as arrays:
        time = arrays["time_s"].copy()
        state = arrays["joint_pos_rad"][:-1, ids].astype(np.float32)
        action = arrays["reference_rad"][1:, ids].astype(np.float32)
    if len(time) < 2 or not np.allclose(np.diff(time), 1 / FPS, atol=1e-9):
        raise ValueError("nonuniform or insufficient real physics snapshots")
    limits = np.asarray(contract["joint_limits_rad"])[ids]
    if not all(np.isfinite(x).all() for x in (state, action)) or np.any(
        (action < limits[:, 0]) | (action > limits[:, 1])
    ):
        raise ValueError("invalid recorded references; no clipping permitted")
    return state, action, [names[i] for i in ids], time[:-1]


def original_video(recording: Path, movies: dict) -> Path:
    """Resolve a retrieved cloud video without accepting replacement or edited bytes."""
    original = Path(movies["videos"]["wide"]["file"])
    if not original.is_file():
        original = recording / "wide/episode.mp4"
    if digest(original) != movies["videos"]["wide"]["sha256"]:
        raise ValueError("recorded video checksum mismatch")
    return original


def export(recording: Path, out: Path) -> dict:
    """Write a real, independently labelled episode; never promote it to a trained policy."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = external_recording_root(out)
    state, action, names, times = aligned_samples(recording)
    summary = json.loads((recording / "summary.json").read_text())
    metadata = json.loads((recording / "metadata.json").read_text())
    movies = json.loads((recording / "recording.json").read_text())
    if movies["status"] != "complete":
        raise ValueError("complete original video evidence required")
    original = original_video(recording, movies)
    out.mkdir(parents=True, exist_ok=False)
    for part in ("meta", "data/chunk-000", "videos/chunk-000/observation.images.external_scene"):
        (out / part).mkdir(parents=True)
    n = len(state)
    rows = [
        {
            "observation.state": state[i].tolist(),
            "action": action[i].tolist(),
            "timestamp": float(times[i]),
            "frame_index": i,
            "episode_index": 0,
            "index": i,
            "task_index": 0,
            "annotation.human.task_description": 0,
            "next.done": i == n - 1,
            "next.reward": float(i == n - 1 and summary["goal_success"]),
        }
        for i in range(n)
    ]
    table = pa.Table.from_pylist(rows)
    for name, values in (("observation.state", state), ("action", action)):
        column = pa.array(values.tolist(), type=pa.list_(pa.float32(), 31))
        table = table.set_column(table.schema.get_field_index(name), name, column)
    for name in ("timestamp", "next.reward"):
        column = table[name].cast(pa.float32())
        table = table.set_column(table.schema.get_field_index(name), name, column)
    pq.write_table(table, out / "data/chunk-000/episode_000000.parquet")
    video = out / "videos/chunk-000/observation.images.external_scene/episode_000000.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(original),
            "-frames:v",
            str(n),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    feature = {"dtype": "float32", "shape": [31], "names": names}
    write_json(
        out / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "dream_g1_rev1_supported",
            "fps": FPS,
            "total_episodes": 1,
            "total_frames": n,
            "total_tasks": 1,
            "total_videos": 1,
            "total_chunks": 1,
            "chunks_size": 1000,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
            "features": {
                "observation.state": feature,
                "action": feature,
                **{
                    name: {"dtype": "int64", "shape": [1], "names": None}
                    for name in (
                        "frame_index",
                        "episode_index",
                        "index",
                        "task_index",
                        "annotation.human.task_description",
                    )
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
                "next.reward": {"dtype": "float32", "shape": [1], "names": None},
                "observation.images.external_scene": {
                    "dtype": "video",
                    "shape": [720, 1280, 3],
                    "names": ["height", "width", "channel"],
                    "info": {
                        "video.fps": FPS,
                        "video.height": 720,
                        "video.width": 1280,
                        "video.channels": 3,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                },
            },
        },
    )
    write_json(
        out / "meta/modality.json",
        {
            "state": {"joints": {"start": 0, "end": 31}},
            "action": {"joints": {"start": 0, "end": 31}},
            "video": {"external_scene": {"original_key": "observation.images.external_scene"}},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        },
    )
    write_json(
        out / "meta/stats.json",
        {
            "observation.state": normalization_statistics(state),
            "action": normalization_statistics(action),
        },
    )
    for filename, value in (
        ("tasks", {"task_index": 0, "task": metadata["config"]["instruction"]}),
        (
            "episodes",
            {"episode_index": 0, "tasks": [metadata["config"]["instruction"]], "length": n},
        ),
    ):
        (out / f"meta/{filename}.jsonl").write_text(json.dumps(value) + "\n", encoding="utf-8")
    result = {
        "source_recording": str(recording.resolve()),
        "source_commit": metadata["base_git_sha"],
        "source_snapshots_sha256": digest(recording / "snapshots.npz"),
        "joint_names": names,
        "units": "radians, named MuJoCo hinge coordinates, no offsets",
        "sampling_hz": FPS,
        "action": "absolute rate-limited reference at t+0.05s",
        "image": "fixed external wide view, offline render of recorded state, clean RGB",
        "goal_success": summary["goal_success"],
        "diagnostic_only": not summary["goal_success"],
        "training_ready": False,
        "normalization_scope": "this episode only; no held-out samples; not training approval",
        "readiness_reason": "One evaluation is not a diverse successful demonstration corpus.",
        "backend": summary["backend"],
        "motor_policy": "classical demonstration data; no VLA calls",
        "parquet_sha256": digest(out / "data/chunk-000/episode_000000.parquet"),
        "video_sha256": digest(video),
        "rows": n,
        "pyarrow": pa.__version__,
    }
    write_json(out / "provenance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.recording, args.out), indent=2))
