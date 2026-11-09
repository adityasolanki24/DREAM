"""Compact manifest and summary of a perception-capture job's raw output (kept outside Git).

Reads a retrieved job output folder (``jobs/<n>/out``, with the guest's MANIFEST.sha256 of every
file, verified at retrieval by cloud/isaac_session.py) and writes, for each captured item
(composed home or kitchen), the frame count, byte totals, the SHA-256 of its ``frames.jsonl``
and one digest over all of its file checksums, plus per-frame statistics (Qwen frames, ground
truth instances per frame). Only this compact file is committed; the raw data stays under
``%LOCALAPPDATA%/DREAM/cloud_runs``.

    python experiments/isaac_capture_manifest.py --job-out <jobs/n/out> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

FORMAT = {
    "per_item_folder": "<item>/ with frames.jsonl, frames/NNNNN.png, frames/NNNNN.npz",
    "rgb": "frames/NNNNN.png, uint8 H x W x 3, 240 x 320, sRGB, head camera after sensor noise",
    "npz_depth": "frames/NNNNN.npz['depth'], float16 H x W, metres along the optical axis "
    "(distance_to_image_plane) after sensor noise; non-finite where there is no return",
    "npz_instance_ids": "frames/NNNNN.npz['instance_ids'], int32 H x W, Isaac instance_id_"
    "segmentation_fast ids; frames.jsonl row 'instance_id_to_prim' maps an id (string) to its "
    "prim path for the placed instances; 0 and ids of the robot or the shell are not listed",
    "frames.jsonl row": {
        "i": "frame index within the item",
        "trajectory": "trajectory index within the item",
        "sim_time_s": "simulated seconds since the item began",
        "odometry_pose": "[x, y, yaw] the AGENT's odometry estimate (oracle-assisted with noise), "
        "metres and radians, in the odometry frame whose origin is the start pose",
        "true_pose": "[x, y, yaw] torso pose in the world frame (EVALUATION ONLY)",
        "true_camera": "world pose of the head camera (EVALUATION ONLY): position (m), forward and "
        "up unit vectors, fovy_deg, aspect; right = forward x up; world frame z up",
        "true_room_id": "composed homes only: generator room id at the robot (EVALUATION ONLY)",
        "gt": "per visible placed instance (EVALUATION ONLY): class, pixel count and tight 2D "
        "box [x1, y1, x2, y2] in pixels (x1, y1 inclusive, x2, y2 exclusive) from the "
        "segmentation, keyed by instance id of scene_truth.json",
        "png_sha256": "checksum of the RGB file",
    },
    "intrinsics": "pinhole, fx = fy = (H/2) / tan(fovy/2) with fovy 58 degrees, cx = W/2, "
    "cy = H/2, principal point at pixel centres of a 320 x 240 image; stored per item in the "
    "run result JSON under 'intrinsics'",
    "frame_conventions": "world: z up, metres; image origin top-left, x right, y down; camera "
    "optical axis = forward, up as given; head camera pitched 35 degrees down on the torso",
    "hires": "capture-only second render product of the same head camera at 640 x 480, rendered "
    "from the identical pose right after the 320 x 240 frame: frames_hires/NNNNN.png (uint8 "
    "480 x 640 x 3), frames_hires/NNNNN.npz (depth float16 metres, instance_ids int32), "
    "frames.jsonl keys gt_hires, instance_id_to_prim_hires, png_hires_sha256; intrinsics scale "
    "with the resolution (fx = fy = 240 / tan(fovy/2) at 640 x 480, cx = 320, cy = 240); the "
    "agent's camera stays 320 x 240; each row has 'resolution': base and hires [w, h]",
    "no_model_output": "Qwen3-VL is not run during capture; detector comparison is offline",
}


def read_checksums(manifest: pathlib.Path) -> dict[str, str]:
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            rows[name.strip()] = digest
    return rows


def summarise_item(item: str, files: dict[str, str], root: pathlib.Path, prefix: str) -> dict:
    mine = {k: v for k, v in files.items() if k.startswith(f"{prefix}/{item}/")}
    frames = pathlib.Path(root) / item / "frames.jsonl"
    rows = (
        [json.loads(x) for x in frames.read_text(encoding="utf-8").splitlines() if x.strip()]
        if frames.is_file()
        else []
    )
    paths = [pathlib.Path(root) / k.split(f"{prefix}/", 1)[1] for k in mine]
    total = sum(p.stat().st_size for p in paths if p.is_file())
    digest = hashlib.sha256("\n".join(f"{v}  {k}" for k, v in sorted(mine.items())).encode())
    return {
        "item": item,
        "files": len(mine),
        "bytes": total,
        "frames": len(rows),
        "frames_with_qwen": sum("qwen" in r for r in rows),
        "frames_with_gt_instance": sum(bool(r.get("gt")) for r in rows),
        "frames_jsonl_sha256": files.get(f"{prefix}/{item}/frames.jsonl"),
        "all_file_checksums_digest": digest.hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-out", type=pathlib.Path, required=True)
    ap.add_argument("--raw-subdir", default="raw")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    files = read_checksums(args.job_out / "MANIFEST.sha256")
    root = args.job_out / args.raw_subdir
    items = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    out = {
        "format": FORMAT,
        "raw_location": str(root),
        "outside_git_and_onedrive": True,
        "manifest_sha256_of_job": hashlib.sha256(
            (args.job_out / "MANIFEST.sha256").read_bytes()
        ).hexdigest(),
        "items": [summarise_item(i, files, root, args.raw_subdir) for i in items],
    }
    out["total_bytes"] = sum(i["bytes"] for i in out["items"])
    out["total_frames"] = sum(i["frames"] for i in out["items"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    main()
