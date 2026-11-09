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
