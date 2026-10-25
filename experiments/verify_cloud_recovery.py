"""Verify original downloads before permitting source-bucket deletion. No GPU operations.

The guest excludes its actively written logs and the manifest itself from SHA-256.
Those three objects are checked against final Cloud Storage sizes and MD5 metadata.
All other files must appear in the guest manifest; unexpected local files fail closed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from cloud.launch import gcloud


def checksum(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def verify(root: Path, remote_metadata: dict) -> dict:
    """Verify every local original, including files omitted from the guest manifest."""
    root = root.resolve()
    manifest = root / "MANIFEST.sha256"
    records = []
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        path = (root / name).resolve()
        if root not in path.parents or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("invalid guest manifest")
        name = path.relative_to(root).as_posix()
        if name in expected:
            raise ValueError("duplicate manifest path")
        expected[name] = sha
    if "EXIT_CODE" not in expected:
        raise ValueError("final guest status missing from manifest")
    local = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    if set(expected) - local.keys():
        raise ValueError("missing downloaded files; retain source bucket")
    if local.keys() - expected.keys() != set(remote_metadata):
        raise ValueError("unverified extra files; retain source bucket")
    for name, path in sorted(local.items()):
        sha = checksum(path)
        if name in expected:
            if sha != expected[name]:
                raise ValueError(f"guest checksum mismatch: {name}")
            method = "guest SHA-256"
        else:
            metadata = remote_metadata[name]
            md5 = base64.b64encode(bytes.fromhex(checksum(path, "md5"))).decode()
            if path.stat().st_size != int(metadata["size"]) or md5 != metadata["md5_hash"]:
                raise ValueError(f"cloud object checksum mismatch: {name}")
            method = "final Cloud Storage MD5/size"
        records.append(
            {"file": name, "bytes": path.stat().st_size, "sha256": sha, "method": method}
        )
    return {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "all_downloaded_files_verified": True,
        "guest_manifest_files": len(expected),
        "cloud_metadata_files": len(remote_metadata),
        "total_files": len(records),
        "total_bytes": sum(r["bytes"] for r in records),
        "manifest_sha256": checksum(manifest),
        "remote_metadata": remote_metadata,
        "files": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = {
        name: json.loads(
            gcloud(
                "storage", "objects", "describe", f"gs://{args.bucket}/out/{name}", "--format=json"
            )
        )
        for name in ("MANIFEST.sha256", "vm.log", "gpu.csv")
    }
    result = verify(args.root, metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "remote_metadata")}))
