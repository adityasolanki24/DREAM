"""Download NVIDIA's Decoupled WBC G1 balance/walk policies into the local cache.

    python experiments/fetch_wbc_policy.py

Pinned to the commit and SHA-256 in experiments/configs/wbc_policy.json; a file whose hash
differs is deleted and the script fails. The cache is outside Git: $DREAM_CACHE, else
~/.cache/dream. Weights are licensed by NVIDIA Corporation under the NVIDIA Open Model License.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.robot.g1_wbc import policy_cache  # noqa: E402


def main() -> None:
    cfg = json.loads((ROOT / "experiments/configs/wbc_policy.json").read_text())
    out = policy_cache()
    out.mkdir(parents=True, exist_ok=True)
    for entry in cfg["files"].values():
        path = out / entry["file"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
            print("cached", path)
            continue
        url = cfg["url"].format(commit=cfg["commit"], file=entry["file"])
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            f.write(r.read())
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            tmp.unlink()
            sys.exit(f"hash mismatch for {entry['file']}: {digest}")
        tmp.rename(path)
        print("fetched", path, digest)


if __name__ == "__main__":
    main()
