"""Fetch NVIDIA's GEAR-SONIC ONNX weights into the local cache.

    python experiments/fetch_sonic_policy.py                 # download from Hugging Face
    python experiments/fetch_sonic_policy.py --from DIR      # copy from a local folder

Pinned to the Hugging Face revision and SHA-256 in experiments/configs/sonic_policy.json; a file
whose hash differs is deleted and the script fails. The cache is outside Git:
$DREAM_CACHE/gear_sonic, else ~/.cache/dream/gear_sonic. The planner is ~740 MB. Weights are
licensed by NVIDIA Corporation under the NVIDIA Open Model License.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.robot.sonic import policy_cache  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", type=pathlib.Path, help="copy from this folder")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "experiments/configs/sonic_policy.json").read_text())
    out = policy_cache()
    out.mkdir(parents=True, exist_ok=True)
    for entry in cfg["files"].values():
        path = out / entry["file"]
        if path.exists() and sha256(path) == entry["sha256"]:
            print("cached", path)
            continue
        tmp = path.with_suffix(".part")
        if args.source is not None:
            shutil.copyfile(args.source / entry["file"], tmp)
        else:
            url = cfg["url"].format(repo=cfg["repo"], revision=cfg["revision"], file=entry["file"])
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        digest = sha256(tmp)
        if digest != entry["sha256"]:
            tmp.unlink()
            sys.exit(f"hash mismatch for {entry['file']}: {digest}")
        tmp.replace(path)
        print("fetched", path, digest)


if __name__ == "__main__":
    main()
