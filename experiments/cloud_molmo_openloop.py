"""One approved GPU workload: real sensor capture, then raw learned action tensors only.

Never launches resources itself, dispatches actions or downloads a second model. The guest's
outer timeout includes capture, checkpoint download and inference. No stand-in policy mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download_checkpoint(cfg: dict, cache: Path) -> str:
    """Include every checksummed file, not only the model loader's file extensions."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        cfg["checkpoint"],
        revision=cfg["revision"],
        cache_dir=cache,
        allow_patterns=[*cfg["checkpoint_source_sha256"], "*.json", "*.safetensors", "*.jinja"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "experiments/configs/molmo_openloop.json").read_text())
    cache = Path(os.environ.get("HF_HOME", "/work/model-cache")).resolve()
    if (
        cache == ROOT
        or ROOT in cache.parents
        or cache == args.out.resolve()
        or args.out.resolve() in cache.parents
        or "onedrive" in str(cache).lower()
    ):
        raise ValueError("model cache must stay outside repository, OneDrive and outputs")
    script = str(ROOT / "experiments/molmo_openloop.py")
    subprocess.run(
        [sys.executable, script, "capture", "--out", str(args.out / "observation")], check=True
    )
    model = download_checkpoint(cfg, cache)
    subprocess.run(
        [
            sys.executable,
            script,
            "infer",
            "--out",
            str(args.out / "molmo"),
            "--observation",
            str(args.out / "observation/observation.npz"),
            "--model-path",
            model,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
