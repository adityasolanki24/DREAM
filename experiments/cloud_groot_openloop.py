"""Pinned REAL_G1 inference only; measured provisional input, no actuator routing.

The official source is downloaded unmodified outside DREAM. The gated backbone must be
accessible before allocating this workload. Never save credentials in results or archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import traceback
import urllib.request
from pathlib import Path

from experiments.g1_dexterous import write_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/groot_openloop.json"


def offline_environment(source: Path, cache: Path) -> dict:
    """Make the named backbone load resolve the exact explicit snapshot-download cache."""
    return {
        **os.environ,
        "PYTHONPATH": str(source) + os.pathsep + str(ROOT),
        "HF_HUB_CACHE": str(cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(CONFIG.read_text())
    write_json(
        args.out / "freeze.json",
        {
            "config": cfg,
            "git_sha": os.environ["DREAM_CODE_SHA"],
            "configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "scope": "Inference only; provisional conventions and external camera; no actuation",
        },
    )
    try:
        from huggingface_hub import snapshot_download

        cache = Path(os.environ["HF_HOME"]).resolve()
        if ROOT in cache.parents or cache == ROOT or args.out.resolve() in (cache, *cache.parents):
            raise ValueError("cache must be outside source/results")
        # Fetch main, verify the exact pinned revision, then prohibit subsequent networking
        # in the inference subprocess. Upstream internally requests the backbone by name.
        backbone = Path(
            snapshot_download(
                cfg["backbone"],
                revision="main",
                cache_dir=cache,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            )
        )
        if backbone.name != cfg["backbone_revision"]:
            raise ValueError(
                "backbone main changed since preflight; preserve failure, do not adapt"
            )
        checkpoint = snapshot_download(
            cfg["checkpoint"],
            revision=cfg["revision"],
            cache_dir=cache,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE"],
        )
        upstream = Path("/work/upstream")
        upstream.mkdir(exist_ok=False)
        archive = upstream / "source.tar.gz"
        urllib.request.urlretrieve(
            f"https://codeload.github.com/NVIDIA/Isaac-GR00T/tar.gz/{cfg['source_revision']}",
            archive,
        )
        with tarfile.open(archive) as stream:
            stream.extractall(upstream, filter="data")
        source = upstream / f"Isaac-GR00T-{cfg['source_revision']}"
        write_json(
            args.out / "source.json",
            {
                "source_revision": cfg["source_revision"],
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "checkpoint_revision": cfg["revision"],
                "backbone_revision": backbone.name,
                "license": "Source Apache-2.0; checkpoint NVIDIA research/evaluation only",
            },
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.capture_groot_input",
                "--out",
                str(args.out / "sensors"),
            ],
            check=True,
        )
        env = offline_environment(source, cache)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.groot_openloop",
                "--input",
                str(args.out / "sensors/observation.npz"),
                "--model-path",
                str(checkpoint),
                "--out",
                str(args.out / "inference"),
            ],
            env=env,
            check=True,
        )
    except Exception as error:
        write_json(
            args.out / "failure.json",
            {
                "error": type(error).__name__,
                "detail": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
