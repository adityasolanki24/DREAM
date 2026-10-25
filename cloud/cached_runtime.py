"""Validate a prebuilt inference environment offline; never install or download packages."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from packaging.requirements import Requirement


def validate(lock: Path) -> dict:
    versions = {}
    for line in lock.read_text().splitlines():
        if not line or line[0].isspace() or line.startswith(("#", "-")):
            continue
        requirement = Requirement(line.removesuffix("\\").strip())
        if requirement.marker and not requirement.marker.evaluate():
            continue
        version = importlib.metadata.version(requirement.name)
        if not requirement.specifier.contains(version, prereleases=True):
            raise ValueError(f"cached dependency differs: {requirement.name}")
        versions[requirement.name] = version
    if platform.python_version() != "3.12.12":
        raise ValueError("cached runtime requires pinned Python 3.12.12")
    return {
        "preflight_passed": True,
        "requirements_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": versions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    result = validate(args.lock)
    if args.cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cached CUDA runtime unavailable; no installation fallback")
        result["gpu"] = torch.cuda.get_device_name(0)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.freeze:
        args.freeze.write_text(
            "\n".join(
                sorted(
                    f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                )
            )
            + "\n",
            encoding="utf-8",
        )
