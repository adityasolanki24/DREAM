"""Install the hash-pinned extras in cloud/semantic.requirements.lock.txt into the running
interpreter's environment, then run a script with the remaining arguments.

    python experiments/cloud_bootstrap.py experiments/semantic_search_eval.py CONFIG --split S

The cloud "dream" image predates the semantic modules' scipy dependency; rebuilding the
image is heavier than one hash-checked wheel per job. The install is recorded in the job log.
"""

from __future__ import annotations

import pathlib
import runpy
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCK = ROOT / "cloud/semantic.requirements.lock.txt"


def install() -> None:
    try:
        import scipy  # noqa: F401

        print(f"bootstrap: scipy {scipy.__version__} already present", flush=True)
        return
    except ImportError:
        pass
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--python", sys.executable, "--no-deps", "--require-hashes",
         "-r", str(LOCK)]
        if uv
        else [sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes",
              "-r", str(LOCK)]
    )  # fmt: skip
    print("bootstrap:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    install()
    script, *rest = sys.argv[1:]
    sys.argv = [script, *rest]
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / script), run_name="__main__")


if __name__ == "__main__":
    main()
