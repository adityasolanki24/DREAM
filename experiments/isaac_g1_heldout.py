"""The 12 frozen held-out NVIDIA kitchen episodes, a few at a time on one GPU.

    python experiments/isaac_g1_heldout.py --config experiments/configs/isaac_g1_kitchen.json \
        --robot-dir <dir> --prepare-with <converter python> --out <dir> [--workers 3]

Each episode is the unchanged ``experiments/isaac_g1_kitchen.py --mode explore`` in its own
process and output folder; this file only schedules them and collects their results against
the pre-registered evaluation gates. Nothing here touches scenes, the agent or thresholds.
Order: start 1 in every kitchen first, so a session that runs out of time still covers all
four kitchens.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]


def episode(args, kitchen: str, start: int) -> dict:
    out = args.out / f"{kitchen}_{start}"
    cmd = [sys.executable, str(ROOT / "experiments/isaac_g1_kitchen.py"), "--config",
           str(args.config), "--mode", "explore", "--kitchen", kitchen, "--start", str(start),
           "--robot-dir", str(args.robot_dir), "--out", str(out)]  # fmt: skip
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "process.log", "w") as log:
        try:
            code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=args.episode_timeout_s).returncode  # fmt: skip
        except subprocess.TimeoutExpired:
            code = 124
    row = {"kitchen": kitchen, "start": start, "exit_code": code, "wall_s": time.time() - t0}
    result = out / "episode.json"
    if result.is_file():
        row["episode"] = json.loads(result.read_text())
    print(json.dumps({k: v for k, v in row.items() if k != "episode"}), flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--robot-dir", type=pathlib.Path, required=True)
    ap.add_argument("--prepare-with", help="converter Python: build the robot USD once first")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--episode-timeout-s", type=float, default=3600)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare_with:
        subprocess.run([args.prepare_with, str(ROOT / "experiments/prepare_g1_usd.py"),
                        "--out", str(args.robot_dir)], check=True)  # fmt: skip
    jobs = [(k, s) for s in cfg["evaluation_starts"] for k in cfg["evaluation_kitchens"]]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda j: episode(args, *j), jobs))
    (args.out / "episodes.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"episodes": len(rows), "with_result": sum("episode" in r for r in rows)}))


if __name__ == "__main__":
    main()
