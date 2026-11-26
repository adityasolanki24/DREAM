"""Run the stratified R1 development gate with separate sensed and scored outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from experiments.g1_grasp_bringup import attempt

BASE = Path(__file__).resolve().parent / "configs" / "g1_walk_grasp.json"
RANGES = ("width_m", "mass_kg", "table_top_m", "sliding_friction", "start_distance_m")


def episodes(spec: dict) -> list[dict]:
    """Stratify every varied scalar independently within each object family."""
    base = json.loads(BASE.read_text(encoding="utf-8"))
    shapes = list(spec["shapes"].items())
    count = spec["episodes_per_shape"]
    start = spec["seed_start"]
    strata = np.random.default_rng(np.random.SeedSequence(start, spawn_key=(0,)))
    strata = np.array([[strata.permutation(count) for _ in RANGES] for _ in shapes]).transpose(
        2, 0, 1
    )
    bounds = np.array([spec[key] for key in RANGES], float)
    cases = []
    for group in strata:
        for (shape, aspect), bins in zip(shapes, group, strict=True):
            seed = start + len(cases)
            rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
            fraction = (bins + rng.random(len(RANGES))) / count
            width, mass, top, friction, distance = bounds[:, 0] + np.diff(bounds)[:, 0] * fraction
            # Approach a varied cardinal face of the existing rectangular table fixture.
            turn = int(rng.integers(4))
            normal = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)][turn])
            tangent = np.array([-normal[1], normal[0]])
            table_xy = rng.uniform(-0.75, 0.75, 2)
            radius = width / 2
            half_depth = base["table_half_size_m"][0]
            inset = radius + rng.uniform(0.045, 0.095)
            object_lateral = rng.uniform(-0.08, 0.08)
            object_xy = table_xy + (inset - half_depth) * normal + object_lateral * tangent
            start_lateral = rng.uniform(-0.50, 0.50)
            start_xy = table_xy - (half_depth + distance) * normal + start_lateral * tangent
            start_yaw = turn * np.pi / 2 + rng.uniform(-0.45, 0.45)
            cases.append(
                dict(
                    base,
                    seed=seed,
                    table_top_m=top,
                    table_center_xy_m=table_xy.tolist(),
                    object_center_xy_m=object_xy.tolist(),
                    object_shape=shape,
                    object_half_extents_m=[radius, radius, aspect * radius],
                    object_mass_kg=mass,
                    object_sliding_friction=friction,
                    starting_stance_xy_yaw=[*start_xy, start_yaw],
                )
            )
    return cases


def failure_stage(result: dict) -> str | None:
    """Locate the first unmet phase from the recorded run, without guessing root cause."""
    if result["grasp_verified"]:
        return None
    phases = result["phases"]
    if "goal" not in phases.get("stance", {}):
        return "stance"
    walk = phases.get("walk", {})
    if walk.get("status") != "arrived" or walk.get("table_contact_samples"):
        return "walk"
    if not phases.get("settle", {}).get("stable"):
        return "settle"
    for name in ("raise", "pregrasp", "reach"):
        if not phases.get(name, {}).get("completed"):
            return name
    if not phases.get("preclose", {}).get("clear"):
        return "preclose"
    if not phases.get("close", {}).get("opposed_touch"):
        return "close"
    return "lift_or_hold"


def _write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_case(config: dict, out: Path) -> dict:
    result = attempt(config)
    rows = result.pop("rows")
    _write(out / "trace.json", rows)
    _write(out / "result.json", result)
    return {
        "seed": config["seed"],
        "shape": config["object_shape"],
        "observed": result["grasp_observed"],
        "verified": result["grasp_verified"],
        "failure_stage": failure_stage(result),
        "stop": result["stop"],
        "peak_lift_m": result["peak_lift_m"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--limit", type=int, help="Smoke subset in interleaved family order")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    cases = episodes(spec)
    required = spec["gate"]["episode_count"]
    if len(cases) != required:
        parser.error("configured development cohort does not match the gate")
    if args.limit is not None and not 1 <= args.limit < required:
        parser.error("--limit must select a nonempty subset of the development cohort")
    cases = cases[: args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    _write(
        args.out / "metadata.json",
        {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
            "source_sha256": {
                path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for folder in ("dream", "experiments")
                for path in sorted(Path(folder).rglob("*.py"))
            },
            "mujoco": mujoco.__version__,
            "config": spec,
            "seeds": [config["seed"] for config in cases],
            "oracle_assisted": ["table geometry", "object pose", "noisy odometry"],
        },
    )
    finished = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for config in cases:
            directory = args.out / str(config["seed"])
            directory.mkdir()
            _write(directory / "config.json", config)
            futures[executor.submit(_run_case, config, directory)] = config
        for future in as_completed(futures):
            config = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "seed": config["seed"],
                    "shape": config["object_shape"],
                    "error": f"{type(error).__name__}: {error}",
                }
            print(json.dumps(row), flush=True)
            finished.append(row)
    finished.sort(key=lambda row: row["seed"])
    verified = sum(row.get("verified", False) for row in finished)
    complete = sum("error" not in row for row in finished)
    summary = {
        "attempted": len(cases),
        "completed": complete,
        "verified": verified,
        "false_positive": sum(
            row.get("observed", False) and not row["verified"] for row in finished
        ),
        "false_negative": sum(
            row.get("verified", False) and not row["observed"] for row in finished
        ),
        "gate_passed": (
            verified / required >= spec["gate"]["minimum_verified_fraction"]
            if complete == required
            else None
        ),
        "cases": finished,
    }
    _write(args.out / "summary.json", summary)
    if complete < len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
