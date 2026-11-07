"""Per-seed table of the vision arm against the oracle reference arm, from saved episodes.

    python experiments/compare_compartment_arms.py results/compartment_vision_closeout \
        results/compartment_oracle_reference_closeout --out <file.json>

Observational: the arms differ in perception only, on the same seeds, but a per-seed
difference does not by itself establish its cause.
"""

import argparse
import json
from pathlib import Path


def rows(folder: Path) -> dict:
    return {
        (r["kind"], r["seed"]): r
        for r in map(json.loads, (folder / "episodes.jsonl").read_text().splitlines())
    }


def ok(r: dict) -> bool:
    return bool(r["correct_rejection"] if r["kind"] == "fixed" else r["success"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vision", type=Path)
    ap.add_argument("oracle", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    v, o = rows(args.vision), rows(args.oracle)
    if set(v) != set(o):
        raise SystemExit("arms cover different kind/seed sets")
    cats = {
        (f["kind"], f["seed"]): f["category"]
        for f in json.loads((args.vision / "closeout_summary.json").read_text())["failures"]
    }
    table = [
        {"kind": k[0], "seed": k[1], "vision_ok": ok(v[k]), "oracle_ok": ok(o[k]),
         "vision_failure_category": cats.get(k)}
        for k in sorted(v)
    ]  # fmt: skip
    openable = [t for t in table if t["kind"] != "fixed"]
    out = {
        "per_seed": table,
        "openable_vision_ok": sum(t["vision_ok"] for t in openable),
        "openable_oracle_ok": sum(t["oracle_ok"] for t in openable),
        "openable_n": len(openable),
        "fixed_declined_vision": sum(t["vision_ok"] for t in table if t["kind"] == "fixed"),
        "fixed_declined_oracle": sum(t["oracle_ok"] for t in table if t["kind"] == "fixed"),
        "seeds_vision_failed_oracle_ok": sum(
            (not t["vision_ok"]) and t["oracle_ok"] for t in table
        ),
        "seeds_oracle_failed": [t for t in table if not t["oracle_ok"]],
        "note": "observational; same seeds, arms differ in perception only",
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
