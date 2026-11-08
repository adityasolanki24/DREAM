"""Compare two sources of object-to-room knowledge on the benchmark's object classes.

    python experiments/compare_room_knowledge.py [--qwen results/qwen_room_priors/priors.json]

Sources: the ConceptNet 5.7 extract the agent uses (third_party/household_knowledge) and,
optionally, raw Qwen3-VL text answers (results/qwen_room_priors, never used by any agent).
Reference: the room types whose furnishing table in dream/sim/home.py lists the class
(FURNISHING_NOMINAL, studio excluded) or, for the eight search targets, their typical room
(TARGETS), mapped to knowledge categories by ROOM_MATCH. This
measures agreement with the benchmark's hand-written furnishing, not with real homes, and
uses evaluation-side data only. A top-1 that ties with another category is reported as a tie,
not a hit.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.semantic_scoring import ROOM_MATCH  # noqa: E402
from dream.semantics.knowledge import load_knowledge  # noqa: E402
from dream.sim.home import CLASSES, FURNISHING_NOMINAL, TARGETS  # noqa: E402


def reference(cls: str) -> set[str]:
    out: set[str] = set()
    for room, items in FURNISHING_NOMINAL.items():
        if room != "studio" and any(name == cls for name, *_ in items):
            out |= ROOM_MATCH[room]
    if cls in TARGETS:  # targets are placed by their TargetInfo, not the furnishing table
        out |= ROOM_MATCH[TARGETS[cls].typical]
    return out


def top(scores: dict[str, float]) -> tuple[list[str], float]:
    best = max(scores.values())
    return sorted(c for c, v in scores.items() if v == best), best


def judge(scores: dict[str, float] | None, ref: set[str]) -> dict:
    if not scores or not ref:
        return {"top": None, "result": "no knowledge" if ref else "no reference"}
    tied, _ = top(scores)
    if len(tied) > 1:
        return {"top": tied, "result": "tie"}
    return {"top": tied[0], "result": "hit" if tied[0] in ref else "miss"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", help="results/qwen_room_priors/priors.json (optional)")
    ap.add_argument("--out")
    args = ap.parse_args()
    kn = load_knowledge()
    qwen = None
    if args.qwen:
        qwen = json.loads(pathlib.Path(args.qwen).read_text())["objects"]
    rows = {}
    for cls in CLASSES:
        ref = reference(cls)
        name = cls if kn.has(cls) else cls.replace("_", " ")
        cn = (
            dict(zip(kn.categories, kn.association(name).tolist(), strict=True))
            if kn.has(name)
            else None
        )
        row = {"reference": sorted(ref), "conceptnet": judge(cn, ref)}
        if qwen is not None:
            entry = qwen.get(cls) or qwen.get(cls.replace("_", " ")) or {}
            table = entry.get("table") if entry.get("valid") else None
            q = {c: float(v) for c, v in (table or {}).items() if c in kn.categories}
            row["qwen"] = judge(q or None, ref)
        rows[cls] = row
    sources = ["conceptnet"] + (["qwen"] if qwen is not None else [])
    totals = {}
    for s in sources:
        counts: dict[str, int] = {}
        for row in rows.values():
            counts[row[s]["result"]] = counts.get(row[s]["result"], 0) + 1
        bench = {}
        for cls in TARGETS:
            res = rows[cls][s]["result"]
            bench[res] = bench.get(res, 0) + 1
        totals[s] = {"all_classes": counts, "benchmark_targets": bench}
    out = {
        "reference": "rooms whose FURNISHING_NOMINAL lists the class (dream/sim/home.py)",
        "classes": len(rows),
        "benchmark_targets": list(TARGETS),
        "totals": totals,
        "per_class": rows,
    }
    text = json.dumps(out, indent=2)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    print(json.dumps(totals, indent=2))
    for cls in TARGETS:
        print(cls, {s: rows[cls][s]["top"] for s in sources}, rows[cls]["reference"])


if __name__ == "__main__":
    main()
