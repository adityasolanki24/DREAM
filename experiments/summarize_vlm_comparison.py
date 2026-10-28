"""Compare saved GPT and Qwen decisions without inference.

The GPT prompt/tool hashes are reconstructed offline after its original run, not captured
at inference time. Output is written only after all input and matching checks pass.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _inputs(run):
    hashes = {key.replace("\\", "/"): value for key, value in run["sha256"].items()}
    selected = {
        key: value
        for key, value in hashes.items()
        if key == "results/phase1_vlm_compare/decisions.json"
        or key.startswith("results/phase1_vlm_compare/frames/")
    }
    if "results/phase1_vlm_compare/decisions.json" not in selected or len(selected) < 2:
        raise ValueError("missing frozen decisions or image hashes")
    return selected


def _rows(run):
    rows = run["rows"]
    by_id = {row["id"]: row for row in rows}
    ids = run["selected_ids"]
    if len(rows) != 11 or len(by_id) != 11 or len(ids) != 11 or set(ids) != set(by_id):
        raise ValueError("expected the same 11 unique completed decisions")
    return by_id


def compare(gpt, reconstruction, qwen):
    frozen = _inputs(gpt)
    if _inputs(reconstruction) != frozen or _inputs(qwen) != frozen:
        raise ValueError("decision or image hashes differ")
    actual, reconstructed, other = map(_rows, (gpt, reconstruction, qwen))
    if set(actual) != set(reconstructed) or set(actual) != set(other):
        raise ValueError("decision IDs differ")
    if reconstruction.get("summary", {}).get("backend") != "offline-provenance-reconstruction":
        raise ValueError("GPT provenance must be the labelled offline reconstruction")
    for ident in actual:
        if not (
            actual[ident]["category"]
            == reconstructed[ident]["category"]
            == other[ident]["category"]
        ):
            raise ValueError(f"category differs for {ident}")
        for field in ("messages_sha256", "tools_sha256"):
            if not reconstructed[ident].get(field) or reconstructed[ident][field] != other[
                ident
            ].get(field):
                raise ValueError(f"{field} differs for {ident}")

    def summarize(rows, run):
        latencies = [row["latency_s"] for row in rows.values() if not row.get("error")]
        categories = sorted({row["category"] for row in rows.values()})
        return {
            "backend": run["summary"].get("backend"),
            "valid": sum(bool(row["valid_tool_call"]) for row in rows.values()),
            "correct": sum(bool(row["correct"]) for row in rows.values()),
            "n": len(rows),
            "by_category": {
                category: {
                    "n": sum(row["category"] == category for row in rows.values()),
                    "correct": sum(
                        bool(row["correct"]) for row in rows.values() if row["category"] == category
                    ),
                }
                for category in categories
            },
            "latency_p50_s": statistics.median(latencies) if latencies else None,
            "latency_max_s": max(latencies) if latencies else None,
            "input_tokens": run["summary"].get("input_tokens"),
            "output_tokens": run["summary"].get("output_tokens"),
            "errors": [row["error"] for row in rows.values() if row.get("error")],
        }

    return {
        "matching": {
            "decision_and_image_hashes": "equal",
            "prompt_and_tool_hashes": (
                "equal to offline GPT reconstruction, not captured in original run"
            ),
        },
        "latency_scope": (
            "per-decision backend call including preprocessing; excludes model startup"
        ),
        "gpt": summarize(actual, gpt),
        "qwen": summarize(other, qwen),
        "decisions": [
            {
                "id": ident,
                "category": actual[ident]["category"],
                "gpt": {
                    "tool": actual[ident]["tool"],
                    "args": actual[ident]["args"],
                    "correct": actual[ident]["correct"],
                },
                "qwen": {
                    "tool": other[ident]["tool"],
                    "args": other[ident]["args"],
                    "correct": other[ident]["correct"],
                },
                "agreement": (actual[ident]["tool"], actual[ident]["args"])
                == (other[ident]["tool"], other[ident]["args"]),
            }
            for ident in gpt["selected_ids"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpt", type=Path)
    parser.add_argument("gpt_reconstruction", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.gpt, args.gpt_reconstruction, args.qwen)
    ]
    result = compare(*runs)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
