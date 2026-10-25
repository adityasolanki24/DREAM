"""Recompute summary.json and plots from a saved results/<name>/episodes.jsonl.

python experiments/summarise.py results/main
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dream.eval.plots import plot_summary  # noqa: E402
from dream.eval.summary import summarise  # noqa: E402


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    budget = json.loads((out / "meta.json").read_text())["config"]["budget"]
    logs = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    summary = summarise(logs, budget)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_summary(logs, summary, budget, out)
    print(f"summarised {len(logs)} episodes -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
