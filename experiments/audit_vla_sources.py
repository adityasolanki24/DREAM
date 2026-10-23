"""Re-fetch pinned small source files for the policy decision. Never fetch model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fetch(source: dict) -> dict:
    request = urllib.request.Request(source["url"], headers={"User-Agent": "DREAM-source-audit"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("source unexpectedly large; refusing to fetch assets")
        digest = hashlib.sha256(data).hexdigest()
        return {**source, "observed_sha256": digest, "matches": digest == source["sha256"]}
    except Exception as error:
        return {**source, "error": str(error), "matches": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("do not overwrite earlier audits")
    sources = json.loads((ROOT / "experiments/configs/vla_sources.json").read_text())
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, sources))
    result = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": records,
        "all_hashes_match": all(r["matches"] for r in records),
        "weights_downloaded": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_hashes_match"]:
        raise SystemExit("source audit failed; inspect recorded errors")


if __name__ == "__main__":
    main()
