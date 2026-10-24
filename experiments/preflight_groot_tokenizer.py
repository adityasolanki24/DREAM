"""Reproduce the recovered tokenizer failure without GPU weights, then test local identity.

Run in the separate Transformers4.57.3 CPU environment. Networking is disabled before import;
this tests real tokenizer loading, not GR00T model inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    report = {"scope": "Real CPU tokenizer only; no weights, GPU or policy actions"}
    try:
        AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", local_files_only=True)
        report["original_remote_identity"] = "unexpectedly loaded"
    except Exception as error:
        report["original_remote_identity"] = type(error).__name__
    tokenizer = AutoTokenizer.from_pretrained(str(args.backbone.resolve()), local_files_only=True)
    ids = tokenizer.encode("Inspect the red object.")
    assert ids and tokenizer.decode(ids)
    report.update(
        local_identity_loaded=True,
        tokenizer_class=type(tokenizer).__name__,
        token_count=len(ids),
        decoded=tokenizer.decode(ids),
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
