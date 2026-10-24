"""Reproduce the recovered tokenizer failure without GPU weights, then test local identity.

Run in the separate Transformers4.57.3 CPU environment. Networking is disabled before import;
this tests real tokenizer loading, not GR00T model inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    report = {"scope": "Real CPU tokenizer only; no weights, GPU or policy actions"}
    if args.upstream:
        source = args.upstream / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
        function = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "get_backbone_cls"
        )
        condition = ast.Expression(function.body[0].test)
        selected = eval(
            compile(condition, str(source), "eval"),
            {"config": SimpleNamespace(model_name=args.backbone.resolve().as_posix())},
        )
        assert selected, "actual upstream dispatch rejects the local backbone path"
        report["actual_upstream_dispatch_condition_passed"] = bool(selected)
        report["dispatch_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
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
