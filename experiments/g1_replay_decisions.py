"""Replay preserved model text against live physics as a diagnostic, never fresh inference.

The output sequence cannot respond to changed observations. This tests controller/tool
corrections against the original failure, not autonomous reasoning or task success.
Every physical attempt uses the ordinary automatic two-view recording pipeline.
"""

import argparse
import json
import tarfile
from pathlib import Path

from experiments.g1_dexterous import write_json
from experiments.g1_explore_unknown import run_case, source_metadata
from experiments.g1_recording import digest


class PreservedOutputs:
    name = "recorded-output-replay-no-new-inference"

    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)

    def set_cancel_event(self, cancel):
        self.cancel = cancel

    def generate(self, messages, tools):
        if self.cancel.is_set():
            raise RuntimeError("cancelled")
        try:
            return next(self.outputs)
        except StopIteration as error:
            raise ValueError("preserved outputs exhausted; no new inference") from error


def run(archive: Path, config_path: Path, case_name: str, out: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    case = next(c for c in cfg["cases"] if c["name"] == case_name)
    prefix = f"out/g1/qwen/{case_name}/inference/"
    with tarfile.open(archive) as saved:
        members = sorted(
            (
                m
                for m in saved.getmembers()
                if m.name.startswith(prefix) and m.name.endswith("_output.json")
            ),
            key=lambda m: m.name,
        )
        outputs = [json.load(saved.extractfile(m))["text"] for m in members]
    if not outputs:
        raise ValueError("no original outputs for requested case")
    cfg = {**cfg, "backend": "qwen", "inference_mode": PreservedOutputs.name}
    out.mkdir(parents=True, exist_ok=False)
    metadata = source_metadata(cfg)
    write_json(
        out / "metadata.json",
        {
            **metadata,
            "archive_sha256": digest(archive),
            "outputs": outputs,
            "source_members": [m.name for m in members],
            "fresh_model_calls": 0,
            "observations_affect_decision_sequence": False,
        },
    )
    result = run_case(case, cfg, out / case_name, backend=PreservedOutputs(outputs))
    write_json(out / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/g1_qwen_comparison.json")
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.config, args.case, args.out), indent=2))
