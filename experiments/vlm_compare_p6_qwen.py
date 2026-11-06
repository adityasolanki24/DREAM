"""P6: Qwen3-VL-4B on the frozen 11-decision set that the ChatGPT run scored (6/11 correct).

    python experiments/vlm_compare_p6_qwen.py --out results/p6_qwen_frozen11_closeout

Same 11 decisions, frames, instruction, tool schemas, planner and scorer as
results/phase1_gpt_luna_none_20261028/run_frozen11.json, one greedy pass, no retries. The
planner is experiments/p6_frozen_mobile_model.py, a verbatim copy of dream/runtime/mobile_model.py
at ba5d547 (the prompt and schemas in force when ChatGPT ran; the live module has since been
rewritten). Scoring is experiments/vlm_compare.py (classify, correct) unchanged. Preconditions
are hash-checked before any inference. The model needs the locked Qwen stack, so on a session
VM this script re-executes itself under DREAM_QWEN_PYTHON.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FROZEN_PLANNER_SHA256 = "22101770902f21e0794ada7afcefb4881612557e8ec220055c2a782d1d854055"
# The ChatGPT run hashed a CRLF Windows checkout; the committed (LF) blob has the second digest.
DECISIONS_SHA256 = {
    "c80778c34b98f1fa745f8b77cc25ad8aa7809c3e7ebb8ffd527f1dc0268b287a",
    "dcaca56786e6e2df1f265f8bbbb5930c6042ecff5a3d587ab4b15324c1db0c79",
}
IDS = ["nav_3", "nav_4", "nav_5", "nav_away", "obj_3", "obj_4", "obj_6", "obj_nocabinet",
       "info_unverified", "info_satisfied", "info_blind"]  # fmt: skip
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def sha_lf(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def stage_inputs() -> None:
    """Cloud archives exclude results/: restore byte-identical copies of the frozen decision
    set and frames (experiments/p6_inputs) to where vlm_compare.py reads them."""
    target = ROOT / "results/phase1_vlm_compare"
    if not (target / "decisions.json").exists():
        shutil.copytree(ROOT / "experiments/p6_inputs", target, dirs_exist_ok=True)
    reference = ROOT / "results/phase1_gpt_luna_none_20261028/run_frozen11.json"
    if not reference.exists():
        reference.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "experiments/p6_inputs/run_frozen11_reference.json", reference)


def check_preconditions() -> None:
    if sha_lf(ROOT / "experiments/p6_frozen_mobile_model.py") != FROZEN_PLANNER_SHA256:
        raise SystemExit("frozen planner differs from mobile_model.py at ba5d547")
    decisions = ROOT / "results/phase1_vlm_compare/decisions.json"
    if hashlib.sha256(decisions.read_bytes()).hexdigest() not in DECISIONS_SHA256:
        raise SystemExit("decision set differs from the one ChatGPT was scored on")
    gpt = json.loads(
        (ROOT / "results/phase1_gpt_luna_none_20261028/run_frozen11.json").read_text("utf-8")
    )
    if gpt["selected_ids"] != IDS:
        raise SystemExit("decision ids differ from the ChatGPT run")
    for key, expected in gpt["sha256"].items():
        frame = ROOT / key.replace("\\", "/")
        if "frames" in key and hashlib.sha256(frame.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"frame differs from the ChatGPT run: {key}")


def main() -> None:
    stage_inputs()
    check_preconditions()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        python = os.environ.get("DREAM_QWEN_PYTHON", "/opt/dream-runtime/bin/python")
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        sys.exit(subprocess.run([python, *sys.argv], cwd=ROOT, env=env).returncode)
    from experiments import p6_frozen_mobile_model, vlm_compare

    vlm_compare.MobileModelPlanner = p6_frozen_mobile_model.MobileModelPlanner
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out is None:
        raise SystemExit("--out is required")
    sys.argv = [
        "vlm_compare.py", "qwen-local", "--device", "cuda", "--model-path", MODEL,
        "--model-revision", REVISION, "--tag", "qwen3vl4b_frozen11", "--ids", *IDS,
        "--out", out,
    ]  # fmt: skip
    vlm_compare.main()


if __name__ == "__main__":
    main()
