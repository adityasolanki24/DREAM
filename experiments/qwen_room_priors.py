"""Room priors from a pretrained VLM's own knowledge (text only), for the lead to compare with
the ConceptNet associations in dream/semantics/knowledge.py. Not used by any agent.

For every object in third_party/household_knowledge/knowledge.json ("objects"), Qwen3-VL-4B
(pinned revision, greedy decoding, no image) is asked once for a JSON object over that file's
place vocabulary plus "none". The raw answer, the parsed table and the prompt verbatim are
saved. A malformed answer is recorded as such, never repaired by a retry.

    python experiments/qwen_room_priors.py --out <dir> [--qwen-port 8767]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KNOWLEDGE = ROOT / "third_party/household_knowledge/knowledge.json"
SYSTEM = (
    "You are a household-knowledge assistant. Answer with a single JSON object and nothing else."
)
USER = (
    'For the household object "{obj}", how likely is it to be found in each type of room? '
    "Room types: {rooms}. Give a number from 0 to 1 for each room type, and a number for "
    '"none" meaning the object is not typically found in any of these rooms. The numbers '
    "must sum to 1. Use exactly these keys: {keys}."
)


def build_messages(obj: str, rooms: list[str]) -> list[dict]:
    keys = [*rooms, "none"]
    user = USER.format(obj=obj.replace("_", " "), rooms=", ".join(rooms), keys=json.dumps(keys))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


def parse_answer(text: str, rooms: list[str]) -> dict:
    """The table in ``text`` over exactly ``rooms + ['none']``, or why it is not valid."""
    keys = [*rooms, "none"]
    m = re.search(r"\{.*\}", text, re.S)
    if m is None:
        return {"valid": False, "problem": "no JSON object in the answer"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as error:
        return {"valid": False, "problem": f"invalid JSON ({error.msg})"}
    if not isinstance(data, dict):
        return {"valid": False, "problem": "not an object"}
    missing = [k for k in keys if k not in data]
    extra = [k for k in data if k not in keys]
    numbers = {k: data[k] for k in keys if isinstance(data.get(k), (int, float))}
    bad = [k for k in keys if k in data and k not in numbers]
    if missing or extra or bad or any(v < 0 for v in numbers.values()):
        return {"valid": False, "problem": "keys or values off", "missing": missing,
                "extra": extra, "non_numeric": bad}  # fmt: skip
    total = float(sum(numbers.values()))
    return {
        "valid": True,
        "table": {k: float(numbers[k]) for k in keys},
        "sum": total,
        "normalised": None if total <= 0 else {k: float(numbers[k]) / total for k in keys},
    }


def main() -> None:  # pragma: no cover - needs the Qwen server
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--qwen-port", type=int, default=8767)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    from dream.runtime.qwen import OpenAICompatibleBackend
    from experiments.isaac_g1_live import QWEN_REVISION
    from experiments.isaac_g1_live_eval import start_model_server, stop_process

    args.out.mkdir(parents=True, exist_ok=True)
    knowledge = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    rooms = list(knowledge["place_vocabulary"])
    cfg = {
        "model_revision": QWEN_REVISION,
        "timeouts_s": {"model_start": 1500},
        "qwen_max_new_tokens": args.max_new_tokens,
    }
    server, log = start_model_server(cfg, args.out, args.qwen_port)
    backend = OpenAICompatibleBackend(f"http://127.0.0.1:{args.qwen_port}", timeout_s=120)
    rows = {}
    try:
        for obj in knowledge["objects"]:
            messages = build_messages(obj, rooms)
            answer = backend.generate(messages, [])
            rows[obj] = {"raw": answer, "usage": backend.last_usage, **parse_answer(answer, rooms)}
            print(obj, rows[obj]["valid"], flush=True)
    finally:
        stop_process(server, 60, term=True)
        if log is not None:
            log.close()
    result = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "model_revision": QWEN_REVISION,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "images": "none (text only)",
        "knowledge_file": str(KNOWLEDGE.relative_to(ROOT)),
        "rooms": rooms,
        "prompt_system": SYSTEM,
        "prompt_user_template": USER,
        "example_messages": build_messages("kettle", rooms),
        "valid": sum(r["valid"] for r in rows.values()),
        "of": len(rows),
        "objects": rows,
        "note": "Raw pretrained-VLM answers; not used by any agent. The numbers are the model's "
        "stated beliefs, uncalibrated, and their sum is recorded (not forced to 1).",
    }
    (args.out / "priors.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
