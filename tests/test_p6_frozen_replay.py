"""The P6 runner must score exactly as the ChatGPT run was scored.

Replays the recorded ChatGPT raw outputs (no model call) through the frozen planner and the
unchanged scorer: valid/tool/args/correct must reproduce run_frozen11.json, and the prompt and
tool digests must equal the offline reconstruction's.
"""

import json
import pathlib

import numpy as np
from PIL import Image

from dream.runtime.agent import ToolCall
from experiments import p6_frozen_mobile_model, vlm_compare, vlm_compare_p6_qwen

ROOT = pathlib.Path(__file__).resolve().parents[1]
GPT = ROOT / "results/phase1_gpt_luna_none_20261028"


class Replay:
    def __init__(self, raw):
        self.raw = raw

    def generate(self, messages, tools):
        return self.raw


def test_frozen_planner_and_decisions_unchanged():
    vlm_compare_p6_qwen.check_preconditions()


def test_replay_reproduces_chatgpt_scoring_and_prompt_digests():
    gpt = json.loads((GPT / "run_frozen11.json").read_text("utf-8"))
    recon = json.loads((GPT / "run_offline-provenance-reconstruction.json").read_text("utf-8"))
    recon_rows = {r["id"]: r for r in recon["rows"]}
    cfg = json.loads((vlm_compare.OUT / "decisions.json").read_text("utf-8"))
    for row, dec in zip(gpt["rows"], cfg["decisions"], strict=True):
        assert row["id"] == dec["id"]
        rec = vlm_compare.Recorder(Replay(row["raw"]))
        image = np.asarray(Image.open(vlm_compare.OUT / dec["image"]).convert("RGB"))
        planner = p6_frozen_mobile_model.MobileModelPlanner(
            rec, cfg["instruction"], lambda image=image: image
        )
        history = [
            type(
                "S",
                (),
                {
                    "call": ToolCall(h["tool"], h["args"]),
                    "result": type("R", (), {"status": h["status"], "message": h["message"]})(),
                },
            )()
            for h in dec["history"]
        ]
        action = planner.next_action(vlm_compare.GOAL, dec["scene"], dec["verdict"], history)
        tool, args, _ = vlm_compare.classify(action, dec["scene"])
        assert tool is not None
        assert (tool, args) == (row["tool"], row["args"]), row["id"]
        assert vlm_compare.correct(tool, args, dec["accept"]) == row["correct"], row["id"]
        assert rec.last["messages_sha256"] == recon_rows[row["id"]]["messages_sha256"], row["id"]
        assert rec.last["tools_sha256"] == recon_rows[row["id"]]["tools_sha256"], row["id"]


def test_staged_inputs_are_byte_identical_to_the_frozen_set():
    import hashlib

    gpt = json.loads((GPT / "run_frozen11.json").read_text("utf-8"))
    staged = ROOT / "experiments/p6_inputs"
    for key, digest in gpt["sha256"].items():
        if "frames" in key:
            name = key.replace("\\", "/").split("phase1_vlm_compare/")[1]
            assert hashlib.sha256((staged / name).read_bytes()).hexdigest() == digest, name
    allowed = vlm_compare_p6_qwen.DECISIONS_SHA256
    assert hashlib.sha256((staged / "decisions.json").read_bytes()).hexdigest() in allowed
