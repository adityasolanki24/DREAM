"""The post-fix diagnostic config: each arm declares its renderer and seeds; seeds match the
frozen fresh and historical sets."""

import json
import pathlib

from experiments.g1_explore_room import select_arm

CFGS = pathlib.Path(__file__).resolve().parents[1] / "experiments/configs"


def test_arms_declare_renderer_and_seeds():
    cfg = json.loads((CFGS / "g1_explore_room_postfix_closeout.json").read_text())
    fresh = {"nominal": [1300, 1320], "shifted": [2300, 2310], "no_cabinet": [3100, 3110]}
    hist = {"nominal": [1000, 1020], "shifted": [2000, 2010], "no_cabinet": [3000, 3006]}
    expect = {
        "R1": ("osmesa", fresh),
        "R2_fresh": ("egl", fresh),
        "R2_hist": ("egl", hist),
        "R3": ("osmesa", hist),
    }
    for arm, (renderer, seeds) in expect.items():
        a = select_arm(cfg, arm)
        assert a["renderer"] == renderer and a["seeds"] == seeds and a["methods"] == ["gain"]
        assert a["time_budget_s"] == 240 and a["max_steps"] == 60
    assert cfg["status"] == "frozen" and "NOT held-out" in cfg["label"]
