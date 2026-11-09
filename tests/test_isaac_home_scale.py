"""The no-squash scale rule on the real asset manifest and the dev homes (CPU)."""

import json
import pathlib

import numpy as np
import pytest

from dream.sim.home import ObjectSpec, Rect, sample_home
from experiments import isaac_home_compose as comp
from experiments.isaac_home_compose import _overlap

CFG = json.loads(
    pathlib.Path(comp.__file__).parent.joinpath("configs/isaac_home_dev.json").read_text()
)


def plan(variant, seed, scenario, target):
    home = sample_home(np.random.default_rng(seed), scenario, target, variant, seed)
    manifest = comp.load_manifest()
    comp.check_case(home, manifest)
    return home, comp.plan_placements(home, manifest)


def check_rule(home, placements):
    walls = [Rect(w.centre[0], w.centre[1], w.half[0], w.half[1]) for w in home.walls]
    rects = {
        p.instance_id: Rect(p.xy[0], p.xy[1], p.size_m[0] / 2, p.size_m[1] / 2, p.yaw)
        for p in placements
    }
    for p in placements:
        base = p.scale[2]
        if p.mode == "fit":
            assert comp.SCALE_MIN - 1e-9 <= base <= comp.SCALE_MAX + 1e-9
            assert all(abs(k / base - 1) <= comp.AXIS_DEV + 1e-9 for k in p.scale)
        else:
            assert p.scale == (1.0, 1.0, 1.0)
            r = Rect(p.xy[0], p.xy[1], p.size_m[0] / 2 - 0.02, p.size_m[1] / 2 - 0.02, p.yaw)
            assert not any(_overlap(r, w) for w in walls)
            assert not any(_overlap(r, o) for i, o in rects.items() if i != p.instance_id)


@pytest.mark.parametrize("h", CFG["homes"], ids=lambda h: f"{h['seed']}{h['scenario']}")
def test_dev_homes_follow_the_rule_or_are_refused_with_a_reason(h):
    try:
        home, placements = plan(CFG["variant"], h["seed"], h["scenario"], h["target"])
    except comp.MissingAssetError as refusal:
        assert refusal.classes  # a refusal always says which class and why
        return
    check_rule(home, placements)


def test_anchored_native_keeps_the_back_face_shifts_laterally_and_refuses_collisions():
    obj = ObjectSpec("x", "sofa", 0, (0.0, 0.0), 0.0, (0.3, 0.5, 0.4), ())
    entry = {"size_m": [0.9, 1.2, 0.8]}
    xy, shift = comp.anchored_native(entry, obj, [], [])
    assert abs((xy[0] - 0.45) - (0.0 - 0.3)) < 1e-9  # back face at x = -0.3 unchanged
    assert np.allclose(shift, (0.15, 0.0))
    block = Rect(0.15, 0.55, 0.1, 0.1)  # an object at one end: a lateral shift clears it
    xy2, shift2 = comp.anchored_native(entry, obj, [block], [])
    assert shift2 != shift or xy2 != xy
    wall = Rect(0.0, 0.0, 5.0, 0.05)
    with pytest.raises(ValueError):
        comp.anchored_native(entry, obj, [], [wall])


def test_fit_rule_is_uniform_when_possible_and_bounded_otherwise():
    obj = ObjectSpec("x", "sofa", 0, (0.0, 0.0), 0.0, (0.5, 0.5, 0.4), ())
    assert comp.fit_scale({"size_m": [1.0, 1.0, 1.0]}, obj) == (1.0, 1.0, 1.0)  # fits at native
    assert comp.fit_scale({"size_m": [10.0, 10.0, 1.0]}, obj) is None  # far too large
    d, w, h = comp.fit_scale({"size_m": [1.4, 1.0, 1.0]}, obj)  # 0.71 and 1.0: uniform 0.75, d 0.71
    assert h == comp.SCALE_MIN and abs(d / h - 1) <= comp.AXIS_DEV + 1e-9
