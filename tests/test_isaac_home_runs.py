"""CPU-side helpers of the composed-home runner: viewpoints, instance boxes, recognisability."""

import json
import math
import pathlib

import numpy as np

from dream.sim.home import footprint, room_at, sample_home
from experiments import isaac_home_runs as r

CFG = json.loads(
    (pathlib.Path(r.__file__).parent / "configs/isaac_home_dev.json").read_text(encoding="utf-8")
)


def homes():
    for h in CFG["homes"]:
        yield sample_home(
            np.random.default_rng(h["seed"]), h["scenario"], h["target"], CFG["variant"], h["seed"]
        )


def test_dev_config_is_development_only_and_uses_dev_seeds():
    assert CFG["status"] == "development"
    assert all(100 <= h["seed"] <= 103 for h in CFG["homes"])


def test_viewpoints_are_in_band_clear_and_face_the_object():
    total = 0
    for home in homes():
        for obj in home.objects:
            for x, y, yaw in r.viewpoints_for(home, obj, 3):
                total += 1
                d = float(footprint(obj).distance(np.array([[x, y]]))[0])
                assert r.NEAR_M - 1e-6 <= d <= r.FAR_M + 1e-6
                assert room_at(home, (x, y)).id == room_at(home, obj.xy).id
                want = math.atan2(obj.xy[1] - y, obj.xy[0] - x)
                assert abs(math.remainder(yaw - want, 2 * math.pi)) < 1e-9
    assert total > 20  # most objects have somewhere to stand in their room


def test_instance_boxes_use_prim_prefix_and_ignore_other_ids():
    ids = np.zeros((10, 10), int)
    ids[2:5, 3:8] = 7
    ids[6:8, 1:3] = 9
    paths = {7: "/World/Home/o001/Geom/mesh", 9: "/World/G1/torso", 0: "BACKGROUND"}
    out = r.instance_boxes(
        ids, paths, {"inst-a": "/World/Home/o001", "inst-b": "/World/Home/o0010"}
    )
    assert out == {"inst-a": {"pixels": 15, "box": [3.0, 2.0, 8.0, 5.0]}}


def test_recognisability_counts_label_and_iou_hits_and_confusions():
    truth = [0, 0, 100, 100]
    rows = [
        {"class": "sink", "truth_box": truth, "detections": [{"label": "sink", "box": truth}]},
        {
            "class": "sink",
            "truth_box": truth,
            "detections": [{"label": "sink", "box": [60, 60, 200, 200]}],
        },
        {
            "class": "sink",
            "truth_box": truth,
            "detections": [{"label": "countertop", "box": truth}],
        },
    ]
    out = r.recognisability(rows)["sink"]
    assert (out["views"], out["label_hit"], out["iou_hit"]) == (3, 2, 1)
    assert out["confused_as"] == {"countertop": 1}
