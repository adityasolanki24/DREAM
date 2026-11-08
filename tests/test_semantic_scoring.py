"""The independent search scorer and the VLM grounding parser."""

import numpy as np
import pytest

from dream.eval.semantic_scoring import TrueInstance, odometry_to_world, score_episode, score_rooms
from dream.perception.vlm_detect import canonical_labels, parse_grounding

FRIDGE = TrueInstance("o1", "refrigerator", np.array([3.0, 0.0]), np.array([0.35, 0.35]), 0.0,
                      "kitchen")  # fmt: skip
WARDROBE = TrueInstance("o2", "wardrobe", np.array([-3.0, 0.0]), np.array([0.3, 0.5]), 0.0,
                        "bedroom")  # fmt: skip
IDENTITY = odometry_to_world((0, 0, 0), (0, 0, 0))


def run(**kw):
    base = dict(
        target="refrigerator", outcome="visually_localized", supported_expected=True,
        claimed=False, completion=None, claim_xy_odom=None, instances=[FRIDGE, WARDROBE],
        to_world=IDENTITY, true_pose_end=(0, 0, 0),
    )  # fmt: skip
    return score_episode(**{**base, **kw})


@pytest.mark.parametrize(
    "kw, result",
    [
        (dict(claimed=True, claim_xy_odom=(2.7, 0.1)), "success"),
        (dict(claimed=True, claim_xy_odom=(-3.0, 0.0)), "false_success"),  # the wardrobe
        (dict(claimed=True, claim_xy_odom=(0.0, 3.0)), "false_success"),  # nothing there
        (dict(completion="search_exhausted"), "false_absence"),
        (dict(completion="target_not_yet_found"), "missed"),
        (dict(completion="search_exhausted", instances=[WARDROBE]), "correct_refusal"),
        (dict(supported_expected=False, completion="goal_unsupported"), "correct_refusal"),
        (dict(supported_expected=False, claimed=True, claim_xy_odom=(3, 0)), "false_success"),
        (dict(completion="goal_unsupported"), "wrong_refusal"),
        (dict(claimed=True, claim_xy_odom=(3, 0), outcome="reached"), "false_success"),
        (
            dict(
                claimed=True, claim_xy_odom=(3, 0), outcome="reached", true_pose_end=(1.8, 0.0, 0.0)
            ),
            "success",
        ),  # fmt: skip
    ],
)
def test_original_request_is_scored(kw, result):
    out = run(**kw)
    assert out["result"] == result
    assert out["false_success_claim"] == (result == "false_success")


def test_identity_error_is_reported():
    out = run(claimed=True, claim_xy_odom=(-3.0, 0.2))
    assert out["identity_error"] and out["claim_nearest"]["cls"] == "wardrobe"


def test_odometry_to_world_compensates_drift():
    f = odometry_to_world((1.0, 2.0, np.pi / 2), (0.0, 0.0, 0.0))
    assert np.allclose(f((1.0, 0.0)), (1.0, 3.0))


def test_room_scoring_counts_ambiguous_apart():
    regions = [{"id": 1, "centroid": [3, 0], "classification": "kitchen"},
               {"id": 2, "centroid": [-3, 0], "classification": "ambiguous"},
               {"id": 3, "centroid": [-3, 1], "classification": "garage"}]  # fmt: skip
    out = score_rooms(regions, IDENTITY, lambda xy: "kitchen" if xy[0] > 0 else "bedroom")
    assert (out["correct"], out["incorrect"], out["ambiguous"]) == (1, 1, 1)


def test_grounding_parser_maps_synonyms_and_scales_boxes():
    labels = canonical_labels({"refrigerator": ("fridge",), "sink": ()})
    text = (
        'Sure: [{"label": "fridge", "bbox_2d": [500, 0, 1000, 1000]}, '
        '{"label": "sofa", "bbox_2d": [0, 0, 10, 10]}, {"label": "sink", "bbox_2d": [1, 2]}]'
    )
    dets, problems = parse_grounding(text, 320, 240, labels)
    assert [d.label for d in dets] == ["refrigerator"]
    assert dets[0].mask[:, 160:].all() and not dets[0].mask[:, :160].any()
    assert len(problems) == 2
    assert parse_grounding("nothing here", 320, 240, labels) == ([], ["no JSON list in the answer"])
    absolute, _ = parse_grounding('[{"label": "sink", "bbox_2d": [10, 10, 50, 50]}]', 320, 240,
                                  labels, coordinates="absolute")  # fmt: skip
    assert absolute[0].mask.sum() == 40 * 40
