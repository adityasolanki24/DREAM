"""CPU tests of the development-kitchen measurements (coordinate check, Qwen room priors)."""

import json
import math

import numpy as np

from dream.perception.vlm_detect import canonical_labels
from experiments import isaac_semantic_dev as dev
from experiments import qwen_room_priors as priors

CAMERA = {
    "position": [0.0, 0.0, 1.0],
    "forward": [1.0, 0.0, 0.0],
    "up": [0.0, 0.0, 1.0],
    "fovy_deg": 58.0,
    "aspect": 320 / 240,
}


def test_projection_of_a_box_straight_ahead_is_centred_and_ordered():
    box = dev.project_box(CAMERA, [2.0, -0.5, 0.5], [3.0, 0.5, 1.5], 320, 240)
    x0, y0, x1, y1 = box["box"]
    assert abs((x0 + x1) / 2 - 160) < 1e-6 and abs((y0 + y1) / 2 - 120) < 1e-6
    assert x0 < x1 and y0 < y1 and box["visible_fraction"] == 1.0
    right = dev.project_box(CAMERA, [2.0, -2.0, 0.5], [3.0, -1.0, 1.5], 320, 240)
    assert right["full_box"][0] > 160  # +y is left of the camera; -y is image right


def test_box_behind_the_camera_has_no_projection_and_clipping_reports_the_fraction():
    assert dev.project_box(CAMERA, [-3.0, -0.5, 0.5], [-2.0, 0.5, 1.5], 320, 240) is None
    wide = dev.project_box(CAMERA, [1.0, -3.0, 0.5], [2.0, 3.0, 1.5], 320, 240)
    assert 0.0 < wide["visible_fraction"] < 1.0


def test_iou():
    assert dev.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert dev.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert math.isclose(dev.iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)


def frame_scores(answer, truth_box):
    labels = canonical_labels({"refrigerator": ["fridge"], "sink": []})
    return {"scores": dev.score_frame(answer, 320, 240, labels, {"refrigerator": truth_box})}


def test_the_convention_with_matching_boxes_is_selected_by_the_preregistered_rule():
    truth = [80.0, 24.0, 240.0, 216.0]  # 25%..75% of 320 x 10%..90% of 240
    relative = json.dumps([{"label": "refrigerator", "bbox_2d": [250, 100, 750, 900]}])
    absolute = json.dumps([{"label": "refrigerator", "bbox_2d": [80, 24, 240, 216]}])
    rel_frames = [frame_scores(relative, truth) for _ in range(3)]
    abs_frames = [frame_scores(absolute, truth) for _ in range(3)]
    assert dev.summarise(rel_frames)["evidence_says"] == "relative1000"
    assert dev.summarise(abs_frames)["evidence_says"] == "absolute"
    assert dev.summarise(rel_frames)["per_convention"]["relative1000"]["matched_at_0.5"] == 3


def test_a_tie_or_no_detection_keeps_the_documented_convention():
    summary = dev.summarise([frame_scores("[]", [0, 0, 100, 100])])
    assert summary["evidence_says"] == "relative1000"


def test_candidate_poses_face_the_object_and_stand_on_free_cells():
    xs, ys = np.meshgrid(np.arange(-4, 4, 0.1), np.arange(-4, 4, 0.1))
    cells = np.column_stack([xs.ravel(), ys.ravel()])
    poses = dev.candidate_poses((-0.2, -0.2), (0.2, 0.2), cells)
    assert 30 <= len(poses) <= 40
    for x, y, yaw in poses:
        gap = np.maximum(
            np.maximum(np.array([-0.2, -0.2]) - [x, y], [x, y] - np.array([0.2, 0.2])), 0
        )
        assert 0.8 <= np.linalg.norm(gap) <= 3.5
        assert math.isclose(math.atan2(-y, -x), yaw, abs_tol=1e-9)
    bearings = {round(math.degrees(math.atan2(y, x)) / 45) for x, y, _ in poses}
    assert len(bearings) >= 7  # spread all around, not clustered
    assert dev.candidate_poses((0, 0), (1, 1), np.array([[0.5, 0.5]])) == []


def test_detection_outside_the_image_scores_zero_not_none():
    labels = canonical_labels({"sink": []})
    answer = json.dumps([{"label": "sink", "bbox_2d": [328, 50, 660, 178]}])
    scored = dev.score_frame(answer, 320, 240, labels, {"sink": [100.0, 0.0, 200.0, 60.0]})
    assert scored["absolute"]["best_iou_by_class"]["sink"] == 0.0
    assert dev.summarise([{"scores": scored}])["per_convention"]["absolute"]["targets_scored"] == 1


ROOMS = ["kitchen", "bedroom"]


def test_priors_parse_valid_and_report_malformed_answers():
    good = priors.parse_answer('{"kitchen": 0.7, "bedroom": 0.2, "none": 0.1}', ROOMS)
    assert good["valid"] and math.isclose(good["sum"], 1.0)
    assert not priors.parse_answer('{"kitchen": 0.7}', ROOMS)["valid"]
    assert not priors.parse_answer("I think kitchen.", ROOMS)["valid"]
    assert not priors.parse_answer('{"kitchen": "high", "bedroom": 0, "none": 0}', ROOMS)["valid"]
    unnormalised = priors.parse_answer('{"kitchen": 2, "bedroom": 2, "none": 0}', ROOMS)
    assert unnormalised["sum"] == 4 and unnormalised["normalised"]["kitchen"] == 0.5


def test_priors_prompt_is_text_only_and_names_every_key():
    messages = priors.build_messages("kettle", ROOMS)
    assert "image" not in {p["type"] for m in messages for p in m["content"]}
    text = json.dumps(messages)
    assert "kettle" in text and "bedroom" in text and "none" in text
