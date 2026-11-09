"""Box detector adapters: validation, coordinate handling and the frame-detector wrapper.

No model is loaded: backends and grounders are stubs."""

import numpy as np
import pytest

from dream.perception.box_detectors import (
    ModelDetector,
    QwenGrounder,
    RawBox,
    infer_conventions,
    nms,
    nms_per_class,
    parse_answer,
    resolve_phrase,
    validate_box,
    validate_boxes,
)
from dream.perception.vlm_detect import canonical_labels, parse_grounding
from dream.types import CameraCalibration, RGBDFrame

W, H = 320, 240
VOCAB = {"sink": ["basin"], "kitchen_cabinet": ["cabinet"], "refrigerator": ["fridge"]}


def frame() -> RGBDFrame:
    cal = CameraCalibration(W, H, 58.0, np.zeros(3), np.eye(3))
    return RGBDFrame(0.0, np.zeros((H, W, 3), np.uint8), np.ones((H, W), np.float32),
                     np.zeros((H, W), bool), cal)  # fmt: skip


@pytest.mark.parametrize(
    "box,reason",
    [
        ((float("nan"), 0, 10, 10), "non-finite"),
        ((50, 50, 20, 80), "inverted"),
        ((50, 80, 90, 80), "inverted"),
        ((400, 10, 500, 50), "outside image"),
        ((300, 10, 600, 50), "mostly outside"),  # 20 of 300 px inside
        ((10, 10, 11, 50), "degenerate"),
    ],
)
def test_rejected_boxes_never_reach_depth(box, reason):
    clipped, why, _ = validate_box(box, W, H)
    assert clipped is None and reason in why


def test_overhang_is_clipped_and_flagged_but_inside_boxes_are_untouched():
    assert validate_box((-3, 10, 100, 60), W, H) == ((0.0, 10.0, 100.0, 60.0), "", True)
    assert validate_box((10, 10, 100, 60), W, H) == ((10.0, 10.0, 100.0, 60.0), "", False)
    assert validate_box((0, 0, W, H), W, H)[0] == (0.0, 0.0, float(W), float(H))


def test_wrong_convention_is_caught_as_mass_rejection():
    """A 0..1000 box read as pixels overflows a 320x240 image; read correctly it fits."""
    answer = '[{"label": "sink", "bbox_2d": [600, 500, 900, 900]}]'
    labels = canonical_labels(VOCAB)
    raw_abs, _ = parse_answer(answer, W, H, "absolute")
    assert validate_boxes(raw_abs, labels, W, H).detections == []
    raw_rel, _ = parse_answer(answer, W, H, "relative1000")
    (det,) = validate_boxes(raw_rel, labels, W, H).detections
    assert det[2] == pytest.approx((192.0, 120.0, 288.0, 216.0))
    assert infer_conventions([(600, 500, 900, 900)], W, H) == ["relative1000"]
    assert infer_conventions([(0.1, 0.2, 0.5, 0.9)], W, H) == [
        "absolute",
        "relative1",
        "relative1000",
    ]


def test_parse_answer_agrees_with_the_existing_parser_on_wellformed_answers():
    answer = (
        '[{"label": "sink", "bbox_2d": [0, 186, 148, 325]},'
        ' {"label": "cabinet", "bbox_2d": [0, 375, 1000, 998]}]'
    )
    labels = canonical_labels(VOCAB)
    old, _ = parse_grounding(answer, W, H, labels, "relative1000")
    new = validate_boxes(parse_answer(answer, W, H, "relative1000")[0], labels, W, H).detections
    assert [d.label for d in old] == [d[0] for d in new]
    for d, (_, _, box, _) in zip(old, new, strict=True):
        ys, xs = np.nonzero(d.mask)
        assert (xs.min(), ys.min()) == (int(box[0]), int(box[1]))


def test_malformed_items_are_reported_not_raised():
    boxes, problems = parse_answer(
        '[1, {"label": "sink", "bbox_2d": [1, 2]}, {"label": "sink", "bbox_2d": [true, 2, 3, 4]}]',
        W,
        H,
        "absolute",
    )
    assert boxes == [] and len(problems) == 3
    assert parse_answer("none", W, H, "absolute")[1]
    assert parse_answer("[{", W, H, "absolute")[1]


def test_nms_keeps_best_of_overlapping_and_both_of_distinct():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    assert nms(boxes, np.array([0.5, 0.9, 0.4]), 0.5) == [1, 2]
    same = [RawBox("a", 0.9, (0, 0, 10, 10)), RawBox("a", 0.8, (0, 0, 10, 10)),
            RawBox("b", 0.7, (0, 0, 10, 10))]  # fmt: skip
    assert sorted((b.label, b.score) for b in nms_per_class(same)) == [("a", 0.9), ("b", 0.7)]


def test_resolve_phrase_is_exact_or_unique():
    classes = ["sink", "kitchen cabinet", "countertop"]
    assert resolve_phrase("Kitchen Cabinet", classes) == "kitchen cabinet"
    assert resolve_phrase("cabinet", classes) == "kitchen cabinet"
    assert resolve_phrase("sink countertop", classes) is None  # two classes in one span
    assert resolve_phrase("bed", classes) is None
    assert resolve_phrase("", classes) is None


class Backend:
    provenance = {"model_id": "stub"}

    def __init__(self, text):
        self.text, self.sent = text, None

    def generate(self, messages, tools):
        self.sent = messages
        return self.text


def test_qwen_grounder_maps_boxes_back_from_an_upscaled_image():
    backend = Backend('[{"label": "sink", "bbox_2d": [100, 100, 500, 500]}]')
    g = QwenGrounder(backend, "q", "Apache-2.0", "rev", image_scale=2.0)
    (box,) = g.detect(np.zeros((H, W, 3), np.uint8), ["sink"])
    assert backend.sent[0]["content"][0]["image"].shape[:2] == (2 * H, 2 * W)
    assert box.box == pytest.approx((32.0, 24.0, 160.0, 120.0))  # relative-1000 is scale-free
    g_abs = QwenGrounder(Backend('[{"label": "sink", "bbox_2d": [64, 48, 320, 240]}]'), "q",
                         "Apache-2.0", "rev", coordinates="absolute", image_scale=2.0)  # fmt: skip
    (b2,) = g_abs.detect(np.zeros((H, W, 3), np.uint8), ["sink"])
    assert b2.box == pytest.approx((32.0, 24.0, 160.0, 120.0))  # absolute is in sent pixels


class StubGrounder:
    name, has_scores = "stub", True

    def detect(self, rgb, classes):
        return [
            RawBox("sink", 0.9, (10, 20, 60, 70)),
            RawBox("basin", 0.2, (-5, 20, 60, 70)),  # synonym, low score, overhangs
            RawBox("bed", 0.8, (10, 20, 60, 70)),  # not in the vocabulary
            RawBox("sink", 0.8, (900, 900, 950, 950)),  # off-image
        ]

    def provenance(self):
        return {}


def test_model_detector_has_the_vlm_detector_interface_and_masks():
    det = ModelDetector(StubGrounder(), VOCAB, min_score=0.3)
    out = det(frame())
    assert det.calls == 1 and out.detections is not None
    assert [d.label for d in out.detections] == ["sink"]  # synonym dropped by score only
    mask = out.detections[0].mask
    assert mask.shape == (H, W) and mask.sum() == 50 * 50 and mask[20, 10] and not mask[70, 60]
    reasons = sorted(r["reason"] for r in det.log[0]["rejected"])
    assert reasons == ["outside image", "unknown label"]
    assert ModelDetector(StubGrounder(), VOCAB)(frame()).detections[1].label == "sink"
