"""Open-vocabulary box detectors behind one interface, with validated pixel boxes.

Why: Isaac evaluation uses learned perception, and the detector is chosen on development data
(experiments/detector_compare.py). Every candidate sits behind ``Grounder.detect(rgb, classes)``
and returns boxes in the pixel coordinates of the image it was given. ``ModelDetector`` wraps a
grounder in the same frame-in, frame-out callable as ``dream.perception.vlm_detect.VLMDetector``
(``detections`` filled, ``log`` and ``calls`` kept), so toolboxes accept either.

Every box passes ``validate_box`` before it can become a mask and reach depth back-projection:
non-finite, inverted, outside-the-image, mostly-outside and degenerate boxes are rejected and
counted; boxes that overhang the border a little are clipped. A wrong coordinate convention
shows up as mass rejection, which the evaluation reports.

Adapters (licence and revision recorded on each; all Apache-2.0, none gated):
- ``QwenGrounder``: Qwen3-VL (4B at ``ebb281ec``, 8B at ``0c351dd0``) through the existing text
  backend; coordinates 0..1000 relative (Qwen3-VL), checked against segmentation truth.
- ``OwlV2Grounder``: google/owlv2-*-ensemble. The model pads the image to a square at the
  bottom/right, so boxes are rescaled to max(height, width) pixels, then validated.
- ``GroundingDinoGrounder``: IDEA-Research/grounding-dino-*, caption of "a. b. c." phrases.

torch and transformers are imported only when an adapter loads, never by this module.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import numpy as np

from dream.perception.vlm_detect import PROMPT, canonical_labels
from dream.types import RGBDFrame, SemanticDetection

MIN_SIDE_PX = 2.0  # a box thinner than this has no depth support (matches vlm_detect)
MIN_VISIBLE_FRACTION = 0.5  # of a box's own area must lie inside the image; less means the
# coordinates are in the wrong convention or the box is about something outside the view.
# Values 0.5 and 2 px are design choices, not calibrated; rejection counts are reported.


@dataclasses.dataclass(frozen=True)
class RawBox:
    """A model output before validation. ``label`` is the model's own surface form."""

    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the input image


@dataclasses.dataclass(frozen=True)
class Rejection:
    label: str
    reason: str
    box: tuple[float, ...]


def validate_box(
    box: Sequence[float], width: int, height: int
) -> tuple[tuple[float, float, float, float] | None, str, bool]:
    """(clipped box or None, reason if rejected, whether it was clipped).

    Rejects: non-finite values, inverted corners (x2 <= x1 or y2 <= y1; a model that returns
    them is in another convention, so they are not silently re-sorted), boxes with less than
    MIN_VISIBLE_FRACTION of their area inside the image, and boxes thinner than MIN_SIDE_PX
    after clipping. Otherwise clips to [0, width] x [0, height]."""
    if len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
        return None, "not four numbers", False
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None, "non-finite", False
    if x2 <= x1 or y2 <= y1:
        return None, "inverted", False
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(width), x2), min(float(height), y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, "outside image", False
    visible = (cx2 - cx1) * (cy2 - cy1) / ((x2 - x1) * (y2 - y1))
    if visible < MIN_VISIBLE_FRACTION:
        return None, f"mostly outside image ({visible:.2f} visible)", False
    if cx2 - cx1 < MIN_SIDE_PX or cy2 - cy1 < MIN_SIDE_PX:
        return None, "degenerate", False
    return (cx1, cy1, cx2, cy2), "", (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Indices kept by greedy non-maximum suppression (boxes (N, 4) x1y1x2y2)."""
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            ix = max(0.0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0]))
            iy = max(0.0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            inter = ix * iy
            union = (
                (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                + (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                - inter
            )
            if union <= 0 or inter / union <= iou:
                rest.append(j)
        order = rest
    return keep


def infer_conventions(boxes: Iterable[Sequence[float]], width: int, height: int) -> list[str]:
    """Coordinate conventions a model's raw numbers can still be (a diagnostic, not a decision:
    the convention is decided against segmentation truth in experiments/detector_compare.py).

    Any value above 1 rules out ``relative1``; any above 1000 rules out ``relative1000``; an x
    above ``width`` or a y above ``height`` rules out ``absolute``."""
    ok = {"relative1", "relative1000", "absolute"}
    for x1, y1, x2, y2 in boxes:
        if max(x1, y1, x2, y2) > 1.0:
            ok.discard("relative1")
        if max(x1, y1, x2, y2) > 1000.0:
            ok.discard("relative1000")
        if max(x1, x2) > width or max(y1, y2) > height:
            ok.discard("absolute")
    return sorted(ok)


def to_pixels(box: Sequence[float], width: int, height: int, coordinates: str) -> tuple:
    x1, y1, x2, y2 = (float(v) for v in box)
    if coordinates == "relative1000":
        return (x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000)
    if coordinates == "relative1":
        return (x1 * width, y1 * height, x2 * width, y2 * height)
    if coordinates == "absolute":
        return (x1, y1, x2, y2)
    raise ValueError(f"unknown coordinate convention {coordinates!r}")


def parse_answer(
    text: str, width: int, height: int, coordinates: str
) -> tuple[list[RawBox], list[str]]:
    """RawBoxes (pixels, unvalidated) from a Qwen-style JSON answer, and the problems met.
    Equivalent to ``vlm_detect.parse_grounding`` up to validation (tests assert it)."""
    m = re.search(r"\[.*\]", text, re.S)
    if m is None:
        return [], ["no JSON list in the answer"]
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], [f"invalid JSON ({e.msg})"]
    out, problems = [], []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            problems.append("non-object item")
            continue
        box = item.get("bbox_2d")
        if not (isinstance(box, list) and len(box) == 4 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)):  # fmt: skip
            problems.append(f"bad box for {item.get('label')!r}")
            continue
        score = item.get("score", 0.5)
        score = float(score) if isinstance(score, (int, float)) else 0.5
        out.append(
            RawBox(str(item.get("label", "")), score, to_pixels(box, width, height, coordinates))
        )
    return out, problems


class Grounder(Protocol):
    """A box detector for a class list. ``detect`` returns boxes in pixels of ``rgb``."""

    name: str
    has_scores: bool  # False: scores are a constant, not a ranking

    def detect(self, rgb: np.ndarray, classes: Sequence[str]) -> list[RawBox]: ...

    def provenance(self) -> dict: ...


class QwenGrounder:
    """Qwen3-VL grounding through a text backend (``generate(messages, tools) -> str``).

    ``image_scale`` > 1 upsamples the image (bicubic) before it is sent, because the 320x240
    head camera gives the vision tower only ~80 patches; boxes are mapped back to the original
    pixels. Scale 1 sends the image unchanged, as dream.perception.vlm_detect does."""

    has_scores = False

    def __init__(self, backend, name: str, license: str, revision: str | None,
                 coordinates: str = "relative1000", image_scale: float = 1.0) -> None:  # fmt: skip
        self.backend, self.name = backend, name
        self.license, self.revision = license, revision
        self.coordinates, self.image_scale = coordinates, float(image_scale)
        self.last_text = ""

    def detect(self, rgb: np.ndarray, classes: Sequence[str]) -> list[RawBox]:
        h, w = rgb.shape[:2]
        sent = rgb
        if self.image_scale != 1.0:
            from PIL import Image

            size = (round(w * self.image_scale), round(h * self.image_scale))
            sent = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BICUBIC))
        sh, sw = sent.shape[:2]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": sent},
            {"type": "text", "text": PROMPT.format(classes=", ".join(classes))},
        ]}]  # fmt: skip
        self.last_text = self.backend.generate(messages, [])
        boxes, _ = parse_answer(self.last_text, sw, sh, self.coordinates)
        s = (sw / w, sh / h)
        return [
            dataclasses.replace(
                b, box=(b.box[0] / s[0], b.box[1] / s[1], b.box[2] / s[0], b.box[3] / s[1])
            )
            for b in boxes
        ]

    def parse_problems(self, width: int, height: int) -> list[str]:
        return parse_answer(self.last_text, width, height, self.coordinates)[1]

    def provenance(self) -> dict:
        return {"name": self.name, "family": "qwen3-vl", "license": self.license,
                "revision": self.revision, "coordinates": self.coordinates,
                "image_scale": self.image_scale,
                "backend": getattr(self.backend, "provenance", None)}  # fmt: skip


def resolve_phrase(phrase: str, classes: Sequence[str]) -> str | None:
    """The one class a Grounding DINO text span names, else None.

    Exact match first. Otherwise the span's words must all occur in exactly one class name
    (\"cabinet\" inside \"kitchen cabinet\"). A span naming several classes (\"sink countertop\")
    or none is ambiguous and dropped; the caller counts it."""
    p = phrase.casefold().strip()
    forms = {c.casefold(): c for c in classes}
    if p in forms:
        return forms[p]
    words = set(p.split())
    if not words:
        return None
    hits = [c for c in classes if words <= set(c.casefold().split())]
    return hits[0] if len(hits) == 1 else None


class OwlV2Grounder:
    """OWLv2 (Apache-2.0) through transformers' ``Owlv2ForObjectDetection``.

    One query per class (\"a photo of a <class>\"). Boxes come back in the pixels of the
    padded square (the processor pads bottom/right), hence target size max(h, w). Detections
    below ``score_floor`` are dropped; per-class NMS at 0.5 IoU."""

    has_scores = True

    def __init__(self, model_id: str, revision: str, device: str = "cuda",
                 score_floor: float = 0.05, name: str | None = None) -> None:  # fmt: skip
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.name, self.model_id, self.revision = name or model_id, model_id, revision
        self.license, self.score_floor, self.device = "Apache-2.0", score_floor, device
        self._torch = torch
        self.processor = Owlv2Processor.from_pretrained(model_id, revision=revision)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model_id, revision=revision).to(device).eval()
        )

    def detect(self, rgb: np.ndarray, classes: Sequence[str]) -> list[RawBox]:
        from PIL import Image

        h, w = rgb.shape[:2]
        side = max(h, w)
        inputs = self.processor(
            text=[[f"a photo of a {c}" for c in classes]],
            images=Image.fromarray(rgb),
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs=outputs, threshold=self.score_floor, target_sizes=[(side, side)]
        )[0]
        return nms_per_class(
            [
                RawBox(classes[int(i)], float(s), tuple(float(v) for v in b))
                for b, s, i in zip(
                    res["boxes"].tolist(),
                    res["scores"].tolist(),
                    res["labels"].tolist(),
                    strict=True,
                )  # fmt: skip
            ]
        )

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "family": "owlv2",
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "score_floor": self.score_floor,
            "coordinates": "pixels of max(h,w) square",
        }


class GroundingDinoGrounder:
    """Grounding DINO (Apache-2.0) through ``AutoModelForZeroShotObjectDetection``.

    The caption is the lower-case class names joined as \"a. b. c.\". Thresholds are floors
    (box 0.2, text 0.2); the operating threshold is applied in the evaluation. Spans are mapped
    to classes with ``resolve_phrase``; ambiguous spans keep their raw text, so the detector
    wrapper reports them as unknown labels."""

    has_scores = True

    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "cuda",
        box_floor: float = 0.2,
        text_floor: float = 0.2,
        name: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.name, self.model_id, self.revision = name or model_id, model_id, revision
        self.license, self.device = "Apache-2.0", device
        self.box_floor, self.text_floor = box_floor, text_floor
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision)
            .to(device)
            .eval()
        )

    def detect(self, rgb: np.ndarray, classes: Sequence[str]) -> list[RawBox]:
        from PIL import Image

        h, w = rgb.shape[:2]
        caption = " ".join(f"{c.casefold()}." for c in classes)
        inputs = self.processor(images=Image.fromarray(rgb), text=caption,
                                return_tensors="pt").to(self.device)  # fmt: skip
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_floor,
            text_threshold=self.text_floor, target_sizes=[(h, w)],
        )[0]  # fmt: skip
        phrases = res.get("text_labels", res.get("labels"))
        out = []
        for b, s, phrase in zip(
            res["boxes"].tolist(), res["scores"].tolist(), phrases, strict=True
        ):
            out.append(RawBox(resolve_phrase(str(phrase), classes) or str(phrase), float(s),
                              tuple(float(v) for v in b)))  # fmt: skip
        return nms_per_class(out)

    def provenance(self) -> dict:
        return {"name": self.name, "family": "grounding-dino", "model_id": self.model_id,
                "revision": self.revision, "license": self.license,
                "box_floor": self.box_floor, "text_floor": self.text_floor,
                "coordinates": "pixels of the input image (post_process target_sizes)"}  # fmt: skip


def nms_per_class(boxes: list[RawBox], iou: float = 0.5) -> list[RawBox]:
    out = []
    for label in sorted({b.label for b in boxes}):
        same = [b for b in boxes if b.label == label]
        keep = nms(np.array([b.box for b in same]), np.array([b.score for b in same]), iou)
        out += [same[i] for i in keep]
    return out


@dataclasses.dataclass
class Validated:
    detections: list[tuple[str, float, tuple[float, float, float, float], bool]]
    rejections: list[Rejection]


def validate_boxes(raw: Iterable[RawBox], labels: Mapping[str, str], width: int,
                   height: int) -> Validated:  # fmt: skip
    """Canonical-label, validated, clipped detections: (class, score, box, was_clipped)."""
    kept, rejected = [], []
    for r in raw:
        label = labels.get(r.label.casefold().strip())
        if label is None:
            rejected.append(Rejection(r.label, "unknown label", tuple(r.box)))
            continue
        box, reason, clipped = validate_box(r.box, width, height)
        if box is None:
            rejected.append(Rejection(label, reason, tuple(r.box)))
        else:
            kept.append((label, r.score, box, clipped))
    return Validated(kept, rejected)


class ModelDetector:
    """Frame-in, frame-out detector over a ``Grounder``: the interface of ``VLMDetector``.

    Boxes below ``min_score`` are dropped (0 keeps all; Qwen has no scores). Each accepted box
    becomes a rectangular mask, exactly as VLMDetector builds them."""

    def __init__(self, grounder: Grounder, vocabulary: Mapping[str, Iterable[str]],
                 min_score: float = 0.0) -> None:  # fmt: skip
        self.grounder, self.min_score = grounder, min_score
        self.labels = canonical_labels(vocabulary)
        self.classes = sorted(c.replace("_", " ") for c in vocabulary)
        self.calls = 0
        self.log: list[dict] = []

    def __call__(self, frame: RGBDFrame) -> RGBDFrame:
        h, w = frame.depth.shape
        raw = self.grounder.detect(frame.rgb, self.classes)
        self.calls += 1
        result = validate_boxes(raw, self.labels, w, h)
        dets = []
        for label, score, (x1, y1, x2, y2), _ in result.detections:
            if score < self.min_score:
                continue
            mask = np.zeros((h, w), bool)
            mask[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = True
            dets.append(SemanticDetection(label, score, mask, self.grounder.name))
        self.log.append(
            {
                "time": frame.time,
                "labels": [d.label for d in dets],
                "rejected": [dataclasses.asdict(r) for r in result.rejections],
            }
        )
        return dataclasses.replace(frame, detections=tuple(dets))
