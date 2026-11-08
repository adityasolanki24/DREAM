"""Open-vocabulary object detection with a vision-language model (Qwen3-VL grounding).

The model is asked, per frame, for boxes of the classes in a fixed vocabulary; its labels are
mapped to canonical classes through the vocabulary's synonyms (anything else is dropped) and
each box becomes a pixel mask. Depth then removes the background inside the box
(dream/perception/semantic.region_points). A model detection is a fallible observation like
any other: never a fact, never verification on its own.

Coordinates: Qwen3-VL reports ``bbox_2d`` in relative units 0..1000 (``COORDINATES =
"relative1000"``); "absolute" pixels is the Qwen2.5-VL convention. The convention in use is
checked against projected ground truth in evaluation before results are reported, never
assumed from documentation alone.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Mapping

import numpy as np

from dream.types import RGBDFrame, SemanticDetection

COORDINATES = "relative1000"
PROMPT = (
    "List every object of these classes that is visible in the image: {classes}. "
    'Answer with JSON only: a list of {{"label": <one of the classes>, "bbox_2d": '
    "[x1, y1, x2, y2]}} with coordinates scaled 0-1000 across the image width and height. "
    "Answer [] if none is visible. Do not list objects you cannot see."
)


def canonical_labels(vocabulary: Mapping[str, Iterable[str]]) -> dict[str, str]:
    """surface form (lower case, spaces) -> canonical class."""
    out = {}
    for canonical, forms in vocabulary.items():
        out[canonical.replace("_", " ")] = canonical
        out[canonical] = canonical
        for f in forms:
            out[f.casefold().strip()] = canonical
    return out


def parse_grounding(
    text: str,
    width: int,
    height: int,
    labels: Mapping[str, str],
    coordinates: str = COORDINATES,
    source: str = "qwen3vl_grounding",
) -> tuple[list[SemanticDetection], list[str]]:
    """Detections from a model answer, and the problems met (malformed or unknown items)."""
    problems: list[str] = []
    m = re.search(r"\[.*\]", text, re.S)
    if m is None:
        return [], ["no JSON list in the answer"]
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], [f"invalid JSON ({e.msg})"]
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            problems.append("non-object item")
            continue
        label = labels.get(str(item.get("label", "")).casefold().strip())
        box = item.get("bbox_2d")
        if label is None:
            problems.append(f"unknown label {item.get('label')!r}")
            continue
        if not (isinstance(box, list) and len(box) == 4 and all(
                isinstance(v, (int, float)) for v in box)):  # fmt: skip
            problems.append(f"bad box for {label}")
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if coordinates == "relative1000":
            x1, x2 = x1 * width / 1000, x2 * width / 1000
            y1, y2 = y1 * height / 1000, y2 * height / 1000
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            problems.append(f"degenerate box for {label}")
            continue
        mask = np.zeros((height, width), bool)
        mask[int(y1) : int(np.ceil(y2)), int(x1) : int(np.ceil(x2))] = True
        score = item.get("score", 0.5)
        out.append(SemanticDetection(label, float(score) if isinstance(score, (int, float))
                                     else 0.5, mask, source))  # fmt: skip
    return out, problems


class VLMDetector:
    """Runs a backend (``generate(messages, tools) -> text``) on a frame's RGB image."""

    def __init__(self, backend, vocabulary: Mapping[str, Iterable[str]],
                 coordinates: str = COORDINATES) -> None:  # fmt: skip
        self.backend, self.coordinates = backend, coordinates
        self.labels = canonical_labels(vocabulary)
        self.classes = sorted(c.replace("_", " ") for c in vocabulary)
        self.calls = 0
        self.log: list[dict] = []

    def __call__(self, frame: RGBDFrame) -> RGBDFrame:
        h, w = frame.depth.shape
        messages = [{"role": "user", "content": [
            {"type": "image", "image": frame.rgb},
            {"type": "text", "text": PROMPT.format(classes=", ".join(self.classes))},
        ]}]  # fmt: skip
        text = self.backend.generate(messages, [])
        self.calls += 1
        dets, problems = parse_grounding(text, w, h, self.labels, self.coordinates)
        self.log.append({"time": frame.time, "answer": text[:600], "problems": problems,
                         "labels": [d.label for d in dets]})  # fmt: skip
        return dataclasses.replace(frame, detections=tuple(dets))
