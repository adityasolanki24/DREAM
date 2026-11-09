"""Detector comparison runner: scoring contracts on tiny hand-made frames and a CPU pipeline
check with stub detectors on real captured frames (no model downloads)."""

import json
import pathlib

import numpy as np
import pytest

from experiments import detector_compare as dc

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = {
    "sink": {"raw": ["sink", "sink_basin", "sink_rim"], "merge": True},
    "countertop": {"raw": ["countertop"]},
}


def inst(cls, box, pixels=1000):
    return {"cls": cls, "box": list(box), "pixels": pixels}


def det(label, box, score=0.9):
    return {"label": label, "score": score, "box": list(box)}


def frame(truth, dets):
    return {
        "truth": truth,
        "dets": dets,
        "raw_truth": [{"class": t["cls"], "box": t["box"], "pixels": t["pixels"]} for t in truth],
    }


def test_merged_truth_is_one_union_instance():
    gt = {
        "a": {"class": "sink_basin", "pixels": 500, "box": [10, 10, 30, 30]},
        "b": {"class": "sink_rim", "pixels": 600, "box": [5, 8, 40, 32]},
        "c": {"class": "countertop", "pixels": 3000, "box": [0, 0, 100, 40]},
    }
    got = dc.truth_instances(gt, CLASSES)
    sink = [t for t in got if t["cls"] == "sink"]
    assert len(sink) == 1 and sink[0]["box"] == [5, 8, 40, 32] and sink[0]["pixels"] == 1100
    assert len([t for t in got if t["cls"] == "countertop"]) == 1


def test_precision_recall_hits_misses_and_small_truths_ignored():
    truth = [
        inst("sink", (0, 0, 100, 100)),
        inst("sink", (200, 0, 300, 100)),
        inst("sink", (0, 200, 20, 220), pixels=50),
    ]
    dets = [
        det("sink", (0, 0, 100, 90)),  # IoU 0.9 hit
        det("sink", (400, 400, 450, 450)),  # false positive
        det("sink", (0, 200, 20, 220)),
    ]  # matches the ignored small truth: neither
    r = dc.score_precision_recall([frame(truth, dets)], {"sink": CLASSES["sink"]}, 0.0, False, 0.5)
    s = r["per_class"]["sink"]
    assert (s["tp"], s["fp"], s["fn"], s["ignored"]) == (1, 1, 1, 1)
    assert r["micro"]["precision"] == 0.5 and r["micro"]["recall"] == 0.5


def test_iou_threshold_changes_the_verdict_and_a_truth_matches_once():
    truth = [inst("sink", (0, 0, 100, 100))]
    loose = det("sink", (0, 0, 100, 40))  # IoU 0.4
    for thr, tp in ((0.5, 0), (0.3, 1)):
        r = dc.score_precision_recall(
            [frame(truth, [loose])], {"sink": CLASSES["sink"]}, 0, False, thr
        )
        assert r["micro"]["tp"] == tp
    twice = [det("sink", (0, 0, 100, 100), 0.9), det("sink", (0, 0, 100, 99), 0.8)]
    r = dc.score_precision_recall([frame(truth, twice)], {"sink": CLASSES["sink"]}, 0, True, 0.5)
    assert (r["micro"]["tp"], r["micro"]["fp"]) == (1, 1)


def test_score_threshold_filters_scored_detectors_only():
    truth = [inst("sink", (0, 0, 100, 100))]
    low = det("sink", (0, 0, 100, 100), 0.2)
    kw = dict(classes={"sink": CLASSES["sink"]}, iou_thr=0.5)
    assert (
        dc.score_precision_recall([frame(truth, [low])], threshold=0.5, has_scores=True, **kw)[
            "micro"
        ]["tp"]
        == 0
    )
    assert (
        dc.score_precision_recall([frame(truth, [low])], threshold=0.5, has_scores=False, **kw)[
            "micro"
        ]["tp"]
        == 1
    )


def test_absent_class_detections_are_false_positives_and_listed():
    r = dc.score_precision_recall([frame([], [det("bed", (0, 0, 10, 10))])], CLASSES, 0, False, 0.5)
    assert r["absent_class_fp"] == {"bed": 1} and r["micro"]["fp"] == 1
    assert r["micro_excluding_absent_classes"]["fp"] == 0


def test_look_alike_confusion_counts_label_on_other_class_instance():
    truth = [inst("countertop", (0, 0, 100, 40))]
    f = frame(truth, [det("sink", (0, 0, 100, 38))])
    f["raw_truth"].append({"class": "backsplash", "box": [0, 60, 100, 100], "pixels": 900})
    f["dets"].append(det("sink", (0, 60, 100, 98)))
    got = dc.confusions([f], "raw_truth", CLASSES, 0, False)
    assert got == {"sink": {"countertop": 1, "backsplash": 1}}


def test_nested_pairs_are_geometric_and_metric_tracks_swallowing():
    outer, inner = inst("countertop", (0, 0, 300, 100)), inst("sink", (100, 20, 160, 70))
    assert dc.nested_pairs([outer, inner]) == [(inner, outer)]
    assert (
        dc.nested_pairs([inst("countertop", (0, 0, 100, 100)), inst("sink", (0, 0, 90, 90))]) == []
    )
    truth = [outer, inner]
    swallowed = frame(truth, [det("countertop", (0, 0, 300, 100))])
    both = frame(truth, [det("countertop", (0, 0, 300, 100)), det("sink", (100, 20, 160, 70))])
    m = dc.nested_metric([swallowed, both], 0, False, 0.5)["sink in countertop"]
    assert m == {
        "pairs": 2,
        "inner_found": 1,
        "outer_found": 2,
        "both_found": 1,
        "nested_detected": 1,
        "swallowed": 1,
    }


def test_convention_rule_prefers_documented_unless_absolute_wins_on_both_counts():
    truth = [inst("sink", (80, 24, 240, 216))]
    right = [det("sink", (80, 24, 240, 216))]
    f = {"truth": truth, "by_convention": {"relative1000": right, "absolute": []}}
    assert dc.convention_evidence([f], CLASSES)["evidence_says"] == "relative1000"
    f["by_convention"] = {"relative1000": [], "absolute": right}
    assert dc.convention_evidence([f], CLASSES)["evidence_says"] == "absolute"
    f["by_convention"] = {"relative1000": [], "absolute": []}
    assert dc.convention_evidence([f], CLASSES)["evidence_says"] == "relative1000"


@pytest.mark.skipif(not (ROOT / "experiments/data/detector_dev").is_dir(), reason="dataset absent")
def test_pipeline_runs_on_two_real_frames_with_stub_detectors(tmp_path):
    cfg = json.loads((ROOT / "experiments/configs/detector_compare.json").read_text("utf-8"))
    frames = dc.load_item(
        ROOT / cfg["dataset"]["default_root"], "kitchen_l_shape_repaired", only={22, 14}
    )
    assert [f["i"] for f in frames] == [14, 22] and frames[0]["rgb"].shape == (240, 320, 3)
    entries = [
        dict(cfg["detectors"][4], name="stub-ok"),
        dict(cfg["detectors"][4], name="stub-bad"),
    ]

    def make(entry):
        if entry["name"] == "stub-bad":
            raise RuntimeError("cannot load")
        return dc.stub_make(entry)

    report = dc.run_comparison(cfg, frames, entries, make, tmp_path)
    assert report["stub-bad"]["status"] == "failed" and "cannot load" in report["stub-bad"]["error"]
    s = report["stub-ok"]["summary"]
    assert report["stub-ok"]["status"] == "completed" and s["latency"]["n"] == 2
    assert s["iou_0.5"]["absent_class_fp"] == {"bed": 2}  # the stub's 'bed' box, both frames
    assert set(s["sweep_iou_0.5"]) == {str(x) for x in dc.SWEEP}
    saved = json.loads((tmp_path / "detections" / "stub-ok.json").read_text("utf-8"))
    assert len(saved["frames"]) == 2 and "truth" in saved["frames"][0]
    assert "stub-ok" in dc.compact_summary(report) and "FAILED" in dc.compact_summary(report)
    assert np.isfinite(s["latency"]["mean_s"])
