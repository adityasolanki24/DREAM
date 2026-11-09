"""Offline comparison of open-vocabulary box detectors on captured Isaac frames.

    python experiments/detector_compare.py --config experiments/configs/detector_compare.json \
        --dataset <capture root> --out <dir> [--detectors a,b] [--max-frames N] [--stub]

Dataset (written by the Isaac capture jobs): ``<root>/<item>/frames.jsonl`` with one JSON row per
frame, ``<root>/<item>/frames/<NNNNN>.png`` (the RGB image the detector sees) and a ``gt`` object
per row: ``{instance: {"class", "pixels", "box": [x1, y1, x2, y2]}}``, the tight pixel box of the
instance segmentation (evaluation side only; the detectors never see it).

Why: the Isaac evaluation uses learned perception, and the detector is chosen on development data
before the evaluation is frozen. For each detector this reports, per class and overall, precision
and recall at IoU 0.5 and 0.3, false positives on look-alikes (a box of one class on another
class's instance) and on classes absent from the scene, latency, GPU memory, and a nested-object
metric (an object whose truth box sits inside another's: sink in countertop).

Protocol (fixed here, before any held-out data exists):
- Ground-truth instances with fewer than MIN_GT_PIXELS visible pixels are *ignored*: they are not
  required to be found, and a detection matching one is neither a hit nor a false positive.
  400 px is ~20x20 px at 320x240, below which a box-level detector has nothing to ground.
- Instances of a class listed with ``merge`` (a sink is a rim, a basin and a bowl) become one
  instance: union box, summed pixels.
- A detection matches the unmatched same-class truth with the highest IoU at or above the
  threshold, detections taken in descending score (Qwen has no score: output order).
- The operating threshold of each detector is in the config. Precision and recall at it are the
  headline numbers; the threshold sweep is printed so a threshold can be *chosen on development
  data* and then frozen. Never tune it on held-out data.
- Qwen's coordinate convention is decided against truth with the pre-registered rule of
  experiments/isaac_semantic_dev.py: "absolute" only if it matches more truths at IoU 0.5 and has
  the higher mean best IoU; otherwise the documented relative-1000 stays.
Latency is wall time of ``detect`` (CUDA-synchronised) after one untimed warm-up image.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_GT_PIXELS = 400  # see module docstring
IOU_THRESHOLDS = (0.5, 0.3)
NEST_CONTAINMENT = 0.8  # share of the inner truth box inside the outer one
NEST_MAX_AREA_RATIO = 0.5  # inner box area / outer box area
CONFUSION_IOU = 0.5
SWEEP = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8)
CONVENTIONS = ("relative1000", "absolute")


# ----------------------------------------------------------------------------------------------
# geometry and ground truth
# ----------------------------------------------------------------------------------------------
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def contained(inner, outer) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return 0.0 if area(inner) <= 0 else ix * iy / area(inner)


def truth_instances(gt: dict, classes: dict) -> list[dict]:
    """Evaluation instances of one frame: ``{"cls", "box", "pixels"}`` per config class entry.

    ``classes``: eval class -> {"raw": [scene classes], "merge": bool}."""
    out = []
    for cls, spec in classes.items():
        mine = [v for v in gt.values() if v["class"] in spec["raw"]]
        if spec.get("merge") and mine:
            boxes = np.array([v["box"] for v in mine], float)
            mine = [
                {
                    "box": [
                        boxes[:, 0].min(),
                        boxes[:, 1].min(),
                        boxes[:, 2].max(),
                        boxes[:, 3].max(),
                    ],
                    "pixels": sum(v["pixels"] for v in mine),
                }
            ]
        out += [
            {"cls": cls, "box": [float(x) for x in v["box"]], "pixels": int(v["pixels"])}
            for v in mine
        ]
    return out


def raw_to_eval_class(raw: str, classes: dict) -> str:
    for cls, spec in classes.items():
        if raw in spec["raw"]:
            return cls
    return raw


# ----------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------
def match_class(dets: list[dict], truths: list[dict], thr: float) -> dict:
    """Greedy matching of one class in one frame. Returns tp, fp, fn, ignored counts, plus the
    best IoU reached by each eligible truth (0 if none) for the nested metric."""
    elig = [t for t in truths if t["pixels"] >= MIN_GT_PIXELS]
    ign = [t for t in truths if t["pixels"] < MIN_GT_PIXELS]
    taken: set[int] = set()
    tp = fp = ignored = 0
    for d in sorted(dets, key=lambda d: -d["score"]):
        best, best_iou = None, thr - 1e-12
        for k, t in enumerate(elig):
            v = iou(d["box"], t["box"])
            if k not in taken and v >= best_iou and v > 0:
                best, best_iou = k, v
        if best is not None:
            taken.add(best)
            tp += 1
        elif any(iou(d["box"], t["box"]) >= thr for t in ign):
            ignored += 1
        else:
            fp += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": len(elig) - len(taken),
        "ignored": ignored,
        "n_truth": len(elig),
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def kept(dets: list[dict], threshold: float, has_scores: bool) -> list[dict]:
    return [d for d in dets if not has_scores or d["score"] >= threshold]


def score_precision_recall(
    frames: list[dict], classes: dict, threshold: float, has_scores: bool, iou_thr: float
) -> dict:
    """Per-class and micro precision/recall at one score threshold and one IoU threshold.

    ``frames``: rows ``{"truth": [instances], "dets": [{"label","score","box"}]}``.
    Detections of a label with no entry in ``classes`` are counted as ``absent_class`` false
    positives (nothing of that label exists in the scene's labelled truth)."""
    per = {c: {"tp": 0, "fp": 0, "fn": 0, "ignored": 0} for c in classes}
    absent: dict[str, int] = {}
    for f in frames:
        dets = kept(f["dets"], threshold, has_scores)
        for c in classes:
            r = match_class(
                [d for d in dets if d["label"] == c],
                [t for t in f["truth"] if t["cls"] == c],
                iou_thr,
            )
            for k in per[c]:
                per[c][k] += r[k]
        for d in dets:
            if d["label"] not in classes:
                absent[d["label"]] = absent.get(d["label"], 0) + 1
    out = {c: prf(v["tp"], v["fp"], v["fn"]) | {"ignored": v["ignored"]} for c, v in per.items()}
    tp, fp, fn = (sum(v[k] for v in per.values()) for k in ("tp", "fp", "fn"))
    n_absent = sum(absent.values())
    return {
        "per_class": out,
        "micro": prf(tp, fp + n_absent, fn),
        "absent_class_fp": absent,
        "micro_excluding_absent_classes": prf(tp, fp, fn),
    }


def confusions(
    frames: list[dict], raw_gt_key: str, classes: dict, threshold: float, has_scores: bool
) -> dict:
    """Look-alike false positives: label -> scene class -> count, for detections whose best-
    overlapping scene instance (>= CONFUSION_IOU) belongs to another eval class or to a scene
    class outside the config (cabinet_door, backsplash, ...)."""
    out: dict[str, dict[str, int]] = {}
    for f in frames:
        for d in kept(f["dets"], threshold, has_scores):
            best, best_iou = None, 0.0
            for g in f[raw_gt_key]:
                if g["pixels"] < MIN_GT_PIXELS:
                    continue
                v = iou(d["box"], g["box"])
                if v > best_iou:
                    best, best_iou = g, v
            if best is not None and best_iou >= CONFUSION_IOU:
                scene = best["class"]
                if raw_to_eval_class(scene, classes) != d["label"]:
                    row = out.setdefault(d["label"], {})
                    row[scene] = row.get(scene, 0) + 1
    return out


def nested_pairs(truth: list[dict]) -> list[tuple[dict, dict]]:
    """(inner, outer) truth pairs of different classes: inner >= NEST_CONTAINMENT inside outer and
    at most NEST_MAX_AREA_RATIO of its area. Geometry only; no class names."""
    elig = [t for t in truth if t["pixels"] >= MIN_GT_PIXELS]
    return [
        (a, b)
        for a in elig
        for b in elig
        if a is not b
        and a["cls"] != b["cls"]
        and contained(a["box"], b["box"]) >= NEST_CONTAINMENT
        and area(a["box"]) <= NEST_MAX_AREA_RATIO * area(b["box"])
    ]


def nested_metric(frames: list[dict], threshold: float, has_scores: bool, iou_thr: float) -> dict:
    """Per (inner class, outer class): pairs in the truth, how often each is found at
    ``iou_thr``, how often the inner box is detected nested inside the detected outer box (the
    case geometric exclusion can act on), and how often the inner object is *swallowed*: an outer
    detection overlaps the outer truth, contains the inner truth, and no inner detection exists."""
    acc: dict[str, dict[str, int]] = {}
    for f in frames:
        dets = kept(f["dets"], threshold, has_scores)
        for inner, outer in nested_pairs(f["truth"]):
            row = acc.setdefault(
                f"{inner['cls']} in {outer['cls']}",
                dict.fromkeys(
                    (
                        "pairs",
                        "inner_found",
                        "outer_found",
                        "both_found",
                        "nested_detected",
                        "swallowed",
                    ),
                    0,
                ),
            )
            din = [d for d in dets if d["label"] == inner["cls"]]
            dout = [d for d in dets if d["label"] == outer["cls"]]
            fin = [d for d in din if iou(d["box"], inner["box"]) >= iou_thr]
            fout = [d for d in dout if iou(d["box"], outer["box"]) >= iou_thr]
            row["pairs"] += 1
            row["inner_found"] += bool(fin)
            row["outer_found"] += bool(fout)
            row["both_found"] += bool(fin and fout)
            row["nested_detected"] += any(
                contained(i["box"], o["box"]) >= NEST_CONTAINMENT for i in fin for o in fout
            )
            row["swallowed"] += bool(
                fout
                and not fin
                and any(contained(inner["box"], o["box"]) >= NEST_CONTAINMENT for o in fout)
            )
    return acc


def convention_evidence(frames: list[dict], classes: dict) -> dict:
    """For a Qwen detector: with each coordinate convention, the mean over (frame, class) truths of
    the best same-class IoU, and how many are matched at 0.5. ``frames`` carry ``by_convention``:
    convention -> detections. Decision rule as in experiments/isaac_semantic_dev.py."""
    out = {}
    for conv in CONVENTIONS:
        best, matched, n = [], 0, 0
        for f in frames:
            for t in f["truth"]:
                if t["pixels"] < MIN_GT_PIXELS:
                    continue
                v = max(
                    (
                        iou(d["box"], t["box"])
                        for d in f["by_convention"][conv]
                        if d["label"] == t["cls"]
                    ),
                    default=0.0,
                )
                best.append(v)
                matched += v >= 0.5
                n += 1
        out[conv] = {
            "truths": n,
            "mean_best_iou": float(np.mean(best)) if best else 0.0,
            "matched_at_0.5": matched,
        }
    rel, ab = out["relative1000"], out["absolute"]
    out["evidence_says"] = (
        "absolute"
        if ab["matched_at_0.5"] > rel["matched_at_0.5"]
        and ab["mean_best_iou"] > rel["mean_best_iou"]
        else "relative1000"
    )
    return out


def latency_stats(times: list[float]) -> dict:
    if not times:
        return {"n": 0}
    a = np.asarray(times)
    return {
        "n": len(a),
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "p90_s": float(np.percentile(a, 90)),
        "max_s": float(a.max()),
    }


def summarise_detector(frames: list[dict], cfg: dict, entry: dict, has_scores: bool) -> dict:
    classes, thr = cfg["gt_classes"], float(entry.get("operating_threshold", 0.0))
    res = {"operating_threshold": thr if has_scores else None, "has_scores": has_scores}
    for t in IOU_THRESHOLDS:
        res[f"iou_{t}"] = score_precision_recall(frames, classes, thr, has_scores, t)
    res["sweep_iou_0.5"] = (
        {str(s): score_precision_recall(frames, classes, s, True, 0.5)["micro"] for s in SWEEP}
        if has_scores
        else None
    )
    res["look_alike_false_positives"] = confusions(frames, "raw_truth", classes, thr, has_scores)
    res["nested"] = {str(t): nested_metric(frames, thr, has_scores, t) for t in IOU_THRESHOLDS}
    n_raw = sum(f["n_raw"] for f in frames)
    n_rej = sum(f["n_rejected"] for f in frames)
    n_clip = sum(f["n_clipped"] for f in frames)
    res["box_validation"] = {
        "raw_boxes": n_raw,
        "rejected": n_rej,
        "clipped": n_clip,
        "rejection_reasons": reason_counts(frames),
    }
    return res


def reason_counts(frames: list[dict]) -> dict:
    out: dict[str, int] = {}
    for f in frames:
        for r in f["rejections"]:
            key = r["reason"].split(" (")[0]
            out[key] = out.get(key, 0) + 1
    return out


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_item(
    root: pathlib.Path, item: str, max_frames: int | None = None, only: set[int] | None = None
) -> list[dict]:
    from PIL import Image

    rows = [
        json.loads(x)
        for x in (root / item / "frames.jsonl").read_text("utf-8").splitlines()
        if x.strip()
    ]
    out = []
    for row in rows:
        if only is not None and row["i"] not in only:
            continue
        if max_frames is not None and len(out) >= max_frames:
            break
        png = root / item / "frames" / f"{row['i']:05d}.png"
        out.append(
            {
                "item": item,
                "i": row["i"],
                "rgb": np.asarray(Image.open(png).convert("RGB")),
                "gt": row.get("gt", {}),
            }
        )
    return out


# ----------------------------------------------------------------------------------------------
# running detectors
# ----------------------------------------------------------------------------------------------
def gpu_state() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "device": torch.cuda.get_device_name(0),
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
        }
    except ImportError:
        return None


def sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def cuda_peak_gb(reset: bool = False) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return None
        return torch.cuda.max_memory_allocated() / 2**30
    except ImportError:
        return None


def run_detector(grounder, entry: dict, frames: list[dict], vocabulary: dict, cfg: dict) -> dict:
    """All frames through one grounder; returns the per-frame rows used for scoring."""
    from dream.perception.box_detectors import parse_answer, validate_boxes
    from dream.perception.vlm_detect import canonical_labels

    labels = canonical_labels(vocabulary)
    classes = sorted(c.replace("_", " ") for c in vocabulary)
    classes_cfg = cfg["gt_classes"]
    if frames:  # untimed warm-up
        grounder.detect(frames[0]["rgb"], classes)
    sync()
    cuda_peak_gb(reset=True)
    rows, times = [], []
    for fr in frames:
        h, w = fr["rgb"].shape[:2]
        t0 = time.perf_counter()
        raw = grounder.detect(fr["rgb"], classes)
        sync()
        times.append(time.perf_counter() - t0)
        v = validate_boxes(raw, labels, w, h)
        dets = [
            {"label": a, "score": s, "box": list(b), "clipped": c} for a, s, b, c in v.detections
        ]
        raw_truth = [
            {"class": g["class"], "box": g["box"], "pixels": g["pixels"]} for g in fr["gt"].values()
        ]
        row = {
            "item": fr["item"],
            "i": fr["i"],
            "dets": dets,
            "rejections": [dataclasses.asdict(r) for r in v.rejections],
            "n_raw": len(raw),
            "n_rejected": len(v.rejections),
            "n_clipped": sum(d["clipped"] for d in dets),
            "truth": truth_instances(fr["gt"], classes_cfg),
            "raw_truth": raw_truth,
        }
        text = getattr(grounder, "last_text", None)
        if text is not None:
            row["raw_answer"] = text[:1200]
            row["by_convention"] = {}
            for conv in CONVENTIONS:
                rb, _ = parse_answer(text, w * grounder.image_scale, h * grounder.image_scale, conv)
                rb = [
                    dataclasses.replace(b, box=tuple(x / grounder.image_scale for x in b.box))
                    for b in rb
                ]
                vv = validate_boxes(rb, labels, w, h)
                row["by_convention"][conv] = [
                    {"label": a, "score": s, "box": list(b)} for a, s, b, _ in vv.detections
                ]
        rows.append(row)
    return {"rows": rows, "latency": latency_stats(times), "peak_inference_gb": cuda_peak_gb()}


def summarise_runs(result: dict, cfg: dict, entry: dict, has_scores: bool) -> dict:
    s = summarise_detector(result["rows"], cfg, entry, has_scores)
    s["latency"] = result["latency"]
    s["peak_gpu_gb_during_inference"] = result["peak_inference_gb"]
    if result["rows"] and "by_convention" in result["rows"][0]:
        s["coordinate_check"] = convention_evidence(result["rows"], cfg["gt_classes"])
        s["coordinate_convention_declared"] = entry.get("coordinates", "relative1000")
    return s


def build_grounder(entry: dict, device: str):
    family = entry["family"]
    if family == "qwen3vl":
        from dream.perception.box_detectors import QwenGrounder
        from dream.runtime.qwen import QwenTransformersBackend

        backend = QwenTransformersBackend(
            entry["model_id"],
            device=device,
            max_new_tokens=512,
            dtype="bfloat16" if device != "cpu" else "float32",
            timeout_s=180.0,
            revision=entry["revision"],
        )
        return QwenGrounder(
            backend,
            entry["name"],
            entry["license"],
            entry["revision"],
            entry.get("coordinates", "relative1000"),
            entry.get("image_scale", 1.0),
        )
    if family == "owlv2":
        from dream.perception.box_detectors import OwlV2Grounder

        return OwlV2Grounder(
            entry["model_id"],
            entry["revision"],
            device,
            entry.get("score_floor", 0.05),
            entry["name"],
        )
    if family == "grounding-dino":
        from dream.perception.box_detectors import GroundingDinoGrounder

        return GroundingDinoGrounder(
            entry["model_id"], entry["revision"], device, name=entry["name"]
        )
    raise ValueError(f"unknown detector family {family!r}")


def run_comparison(
    cfg: dict,
    frames: list[dict],
    entries: list[dict],
    make_grounder,
    out: pathlib.Path | None = None,
) -> dict:
    """Run ``entries`` over ``frames`` with grounders from ``make_grounder(entry)``. A detector
    that fails to load or run is recorded and skipped; the others still run."""
    from dream.semantics.knowledge import load_knowledge
    from experiments import isaac_semantic as sem

    vocabulary = sem.detector_vocabulary(load_knowledge())
    report = {}
    for entry in entries:
        name = entry["name"]
        record: dict = {"entry": entry}
        t0 = time.perf_counter()
        try:
            cuda_peak_gb(reset=True)
            grounder = make_grounder(entry)
            sync()
            record["load_s"] = time.perf_counter() - t0
            record["gpu_gb_after_load"] = cuda_peak_gb()
            record["provenance"] = grounder.provenance()
            result = run_detector(grounder, entry, frames, vocabulary, cfg)
            record["summary"] = summarise_runs(result, cfg, entry, grounder.has_scores)
            record["status"] = "completed"
            if out is not None:
                (out / "detections").mkdir(parents=True, exist_ok=True)
                (out / "detections" / f"{name}.json").write_text(
                    json.dumps({"detector": name, "frames": result["rows"]}), encoding="utf-8"
                )
            del grounder
        except Exception as error:  # recorded, never hidden
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()[-3000:]
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        report[name] = record
    return report


# ----------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------
def fmt(v) -> str:
    return "  n/a" if v is None else f"{v:5.2f}"


def compact_summary(report: dict) -> str:
    lines = []
    for name, rec in report.items():
        if rec["status"] != "completed":
            lines.append(f"{name}: FAILED {rec['error']}")
            continue
        s = rec["summary"]
        m5, m3 = s["iou_0.5"]["micro"], s["iou_0.3"]["micro"]
        lat = s["latency"]
        lines.append(
            f"{name}: P/R@0.5 {fmt(m5['precision'])}/{fmt(m5['recall'])}  "
            f"P/R@0.3 {fmt(m3['precision'])}/{fmt(m3['recall'])}  "
            f"latency {lat.get('median_s', float('nan')):.2f}s/img  "
            f"gpu load {rec.get('gpu_gb_after_load') or 0:.1f} GB peak "
            f"{s['peak_gpu_gb_during_inference'] or 0:.1f} GB  "
            f"rejected {s['box_validation']['rejected']}/{s['box_validation']['raw_boxes']}"
        )
        for c, v in s["iou_0.5"]["per_class"].items():
            lines.append(
                f"    {c:16s} P {fmt(v['precision'])} R {fmt(v['recall'])} "
                f"(tp {v['tp']} fp {v['fp']} fn {v['fn']})"
            )
        for key, v in s["nested"]["0.5"].items():
            lines.append(
                f"    nested {key}: pairs {v['pairs']} inner@0.5 {v['inner_found']} "
                f"outer@0.5 {v['outer_found']} nested-detected {v['nested_detected']} "
                f"swallowed {v['swallowed']}"
            )
        if s["look_alike_false_positives"]:
            lines.append(f"    look-alike FPs: {json.dumps(s['look_alike_false_positives'])}")
        if s["iou_0.5"]["absent_class_fp"]:
            lines.append(f"    absent-class FPs: {json.dumps(s['iou_0.5']['absent_class_fp'])}")
        if "coordinate_check" in s:
            c = s["coordinate_check"]
            lines.append(
                f"    coordinates declared {s['coordinate_convention_declared']}; truth says "
                f"{c['evidence_says']} (rel1000 mean IoU "
                f"{c['relative1000']['mean_best_iou']:.2f}, abs "
                f"{c['absolute']['mean_best_iou']:.2f})"
            )
    return "\n".join(lines)


def versions() -> dict:
    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    for mod in ("torch", "transformers", "PIL", "scipy", "torchvision", "accelerate"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def ensure_scipy() -> str:
    """OWLv2's image processor needs scipy; the cloud runtime image may lack it. Installs the
    hash-pinned wheel of cloud/semantic.requirements.lock.txt (no resolution); says what it did."""
    try:
        import scipy

        return f"scipy {scipy.__version__} present"
    except ImportError:
        pass
    try:
        from experiments import cloud_bootstrap

        cloud_bootstrap.install()
        return "scipy installed from cloud/semantic.requirements.lock.txt"
    except Exception as error:
        return f"scipy unavailable: {type(error).__name__}: {error}"


def maybe_reexec_under_runtime() -> None:
    """Cloud jobs start in the Isaac or DREAM interpreter; torch+transformers live in the pinned
    Qwen runtime image (DREAM_QWEN_PYTHON), as for experiments/vlm_compare_p6_qwen.py."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        python = os.environ.get("DREAM_QWEN_PYTHON", "/opt/dream-runtime/bin/python")
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        sys.exit(subprocess.run([python, *sys.argv], cwd=ROOT, env=env).returncode)


def stub_make(entry: dict):
    """CPU pipeline check: a fixed, truth-blind grounder. Not a detector."""
    from dream.perception.box_detectors import RawBox

    class Stub:
        name, has_scores, image_scale = entry["name"], True, 1.0

        def detect(self, rgb, classes):
            return [
                RawBox("sink", 0.6, (10, 60, 70, 100)),
                RawBox("bed", 0.5, (0, 0, 50, 50)),
                RawBox("countertop", 0.4, (0, 20, 320, 110)),
            ]

        def provenance(self):
            return {"name": entry["name"], "stub": True}

    return Stub()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--config", type=pathlib.Path, default=ROOT / "experiments/configs/detector_compare.json"
    )
    ap.add_argument("--dataset", type=pathlib.Path, default=None)
    ap.add_argument("--items", default=None, help="comma-separated; default: the config's items")
    ap.add_argument("--detectors", default=None, help="comma-separated names; default: all")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--stub", action="store_true", help="truth-blind stub detectors (CPU pipeline check)"
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text("utf-8"))
    entries = [
        e
        for e in cfg["detectors"]
        if args.detectors is None or e["name"] in args.detectors.split(",")
    ]
    if not entries:
        raise SystemExit("no detector selected")
    if not args.stub:
        maybe_reexec_under_runtime()
        os.environ.setdefault(
            "HF_HOME", str(pathlib.Path(os.environ.get("DREAM_CACHE", ROOT / "cache")) / "hf")
        )
    sys.path.insert(0, str(ROOT))
    data = args.dataset or ROOT / cfg["dataset"]["default_root"]
    items = args.items.split(",") if args.items else cfg["dataset"]["items"]
    frames = [f for it in items for f in load_item(data, it, args.max_frames)]
    if not frames:
        raise SystemExit("no frames loaded")
    args.out.mkdir(parents=True, exist_ok=True)
    from experiments.code_version import code_dirty, code_sha

    notes = [] if args.stub else [ensure_scipy()]
    report = run_comparison(
        cfg,
        frames,
        entries,
        stub_make if args.stub else (lambda e: build_grounder(e, args.device)),
        args.out,
    )
    meta = {
        "config": cfg,
        "config_file": str(args.config),
        "dataset_root": str(data),
        "items": items,
        "frames": len(frames),
        "min_gt_pixels": MIN_GT_PIXELS,
        "stub": args.stub,
        "code_sha": code_sha(ROOT),
        "code_dirty": code_dirty(ROOT),
        "versions": versions(),
        "gpu": gpu_state(),
        "notes": notes,
    }
    (args.out / "detector_compare.json").write_text(
        json.dumps({"meta": meta, "detectors": report}, indent=1), encoding="utf-8"
    )
    text = compact_summary(report)
    (args.out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    sys.exit(0 if any(r["status"] == "completed" for r in report.values()) else 1)


if __name__ == "__main__":
    main()
