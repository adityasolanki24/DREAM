import pytest

from dream.eval.adequacy import alert_flags, summarize_detection, wilson_interval


def test_early_alarms_are_suppressed():
    assert alert_flags([0, 0.4, 0.5, 0.1, 0.5], 0.28, 3) == [False, False, False, False, True]


def test_out_of_class_is_not_suppressed_after_minimum_evidence():
    cases = [
        {"out_of_class": False, "scores": [0, 0.4, 0.1, 0.1]},
        {"out_of_class": True, "scores": [0, 0.4, 0.7, 0.9]},
    ]
    out = summarize_detection(cases, 0.28, 3)
    assert out["in_class"]["episode_alert_rate"] == 0
    assert out["out_of_class"]["episode_alert_rate"] == 1
    assert out["out_of_class"]["first_alert_probe"] == [3]


@pytest.mark.parametrize("min_probes", [0, -1])
def test_invalid_evidence_floor(min_probes):
    with pytest.raises(ValueError):
        alert_flags([0, 1], 0.28, min_probes)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        alert_flags([0, 1], 0, 1)


def _load_runner():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "validate_adequacy.py"
    spec = importlib.util.spec_from_file_location("validate_adequacy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_instances_have_independent_seeds():
    """Regression: seeding by len(family) gave e.g. bottom_hinge and rising_hinge identical
    random streams, so 'fresh independent instances' were correlated across families."""
    runner = _load_runner()
    jobs = runner.make_jobs()
    mech = [tuple(runner.instance_seeds(job)[0]) for job in jobs]
    episode = [runner.instance_seeds(job)[1] for job in jobs]
    assert len(set(mech)) == len(jobs)
    assert len(set(episode)) == len(jobs)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 12)
    assert lo == 0.0 and hi == pytest.approx(0.2425, abs=1e-3)
    lo, hi = wilson_interval(12, 12)
    assert hi == 1.0 and lo == pytest.approx(0.7575, abs=1e-3)


def test_latency_is_reported_for_detected_cases_only():
    cases = [
        {"out_of_class": True, "scores": [0, 0.1, 0.1, 0.5, 0.9]},
        {"out_of_class": True, "scores": [0, 0.1, 0.1, 0.1, 0.1]},
    ]
    out = summarize_detection(cases, 0.28, 1)["out_of_class"]
    assert out["first_alert_probe"] == [3, None]
    assert out["median_first_alert_probe"] == 3
