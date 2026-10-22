import pytest

from dream.eval.adequacy import alert_flags, summarize_detection


def test_early_alarms_are_suppressed():
    assert alert_flags([0, 0.4, 0.5, 0.1, 0.5], 0.28, 3) == [
        False, False, False, False, True
    ]


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
