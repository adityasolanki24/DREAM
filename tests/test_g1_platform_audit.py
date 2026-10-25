"""Missing RGB-D depth pixels must stay explicit in offline platform comparisons."""

import json
from types import SimpleNamespace

import numpy as np

from experiments import compare_g1_rendering as audit


def test_missing_depth_does_not_produce_a_nan_report(tmp_path, monkeypatch):
    cloud, local = tmp_path / "cloud", tmp_path / "local"
    for name in ("free_shift", "free_heavier"):
        for folder in (cloud / "deterministic" / name, local / name):
            (folder / "frames").mkdir(parents=True)
            np.savez(folder / "frames/0000.npz", rgb=np.zeros((1, 2, 3)), depth=[[np.nan, 1]])
            (folder / "physics.jsonl").write_text("")
        (cloud / "deterministic" / name / "frames.json").write_text(json.dumps([]))
    monkeypatch.setattr(audit, "estimate", lambda p: SimpleNamespace(position=np.zeros(3)))
    result = audit.compare(cloud, local)
    assert np.isfinite(result["free_shift"]["initial_depth_rms_difference_m"])
