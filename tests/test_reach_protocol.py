import hashlib
import json
import sys
from pathlib import Path

import pytest

from dream.eval import reach
from experiments import g1_reach_heights as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/g1_reach_heights_v2.json"


def test_v2_success_definition_matches_evaluator():
    cfg = json.loads(CONFIG.read_text())
    success = cfg["success"]
    assert cfg["success_definition"] == success["definition"]
    assert success["mode"] == "contact"
    assert success["contact_min_s"] == reach.CONTACT_MIN_S
    assert success["peak_contact_force_max_n"] == reach.CONTACT_FORCE_MAX


def launch(monkeypatch, out):
    class EmptyPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, fn, jobs):
            assert not jobs
            return []

    monkeypatch.setattr(runner, "ProcessPoolExecutor", EmptyPool)
    monkeypatch.setattr(runner, "summarize_set", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", str(CONFIG), "--set", "dev", "--out", str(out), "--only", "not-an-episode"],
    )
    runner.main()


def test_launch_hash_and_matching_legacy_resume(monkeypatch, tmp_path):
    cfg = json.loads(CONFIG.read_text())
    tmp_path.joinpath("meta.json").write_text(json.dumps({"config": cfg}))
    launch(monkeypatch, tmp_path)
    assert json.loads((tmp_path / "meta.json").read_text()) == {"config": cfg}
    record = json.loads((tmp_path / "resumes.jsonl").read_text())
    assert record["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_new_meta_records_hash(monkeypatch, tmp_path):
    launch(monkeypatch, tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_mismatching_resume_refuses_without_mutation(monkeypatch, tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"config": {"different": True}}))
    (tmp_path / "resumes.jsonl").write_text("existing\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(SystemExit, match="config"):
        launch(monkeypatch, tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
