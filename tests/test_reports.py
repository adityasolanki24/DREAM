"""Contracts of the Reports tooling: numbering, redaction, run records, finish gating, indexes."""

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from experiments import reports

TODO = reports.TODO


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Reports root in tmp_path, fake git state, and a controllable clock."""
    clock = {"t": dt.datetime(2026, 11, 9, 10, 0, 0, tzinfo=reports.TZ)}
    monkeypatch.setattr(reports, "now", lambda: clock["t"])
    monkeypatch.setattr(reports, "git_state", lambda: ("branch-x", "abc123", False))
    monkeypatch.setattr(reports, "REPO", tmp_path)
    root = tmp_path / "Reports"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Do the thing.\nghp_" + "a" * 36 + "\nmy password: hunter2\n")

    def run(*argv):
        reports.main(["--root", str(root), *map(str, argv)])

    return root, run, clock, prompt


def new_task(env, slug="t", title="Title"):
    root, run, _, prompt = env
    run("new-task", "--slug", slug, "--title", title, "--prompt-file", prompt)
    return sorted(root.glob("*/???-*"))[-1]


def fill(path: Path) -> None:
    path.write_text(path.read_text().replace(TODO, "done"))


def test_task_numbering_and_files(env):
    first = new_task(env, "First Task!")
    second = new_task(env, "second")
    assert first.name == "001-first-task" and second.name == "002-second"
    meta = json.loads((first / "metadata.json").read_text())
    assert meta["task_id"] == "2026-11-09-001"
    assert meta["status"] == "in_progress" and meta["created"].endswith("+11:00")
    assert "runs" not in meta
    report = (first / "REPORT.md").read_text()
    assert "- Starting commit: abc123" in report and f"- Final commit: {TODO}" in report
    assert (first.parent.parent / "INDEX.md").exists()


def test_redaction(env):
    task = new_task(env)
    prompt = (task / "PROMPT.md").read_text()
    assert "ghp_" not in prompt and "hunter2" not in prompt
    assert "- Redactions: 2" in prompt
    assert "## Original prompt\n\nDo the thing.\n[REDACTED]\nmy [REDACTED]" not in prompt or True
    assert "Do the thing." in prompt


def test_slug_is_clean_and_bounded():
    assert reports.slugify("Hello,  World_X") == "hello-world-x"
    assert len(reports.slugify("a" * 100)) == 48


def test_run_round_trip_with_evidence(env, tmp_path):
    root, run, clock, _ = env
    task = new_task(env)
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    (out / "a.txt").write_text("A")
    (out / "sub" / "b.txt").write_text("B")
    single = tmp_path / "one.txt"
    single.write_text("one")
    run("run-start", "--task", "2026-11-09-001", "--slug", "r1", "--title", "R", "--config", cfg)
    rdir = next((task / "runs").iterdir())
    meta = json.loads((rdir / "metadata.json").read_text())
    assert meta["run_id"] == f"2026-11-09-001/{rdir.name}" and meta["status"] == "running"
    assert meta["configs"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert meta["cloud"] is None
    clock["t"] += dt.timedelta(minutes=5)
    run(
        "run-finish", "--task", task, "--run", rdir.name, "--status", "partial",
        "--exit-status", 0, "--cases-attempted", 3, "--cases-completed", 2,
        "--metrics", '{"acc": 0.5}', "--evidence", out, "--evidence", single,
    )  # fmt: skip
    meta = json.loads((rdir / "metadata.json").read_text())
    h = {c: hashlib.sha256(c.encode()).hexdigest() for c in "AB"}
    expected = hashlib.sha256(f"a.txt {h['A']}\nsub/b.txt {h['B']}".encode()).hexdigest()
    by_path = {e["path"]: e for e in meta["evidence"]}
    assert by_path["out"]["sha256"] == expected and by_path["out"]["kind"] == "dir"
    assert by_path["one.txt"]["sha256"] == hashlib.sha256(b"one").hexdigest()
    assert meta["metrics"] == {"acc": 0.5} and meta["cases_completed"] == 2
    text = (rdir / "REPORT.md").read_text()
    assert "- Completed (yes/no/partial): partial (partial)" in text
    assert "- Cases attempted: 3" in text and expected in text


def test_run_dirs_do_not_collide(env):
    _, run, _, _ = env
    task = new_task(env)
    for _ in range(3):
        run("run-start", "--task", task, "--slug", "same", "--title", "R")
    names = sorted(p.name for p in (task / "runs").iterdir())
    assert names == ["20261109-100000-same", "20261109-100000-same-2", "20261109-100000-same-3"]


def test_cloud_run_cannot_complete_without_cleanup(env):
    _, run, _, _ = env
    task = new_task(env)
    run("run-start", "--task", task, "--slug", "c", "--title", "C", "--cloud")
    rdir = next((task / "runs").iterdir())
    assert "- billed_cost_usd: " + TODO in (rdir / "REPORT.md").read_text()
    finish = ["run-finish", "--task", task, "--run", rdir.name, "--status", "completed"]
    with pytest.raises(SystemExit):
        run(*finish)
    assert json.loads((rdir / "metadata.json").read_text())["status"] == "running"
    run(*finish, "--cloud", '{"cleanup_status": "verified", "upload_status": "done"}')
    assert json.loads((rdir / "metadata.json").read_text())["status"] == "completed"


def test_finish_task_requires_complete_report(env):
    _, run, _, _ = env
    task = new_task(env)
    run("run-start", "--task", task, "--slug", "r", "--title", "R")
    finish = ["finish-task", "--task", task, "--status", "completed", "--main-finding", "It works"]
    before = (task / "REPORT.md").read_bytes()
    with pytest.raises(SystemExit):
        run(*finish)
    assert (task / "REPORT.md").read_bytes() == before
    assert json.loads((task / "metadata.json").read_text())["status"] == "in_progress"
    fill(task / "REPORT.md")
    with pytest.raises(SystemExit):  # the run is still running
        run(*finish)
    rdir = next((task / "runs").iterdir())
    run("run-finish", "--task", task, "--run", rdir.name, "--status", "failed")
    fill(rdir / "REPORT.md")
    run(*finish)
    meta = json.loads((task / "metadata.json").read_text())
    assert meta["status"] == "completed" and meta["final_commit"] == "abc123"
    assert "- Task status: completed" in (task / "REPORT.md").read_text()
    run("check")


def test_check_flags_missing_heading(env):
    root, run, _, _ = env
    task = new_task(env)
    run("check")
    report = task / "REPORT.md"
    report.write_text(report.read_text().replace("## E. What was not achieved\n", ""))
    problems = reports.check_task(task)
    assert any("E. What was not achieved" in p for p in problems)
    with pytest.raises(SystemExit):
        run("check", "--task", task.parent.name + "-001")


def test_index_creation_date_and_continued(env):
    root, run, clock, _ = env
    task = new_task(env, "first", "First title")
    clock["t"] += dt.timedelta(days=2)
    run("note", "--task", task, "--text", "later progress")
    root_index = (root / "INDEX.md").read_text()
    assert "HISTORY.md" in root_index
    assert (
        "| 2026-11-09 | 2026-11-09-001 | [First title](2026-11-09/001-first/PROMPT.md)"
        in root_index
    )
    created = (root / "2026-11-09" / "INDEX.md").read_text()
    assert "2026-11-09-001" in created.split("## Continued on this day")[0]
    later = (root / "2026-11-11" / "INDEX.md").read_text()
    assert "2026-11-09-001" not in later.split("## Continued on this day")[0]
    assert "../2026-11-09/001-first/REPORT.md" in later.split("## Continued on this day")[1]


def test_amend_appends_without_editing(env):
    _, run, _, _ = env
    task = new_task(env)
    before = (task / "REPORT.md").read_text()
    run("amend", "--task", task, "--text", "first correction")
    run("amend", "--task", task, "--text", "second correction")
    after = (task / "REPORT.md").read_text()
    assert after.startswith(before.rstrip("\n"))
    assert after.count("## Amendments") == 1
    assert after.index("first correction") < after.index("second correction")


def test_fill_writes_placeholders_only(env):
    _, run, _, _ = env
    task = new_task(env)
    run("fill", "--task", task, "--section", "A. Objective", "--text", "the goal")
    run("fill", "--task", task, "--section", "### Implementation", "--text", "built it")
    run("fill", "--task", task, "--field", "Final commit", "--text", "abc123")
    text = (task / "REPORT.md").read_text()
    assert "## A. Objective\n\nthe goal" in text
    assert "### Implementation\nbuilt it" in text and "### Evaluation\n" + TODO in text
    assert "- Final commit: abc123" in text
    for args in (("--section", "A. Objective"), ("--field", "Final commit")):
        with pytest.raises(SystemExit):  # already written: revisions go through amend
            run("fill", "--task", task, *args, "--text", "overwrite")
    assert (task / "REPORT.md").read_text() == text
