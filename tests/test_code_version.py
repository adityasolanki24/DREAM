from experiments.code_version import code_dirty, code_sha


def test_archive_snapshot_without_git_uses_recorded_sha(tmp_path, monkeypatch):
    # A cloud job runs from `git archive` output: no .git, SHA passed by the session guest.
    monkeypatch.setenv("DREAM_CODE_SHA", "abc123")
    assert code_sha(tmp_path) == "abc123"
    assert code_dirty(tmp_path) is False
