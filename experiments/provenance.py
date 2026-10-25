"""Which code ran: the Git checkout's HEAD, or the committed archive a cloud job unpacked."""

from __future__ import annotations

import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def code_revision() -> tuple[str, bool]:
    """(commit SHA, dream/ has uncommitted changes). Cloud jobs run ``git archive`` of a clean
    commit (cloud/isaac_session.py) and pass its SHA in DREAM_CODE_SHA; there is no .git."""
    sha = os.environ.get("DREAM_CODE_SHA")
    if sha:
        return sha, False
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "dream"], cwd=ROOT))
    return head, dirty
