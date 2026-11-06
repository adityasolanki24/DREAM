"""Code identity for evaluation runners, also inside a ``git archive`` snapshot.

A cloud job runs from an archive without ``.git``; the session guest passes the archived
commit as DREAM_CODE_SHA. Locally the repository is asked, as before.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def code_sha(root: Path) -> str:
    return (
        os.environ.get("DREAM_CODE_SHA")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    )


def code_dirty(root: Path) -> bool:
    """False for an archived snapshot (it is a clean tracked tree by construction)."""
    if os.environ.get("DREAM_CODE_SHA"):
        return False
    return bool(subprocess.check_output(["git", "status", "--porcelain", "dream"], cwd=root))
