"""Import DREAM before any test imports MuJoCo, so the offscreen GL backend is selected
(dream/__init__.py) regardless of test collection order."""

import dream  # noqa: F401
