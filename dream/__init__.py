import os as _os

# Headless rendering (camera frames, replays) needs an offscreen GL backend, chosen when MuJoCo
# is first imported; every module imports this package first. A display, or an explicit
# MUJOCO_GL, takes precedence.
if _os.name != "nt" and "DISPLAY" not in _os.environ:
    _os.environ.setdefault("MUJOCO_GL", "osmesa")
    _os.environ.setdefault("PYOPENGL_PLATFORM", _os.environ["MUJOCO_GL"])
