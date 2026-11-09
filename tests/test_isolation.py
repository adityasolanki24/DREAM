"""Decision-making code must not be able to see simulator ground truth (AGENTS.md, invariant 1).

Every decision-making package is listed in DECISION_PACKAGES. Add new ones (tools, runtime,
perception, memory) when they are created. The robot adapter may use MuJoCo for the robot's own
kinematics and sensing, but must not import the scene or evaluation code.
"""

import ast
import pathlib

import pytest

DREAM = pathlib.Path(__file__).resolve().parents[1] / "dream"
DECISION_PACKAGES = (
    "agent", "skills", "tools", "runtime", "perception", "navigation", "memory", "semantics",
)  # fmt: skip
FORBIDDEN_FOR_DECISIONS = ("mujoco", "dream.sim", "dream.eval", "dream.robot")
FORBIDDEN_FOR_ROBOT = ("dream.sim", "dream.eval", "dream.agent", "dream.skills")
FORBIDDEN_NAMES = (
    "privileged", "MechanismSpec", "TabletopSpec", "CompartmentSpec", "_spec", "HomeSpec",
    "ObjectSpec", "GroundTruthView",
)  # fmt: skip


def _files(package: str) -> list[pathlib.Path]:
    return sorted((DREAM / package).rglob("*.py"))


def _package_of(path: pathlib.Path) -> list[str]:
    """Dotted package parts of the module a file defines, e.g. dream/agent/x.py -> dream, agent."""
    return list(path.relative_to(DREAM.parent).with_suffix("").parts[:-1])


def _imports_in(source: str, package: list[str]) -> list[str]:
    """Every module an import statement in ``source`` can bring in, as absolute dotted names.

    ``from dream import sim`` yields ``dream`` and ``dream.sim``; ``from ..sim import x`` in
    ``dream/agent/y.py`` yields ``dream.sim`` and ``dream.sim.x``; ``importlib.import_module``
    and ``__import__`` with a string literal are included.
    """
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                base = package[: max(keep, 0)] + ([node.module] if node.module else [])
                module = ".".join(base)
            else:
                module = node.module or ""
            names.append(module)
            names += [f"{module}.{a.name}" if module else a.name for a in node.names]
        elif isinstance(node, ast.Call):
            fn = node.func
            called = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if called in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.append(arg.value)
    return names


def _string_attribute_uses(source: str) -> list[str]:
    """String literals used as attribute names: getattr / hasattr / setattr / delattr."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) in ("getattr", "hasattr", "setattr", "delattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            out.append(node.args[1].value)
    return out


def _imports(path: pathlib.Path) -> list[str]:
    return _imports_in(path.read_text(), _package_of(path))


def import_violations(source: str, package: list[str], forbidden: tuple[str, ...]) -> list[str]:
    return [n for n in _imports_in(source, package) if n.startswith(forbidden)]


def name_violations(source: str) -> list[str]:
    """Privileged identifiers used as attributes, names, or string-literal getattr targets."""
    found = []
    for node in ast.walk(ast.parse(source)):
        ident = getattr(node, "attr", None) or getattr(node, "id", None)
        if ident and any(f in ident for f in FORBIDDEN_NAMES):
            found.append(ident)
    found += [n for n in _string_attribute_uses(source) if any(f in n for f in FORBIDDEN_NAMES)]
    return found


def test_decision_packages_exist():
    assert _files("agent") and _files("skills")


@pytest.mark.parametrize("package", DECISION_PACKAGES)
def test_decision_code_does_not_import_simulator_robot_internals_or_evaluation(package):
    for path in _files(package):
        bad = import_violations(path.read_text(), _package_of(path), FORBIDDEN_FOR_DECISIONS)
        assert not bad, f"{package}/{path.name} imports {bad}"


@pytest.mark.parametrize("package", DECISION_PACKAGES)
def test_decision_code_never_touches_privileged_accessors(package):
    for path in _files(package):
        bad = name_violations(path.read_text())
        assert not bad, f"{package}/{path.name} uses {bad}"


def test_robot_adapter_does_not_import_scene_or_evaluation():
    for path in _files("robot"):
        bad = import_violations(path.read_text(), _package_of(path), FORBIDDEN_FOR_ROBOT)
        assert not bad, f"robot/{path.name} imports {bad}"


def test_task_scoring_is_evaluation_only_and_no_decision_package_imports_it():
    """dream/eval/task_scoring.py reads ground truth (true pose, labelled handles, scene boxes,
    contacts); it may be imported only from evaluation and experiment code."""
    assert (DREAM / "eval/task_scoring.py").is_file()
    for package in DECISION_PACKAGES:
        for path in _files(package):
            assert not any("task_scoring" in name for name in _imports(path)), path


# The checker itself: each pattern, fed as synthetic source, must be flagged.
AGENT_PKG = ["dream", "agent"]


@pytest.mark.parametrize(
    "source",
    [
        "import mujoco",
        "from dream.sim import x",
        "from dream import sim",
        "from dream import eval",
        "from dream import robot as r",
        "from ..sim import x",
        "from .. import sim",
        "from ..eval.task_scoring import score",
        "import importlib; importlib.import_module('dream.sim.kinematic_base')",
        "m = __import__('mujoco')",
    ],
)
def test_checker_flags_forbidden_imports(source):
    assert import_violations(source, AGENT_PKG, FORBIDDEN_FOR_DECISIONS)


@pytest.mark.parametrize(
    "source",
    [
        "from dream import types",
        "from ..tools import mobile",
        "from . import sibling",
        "import importlib; importlib.import_module('dream.tools.mobile')",
        "x = __import__(name)",
    ],
)
def test_checker_passes_allowed_imports(source):
    assert not import_violations(source, AGENT_PKG, FORBIDDEN_FOR_DECISIONS)


@pytest.mark.parametrize(
    "source", ["getattr(env, 'privileged')", "hasattr(x, 'GroundTruthView')", "getattr(x, '_spec')"]
)
def test_checker_flags_string_getattr_of_forbidden_names(source):
    assert name_violations(source)


def test_checker_passes_innocent_getattr():
    assert not name_violations("getattr(x, 'time', None)")
