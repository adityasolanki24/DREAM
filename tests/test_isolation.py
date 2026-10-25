"""Decision-making code must not be able to see simulator ground truth (AGENTS.md, invariant 1).

Every decision-making package is listed in DECISION_PACKAGES. Add new ones (tools, runtime,
perception, memory) when they are created. The robot adapter may use MuJoCo for the robot's own
kinematics and sensing, but must not import the scene or evaluation code.
"""

import ast
import pathlib

import pytest

DREAM = pathlib.Path(__file__).resolve().parents[1] / "dream"
DECISION_PACKAGES = ("agent", "skills", "tools", "runtime", "perception", "navigation", "memory")
FORBIDDEN_FOR_DECISIONS = ("mujoco", "dream.sim", "dream.eval", "dream.robot")
FORBIDDEN_FOR_ROBOT = ("dream.sim", "dream.eval", "dream.agent", "dream.skills")
FORBIDDEN_NAMES = ("privileged", "MechanismSpec", "TabletopSpec", "CompartmentSpec", "_spec")


def _files(package: str) -> list[pathlib.Path]:
    return sorted((DREAM / package).rglob("*.py"))


def _imports(path: pathlib.Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_decision_packages_exist():
    assert _files("agent") and _files("skills")


@pytest.mark.parametrize("package", DECISION_PACKAGES)
def test_decision_code_does_not_import_simulator_robot_internals_or_evaluation(package):
    for path in _files(package):
        for name in _imports(path):
            assert not name.startswith(FORBIDDEN_FOR_DECISIONS), (
                f"{package}/{path.name} imports {name}"
            )


@pytest.mark.parametrize("package", DECISION_PACKAGES)
def test_decision_code_never_touches_privileged_accessors(package):
    for path in _files(package):
        for node in ast.walk(ast.parse(path.read_text())):
            ident = getattr(node, "attr", None) or getattr(node, "id", None)
            if ident:
                assert not any(f in ident for f in FORBIDDEN_NAMES), (
                    f"{package}/{path.name} uses {ident}"
                )


def test_robot_adapter_does_not_import_scene_or_evaluation():
    for path in _files("robot"):
        for name in _imports(path):
            assert not name.startswith(FORBIDDEN_FOR_ROBOT), f"robot/{path.name} imports {name}"
