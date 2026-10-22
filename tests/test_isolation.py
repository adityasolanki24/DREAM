"""The agent must not be able to see simulator ground truth (AGENTS.md, integrity rule 1)."""

import ast
import pathlib

AGENT_DIR = pathlib.Path(__file__).resolve().parents[1] / "dream" / "agent"
FORBIDDEN_MODULES = ("mujoco", "dream.sim", "dream.eval")
FORBIDDEN_NAMES = ("privileged", "MechanismSpec", "_spec")


def _agent_files():
    files = sorted(AGENT_DIR.rglob("*.py"))
    assert files, "no agent modules found"
    return files


def test_agent_does_not_import_simulator_or_evaluation():
    for path in _agent_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith(FORBIDDEN_MODULES), f"{path.name} imports {name}"


def test_agent_never_touches_privileged_accessors():
    for path in _agent_files():
        for node in ast.walk(ast.parse(path.read_text())):
            ident = getattr(node, "attr", None) or getattr(node, "id", None)
            if ident:
                assert not any(f in ident for f in FORBIDDEN_NAMES), f"{path.name} uses {ident}"
