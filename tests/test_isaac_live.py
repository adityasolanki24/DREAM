"""CPU regression for physics ownership across idle and task execution."""

import io
import json
from types import SimpleNamespace

import pytest

from experiments import isaac_g1_live as live
from experiments.isaac_g1_live import run_tasks


def test_waiting_for_a_goal_never_steps_uncontrolled_physics():
    ticks = iter([True] * 20 + [False])
    physics = {"steps": 0, "controlled": 0}

    def uncontrolled_update():
        physics["steps"] += 1

    def task_step():
        # One queued action after a period waiting for user input.
        if len(waits) == 10 and physics["controlled"] == 0:
            physics["steps"] += 4
            physics["controlled"] += 4
            return "task"
        return None

    waits = []
    app = SimpleNamespace(is_running=lambda: next(ticks), update=uncontrolled_update)
    run_tasks(app, SimpleNamespace(step=task_step), sleep=waits.append)
    assert physics["controlled"] == 4
    assert physics["steps"] == physics["controlled"], "idle frames advanced unowned physics"
    assert len(waits) == 19


def test_independent_inspection_score_checks_the_selected_instance():
    import numpy as np

    from experiments.isaac_g1_kitchen import score

    candidates = [
        SimpleNamespace(id=1, handle=np.array([0.0, 2.0, 0.0]), handle_views=10),
        SimpleNamespace(id=2, handle=np.array([1.0, 0.0, 0.0]), handle_views=1),
    ]
    env = SimpleNamespace(true_pose=lambda: [0.0, 0.0, 0.0], odometry=lambda: [0.0, 0.0, 0.0])
    toolbox = SimpleNamespace(memory=SimpleNamespace(candidates=candidates))
    handles = [
        {"centre": [1.0, 0.0, 0.0], "cabinet_centre": [2.0, 0.0, 0.0], "path": "cabinet/handle"}
    ]
    assert score(env, toolbox, True, handles)["false_claim"]
    assert score(env, toolbox, True, handles, candidate_id=2)["success"]


@pytest.mark.parametrize("revision", [live.QWEN_REVISION, "different-checkpoint"])
def test_cached_model_reuse_requires_exact_ready_revision(monkeypatch, tmp_path, revision):
    health = {"ready": True, "revision": revision}
    monkeypatch.setattr(
        live.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(health).encode())
    )

    def launch(*args, **kwargs):
        raise RuntimeError("new model process required")

    monkeypatch.setattr(live.subprocess, "Popen", launch)
    args = SimpleNamespace(qwen_port=8767)
    if revision == live.QWEN_REVISION:
        assert live.start_qwen(args, tmp_path) == (None, None)
        assert json.loads((tmp_path / "qwen_reused.json").read_text()) == health
    else:
        with pytest.raises(RuntimeError, match="new model process required"):
            live.start_qwen(args, tmp_path)
        assert not (tmp_path / "qwen_reused.json").exists()
