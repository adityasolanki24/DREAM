"""Per-task model choice in the phase 1 interface: provider registry, POST /tasks validation,
and the planner each task receives. No network, simulator or real model: backends are fakes
and TaskAgent is replaced by a recorder."""

from __future__ import annotations

import argparse
import json
import threading
import time
from types import SimpleNamespace

import pytest

import dream.runtime.openai_plan as openai_plan
import experiments.phase1_interface as phase1
from dream.runtime.mobile_model import MobileModelPlanner
from dream.runtime.openai_plan import ChatGPTPlanBackend
from dream.runtime.phone_api import PhoneAPI
from dream.runtime.scheduler import TaskScheduler
from dream.runtime.search import PlacePlanner, RoomSearchPlanner
from experiments.g1_explore_room import Budgeted

TOKEN = "t" * 32
SECRET = "secret-access-token-value-12345"


class FakeBackend:
    def __init__(self, model):
        self.model, self.cancel = model, None

    def set_cancel_event(self, cancel):
        self.cancel = cancel


def provider(pid, models=("m1", "m2"), default="m1", backend=None):
    return phase1.Provider(
        pid,
        pid,
        True,
        models=list(models),
        default_model=default,
        supports=phase1.MODEL_SUPPORTS,
        backend_for=backend or (lambda model: FakeBackend(model)),
    )


def structured():
    return phase1.Provider("structured", "S", True, supports=phase1.STRUCTURED_SUPPORTS)


def api_for(tmp_path, providers, default="structured"):
    s = TaskScheduler(str(tmp_path / "tasks.sqlite"), lambda *a: None, phase1.mobile_goal)
    api = PhoneAPI(
        s,
        TOKEN,
        provider=default,
        validate_task=phase1.make_validator(providers, default),
    )
    s.notify = api.notify
    return api


def post(api, body):
    raw = json.dumps(body).encode()
    return api.handle("POST", "/tasks", f"Bearer {TOKEN}", raw)


def registry():
    return {
        "structured": structured(),
        "qwen": provider("qwen", models=["q1"], default="q1"),
        "openai_chatgpt_plan": provider("openai_chatgpt_plan", models=["g1", "g2"], default="g1"),
    }


def test_unknown_and_unavailable_providers_are_refused(tmp_path):
    unavailable = phase1.Provider(
        "qwen",
        "Qwen",
        False,
        reason="start the backend with --qwen-url",
        supports=phase1.MODEL_SUPPORTS,
    )
    api = api_for(tmp_path, {"structured": structured(), "qwen": unavailable})
    assert post(api, {"text": "inspect cabinet", "provider": "nope"}) == (
        422,
        {"error": "unknown provider: nope"},
    )
    assert post(api, {"text": "inspect cabinet", "provider": "qwen"}) == (
        422,
        {"error": "provider is not available: start the backend with --qwen-url"},
    )
    assert api.handle("GET", "/tasks", f"Bearer {TOKEN}", b"")[1] == []


def test_model_checks_goal_form_and_default_model(tmp_path):
    api = api_for(tmp_path, registry())
    assert post(api, {"text": "inspect cabinet", "provider": "qwen", "model": "nope"}) == (
        422,
        {"error": "model is not available for qwen: nope"},
    )
    assert post(api, {"text": "go to the entrance", "provider": "qwen", "model": "q1"})[0] == 201
    assert post(api, {"text": "inspect cabinet", "provider": "structured", "model": "q1"}) == (
        422,
        {"error": "model must be omitted for structured"},
    )
    assert post(api, {"text": "fly to the moon", "provider": "qwen", "model": "q1"})[0] == 201
    assert post(api, {"text": "fly to the moon", "provider": "structured"})[0] == 422
    assert post(api, {"text": "inspect cabinet", "provider": "qwen"})[0] == 201  # default q1
    assert (
        post(api, {"text": "inspect cabinet", "provider": "openai_chatgpt_plan", "model": "g2"})[0]
        == 201
    )
    assert post(api, {"text": "goto entrance"})[0] == 201  # default structured parser
    tasks = api.scheduler.tasks()
    assert [t["meta"] for t in tasks] == [
        {"provider": "qwen", "model": "q1"},
        {"provider": "qwen", "model": "q1"},
        {"provider": "qwen", "model": "q1"},
        {"provider": "openai_chatgpt_plan", "model": "g2"},
        {"provider": "structured", "model": None},
    ]


def test_model_must_be_named_when_there_is_no_default_and_is_checked_when_listed(tmp_path):
    no_default = provider("qwen", models=(), default=None)
    api = api_for(tmp_path, {"structured": structured(), "qwen": no_default})
    assert post(api, {"text": "inspect cabinet", "provider": "qwen"}) == (
        422,
        {"error": "model is required for qwen"},
    )
    assert post(api, {"text": "inspect cabinet", "provider": "qwen", "model": "any"})[0] == 201


def test_accepted_event_names_the_provider_and_model(tmp_path):
    api = api_for(tmp_path, registry())
    post(api, {"text": "inspect cabinet", "provider": "openai_chatgpt_plan", "model": "g2"})
    accepted = [e for e in api.events if e["kind"] == "accepted"]
    assert accepted[0]["provider"] == "openai_chatgpt_plan" and accepted[0]["model"] == "g2"


def test_build_providers_lists_unconfigured_providers_with_reasons():
    args = argparse.Namespace(qwen_url=None, openai_token_file=None, openai_model=None)
    providers = phase1.build_providers(args)
    assert list(providers) == ["structured", "qwen", "openai_chatgpt_plan"]
    assert providers["structured"].describe() == {
        "id": "structured",
        "label": "Structured goal parser (no language model)",
        "available": True,
        "reason": None,
        "models": [],
        "default_model": None,
        "supports": ["inspect cabinet", "goto <place>"],
    }
    assert providers["qwen"].describe()["reason"] == "start the backend with --qwen-url"
    assert providers["qwen"].available is False
    assert providers["openai_chatgpt_plan"].reason == (
        "sign in with experiments/openai_plan_login.py and start with --openai-token-file"
    )
    assert providers["openai_chatgpt_plan"].describe()["supports"] == []


def test_qwen_offers_the_single_served_model(tmp_path):
    args = argparse.Namespace(
        qwen_url="http://127.0.0.1:9", openai_token_file=None, openai_model=None
    )
    qwen = phase1.build_providers(args)["qwen"]
    assert qwen.available and len(qwen.models) == 1
    assert qwen.default_model == qwen.models[0]


def write_token(path, expires_in=3600, scopes=("chatgpt.tokens.use.direct",)):
    path.write_text(
        json.dumps(
            {
                "access_token": SECRET,
                "scopes": list(scopes),
                "expires_at": time.time() + expires_in,
            }
        ),
        encoding="utf-8",
    )
    return path


def fake_catalogue(monkeypatch, status=200, models=None, error=None):
    calls = []

    def list_models(token, timeout_s=30.0):
        calls.append(token)
        if error is not None:
            raise error
        listed = models if models is not None else [
            {"slug": "gpt-a", "visibility": "list"},
            {"id": "gpt-b"},
            {"slug": "gpt-a"},
        ]  # fmt: skip
        return status, None, listed

    monkeypatch.setattr(openai_plan, "list_models", list_models)
    return calls


def test_chatgpt_models_come_from_one_catalogue_call_and_never_expose_the_token(
    tmp_path, monkeypatch
):
    calls = fake_catalogue(monkeypatch)
    args = argparse.Namespace(
        qwen_url=None, openai_token_file=write_token(tmp_path / "t.json"), openai_model="gpt-b"
    )
    providers = phase1.build_providers(args)
    chatgpt = providers["openai_chatgpt_plan"]
    assert calls == [SECRET]  # exactly one listing call, made at start-up
    assert chatgpt.available and chatgpt.models == ["gpt-a", "gpt-b"]
    assert chatgpt.default_model == "gpt-b"
    assert SECRET not in json.dumps([p.describe() for p in providers.values()])


def test_chatgpt_default_is_the_first_listed_model_without_openai_model(tmp_path, monkeypatch):
    fake_catalogue(monkeypatch)
    args = argparse.Namespace(
        qwen_url=None, openai_token_file=write_token(tmp_path / "t.json"), openai_model=None
    )
    assert phase1.build_providers(args)["openai_chatgpt_plan"].default_model == "gpt-a"


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ({"status": 401}, "model list request failed: HTTP 401"),
        ({"error": ConnectionError("no route")}, "model list unavailable: ConnectionError"),
        ({"expires_in": -10}, "SIWC access token expired; sign in again"),
        (
            {"scopes": ("openid",)},
            "credential does not grant chatgpt.tokens.use.direct; sign in again",
        ),
    ],
)
def test_failed_listing_marks_chatgpt_unavailable_without_inference(
    tmp_path, monkeypatch, setup, reason
):
    token_options = {k: setup[k] for k in ("expires_in", "scopes") if k in setup}
    calls = fake_catalogue(monkeypatch, status=setup.get("status", 200), error=setup.get("error"))
    args = argparse.Namespace(
        qwen_url=None,
        openai_token_file=write_token(tmp_path / "t.json", **token_options),
        openai_model="gpt-b",
    )
    chatgpt = phase1.build_providers(args)["openai_chatgpt_plan"]
    assert chatgpt.available is False and chatgpt.reason == reason
    assert SECRET not in chatgpt.reason
    assert (calls == []) == ("expires_in" in setup or "scopes" in setup)


def test_chatgpt_backend_is_created_once_per_model(tmp_path):
    factory = phase1.chatgpt_backend_factory(write_token(tmp_path / "t.json"))
    first = factory("gpt-a")
    assert isinstance(first, ChatGPTPlanBackend) and first.model == "gpt-a"
    assert factory("gpt-a") is first
    assert factory("gpt-b") is not first and factory("gpt-b").model == "gpt-b"


def test_each_task_gets_the_planner_its_meta_names(tmp_path, monkeypatch):
    runs = []

    class RecordingAgent:
        def __init__(self, toolbox, planner, max_steps):
            self.planner = planner

        def run(self, goal, on_event=None, history=None, preempt=None):
            runs.append(self.planner)
            return goal

    monkeypatch.setattr(phase1, "TaskAgent", RecordingAgent)
    qwen_backend = FakeBackend("q1")
    providers = {
        "structured": structured(),
        "qwen": provider("qwen", models=["q1"], default="q1", backend=lambda m: qwen_backend),
        "openai_chatgpt_plan": provider(
            "openai_chatgpt_plan",
            models=["g1", "g2"],
            default="g1",
            backend=phase1.chatgpt_backend_factory(write_token(tmp_path / "t.json")),
        ),
    }
    emitted = []
    run_task = phase1.make_run_task(
        SimpleNamespace(), SimpleNamespace(time=0.0), SimpleNamespace(last_frame=None),
        providers, lambda kind, data: emitted.append((kind, data)),
    )  # fmt: skip
    cancel, preempt = threading.Event(), threading.Event()

    def task(meta, text="inspect cabinet"):
        return SimpleNamespace(id="t1", text=text, goal=phase1.mobile_goal(text), meta=meta)

    run_task(task({"provider": "qwen", "model": "q1"}), [], preempt, cancel)
    assert isinstance(runs[-1], MobileModelPlanner) and runs[-1].backend is qwen_backend
    assert qwen_backend.cancel is cancel

    run_task(task({"provider": "openai_chatgpt_plan", "model": "g2"}), [], preempt, cancel)
    assert isinstance(runs[-1], MobileModelPlanner)
    assert isinstance(runs[-1].backend, ChatGPTPlanBackend) and runs[-1].backend.model == "g2"

    for meta in ({}, {"provider": "structured", "model": None}):  # no meta means structured
        run_task(task(meta), [], preempt, cancel)
        assert isinstance(runs[-1], Budgeted) and isinstance(runs[-1].planner, RoomSearchPlanner)

    run_task(task({"provider": "qwen", "model": "q1"}, "goto entrance"), [], preempt, cancel)
    assert isinstance(runs[-1], PlacePlanner)

    with pytest.raises(RuntimeError, match="provider not configured: missing"):
        run_task(task({"provider": "missing", "model": "x"}), [], preempt, cancel)
