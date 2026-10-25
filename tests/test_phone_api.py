"""Phone API: authentication, task submission/cancel/priority through the scheduler only."""

from __future__ import annotations

import json
import urllib.request

import pytest

from dream.runtime.phone_api import PhoneAPI, load_or_create_token
from dream.runtime.scheduler import TaskScheduler

TOKEN = "t" * 32


def parse(text):
    if text.startswith("fly"):
        raise ValueError("this robot cannot fly")
    return {"text": text}


@pytest.fixture
def api(tmp_path):
    s = TaskScheduler(str(tmp_path / "t.sqlite"), lambda *a: None, parse)
    a = PhoneAPI(s, TOKEN)
    s.notify = a.notify
    return a


def call(api, method, url, body=None, token=TOKEN):
    raw = json.dumps(body).encode() if body is not None else b""
    return api.handle(method, url, f"Bearer {token}" if token else None, raw)


def test_every_control_endpoint_requires_the_token(api):
    for method, url in (("GET", "/tasks"), ("POST", "/tasks"), ("GET", "/status")):
        assert call(api, method, url, {"text": "x"}, token=None)[0] == 401
        assert call(api, method, url, {"text": "x"}, token="wrong" * 8)[0] == 401
    assert call(api, "GET", "/")[0] == 200  # the page itself holds no data


def test_submit_list_priority_cancel(api):
    code, out = call(api, "POST", "/tasks", {"text": "bring my bottle", "priority": 5})
    assert code == 201
    tid = out["id"]
    assert call(api, "GET", "/tasks")[1][0]["priority"] == 5
    assert call(api, "POST", f"/tasks/{tid}/priority", {"priority": 1})[0] == 200
    assert call(api, "GET", "/tasks")[1][0]["priority"] == 1
    assert call(api, "POST", f"/tasks/{tid}/cancel", {})[1] == {"result": "cancelled"}
    assert call(api, "POST", "/tasks/nope/priority", {"priority": 1})[0] == 404
    events = call(api, "GET", "/events?after=0")[1]
    assert [e["kind"] for e in events] == ["accepted"]


def test_unrepresentable_and_malformed_requests_are_rejected(api):
    assert call(api, "POST", "/tasks", {"text": "fly to the moon"})[0] == 422
    assert call(api, "POST", "/tasks", {"text": ""})[0] == 400
    assert api.handle("POST", "/tasks", f"Bearer {TOKEN}", b"{not json")[0] == 400
    assert call(api, "GET", "/tasks")[1] == []


def test_served_over_http_on_localhost(api):
    server = api.serve(port=0)
    try:
        port = server.server_address[1]
        assert server.server_address[0] == "127.0.0.1"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tasks", data=json.dumps({"text": "tidy"}).encode(),
            headers={"Authorization": f"Bearer {TOKEN}"}, method="POST",
        )  # fmt: skip
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 201
    finally:
        server.shutdown()


def test_token_is_created_once_and_reused(tmp_path):
    path = tmp_path / "token"
    a = load_or_create_token(path)
    assert len(a) >= 32 and load_or_create_token(path) == a
