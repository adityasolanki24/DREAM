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
    assert [e["kind"] for e in events] == ["accepted", "cancelled"]


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


def test_invalid_request_objects_and_providers_are_rejected(api):
    for body in (b"null", b"[]", b'"text"'):
        assert api.handle("POST", "/tasks", f"Bearer {TOKEN}", body)[0] == 400
    assert call(api, "POST", "/tasks", {"text": "tidy", "provider": "qwen"})[0] == 422
    assert call(api, "GET", "/events?after=bad")[0] == 422


def test_snapshot_is_authenticated_and_events_remain_monotonic(api):
    api.snapshot = lambda: {"camera": None}
    assert call(api, "GET", "/snapshot", token=None)[0] == 401
    assert call(api, "GET", "/snapshot")[1] == {"camera": None}
    for _ in range(2002):
        api.notify("update", {})
    events = call(api, "GET", "/events?after=1")[1]
    assert len(events) == 2000 and events[0]["n"] == 3 and events[-1]["n"] == 2002


def test_ag_ui_maps_real_events_and_keeps_cursor(api):
    api.notify("started", {"id": "t"})
    api.notify("action", {"id": "t", "tool": "look_around", "args": {}})
    api.notify("result", {"id": "t", "tool": "look_around", "status": "succeeded", "message": ""})
    api.notify("succeeded", {"id": "t", "reason": "observed"})
    code, page = call(api, "GET", "/ag-ui/events?after=0")
    assert code == 200 and page["cursor"] == 4
    assert [e["type"] for e in page["events"]] == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "RUN_FINISHED",
    ]
    assert page["events"][1]["toolCallId"] == page["events"][4]["toolCallId"]
    tail = call(api, "GET", "/ag-ui/events?after=2")[1]
    assert tail["cursor"] == 4 and tail["events"][0]["type"] == "TOOL_CALL_RESULT"
    assert call(api, "GET", "/ag-ui/events", token=None)[0] == 401


def test_ag_ui_run_end_keeps_the_real_outcome(api):
    for tid, state in (("a", "cancelled"), ("b", "failed"), ("c", "succeeded")):
        api.notify("started", {"id": tid})
        api.notify(state, {"id": tid, "reason": f"{state} reason"})
    events = call(api, "GET", "/ag-ui/events?after=0")[1]["events"]
    ends = [e for e in events if e["type"] in ("RUN_FINISHED", "RUN_ERROR")]
    assert [e["type"] for e in ends] == ["RUN_FINISHED", "RUN_ERROR", "RUN_FINISHED"]
    assert ends[0]["result"] == {"state": "cancelled", "reason": "cancelled reason"}
    assert ends[1]["message"] == "failed reason" and ends[2]["result"]["state"] == "succeeded"


def test_ag_ui_stream_id_is_stable_per_api_and_changes_on_restart(api):
    first = call(api, "GET", "/ag-ui/events?after=0")[1]["stream_id"]
    assert first == call(api, "GET", "/ag-ui/events?after=0")[1]["stream_id"]
    restarted = PhoneAPI(api.scheduler, TOKEN)
    assert first != call(restarted, "GET", "/ag-ui/events?after=0")[1]["stream_id"]
