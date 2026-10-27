import json

import pytest

from dream.runtime.openai_plan import response_output_text, responses_request


def test_tools_are_developer_additional_tools_input():
    body = responses_request(
        "model",
        [],
        [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}],
    )
    assert "tools" not in body
    assert responses_request("gpt-6-luna", [], [], "none")["reasoning"] == {"effort": "none"}
    assert body["input"][0]["type"] == "additional_tools"
    assert body["input"][0]["role"] == "developer"
    assert body["input"][0]["tools"][0]["name"] == "inspect"


def test_unknown_namespace_fails_closed():
    with pytest.raises(ValueError):
        response_output_text(
            {
                "output": [
                    {
                        "type": "function_call",
                        "namespace": "other",
                        "name": "inspect",
                        "arguments": "{}",
                    }
                ]
            }
        )


def test_streamed_items_are_used_when_completion_omits_output():
    from dream.runtime.openai_plan import completed_with_stream_items

    streamed = [
        {
            "id": "fc_canary",
            "type": "function_call",
            "status": "completed",
            "arguments": '{"index":0}',
            "call_id": "call_canary",
            "name": "explore_frontier",
        }
    ]
    result = completed_with_stream_items({"status": "completed", "output": []}, streamed)
    assert "<tool_call>" in response_output_text(result)


def test_stream_requires_completion_and_retains_all_calls(monkeypatch):
    import dream.runtime.siwc_auth as auth
    from dream.runtime.openai_plan import ChatGPTPlanBackend

    class Response:
        ok = True
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_lines(self):
            for event in [
                {
                    "type": "response.output_item.done",
                    "item": {"type": "function_call", "name": "inspect", "arguments": "{}"},
                },
                {
                    "type": "response.output_item.done",
                    "item": {"type": "function_call", "name": "finish", "arguments": "{}"},
                },
                {"type": "response.completed", "response": {"status": "completed", "output": []}},
            ]:
                yield ("data: " + json.dumps(event)).encode()

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(auth, "http_session", lambda: Session())
    backend = ChatGPTPlanBackend("token", "model")
    assert backend.generate([], []).count("<tool_call>") == 2
    assert len(backend.last_stream_items) == 2


def test_incomplete_stream_fails_and_resets_metadata(monkeypatch):
    import dream.runtime.siwc_auth as auth
    from dream.runtime.openai_plan import ChatGPTPlanBackend

    class Response:
        ok = True
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_lines(self):
            yield (
                b"data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {"type": "function_call", "name": "inspect", "arguments": "{}"},
                    }
                ).encode()
            )

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(auth, "http_session", lambda: Session())
    backend = ChatGPTPlanBackend("token", "model")
    backend.last_usage = {"input_tokens": 100}
    backend.last_response_id = "old"
    with pytest.raises(ValueError, match="without response.completed"):
        backend.generate([], [])
    assert backend.last_usage is None
    assert backend.last_response_id is None
