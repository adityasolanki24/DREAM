"""ChatGPT-plan Responses backend using a caller-provided SIWC OAuth access token.

The token must come from this application's documented Sign in with ChatGPT flow; Codex
credentials are deliberately unsupported. Authentication and account registration are
performed separately so the robot runtime never opens a browser or stores credentials.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path

import numpy as np


def _image_data_url(image: np.ndarray) -> str:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def responses_input(messages: list[dict]) -> list[dict]:
    """Convert the shared VLM chat format into Responses API input items."""
    converted = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part["type"] == "image":
                content.append({"type": "input_image", "image_url": _image_data_url(part["image"])})
            else:
                kind = "output_text" if message["role"] == "assistant" else "input_text"
                content.append({"type": kind, "text": part["text"]})
        role = "developer" if message["role"] == "system" else message["role"]
        converted.append({"role": role, "content": content})
    return converted


def response_tool_schemas(tools: list[dict]) -> list[dict]:
    return [{"type": "function", **t["function"], "strict": False} for t in tools]


def response_output_text(data: dict) -> str:
    """Preserve all function calls so the shared parser rejects batches atomically."""
    output = data.get("output")
    if not isinstance(output, list):
        raise ValueError("malformed Responses API output")
    calls, text = [], []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            calls.append({"name": item.get("name"), "arguments": item.get("arguments", "{}")})
        elif item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    text.append(part["text"])
    if calls:
        return "\n".join(f"<tool_call>{json.dumps(c)}</tool_call>" for c in calls)
    return "\n".join(text)


class ChatGPTPlanBackend:
    """Vision + typed function-call backend, authenticated with a SIWC plan token."""

    def __init__(self, access_token: str, model: str, timeout_s: float = 90.0):
        if not access_token or not model:
            raise ValueError("a SIWC access token and model slug are required")
        self.name = f"chatgpt-plan:{model}"
        self._token, self.model, self.timeout_s = access_token, model, timeout_s
        self.cancel = threading.Event()
        self.last_http_status = self.last_request_id = self.last_usage = None
        self.last_rate_limits: dict = {}
        self.last_error_body = None

    @classmethod
    def from_token_file(cls, path: str | Path, model: str, timeout_s: float = 90.0):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if "chatgpt.tokens.use.direct" not in record.get("scopes", []):
            raise ValueError("credential does not grant chatgpt.tokens.use.direct")
        return cls(record.get("access_token", ""), model, timeout_s)

    def set_cancel_event(self, cancel: threading.Event) -> None:
        self.cancel = cancel

    def generate(self, messages: list[dict], tools: list[dict]) -> str:
        if self.cancel.is_set():
            raise RuntimeError("inference cancelled")
        body = {"model": self.model, "input": responses_input(messages),
                "tools": response_tool_schemas(tools), "parallel_tool_calls": False,
                "store": False, "stream": True}
        started = time.monotonic()
        from dream.runtime.siwc_auth import http_session

        completed = None
        with http_session().post("https://api.openai.com/v1/responses", json=body,
                                 headers={"Authorization": f"Bearer {self._token}"},
                                 timeout=self.timeout_s, stream=True) as resp:
            self.last_http_status = resp.status_code
            self.last_request_id = resp.headers.get("x-request-id")
            self.last_rate_limits = {k: v for k, v in resp.headers.items()
                                     if k.lower().startswith("x-ratelimit-")}
            if not resp.ok:
                self.last_error_body = resp.text[:600]
                raise RuntimeError(f"Responses HTTP {resp.status_code}: {self.last_error_body}")
            for line in resp.iter_lines(decode_unicode=True):
                if self.cancel.is_set():
                    raise RuntimeError("inference cancelled")
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("type") == "response.failed":
                    error = event.get("response", {}).get("error", {})
                    raise RuntimeError("Responses API failed: " + str(error.get("code", "unknown")))
                if event.get("type") == "response.completed":
                    completed = event.get("response")
            if completed is None:
                raise ValueError("Responses stream ended without response.completed")
        if self.cancel.is_set():
            raise RuntimeError("inference cancelled")
        if time.monotonic() - started > self.timeout_s:
            raise TimeoutError("Responses API exceeded time budget")
        if completed.get("status") != "completed":
            raise ValueError(f"Responses API status {completed.get('status')!r}")
        self.last_response_id = completed.get("id")
        self.last_usage = completed.get("usage")
        return response_output_text(completed)


def list_models(access_token: str, timeout_s: float = 30.0) -> tuple[int, str | None, list[dict]]:
    """GET /v1/models for the signed-in account; returns (status, request_id, visible models)."""
    from dream.runtime.siwc_auth import http_session

    resp = http_session().get("https://api.openai.com/v1/models", timeout=timeout_s,
                              headers={"Authorization": f"Bearer {access_token}"})
    models = []
    if resp.ok:
        data = resp.json()
        items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
        models = [m for m in items if isinstance(m, dict) and m.get("visibility", "list") == "list"]
    return resp.status_code, resp.headers.get("x-request-id"), models
