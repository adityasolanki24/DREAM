"""Small loopback OpenAI-compatible tool endpoint for the locked local Qwen3-VL model."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def decode_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if not url.startswith("data:image/png;base64,"):
                    raise ValueError("only inline PNG image observations are supported")
                raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                parts.append({"type": "image", "image": image})
            else:
                raise ValueError("unsupported message content")
        converted.append({"role": message["role"], "content": parts})
    return converted


def serve(args) -> None:
    import torch

    from dream.runtime.qwen import QwenTransformersBackend

    backend = QwenTransformersBackend(
        args.model,
        device="cuda:0",
        dtype="bfloat16",
        max_new_tokens=args.max_new_tokens,
        timeout_s=args.timeout_s,
        revision=args.revision,
    )
    if not torch.cuda.is_available() or backend.provenance["model_revision"] != args.revision:
        raise RuntimeError("Qwen CUDA device or pinned model revision failed preflight")
    print(json.dumps({"ready": True, "model": args.model, "revision": args.revision,
                      "device": backend.provenance["device"],
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return  # prompts, image payloads, and tool outputs are never logged by the server

        def do_GET(self):  # noqa: N802
            if self.path == "/healthz":
                body = json.dumps({"ready": True, "model": args.model,
                                   "revision": args.revision}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 12 * 1024 * 1024:
                self.send_error(413)
                return
            started = time.perf_counter()
            try:
                request = json.loads(self.rfile.read(size))
                response_text = backend.generate(
                    decode_messages(request["messages"]), request["tools"]
                )
                body = json.dumps({
                    "id": "dream-qwen-tool-call",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": args.model,
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": response_text,
                    }}],
                    "usage": backend.last_usage or {},
                    "latency_s": round(time.perf_counter() - started, 3),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            except Exception as error:  # no prompt, image or credential is written to output
                body = json.dumps({"error": type(error).__name__}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--timeout-s", type=float, default=75.0)
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
