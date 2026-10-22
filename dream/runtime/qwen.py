"""Qwen3-VL backends for ``VLMAgent`` (see dream/runtime/vlm.py).

Two ways to run the selected model, Qwen/Qwen3-VL-4B-Instruct (Apache-2.0):
- ``QwenTransformersBackend``: a local checkpoint through Hugging Face transformers
  (``Qwen3VLForConditionalGeneration`` and the model's own chat template with ``tools``). Needs
  ``torch`` and ``transformers>=4.57`` (optional extra ``vlm``), and the checkpoint on disk.
  About 9 GB of memory in bfloat16. CPU works but is slow.
- ``OpenAICompatibleBackend``: a separately hosted server with an OpenAI-style
  ``/v1/chat/completions`` API (for example vLLM serving the same checkpoint with the hermes
  tool parser). Narrow interface: messages with images as PNG data URLs, plus tool schemas.

Neither is imported by the test suite, and neither has been run in this repository yet: the
development environment had no route to huggingface.co (README, foundation models).
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request

import numpy as np

QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def _pil(image: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.asarray(image, dtype=np.uint8))


class QwenTransformersBackend:
    """Local inference with the official transformers integration."""

    def __init__(
        self,
        model_path: str = QWEN_MODEL_ID,
        device: str = "cpu",
        max_new_tokens: int = 384,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.name = f"qwen3-vl:{model_path}"
        self._torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict], tools: list[dict]) -> str:
        msgs = [
            {
                "role": m["role"],
                "content": [
                    {"type": "image", "image": _pil(p["image"])} if p["type"] == "image" else p
                    for p in m["content"]
                ],
            }
            for m in messages
        ]
        inputs = self.processor.apply_chat_template(
            msgs,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1] :]
        # Keep <tool_call> markers; drop end-of-turn tokens.
        text = self.processor.decode(new, skip_special_tokens=False)
        return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


class OpenAICompatibleBackend:
    """A hosted model behind an OpenAI-style chat-completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = QWEN_MODEL_ID,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        max_tokens: int = 384,
    ) -> None:
        self.name = f"openai-compatible:{model}@{base_url}"
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model, self.api_key = model, api_key
        self.timeout_s, self.max_tokens = timeout_s, max_tokens

    def generate(self, messages: list[dict], tools: list[dict]) -> str:
        body = {
            "model": self.model,
            "messages": [_openai_message(m) for m in messages],
            "tools": tools,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}
            | ({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            choice = json.loads(resp.read())["choices"][0]["message"]
        calls = choice.get("tool_calls") or []
        if calls:  # a server-side tool parser: re-emit in the model's own format
            fn = calls[0]["function"]
            args = fn.get("arguments", "{}")
            args = json.loads(args) if isinstance(args, str) else args
            return f"<tool_call>{json.dumps({'name': fn['name'], 'arguments': args})}</tool_call>"
        return choice.get("content") or ""


def _openai_message(m: dict) -> dict:
    parts = []
    for p in m["content"]:
        if p["type"] == "image":
            buf = io.BytesIO()
            _pil(p["image"]).save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": p["text"]})
    return {"role": m["role"], "content": parts}
