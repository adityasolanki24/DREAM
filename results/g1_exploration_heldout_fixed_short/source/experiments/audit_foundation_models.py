"""Record whether the selected foundation models can be obtained and run in this environment.

    python experiments/audit_foundation_models.py

Writes results/foundation_model_audit/audit.json: CPU, memory, disk, GPU, installed inference
packages, and whether each host that serves the checkpoints answers. Model-integration claims
in the README are bounded by this file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]

HOSTS = {
    "huggingface.co (Qwen3-VL-4B-Instruct)": "https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct",
    "huggingface.co (GR00T-N1.7-LIBERO)": "https://huggingface.co/api/models/nvidia/GR00T-N1.7-LIBERO",
    "github.com (Isaac-GR00T)": "https://github.com/NVIDIA/Isaac-GR00T",
    "github.com (Qwen3-VL)": "https://github.com/QwenLM/Qwen3-VL",
    "modelscope.cn (Qwen mirror)": "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct",
    "download.pytorch.org (CPU wheels)": "https://download.pytorch.org/whl/cpu/torch/",
    "pypi.org (packages)": "https://pypi.org/simple/transformers/",
}


def _reach(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - any failure to connect is the finding
        return f"unreachable: {type(e).__name__}: {str(e)[:120]}"


def memory_bytes() -> tuple[int, int]:
    """Total and currently available physical memory, without an extra dependency."""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended_available",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.total, status.available
    with open("/proc/meminfo") as f:
        info = {line.split()[0].rstrip(":"): int(line.split()[1]) * 1024 for line in f}
    return info["MemTotal"], info["MemAvailable"]


def model_metadata(model: str) -> dict:
    """Record availability and immutable revision; do not download model weights."""
    url = f"https://huggingface.co/api/models/{model}"
    with urllib.request.urlopen(url, timeout=20) as response:
        raw = response.read()
    meta = json.loads(raw)
    return {
        "url": url,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "revision": meta.get("sha"),
        "gated": meta.get("gated"),
        "license": meta.get("cardData", {}).get("license"),
        "files": [f["rfilename"] for f in meta.get("siblings", [])],
    }


def source_contracts(models: dict) -> dict:
    """Inspect pinned, small official files only; no weights or upstream code is vendored."""
    hashes = {}

    def fetch(url):
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read()
        hashes[url] = hashlib.sha256(data).hexdigest()
        return data

    qwen_revision = models["Qwen/Qwen3-VL-4B-Instruct"]["revision"]
    qwen_root = f"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/{qwen_revision}"
    template = json.loads(fetch(qwen_root + "/chat_template.json"))["chat_template"]
    index = json.loads(fetch(qwen_root + "/model.safetensors.index.json"))
    qwen = {
        "weight_bytes": index["metadata"]["total_size"],
        "template_accepts_tools": "if tools" in template,
        "template_uses_tool_call_markers": "<tool_call>" in template,
        "template_accepts_image_parts": "content.type == 'image'" in template,
        "inference_run": False,
    }
    gr_revision = models["nvidia/GR00T-N1.7-LIBERO"]["revision"]
    gr_root = f"https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/resolve/{gr_revision}/libero_10"
    processor = json.loads(fetch(gr_root + "/processor_config.json"))
    gr_index = json.loads(fetch(gr_root + "/model.safetensors.index.json"))
    upstream = json.loads(fetch("https://api.github.com/repos/NVIDIA/Isaac-GR00T/commits/main"))[
        "sha"
    ]
    upstream_root = f"https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/{upstream}"
    modality = json.loads(fetch(upstream_root + "/examples/LIBERO/modality.json"))
    for path in ("gr00t/eval/sim/LIBERO/libero_env.py", "README.md", "pyproject.toml"):
        fetch(upstream_root + "/" + path)
    return {
        "qwen": qwen,
        "gr00t": {
            "upstream_revision": upstream,
            "checkpoint_subfolder": "libero_10",
            "weight_bytes": gr_index["metadata"]["total_size"],
            "modality": modality,
            "processor_config": processor["processor_kwargs"]["modality_configs"]["libero_sim"],
            "inference_run": False,
            "unresolved_for_dream": [
                "No eye-in-hand camera or validated camera calibration mapping",
                "LIBERO OSC scaling, reference frame and control frequency need verification",
                "DREAM down-facing skills have no validated arbitrary-rotation policy mapping",
                "No per-finger joint positions in the current Embodiment interface",
                "No CUDA GPU or configured inference endpoint",
            ],
        },
        "source_sha256": hashes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="foundation_model_audit")
    args = ap.parse_args()
    out = ROOT / "results" / args.name
    if (out / "audit.json").exists():
        raise SystemExit("audit already exists; choose a new --name to preserve earlier evidence")
    total, available = memory_bytes()
    gpu = shutil.which("nvidia-smi")
    audit = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "dream", "experiments"], cwd=ROOT, text=True
            )
        ),
        "audit_script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
        "platform": platform.platform(),
        "memory_gb": round(total / 1e9, 2),
        "available_memory_gb": round(available / 1e9, 2),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1),
        "nvidia_smi": gpu,
        "gpu": subprocess.run([gpu, "-L"], capture_output=True, text=True).stdout.strip()
        if gpu
        else None,
        "packages": {
            m: bool(importlib.util.find_spec(m))
            for m in ("torch", "transformers", "accelerate", "flash_attn", "gr00t")
        },
        "hosts": {name: _reach(url) for name, url in HOSTS.items()},
    }
    models = {}
    for model in ("Qwen/Qwen3-VL-4B-Instruct", "nvidia/GR00T-N1.7-LIBERO"):
        try:
            models[model] = model_metadata(model)
        except (OSError, ValueError) as e:
            models[model] = {"error": type(e).__name__}
    audit["model_metadata"] = models
    try:
        audit["source_contracts"] = source_contracts(models)
    except (OSError, ValueError, KeyError) as e:
        audit["source_contracts"] = {"error": type(e).__name__}
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
