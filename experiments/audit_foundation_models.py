"""Record whether the selected foundation models can be obtained and run in this environment.

    python experiments/audit_foundation_models.py

Writes results/foundation_model_audit/audit.json: CPU, memory, disk, GPU, installed inference
packages, and whether each host that serves the checkpoints answers. Model-integration claims
in the README are bounded by this file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
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


def main() -> None:
    mem_kb = next(
        int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemTotal")
    )
    gpu = shutil.which("nvidia-smi")
    audit = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
        "memory_gb": round(mem_kb / 1e6, 1),
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
    out = ROOT / "results" / "foundation_model_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
