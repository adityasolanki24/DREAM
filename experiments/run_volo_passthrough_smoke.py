"""Run VoLoAgent's official schema client through its CLI proxy and a mock VLA.

This verifies local protocol pass-through only. The mock emits zero joint targets, and the
client never connects to a robot or simulator. Run in VoLoAgent's separate environment:

    python experiments/run_volo_passthrough_smoke.py <VoLoAgent clone> <out.json>
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import websockets.sync.server as ws_server


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(clone: Path, out: Path) -> None:
    clone = clone.resolve()
    py = Path(sys.executable)
    sys.path.insert(0, str(clone))
    from vlm_orchestrator.utils import codec

    received: list[dict] = []
    packer = codec.Packer()

    def handler(ws):
        ws.send(packer.pack({"model": "mock-openpi-vla", "version": "cpu-smoke"}))
        while True:
            try:
                observation = codec.unpackb(ws.recv())
                received.append(
                    {
                        "prompt": observation.get("prompt"),
                        "keys": sorted(observation),
                        "step": observation.get("__step"),
                    }
                )
                ws.send(packer.pack({"actions": np.zeros((8, 8), dtype=np.float32)}))
            except Exception:
                return

    vla = ws_server.serve(handler, "127.0.0.1", 0, compression=None, max_size=None)
    vla_port = vla.socket.getsockname()[1]
    threading.Thread(target=vla.serve_forever, daemon=True).start()
    proxy_port = _free_port()
    proxy_cmd = [
        str(py),
        "-m",
        "vlm_orchestrator.cli",
        "--mode",
        "passthrough",
        "--host",
        "127.0.0.1",
        "--port",
        str(proxy_port),
        "--vla-host",
        "127.0.0.1",
        "--vla-port",
        str(vla_port),
        "--frontend",
        "openpi-ws",
        "--backend",
        "openpi-ws",
        "--robolab-output-dir",
        "",
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(clone))
    proxy = subprocess.Popen(
        proxy_cmd,
        cwd=clone,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    stdout = ""
    client_cmd = [
        str(py),
        str(clone / "examples" / "schema_smoke_test.py"),
        "--orch-host",
        "127.0.0.1",
        "--orch-port",
        str(proxy_port),
        "--steps",
        "3",
        "--grasp",
    ]
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proxy.poll() is not None:
                raise RuntimeError("VoLo CLI proxy exited before the smoke client connected")
            try:
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("VoLo CLI proxy did not start within 20 seconds")
        client = subprocess.run(
            client_cmd,
            cwd=clone,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        stdout = client.stdout + client.stderr
        if client.returncode:
            raise RuntimeError(f"schema client exited {client.returncode}: {stdout}")
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
            proxy.wait(timeout=5)
        if proxy.stdout:
            stdout += proxy.stdout.read()
        vla.shutdown()

    sha = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    payload = {
        "result": "PASS",
        "scope": (
            "local protocol pass-through with synthetic observations and mock VLA; "
            "no robot, simulator, model, or VLM"
        ),
        "upstream_commit": sha,
        "python": sys.version.split()[0],
        "dependencies": {
            "numpy": np.__version__,
            "websockets": importlib.metadata.version("websockets"),
            "vlm_orchestrator": importlib.metadata.version("vlm-orchestrator"),
        },
        "mock_vla": "mock-openpi-vla cpu-smoke; returns float32 zero actions [8,8]",
        "proxy_command": proxy_cmd,
        "client_command": client_cmd,
        "client_output": stdout,
        "mock_received": received,
        "expected_client_steps": 3,
        "interpretation": (
            "The official smoke client supplies synthetic grasp schema fields via --grasp. "
            "The proxy removes camera/depth fields before forwarding to the mock VLA as "
            "documented in vlm_orchestrator/proxy.py; this validates protocol routing and "
            "filtering, not grasp execution."
        ),
        "checks": {
            "steps_reached_mock_vla": len(received) == 3,
            "proxy_only_grasp_fields_stripped_before_vla": all(
                "observation/camera_K" not in item["keys"]
                and "observation/ee_quat" in item["keys"]
                and not any(key.startswith("observation/depth_") for key in item["keys"])
                for item in received
            ),
            "official_client_passed": "[PASS] all assertions held" in stdout,
        },
    }
    if not all(payload["checks"].values()):
        payload["result"] = "FAIL"
        raise RuntimeError(json.dumps(payload, indent=2))
    with out.open("x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} (VoLoAgent {sha})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
