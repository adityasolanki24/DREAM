"""Validate the exact CUDA/Qwen/G1 runtime and bake its immutable image."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from cloud.cached_runtime import validate
from experiments.cloud_g1_qwen import validate_evidence

LOCK = Path("cloud/requirements.lock.txt")
CONFIG = Path("experiments/configs/g1_runtime_build_demo.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-disk", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import mujoco
    import torch

    preflight = validate(LOCK)
    if not torch.cuda.is_available() or not torch.cuda.get_device_name(0):
        raise RuntimeError("CUDA device is unavailable")
    if mujoco.__version__ != "3.16.0":
        raise RuntimeError(f"unexpected MuJoCo version: {mujoco.__version__}")

    result = Path("/work/out/g1")
    validate_evidence(result, CONFIG)
    model = json.loads((result / "model.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (
        model.get("checkpoint") != config["model"]["checkpoint"]
        or model.get("revision") != config["model"]["revision"]
        or model.get("device") != "cuda:0"
        or not model.get("checkpoint_files_sha256")
    ):
        raise RuntimeError("G1/Qwen experiment did not validate exact CUDA checkpoint inference")

    receipt = {
        **preflight,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "mujoco": mujoco.__version__,
        "checkpoint": model["checkpoint"],
        "checkpoint_revision": model["revision"],
        "inference_and_physics_validated": True,
        "demo_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
    }
    description = json.dumps(
        {
            "requirements_lock_sha256": receipt["requirements_lock_sha256"],
            "preflight_passed": True,
            "inference_and_physics_validated": True,
            "checkpoint_revision": receipt["checkpoint_revision"],
            "mujoco": receipt["mujoco"],
        },
        separators=(",", ":"),
    )
    if len(description) > 256:
        raise ValueError("GCE image description exceeds its limit")

    import urllib.request

    def metadata(path: str) -> str:
        request = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/" + path,
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.read().decode()

    bucket = metadata("instance/attributes/dream-bucket")
    project = metadata("project/project-id")
    # The snapshot boots as a fresh VM. Never bake the source VM's one-shot
    # startup guard into the image or the fresh guest will skip provisioning.
    Path("/var/dream_started").unlink(missing_ok=True)
    subprocess.run(["sync"], check=True, timeout=120)
    subprocess.run(
        ["gsutil", "-m", "rsync", "-r", "/work/out", f"gs://{bucket}/out"],
        check=True,
        timeout=600,
    )
    zone = metadata("instance/zone").rsplit("/", 1)[-1]
    command = [
        "gcloud",
        "compute",
        "images",
        "create",
        args.image_name,
        "--project",
        project,
        "--source-disk",
        args.source_disk,
        "--source-disk-zone",
        zone,
        "--force",
        "--description",
        description,
        "--labels",
        "dream=1,purpose=g1-qwen-runtime",
        "--format=json",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    created = json.loads(completed.stdout)
    if isinstance(created, list):
        if len(created) != 1 or not isinstance(created[0], dict):
            raise RuntimeError("GCE returned an unexpected image-create response")
        image = created[0]
    elif isinstance(created, dict):
        image = created
    else:
        raise RuntimeError("GCE returned an unexpected image-create response")
    image_id = str(image.get("id", ""))
    if not image_id.isdigit():
        raise RuntimeError("GCE did not return an immutable numeric image ID")
    deadline = time.monotonic() + 600
    while True:
        verified = json.loads(
            subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "images",
                    "describe",
                    args.image_name,
                    "--project",
                    project,
                    "--format=json",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            ).stdout
        )
        if verified.get("status") == "READY":
            break
        if verified.get("status") == "FAILED" or time.monotonic() >= deadline:
            raise RuntimeError(f"created image did not become READY: {verified.get('status')}")
        time.sleep(10)
    if (
        str(verified.get("id")) != image_id
        or verified.get("status") != "READY"
        or json.loads(verified.get("description", "{}")).get("requirements_lock_sha256")
        != receipt["requirements_lock_sha256"]
    ):
        raise RuntimeError("created image identity or embedded preflight receipt differs")
    receipt["image"] = {
        "name": args.image_name,
        "id": image_id,
        "disk_size_gb": verified.get("diskSizeGb"),
    }
    args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
