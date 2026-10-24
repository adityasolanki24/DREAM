"""File-scoped expiring download capabilities; the user's HF login stays local.

Only the private input bucket receives signed weight URLs. Never print them or place them
in results. Metadata and weight hashes are public evidence; credentials are not. Both
upstream loaders receive a local backbone identity to avoid tokenizer metadata networking.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

FILES = {
    "backbone": {
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
        "model.safetensors",
    },
    "checkpoint": {
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "LICENSE",
    },
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def localize_checkpoint(checkpoint: Path, backbone: Path, destination: Path) -> Path:
    """Derive loader configuration only; leave upstream files and all weights unchanged."""
    destination.mkdir(exist_ok=False)
    # get_backbone_cls dispatches on this literal name even for local directories.
    # Hard links retain the pinned bytes without duplicating multi-GB weights.
    local_backbone = destination / "nvidia" / "Cosmos-Reason2-2B"
    local_backbone.mkdir(parents=True)
    for path in backbone.iterdir():
        if path.is_file():
            os.link(path, local_backbone / path.name)
    identity = local_backbone.resolve().as_posix()
    for path in checkpoint.iterdir():
        if path.is_file():
            os.link(path, destination / path.name)
    for name in ("config.json", "processor_config.json"):
        target = destination / name
        original = json.loads(target.read_text())
        if name == "config.json":
            original["model_name"] = identity
        else:
            original.setdefault("processor_kwargs", {})["model_name"] = identity
        target.unlink()  # Break the metadata hard link before writing a derived configuration.
        target.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
    return destination


def prepare_assets(directory: Path, config: dict) -> dict:
    """Authenticate locally; stage metadata and expiring CDN URLs, never the HF token."""
    from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url

    directory.mkdir(parents=True, exist_ok=False)
    token = get_token()
    if not token:
        raise RuntimeError("authorized local Hugging Face login required")
    api = HfApi(token=token)
    weights, metadata = [], []
    for group, repo, revision in (
        ("backbone", config["backbone"], config["backbone_revision"]),
        ("checkpoint", config["checkpoint"], config["revision"]),
    ):
        info = api.model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError("dependency revision mismatch")
        files = {s.rfilename: s for s in info.siblings}
        if not FILES[group] <= files.keys():
            raise ValueError("required dependency file absent")
        (directory / group).mkdir()
        for name in sorted(FILES[group]):
            item = files[name]
            if name.endswith(".safetensors"):
                request = urllib.request.Request(
                    hf_hub_url(repo, name, revision=revision)
                    + f"?download=true&t={time.time_ns()}",
                    headers={"Authorization": "Bearer " + token},
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    url = response.geturl()
                expires = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["Expires"][0])
                if expires - time.time() < 2700 or token in url:
                    raise RuntimeError("download capability lifetime insufficient")
                weights.append(
                    {
                        "group": group,
                        "file": name,
                        "url": url,
                        "sha256": item.lfs.sha256,
                        "size": item.size,
                        "expires": expires,
                    }
                )
            else:
                path = Path(hf_hub_download(repo, name, revision=revision, token=token))
                shutil.copyfile(path, directory / group / name)
                metadata.append({"group": group, "file": name, "sha256": digest(path)})
    private = directory / "weights.private.json"
    private.write_text(json.dumps(weights), encoding="utf-8")
    assert token not in private.read_text()
    report = {
        "weights": [{k: v for k, v in w.items() if k != "url"} for w in weights],
        "metadata": metadata,
        "hf_token_transferred": False,
        "authorization": "Temporary file-scoped CDN capabilities; private input only",
    }
    (directory / "asset-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def download_weights(entries: list[dict], directory: Path) -> list[dict]:
    """Stream official CDN files with integrity checks; sanitize all download failures."""
    evidence = []
    for entry in entries:
        host = urllib.parse.urlsplit(entry["url"]).hostname or ""
        if not (host.endswith(".hf.co") or host.endswith(".huggingface.co")):
            raise ValueError("untrusted weight download host")
        if entry["group"] not in FILES or entry["file"] not in FILES[entry["group"]]:
            raise ValueError("unexpected weight file")
        if time.time() >= entry["expires"]:
            raise RuntimeError("file-scoped download capability expired")
        target = directory / entry["group"] / entry["file"]
        try:
            with (
                urllib.request.urlopen(entry["url"], timeout=120) as response,
                target.open("xb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        except Exception as error:
            raise RuntimeError(
                f"weight download failed: {entry['file']} ({type(error).__name__})"
            ) from None
        if target.stat().st_size != entry["size"] or digest(target) != entry["sha256"]:
            raise ValueError("weight file size/checksum mismatch")
        evidence.append({k: v for k, v in entry.items() if k != "url"})
    return evidence
