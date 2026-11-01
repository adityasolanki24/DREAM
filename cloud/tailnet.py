"""Join a cloud session VM to the owner's private Tailscale network (tailnet).

Why: the phone and laptop then reach the VM's phone API directly over WireGuard (UDP, direct or
relayed) instead of through the IAP TCP tunnel, where every new connection cost about 1.5 s in the
31 Oct live session. Nothing is published to the internet: the VM is only reachable inside the
tailnet, the API still requires its bearer token, and the IAP path keeps working as a fallback.

The auth key never enters Git, job arguments or result files. ``upload_key`` (laptop) puts it in
the session's private bucket; ``join`` (VM) reads it once, deletes the object, and joins as an
ephemeral node, which Tailscale removes when the VM goes away.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

KEY_OBJECT = "secrets/tailscale_authkey"
INSTALL_URL = "https://tailscale.com/install.sh"
HOSTNAME_RE = re.compile(r"[a-z][a-z0-9-]{1,61}[a-z0-9]")


def key_uri(bucket: str) -> str:
    return f"gs://{bucket}/{KEY_OBJECT}"


def upload_key(bucket: str, key_file: Path) -> None:
    """Laptop side: copy the key file into the session bucket without printing it."""
    key = key_file.read_text(encoding="utf-8").strip()
    if not key.startswith("tskey-"):
        raise ValueError("not a Tailscale auth key file")
    subprocess.run(
        ["gsutil", "-q", "cp", "-", key_uri(bucket)],
        input=key,
        text=True,
        check=True,
        capture_output=True,
    )


def join_commands(hostname: str) -> list[list[str]]:
    """The VM-side commands after the key is read (the key itself is passed on stdin-free argv)."""
    if not HOSTNAME_RE.fullmatch(hostname):
        raise ValueError("tailnet hostname must be a DNS label")
    return [
        ["tailscale", "up", "--hostname", hostname, "--accept-dns=false", "--ssh=false"],
    ]


def join(bucket: str, hostname: str, out: Path) -> dict:
    """VM side: read and delete the key, install Tailscale if needed, join, record the address."""
    got = subprocess.run(["gsutil", "-q", "cat", key_uri(bucket)], capture_output=True, text=True)
    if got.returncode != 0 or not got.stdout.strip():
        raise RuntimeError(
            "no Tailscale auth key in the session bucket (run isaac_session.py tailnet-key)"
        )
    key = got.stdout.strip()
    subprocess.run(["gsutil", "-q", "rm", key_uri(bucket)], capture_output=True, check=False)
    if shutil.which("tailscale") is None:
        subprocess.run(
            ["sh", "-c", f"curl -fsSL {INSTALL_URL} | sh"], check=True, capture_output=True
        )
    (cmd,) = join_commands(hostname)
    proc = subprocess.run([*cmd, f"--auth-key={key}"], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        # Never echo the key: report only Tailscale's own message.
        raise RuntimeError("tailscale up failed: " + proc.stderr.replace(key, "<key>")[-400:])
    status = json.loads(subprocess.run(["tailscale", "status", "--json"], capture_output=True,
                                       text=True, check=True).stdout)  # fmt: skip
    me = status.get("Self", {})
    record = {
        "hostname": hostname,
        "dns_name": (me.get("DNSName") or "").rstrip("."),
        "tailnet_ips": me.get("TailscaleIPs", []),
        "note": "tailnet-only address of this session VM; the phone API still requires its token",
    }
    (out / "tailnet.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def leave() -> None:
    """Best-effort: ephemeral nodes are removed by Tailscale anyway once the VM is gone."""
    subprocess.run(["tailscale", "logout"], capture_output=True, check=False, timeout=30)
