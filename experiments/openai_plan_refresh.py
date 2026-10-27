"""Refresh one explicit SIWC credential; never print token values.

https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from dream.runtime.siwc_auth import ISSUER, RESOURCE, http_session


def refresh(path: Path, http=None) -> None:
    """Serialize rotation; a stale lock requires manual investigation and removal."""
    lock = path.with_suffix(path.suffix + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    tmp = None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("issuer") != ISSUER
            or not record.get("client_id", "").startswith("oaiapp_")
            or not record.get("refresh_token")
            or "chatgpt.tokens.use.direct" not in record.get("scopes", [])
        ):
            raise ValueError("invalid SIWC credential record")
        response = (http or http_session()).post(
            ISSUER + "/api/accounts/oauth/token",
            timeout=30,
            data={
                "grant_type": "refresh_token",
                "client_id": record["client_id"],
                "refresh_token": record["refresh_token"],
                "resource": RESOURCE,
            },
        )
        if not response.ok:
            raise RuntimeError(f"refresh failed: HTTP {response.status_code}")
        token = response.json()
        scopes = token.get("scope", " ".join(record["scopes"])).split()
        if (
            not token.get("access_token")
            or not token.get("refresh_token")
            or int(token.get("expires_in", 0)) <= 0
            or "chatgpt.tokens.use.direct" not in scopes
        ):
            raise ValueError("invalid refresh response; credential not updated")
        record.update(
            access_token=token["access_token"],
            refresh_token=token["refresh_token"],
            expires_in=int(token["expires_in"]),
            expires_at=int(time.time()) + int(token["expires_in"]),
            scopes=scopes,
            saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as f:
            tmp = Path(f.name)
            if os.name != "nt":
                tmp.chmod(0o600)
            json.dump(record, f, indent=2)
        tmp.replace(path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("token_file", type=Path)
    args = ap.parse_args()
    refresh(args.token_file)
    print("SIWC credential refreshed; token values not printed.")


if __name__ == "__main__":
    main()
