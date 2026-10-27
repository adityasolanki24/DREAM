"""Run the documented SIWC dynamic-client loopback login for DREAM.

Official contract: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
Starts a 127.0.0.1 callback listener, opens the system browser, validates OAuth state,
exchanges the code (TLS verification on), verifies the ID token against OpenAI's JWKS and
requires the chatgpt.tokens.use.direct scope. The credential record is saved outside the
repository under %LOCALAPPDATA%/DREAM/openai-chatgpt-plan. Token values are never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import time
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dream.runtime.siwc_auth import (
    ISSUER,
    RESOURCE,
    SCOPES,
    authorization_url,
    http_session,
    make_pkce,
    validate_callback,
)

CRED_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DREAM" / "openai-chatgpt-plan"


def _listen(port: int, state: str, client_id: str | None, result: dict) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            result.update({k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items() if v})
            try:
                validate_callback(result, state, client_id)
                ok = True
            except ValueError:
                ok = False
            body = (b"DREAM sign-in received. You may close this tab." if ok
                    else b"DREAM sign-in failed. Return to the terminal.")
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            return

    try:
        server = HTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise SystemExit(f"Cannot listen on 127.0.0.1:{port} ({exc}); pass --port.") from exc
    server.timeout = 5
    return server


def _exchange(http, issued_id: str, code: str, verifier: str, callback: str) -> dict:
    import requests

    try:
        resp = http.post(ISSUER + "/api/accounts/oauth/token", timeout=30, data={
            "grant_type": "authorization_code", "client_id": issued_id, "code": code,
            "code_verifier": verifier, "redirect_uri": callback, "resource": RESOURCE})
    except requests.RequestException as exc:
        raise SystemExit(f"Token exchange network/TLS error: {type(exc).__name__}: "
                         f"{str(exc)[:200]}") from exc
    if not resp.ok:
        try:
            err = resp.json().get("error")
        except ValueError:
            err = None
        raise SystemExit(f"Token exchange failed: HTTP {resp.status_code} error={err}; "
                         "no credentials saved (an invalid_grant means restart sign-in).")
    return resp.json()


def _verify_identity(http, token: dict, issued_id: str, nonce: str) -> dict:
    try:
        import jwt
    except ImportError as exc:
        raise SystemExit("pip install 'PyJWT[crypto]' requests truststore") from exc
    id_token = token.get("id_token")
    if not id_token:
        raise SystemExit("Token response lacks an ID token; credentials not saved.")
    meta = http.get(ISSUER + "/.well-known/openid-configuration", timeout=15).json()
    jwks = http.get(meta["jwks_uri"], timeout=15).json()
    kid = jwt.get_unverified_header(id_token).get("kid")
    key = next(k for k in jwks["keys"] if k.get("kid") == kid)
    identity = jwt.decode(
        id_token, jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key)),
        algorithms=[key.get("alg", "RS256")], audience=issued_id, issuer=ISSUER,
        options={"require": ["sub", "iss", "aud", "exp", "iat", "nonce"]})
    if identity.get("nonce") != nonce:
        raise SystemExit("ID token nonce mismatch; credentials not saved.")
    return identity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", help="Previously issued client ID (reauthorization)")
    ap.add_argument("--port", type=int, default=1455)
    ap.add_argument("--wait-s", type=int, default=600)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    callback = f"http://127.0.0.1:{args.port}/auth/callback"
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    host_id = "urn:uuid:" + str(uuid.uuid4())
    verifier, _ = make_pkce()
    result: dict = {}
    server = _listen(args.port, state, args.client_id, result)
    url = authorization_url(redirect_uri=callback, state=state, nonce=nonce, verifier=verifier,
                            host_id=host_id, issued_client_id=args.client_id)
    print("Official ChatGPT authorization page; scopes: " + SCOPES, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    print(f"Waiting up to {args.wait_s}s for sign-in on 127.0.0.1:{args.port}.", flush=True)
    deadline = time.monotonic() + args.wait_s
    while not result and time.monotonic() < deadline:
        server.handle_request()
    server.server_close()
    try:
        code = validate_callback(result, state, args.client_id)
    except ValueError as exc:
        raise SystemExit(f"Sign-in denied, invalid or timed out ({exc}); nothing saved.") from exc
    issued_id = result.get("client_id") or args.client_id
    if not issued_id or (not args.client_id and not issued_id.startswith("oaiapp_")):
        raise SystemExit("Registration did not return an issued client ID.")
    http = http_session()
    token = _exchange(http, issued_id, code, verifier, callback)
    scopes = token.get("scope", "").split()
    if "chatgpt.tokens.use.direct" not in scopes or not token.get("access_token"):
        raise SystemExit("Plan usage not granted (no chatgpt.tokens.use.direct); not saved.")
    identity = _verify_identity(http, token, issued_id, nonce)
    record = {
        "issuer": ISSUER, "subject": identity["sub"], "client_id": issued_id,
        "ext_agent_host_id": host_id, "id_token": token["id_token"],
        "email": identity.get("email"),
        "expires_at": int(time.time()) + int(token.get("expires_in", 0)),
        "access_token": token["access_token"], "refresh_token": token.get("refresh_token"),
        "token_type": token.get("token_type", "Bearer"), "expires_in": token.get("expires_in"),
        "scopes": scopes, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    CRED_DIR.mkdir(parents=True, exist_ok=True)
    path = CRED_DIR / (issued_id + ".json")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if os.name != "nt":
        tmp.chmod(0o600)
    tmp.replace(path)
    print(f"Credential saved to {path}; scopes={scopes}; token values not printed.", flush=True)


if __name__ == "__main__":
    main()
