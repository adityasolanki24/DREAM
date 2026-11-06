"""Documented Sign in with ChatGPT OAuth helpers for local DREAM tooling."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import urllib.parse

SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
RESOURCE = "https://api.openai.com/v1"
ISSUER = "https://auth.openai.com"


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def make_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    return verifier, _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def authorization_url(
    *,
    redirect_uri: str,
    state: str,
    nonce: str,
    verifier: str,
    host_id: str,
    issued_client_id: str | None = None,
) -> str:
    query = {
        "client_id": issued_client_id or "dynamic_agent_client",
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": _b64url(hashlib.sha256(verifier.encode("ascii")).digest()),
    }
    if not issued_client_id:
        query.update(agent_name_hint="DREAM", ext_agent_host_id=host_id)
    return "https://auth.openai.com/api/accounts/authorize?" + urllib.parse.urlencode(query)


def validate_callback(
    query: dict[str, str], expected_state: str, issued_client_id: str | None = None
) -> str:
    """Validate OAuth callback state/error/client binding and return its one-time code."""
    if not hmac.compare_digest(query.get("state", ""), expected_state):
        raise ValueError("OAuth state mismatch")
    if query.get("error"):
        raise ValueError("OAuth authorization failed: " + query["error"])
    if issued_client_id and query.get("client_id") not in (None, issued_client_id):
        raise ValueError("OAuth callback client ID changed")
    if not query.get("code"):
        raise ValueError("OAuth callback has no authorization code")
    return query["code"]


def http_session():
    """requests.Session with TLS verification ON.

    Prefers the operating-system trust store (truststore) so corporate or antivirus TLS
    inspection roots are honoured; falls back to the certifi bundle that requests ships with.
    Never disables verification.
    """
    import requests

    session = requests.Session()
    try:
        import ssl

        import truststore
        from requests.adapters import HTTPAdapter

        class _Adapter(HTTPAdapter):
            def init_poolmanager(self, *a, **k):
                k["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                super().init_poolmanager(*a, **k)

        session.mount("https://", _Adapter())
    except ImportError:
        pass  # requests uses certifi by default
    return session
