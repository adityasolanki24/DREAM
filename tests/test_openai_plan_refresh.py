import json

import pytest

from dream.runtime.siwc_auth import ISSUER, RESOURCE
from experiments.openai_plan_refresh import refresh


def test_refresh_rotates_and_preserves_identity(tmp_path):
    path = tmp_path / "credential.json"
    path.write_text(
        json.dumps(
            {
                "issuer": ISSUER,
                "client_id": "oaiapp_test",
                "subject": "s",
                "refresh_token": "old",
                "scopes": ["chatgpt.tokens.use.direct"],
            }
        )
    )

    class Response:
        ok = True

        def json(self):
            return {"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 60}

    class Http:
        def post(self, url, timeout, data):
            assert url == ISSUER + "/api/accounts/oauth/token"
            assert data == {
                "grant_type": "refresh_token",
                "client_id": "oaiapp_test",
                "refresh_token": "old",
                "resource": RESOURCE,
            }
            return Response()

    refresh(path, Http())
    result = json.loads(path.read_text())
    assert result["subject"] == "s"
    assert result["refresh_token"] == "new-refresh"
    assert result["expires_at"] > 0
    assert not path.with_suffix(".json.lock").exists()


def test_refresh_lock_and_failure_preserve_credential(tmp_path):
    path = tmp_path / "credential.json"
    original = json.dumps(
        {
            "issuer": ISSUER,
            "client_id": "oaiapp_test",
            "refresh_token": "old",
            "scopes": ["chatgpt.tokens.use.direct"],
        }
    )
    path.write_text(original)
    lock = path.with_suffix(".json.lock")
    lock.touch()
    with pytest.raises(FileExistsError):
        refresh(path)
    assert path.read_text() == original
    lock.unlink()

    class Http:
        def post(self, *args, **kwargs):
            return type("Response", (), {"ok": False, "status_code": 401})()

    with pytest.raises(RuntimeError, match="HTTP 401"):
        refresh(path, Http())
    assert path.read_text() == original
    assert not lock.exists()
