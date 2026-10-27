import pytest

from dream.runtime.siwc_auth import http_session, validate_callback


def test_callback_validation_rejects_bad_state_error_and_client_swap():
    ok = {"state": "s", "code": "c", "client_id": "oaiapp_x"}
    assert validate_callback(ok, "s", None) == "c"
    with pytest.raises(ValueError):
        validate_callback({**ok, "state": "t"}, "s")
    with pytest.raises(ValueError):
        validate_callback({"state": "s", "error": "access_denied"}, "s")
    with pytest.raises(ValueError):
        validate_callback(ok, "s", "oaiapp_other")


def test_http_session_keeps_tls_verification_on():
    assert http_session().verify is True
