import json

import numpy as np

from dream.runtime.openai_plan import response_output_text, response_tool_schemas, responses_input
from dream.runtime.siwc_auth import authorization_url, make_pkce
from dream.runtime.volo_contract import inspect_volo_contract


def test_siwc_authorization_uses_pkce_dynamic_registration_and_required_scopes():
    verifier, challenge = make_pkce()
    url = authorization_url(redirect_uri="http://127.0.0.1:1455/auth/callback", state="s",
                            nonce="n", verifier=verifier, host_id="urn:uuid:test")
    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(url).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert set(query["scope"][0].split()) == {
        "openid", "profile", "email", "offline_access", "resource.invoke",
        "chatgpt.tokens.use.direct",
    }
    assert query["code_challenge"] == [challenge]
    assert query["ext_agent_host_id"] == ["urn:uuid:test"]


def test_responses_payload_converts_camera_and_tool_schemas():
    messages = [{"role": "system", "content": [{"type": "text", "text": "system"}]},
                {"role": "user", "content": [{"type": "image", "image": np.zeros((2, 3, 3),
                                                                           dtype=np.uint8)}]}]
    converted = responses_input(messages)
    assert converted[0]["role"] == "developer"
    assert converted[1]["content"][0]["type"] == "input_image"
    assert converted[1]["content"][0]["image_url"].startswith("data:image/png;base64,")
    tools = response_tool_schemas([{"type": "function", "function": {
        "name": "inspect", "description": "inspect", "parameters": {"type": "object"}}}])
    assert tools[0]["type"] == "function" and tools[0]["name"] == "inspect"


def test_responses_output_preserves_batches_for_shared_rejection():
    output = [{"type": "function_call", "name": "inspect", "arguments": "{}"},
              {"type": "function_call", "name": "finish", "arguments": json.dumps({"success": False})}]
    text = response_output_text({"status": "completed", "output": output})
    assert text.count("<tool_call>") == 2


def test_volo_adapter_only_accepts_passive_status_and_rejects_actuation():
    assert inspect_volo_contract({"status": "running", "completed": 1, "total": 4}).supported
    assert not inspect_volo_contract({"joint_targets": [0.0]}).supported
    assert not inspect_volo_contract({"status": "done", "arbitrary": 1}).supported
