# ChatGPT (Sign in with ChatGPT) as DREAM's VLM: status and verdict

Status as of 2026-10-27: authentication works and the model catalog was read; **no inference
succeeded** because the account's ChatGPT plan usage limit was already exhausted. The GPT and Qwen
comparison is therefore **not measured**. This document records what is established and what is not.

## Official method (docs fetched 2026-10-28)

- Overview: https://developers.openai.com/siwc/quickstart (eligible Plus/Pro users, ChatGPT plan
  usage open to all open-source partners; OIDC with PKCE, no client secret or API key).
- Registration/sign-in contract: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
  - authorize `https://auth.openai.com/api/accounts/authorize`, token
    `https://auth.openai.com/api/accounts/oauth/token`;
  - first sign-in uses `client_id=dynamic_agent_client` plus `agent_name_hint` and a persisted
    `ext_agent_host_id`; the callback returns an `oaiapp_*` client ID to store;
  - scopes `openid profile email offline_access resource.invoke chatgpt.tokens.use.direct`,
    `resource=https://api.openai.com/v1`; loopback redirect `http://127.0.0.1:<port>/auth/callback`;
  - the ID token must be verified (JWKS, iss, aud, exp, nonce).
- Inference: https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
  `POST https://api.openai.com/v1/responses`, bearer token, `stream: true`, `store: false`;
  models from `GET /v1/models` (`visibility == "list"`).
- Preview limits: .../preview-limitations (images accepted; no `system` role items; omit
  `max_output_tokens`, `temperature`, `metadata`; function tools must be in namespaces or
  `additional_tools`, format not documented there).
- Errors: .../errors-and-recovery (`subscription_sharing_usage_limit_exceeded`, HTTP 429, "pause new
  plan-based requests ... do not infer a reset time").
- Not documented: the refresh-token request, numeric usage limits, open-source ToS (the docs point to
  openai.com/policies). Access tokens last 1 hour, refresh tokens 30 days.

Codex credentials, `backend-api` and scraping are not used.

## Setup

    pip install -e ".[openai-plan]"      # PyJWT[crypto], requests, truststore
    python experiments/openai_plan_login.py   # browser sign-in, 10 min wait

The credential is saved to `%LOCALAPPDATA%\DREAM\openai-chatgpt-plan\<client_id>.json` (outside the
repository, never printed). TLS verification is always on (`dream.runtime.siwc_auth.http_session`:
OS trust store via truststore, certifi fallback). Backend: `dream.runtime.openai_plan.ChatGPTPlanBackend`
(same `generate(messages, tools)` interface as the Qwen backends in `dream/runtime/qwen.py`, so Qwen
remains selectable). Comparison: `experiments/vlm_compare.py {gpt|qwen-http|qwen-local}`.

## Results

Auth: succeeded (all six scopes granted, ID token verified). Models listed (`GET /v1/models`, HTTP 200,
all `text`+`image` input): gpt-6.1-sol, gpt-6-astra, gpt-6.1-terra, gpt-6-sol, gpt-6-luna,
gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna. No `x-request-id` or rate-limit headers were returned.

Pre-registered set: 11 decisions on local MuJoCo G1 head-camera renders (`results/phase1_vlm_compare/
decisions.json`, committed before inference): 4 navigation, 4 object identification, 3 information
gathering, run through DREAM's `MobileModelPlanner` and tool schemas.

| backend | valid | correct | p50 / max latency | tokens | errors |
|---|---|---|---|---|---|
| gpt-6.1-sol (plan) | 0/11 | 0/11 | n/a (4 s to rejection) | none consumed | 11x `subscription_sharing_usage_limit_exceeded` |
| Qwen3-VL (GCP GPU) | not run | not run | | | see below |

The usage error arrived as `response.failed` after HTTP 200, so the runner's "stop on HTTP error"
did not trigger and all 11 calls were attempted once each (one attempt per decision, no retries) before
the code was fixed to stop on any `subscription_sharing_*` error. Whether function-tool namespaces
are required was **not tested** (the request never got that far).

Qwen3-VL was not run: no session VM was live, and `cloud/launch.py` only supports its whitelisted
workloads, so a new workload for this set needs changes to safeguarded cloud code and a costed GPU
session (US$6 ceiling). Existing Qwen3-VL evidence is in the README (G1 exploration: Qwen 0/4 vs
deterministic 1/4 on development cases).

## Verdict

Not decided. Feasibility of the route is confirmed (official docs, working auth, vision-capable
catalog). Suitability as the primary high-level VLM is unmeasured. Known caveats: the ChatGPT plan
quota is shared with the owner's own use and was already exhausted; usage limits are opaque and the
docs forbid inferring reset times; hosted inference needs the internet (no offline robot use) and
Qwen3-VL stays the offline fallback; the open-source ToS for robot use was not found in the docs and
must be checked at openai.com/policies; tokens last 1 h and the documented refresh request is missing,
so unattended operation is not currently supportable. Rerun after the usage window resets.
