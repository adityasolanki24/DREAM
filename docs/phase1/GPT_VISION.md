# ChatGPT (Sign in with ChatGPT) as DREAM's VLM: status and verdict

Status as of 2026-10-28: a small GPT interface pilot completed, reported below. This is not a
general vision evaluation or evidence of autonomous robot performance.

## 2026-10-28 interface pilot

On the same frozen 11 decisions, `gpt-6-luna` with explicit `reasoning.effort=none` produced
11/11 valid tool calls and 6/11 registered correct decisions: navigation 1/4, object
identification 2/4 and information gathering 3/3. Median latency was 3.918 s, maximum 5.477 s;
5,213 input and 213 output tokens were reported, with zero reasoning tokens and no errors.
At the published standard API token rates this is approximately $0.000628; actual ChatGPT plan
charges or quota consumption are unknown. The source commit was `a080cac`; source, decision and
image hashes, response items, usage and errors are in
`results/phase1_gpt_luna_none_20261028/run_frozen11.json`. The working tree contained the recorded
adapter and runner changes. Candidate proposals are ground-truth-assisted and positive object
candidates always use ID 0, so a shortcut can exploit the set. There was no physical actuation.

Earlier three Sol parser failures were pipeline failures and are excluded from model-quality
counts. One subsequent Sol canary produced a valid but incorrect navigation decision; its saved
stream items motivated the adapter fix. Results have not been overwritten.

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
- The documented refresh flow is at https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions.
  Numeric usage limits and the open-source ToS remain unclear in the cited documents.

Codex credentials, `backend-api` and scraping are not used.

## Setup

    pip install -e ".[openai-plan]"      # PyJWT[crypto], requests, truststore
    python experiments/openai_plan_login.py   # browser sign-in, 10 min wait
    python -m experiments.openai_plan_refresh <explicit-credential-file>

The credential is saved to `%LOCALAPPDATA%\DREAM\openai-chatgpt-plan\<client_id>.json` (outside the
repository, never printed). TLS verification is always on (`dream.runtime.siwc_auth.http_session`:
OS trust store via truststore, certifi fallback). Backend: `dream.runtime.openai_plan.ChatGPTPlanBackend`
(same `generate(messages, tools)` interface as the Qwen backends in `dream/runtime/qwen.py`, so Qwen
remains selectable). Comparison: `experiments/vlm_compare.py {gpt|qwen-http|qwen-local}`.

## Earlier results (2026-10-27)

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
the code was fixed to stop on any `subscription_sharing_*` error. At that time function-tool
handling had not been tested because the request never got that far. The subsequent pilot used
developer `additional_tools` and produced valid calls.

At the time of that earlier report, Qwen3-VL had not been run on this decision set; a
`vlm-compare` cloud workload was added in commit 874eb58 but not launched then. Other Qwen3-VL
evidence is in the README (G1 exploration: Qwen 0/4 vs deterministic 1/4 on development cases).

## Verdict

Not decided. Feasibility of the route is confirmed (official docs, working auth, vision-capable
catalog). Suitability as the primary high-level VLM is not established. Known caveats: the ChatGPT plan
quota is shared with the owner's own use; an earlier request encountered a quota denial and the
later pilot succeeded. Usage limits are opaque and the docs forbid inferring reset times; hosted
inference needs the internet (no offline robot use) and
Qwen3-VL stays the offline fallback; the open-source ToS for robot use was not found in the docs and
must be checked at openai.com/policies; refresh credentials need protected local storage. The
comparison is limited to this pilot.
