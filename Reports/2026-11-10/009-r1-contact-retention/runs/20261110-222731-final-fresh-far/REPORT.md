# Run report: Recheck fresh_far on the shape-generalized selector

- Run ID: 2026-11-10-009/20261110-222731-final-fresh-far
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:27:31+11:00
- Ended: 2026-11-10T22:27:43+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_far.json (326650453c1aa9f8abf96ba034eb795cae52a370c6509ec1dfcf986ddc1d964f)
- Seeds: 31
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_far.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_far`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Verify the final shared geometry selector on the matched cube fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"fresh_far","stop":null,"peak":0.13671912608233272,"verified":true}

## Regressions observed

Matched comparison recorded in the aggregate evidence.

## Interruptions and failures

Execution completed. The physical outcome is recorded in Results.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_far (dir, sha256 244f52b6b0c4d6a91f88d2ec836e32cf962354f0b3b7e2034ea48dfc1c43765f)
