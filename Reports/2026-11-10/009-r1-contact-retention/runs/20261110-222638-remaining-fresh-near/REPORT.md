# Run report: Compare geometry ranking on fresh_near

- Run ID: 2026-11-10-009/20261110-222638-remaining-fresh-near
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:26:38+11:00
- Ended: 2026-11-10T22:26:54+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_near.json (d42e7b5ca12ea52327baaecd3c3e7bc92b5b5aa33863d88d3a6b22748105c4c8)
- Seeds: 29
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_near.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_near`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Measure transfer and possible regression on the remaining established cube fixtures.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"fresh_near","pose":{"candidates":15,"offset_m":[0.0,0.0,0.025],"rotation":[[1.0,-0.0,0.0],[0.0,0.9659258262890682,0.25881904510252074],[-0.0,-0.25881904510252074,0.9659258262890682]],"score":1.1007298714103,"support_links":2,"predicted_closure_fraction":0.4375},"stop":null,"phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,descend,preclose,close,probe,lift","peak":0.13569821693369466,"verified":true}

## Regressions observed

Compare with the prior controller in the task report.

## Interruptions and failures

Execution completed. See Results and the trace for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/fresh_near (dir, sha256 293d00979aa5dbe6714d9d55da42fa00c8c739ad6ad78104341a04e32f036b7e)
