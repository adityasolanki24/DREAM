# Run report: Compare geometry ranking on lateral

- Run ID: 2026-11-10-009/20261110-222621-remaining-lateral
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:26:21+11:00
- Ended: 2026-11-10T22:26:37+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/lateral.json (df23c2a61fe4328bb4dce38444190ee52077cdcb4e7b7619829a3feabb656577)
- Seeds: 23
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/lateral.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/lateral`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Measure transfer and possible regression on the remaining established cube fixtures.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"lateral","pose":{"candidates":15,"offset_m":[0.0,0.0,0.025],"rotation":[[1.0,-0.0,0.0],[0.0,0.9659258262890682,0.25881904510252074],[-0.0,-0.25881904510252074,0.9659258262890682]],"score":1.100729871410293,"support_links":2,"predicted_closure_fraction":0.4375},"stop":null,"phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,align_2,descend,preclose,close,probe,lift","peak":0.1364168488821902,"verified":true}

## Regressions observed

Compare with the prior controller in the task report.

## Interruptions and failures

Execution completed. See Results and the trace for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/lateral (dir, sha256 7087be54902665faaaf2651ea16827099ee7b8b943e9b1b63ca5243214942fc6)
