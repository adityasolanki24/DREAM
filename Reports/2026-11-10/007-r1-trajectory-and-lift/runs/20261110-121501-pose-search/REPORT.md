# Run report: Evaluate geometry-derived alternate arm paths

- Run ID: 2026-11-10-007/20261110-121501-pose-search
- Parent task: 2026-11-10-007
- Started: 2026-11-10T12:15:01+11:00
- Ended: 2026-11-10T12:16:14+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with pose and trajectory selection
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-003`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Search other heights, retreat distances and wrist orientations using observed table and box dimensions.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All 47 planned or previewed options were refused. The live pose was held and no new reach occurred.

## Regressions observed

The approach stayed behind the live guard.

## Interruptions and failures

Changing the endpoint did not produce a transition around the table edge.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-003 (dir, sha256 f38bd1e61d4250bb3b4e052e425d41d0e787341003f1d62c22ba92f1d21c33d5)
