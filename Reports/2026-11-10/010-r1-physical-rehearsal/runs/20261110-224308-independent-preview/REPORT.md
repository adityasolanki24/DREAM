# Run report: Verify the whole-body preview with independent odometry noise

- Run ID: 2026-11-10-010/20261110-224308-independent-preview
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:43:08+11:00
- Ended: 2026-11-10T22:45:44+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: results/g1_r1_contact_rank_dev_20261110/summary.json (7ae701b1b4f3fd899c559cfca9586f25dde72bb2819d42cdb6975603b4964b70)
- Seeds: same nine live cases with a fixed independent planning stream
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, 2 workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-independent-preview-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 9
- Cases completed: 9

## Purpose

Rerun the full comparison with a deterministic planning stream independent of live evaluation noise.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The live run passed 5/6 cubes and 1/3 additional geometries. The outward cube and rectangular box were the recovered holds.

## Regressions observed

All four previously verified holds passed. Generalisation to a new distribution was not measured.

## Interruptions and failures

All cases finished. The inward cube, cylinder and sphere failed the independent live hold gate.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-independent-preview-001 (dir, sha256 7b290aa091a2826b7bc1b91755c7b259f20e038a1f326ed1e4c20f9e5b01ac2f)
