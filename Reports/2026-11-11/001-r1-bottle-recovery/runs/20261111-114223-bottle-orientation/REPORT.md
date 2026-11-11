# Run report: Test orientation-diverse grasp retention and observed contact loss

- Run ID: 2026-11-11-001/20261111-114223-bottle-orientation
- Parent task: 2026-11-11-001
- Started: 2026-11-11T11:42:23+11:00
- Ended: 2026-11-11T11:47:22+11:00
- Git commit: d80aa8d5cd282ebb51e50407b9fcb074d2010a61 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle.json (e9e3e2e3a528fb665aa007f2b37d78488e6b5e2e64474d88dfc3ad0dbeb3c586); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle_shift.json (32a8ae767f81d5ee3490accff761aa49b010822be2f73fdf76093d8edd7b4ec5)
- Seeds: 53,61 with an independent planning stream
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-bottle-orientation-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Check more wrist orientations after the first grasp predictions failed.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Neither bottle held. A different pose held the heavier bottle in nominal prediction but the actual heavier, lower-friction object still slipped.

## Regressions observed

No recovered live hold was demonstrated. This expanded orientation search was removed from the final selector.

## Interruptions and failures

Both runs completed. The heavier bottle reached about 0.149 m peak lift and then lost its grasp.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-bottle-orientation-001 (dir, sha256 863fbe33737b2f89e28f2de0c4279f556dc1908b4b1e09e7eb0c137b82245f27)
