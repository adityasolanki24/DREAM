# Run report: Test on fresh box and cylinder geometry, mass and table shifts

- Run ID: 2026-11-10-011/20261110-225450-geometry-shift
- Parent task: 2026-11-10-011
- Started: 2026-11-10T22:54:50+11:00
- Ended: 2026-11-10T22:57:40+11:00
- Git commit: c04ffd81e7a2adffeb84f65a47e64afee342467e (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-depth-geometry-shift-001/cube_shift_baseline.json (ccb65c65c376de45f005cdd5c7b444d16c0d5ddfbf9268d027e186e6b274730c); C:/Users/24adi/DREAM-r1-evidence/r1-depth-geometry-shift-001/cube_shift_expanded.json (1082ce2d451246ae8e5b5dd3d829492c51572b492b58c13708500d0c9c1c28dc); C:/Users/24adi/DREAM-r1-evidence/r1-depth-geometry-shift-001/cylinder_shift_baseline.json (cf1d0e81b76d9b55eb6a9a2c7830cfa01d5dc5c9c3ef3c496288c0fce78917ae); C:/Users/24adi/DREAM-r1-evidence/r1-depth-geometry-shift-001/cylinder_shift_expanded.json (5bb7a991198772e354049745d7ace316e8452a6d9d0e73dea286409a83a48239)
- Seeds: 103,107
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, serial
- Command: `python experiments/g1_grasp_bringup.py with each of the four recorded configs`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 4
- Cases completed: 4

## Purpose

Compare the two selectors on a new cube and cylinder, changing their dimensions, mass, position and table height from the earlier cases.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The previous selector passed one of two trials. The expanded selector passed both. The changed cylinder reached about 0.135 m peak lift with the expanded selector and held for three seconds.

## Regressions observed

The cube passed with both selectors. The shifted cylinder was recovered.

## Interruptions and failures

All four episodes completed. The previous cylinder attempt slipped despite exceeding the lift-height threshold.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-geometry-shift-001 (dir, sha256 777cf7aebb9328519041d7a243a5f6fc96a6f1c5dfd032ff750c8f544c846c6f)
