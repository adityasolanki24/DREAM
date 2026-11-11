# Run report: Verify bottom clearance, sensed hold and actual compound-object contact friction

- Run ID: 2026-11-10-012/20261110-230836-clearance-and-material
- Parent task: 2026-11-10-012
- Started: 2026-11-10T23:08:36+11:00
- Ended: 2026-11-10T23:16:05+11:00
- Git commit: acc45456ad2c216390d156e2dc43ae01b72debab (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: results/g1_r1_depth_dev_20261110/summary.json (68b8425b9e245a420f88e6a23fe3a3aaa9a889603c98d5bb94431c9f586de829); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/bottle.json (e9e3e2e3a528fb665aa007f2b37d78488e6b5e2e64474d88dfc3ad0dbeb3c586); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/mug.json (10a40933002fa08e55aff43aa3fc856fb2c099ab37132e37a7d1192267acc5c2); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/bottle_shift.json (32a8ae767f81d5ee3490accff761aa49b010822be2f73fdf76093d8edd7b4ec5); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/mug_shift.json (ee32a6c44c80b82f320af345b4a47ea228dc128410abade6264cd1e9103bc650)
- Seeds: nine matched, two prior shifts and four compound configurations
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 15
- Cases completed: 15

## Purpose

Rerun the old fixtures and compound cases using explicit contact friction, complete geometric clearance and an independently recorded sensed hold.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
8/9 matched, 2/2 earlier shifted fixtures and 2/4 new compound fixtures passed both sensed and independent hold criteria. Both mugs passed. Both bottles and the sphere failed.

## Regressions observed

All ten earlier holds were retained. Sensed and independently verified outcomes agreed in all 15 cases.

## Interruptions and failures

The run completed. The sphere and bottle failures remain preserved in their physical traces.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001 (dir, sha256 ff011773aa494ccbcc8b27d486c4bdf652fe9a193480bd5d10a062705a8a171d)
