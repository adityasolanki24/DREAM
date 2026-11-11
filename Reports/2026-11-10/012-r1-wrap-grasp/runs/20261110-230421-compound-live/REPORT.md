# Run report: Run the physical grasp controller on bottle and open mug geometry

- Run ID: 2026-11-10-012/20261110-230421-compound-live
- Parent task: 2026-11-10-012
- Started: 2026-11-10T23:04:21+11:00
- Ended: 2026-11-10T23:06:34+11:00
- Git commit: acc45456ad2c216390d156e2dc43ae01b72debab (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/bottle.json (e9e3e2e3a528fb665aa007f2b37d78488e6b5e2e64474d88dfc3ad0dbeb3c586); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/mug.json (10a40933002fa08e55aff43aa3fc856fb2c099ab37132e37a7d1192267acc5c2); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/bottle_shift.json (32a8ae767f81d5ee3490accff761aa49b010822be2f73fdf76093d8edd7b4ec5); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/mug_shift.json (ee32a6c44c80b82f320af345b4a47ea228dc128410abade6264cd1e9103bc650)
- Seeds: 53,59,61,67
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 4
- Cases completed: 4

## Purpose

Test whether the existing geometry-driven controller can handle rigid compound shapes with a real bottle neck and open mug cavity and handle.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
One of the four pilot episodes held the mug. Both bottles and the heavier mug failed. The lighter bottle reached 0.14 m and then slipped.

## Regressions observed

Previous shapes were not rerun in this pilot.

## Interruptions and failures

The first pilot used equal contact priority. A regression test showed that the configured lower friction was masked by the table or hand coefficient. The corrected replay supersedes this pilot for material-shift claims.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001 (dir, sha256 14b7fa527eac5e16a32b997d9bc73fc3bd3fb1edde6ed78f6697bd73fbcbd49a)
