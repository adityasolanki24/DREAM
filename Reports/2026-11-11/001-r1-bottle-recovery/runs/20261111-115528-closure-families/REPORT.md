# Run report: Compare proximal wrapping and distal hooking on the failed bottles

- Run ID: 2026-11-11-001/20261111-115528-closure-families
- Parent task: 2026-11-11-001
- Started: 2026-11-11T11:55:28+11:00
- Ended: 2026-11-11T11:59:14+11:00
- Git commit: d80aa8d5cd282ebb51e50407b9fcb074d2010a61 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle.json (e9e3e2e3a528fb665aa007f2b37d78488e6b5e2e64474d88dfc3ad0dbeb3c586); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle_shift.json (32a8ae767f81d5ee3490accff761aa49b010822be2f73fdf76093d8edd7b4ec5); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/cylinder.json (78807ab15e534dfddabbdcd5becd291d03a04ca300ac72f8e944ae5291e2abeb)
- Seeds: 53,61,41
- Model and checkpoint: Dex3 joint targets recorded per profile
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC, physical stage selection
- Hardware: local CPU, serial
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-bottle-closure-families-001/run.py --case CASE --profile PROFILE for the five recorded pairs`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 5
- Cases completed: 5

## Purpose

Compare proximal curling with distal hooking through geometric rehearsal and the same physical grasp controller.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The proximal-wrap family held the lighter bottle for 150 ticks with at least 0.123 m of complete-object table clearance. The heavier wrap and all three distal-hook controls failed.

## Regressions observed

Only proximal wrapping supplied a new live hold. Distal hooking was left as experimental evidence.

## Interruptions and failures

All five trials completed. These explicit-family executions motivated testing automatic family choice on fresh bottles.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-bottle-closure-families-001 (dir, sha256 511af1965effc886baae58712c4db1b5207fa47e5665edd08e7fc5ebc5ab60f9)
