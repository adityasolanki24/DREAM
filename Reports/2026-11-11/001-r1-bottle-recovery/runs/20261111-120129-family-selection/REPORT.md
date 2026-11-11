# Run report: Validate automatic pinch or wrap choice and transfer to new bottles

- Run ID: 2026-11-11-001/20261111-120129-family-selection
- Parent task: 2026-11-11-001
- Started: 2026-11-11T12:01:29+11:00
- Ended: 2026-11-11T12:15:41+11:00
- Git commit: d80aa8d5cd282ebb51e50407b9fcb074d2010a61 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: results/g1_r1_depth_dev_20261110/summary.json (68b8425b9e245a420f88e6a23fe3a3aaa9a889603c98d5bb94431c9f586de829); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle.json (e9e3e2e3a528fb665aa007f2b37d78488e6b5e2e64474d88dfc3ad0dbeb3c586); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle_shift.json (32a8ae767f81d5ee3490accff761aa49b010822be2f73fdf76093d8edd7b4ec5); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/mug.json (10a40933002fa08e55aff43aa3fc856fb2c099ab37132e37a7d1192267acc5c2); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/mug_shift.json (ee32a6c44c80b82f320af345b4a47ea228dc128410abade6264cd1e9103bc650)
- Seeds: same fifteen live cases plus fresh bottle seeds 71 and 73
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC, candidate-selected closure family
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-family-selection-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 17
- Cases completed: 17

## Purpose

Verify automatic closure choice on the complete matched set and measure transfer to two bottles with new dimensions, mass, friction and table pose.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
13/15 matched episodes passed, compared with 12/15 before the change. The new 0.12 kg bottle held with the second wrap proposal and had at least 0.113 m of clearance throughout the three-second hold. The 0.22 kg bottle failed. Sensed and independently verified outcomes agreed in all 17 cases.

## Regressions observed

All twelve previously successful cases still passed. The additional predictions increase planning work when the pinch proposals fail.

## Interruptions and failures

All seventeen episodes completed. Both heavier bottles and the sphere remain failed physical executions.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-family-selection-001 (dir, sha256 dc3667addf6ebce0516b2dcea434cf4844f169813759ec72f39569aafa1bbe4f)
