# Run report: Check force headroom and the hold gate on all walking fixtures

- Run ID: 2026-11-11-002/20261111-125117-force-headroom
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:51:17+11:00
- Ended: 2026-11-11T12:57:11+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b); experiments/configs/g1_grasp_bringup_cylinder.json (ecb6de06dd040e4a769537eaa933a0462ccfe70d962cf308c713a20fb345dcd8)
- Seeds: 81,83,89,97
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, serial
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-preload-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 4
- Cases completed: 4

## Purpose

Measure whether backing off preload near the hard contact limit recovers the rotated hold.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The rotated cube reached about 0.142 m without the earlier force stop and then slipped. Both original starts held. The cylinder also slipped.

## Regressions observed

The matched holds remained successful. The pressure change did not add a verified hold and was removed from the final controller.

## Interruptions and failures

All four cases completed. The rotated hold remained an open failure in this variant.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-preload-001 (dir, sha256 6b1f1175ec62cc6b3bedbcadaeaaf6193455a74aea1864c3167067c6175ac5ef)
