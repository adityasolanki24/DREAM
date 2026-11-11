# Run report: Verify the corrected stance ranking under a scene rotation

- Run ID: 2026-11-11-002/20261111-124518-frame-invariant-stance
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:45:18+11:00
- Ended: 2026-11-11T12:49:24+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b); experiments/configs/g1_grasp_bringup_cylinder.json (ecb6de06dd040e4a769537eaa933a0462ccfe70d962cf308c713a20fb345dcd8)
- Seeds: 81,83,89,97
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-frame-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 4
- Cases completed: 4

## Purpose

Check a ranking rule that treats submillimetre shoulder differences as equivalent and favours approach alignment with the table face.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The two original holds passed. The rotated stance was the rigid transform of the original goal and the cube lifted about 0.115 m. The 35 N guard stopped the lift after a wrist load spike. The cylinder slipped.

## Regressions observed

The matched holds were retained. The corrected ranking recovered rotated approach and lift, with the hold gate still unmet.

## Interruptions and failures

All four cases completed. The rotated episode stopped on its observed contact load.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-frame-001 (dir, sha256 d5b1218424bdfe7b46e450e60a803c21f620c4e8378dbbe4a686c3be63b80d45)
