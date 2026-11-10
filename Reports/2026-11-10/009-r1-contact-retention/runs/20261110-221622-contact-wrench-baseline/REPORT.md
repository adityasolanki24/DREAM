# Run report: Attribute contact loss during the hold

- Run ID: 2026-11-10-009/20261110-221622-contact-wrench-baseline
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:16:22+11:00
- Ended: 2026-11-10T22:18:42+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b); experiments/configs/g1_grasp_bringup_shift_out.json (6726d0e381ba652a17b281f6538e558b18dd1a4b3adb57dbb9f8f4ce54260535); experiments/configs/g1_grasp_bringup_fresh_far.json (ef003711da9ddb4d4c7692ef750c102396e37dcaa59e4bcca61f7c8c8853f8be)
- Seeds: 0,17,31
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen R1 controller
- Hardware: Local CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-retention-contact-001/probe.py C:/Users/24adi/DREAM-r1-evidence/r1-retention-contact-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 3
- Cases completed: 3

## Purpose

Compare the contact distribution and box motion before failed holds with the nominal pass.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All outcomes reproduced. The nominal hold retained thumb and distal middle contact plus proximal middle support. Both failing holds lacked proximal support, and the thumb normal turned downward as the box rotated off the pad despite near-target load.

## Regressions observed

None measured from evaluator instrumentation. The original two hold failures remain.

## Interruptions and failures

The larger and farther boxes slipped at about 2.1 and 2.6 s into hold.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-retention-contact-001 (dir, sha256 4b56210755d6fe1cf535ab69d07b03519fdeb4a49d44295312c44f62809e73ef)
