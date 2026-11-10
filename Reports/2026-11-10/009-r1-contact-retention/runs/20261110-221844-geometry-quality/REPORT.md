# Run report: Compare opposing pad coverage over candidate wrist poses

- Run ID: 2026-11-10-009/20261110-221844-geometry-quality
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:18:44+11:00
- Ended: 2026-11-10T22:20:05+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b); experiments/configs/g1_grasp_bringup_shift_out.json (6726d0e381ba652a17b281f6538e558b18dd1a4b3adb57dbb9f8f4ce54260535); experiments/configs/g1_grasp_bringup_fresh_far.json (ef003711da9ddb4d4c7692ef750c102396e37dcaa59e4bcca61f7c8c8853f8be)
- Seeds: Not applicable
- Model and checkpoint: Dex3 articulated geometry
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: G1 Dex3 planning model
- Robot controller: Static contact prediction only
- Hardware: Local CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-geometric-contact-001/probe.py C:/Users/24adi/DREAM-r1-evidence/r1-geometric-contact-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 3
- Cases completed: 3

## Purpose

Check whether the articulated geometry admits alternative pinches with opposed normals and a smaller gravity moment.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All three box fixtures had feasible alternatives with higher geometric scores. The largest candidate changes the height and lateral position. The fixed-width fixtures also have a viable wrist roll. No physical lift was executed in this planning-only check.

## Regressions observed

Not measured. This execution only evaluated candidate geometry.

## Interruptions and failures

None.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-geometric-contact-001 (dir, sha256 013d5b48e33f1384fa53ec0841250aeef656ac1d02fa95508ce35e23c354f1a2)
