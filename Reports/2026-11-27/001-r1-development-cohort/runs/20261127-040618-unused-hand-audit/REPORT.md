# Run report: Reproduce the unused-hand table-contact exemption in both directions

- Run ID: 2026-11-27-001/20261127-040618-unused-hand-audit
- Parent task: 2026-11-27-001
- Started: 2026-11-27T04:06:18+11:00
- Ended: 2026-11-27T04:09:49+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: results/g1_wbc_envelope/envelope.json (d2b642f02ab8c6c2d84f7f03e425eaf6727e9f9fe2df55a24d52f7fe3228cdcf)
- Seeds: None, deterministic static geometry
- Model and checkpoint: G1 robot-only posture model
- Simulator and version: MuJoCo 3.16.0, static geometry only
- Renderer: None
- Robot embodiment: G1 posture planning model
- Robot controller: None, no physical execution
- Hardware: Local CPU, one geometry audit
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-unused-hand-audit-20261127/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Check whether the support-surface exception hides an unused-hand collision in both arm roles.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both static configurations have an unused-wrist overlap (0.028659 m and 0.030994 m penetration), while reported clearances remain positive (0.007812 m and 0.006861 m). The reported failure reason is outside_capabilities and omits collision. This establishes a collision-metric defect independently of physical execution.

## Regressions observed

No dynamic execution was attempted. The audit does not prove that fixing the metric alone will recover the failed physical approaches.

## Interruptions and failures

No process interruption. Negative results are preserved in the per-case records.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-unused-hand-audit-20261127 (dir, sha256 569881f6d891b7e837cd0cc3898b3b930da36309aab5c42fc532dd9cdc812289)
