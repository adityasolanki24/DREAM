# Run report: Reproduce the front walking grasp whose saved output was corrupted

- Run ID: 2026-11-27-001/20261127-031601-front-recovery
- Parent task: 2026-11-27-001
- Started: 2026-11-27T03:16:01+11:00
- Ended: 2026-11-27T03:22:05+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-tool-walk-001/front.json (94b4aaf4da04670231f05f033412dfca79bfd96a0c87febc97880d6423c0e357)
- Seeds: 81 development replay
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Decoupled WBC with supervised posture and grasp
- Hardware: Local CPU, one worker
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-tool-walk-001/front.json --out C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-front-20261127`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recover verifiable evidence for the front walking grasp using the same saved configuration and the live humanoid toolbox.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The sensed tool and independent scorer both pass. Peak centre displacement is 0.1435134074 m. The actual simulated tool duration was 30 s.

## Regressions observed

The original front-case pass is reproduced. This establishes the matched integration check on a fresh output directory.

## Interruptions and failures

No interruption or execution failure occurred. The original corrupted files remain preserved.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-front-20261127 (dir, sha256 080fc52ccb5efffc74a584a005cd392fd43a05d10e3c129ed9727df8894e16b7)
