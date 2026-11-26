# Run report: Replay four matched walking grasps through the humanoid tool

- Run ID: 2026-11-11-003/20261111-214921-walking-tool-replay
- Parent task: 2026-11-11-003
- Started: 2026-11-11T21:49:21+11:00
- Ended: 2026-11-27T03:14:18+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b); experiments/configs/g1_grasp_bringup_cylinder.json (ecb6de06dd040e4a769537eaa933a0462ccfe70d962cf308c713a20fb345dcd8)
- Seeds: 81, 83, 89, 97
- Model and checkpoint: Cached NVIDIA Decoupled WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: Camera mounted; headless rollout
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: NVIDIA Decoupled WBC, cached ONNX
- Hardware: Local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-tool-walk-001/run.py`
- Exit status: Not recorded
- Completed (yes/no/partial): partial (partial)
- Cases attempted: 4
- Cases completed: 3

## Purpose

Check the live humanoid tool on four matched walking grasp scenes with the independent hold evaluator.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Two intact cube results (offset and rotated) pass the sensed and independently scored hold. The intact cylinder result fails both and stops on grip loss. All three intact cases reached the stance with zero walking table-contact samples.

## Regressions observed

Original front-case evidence is corrupted and cannot establish its outcome. The two surviving previous passes remain passes.

## Interruptions and failures

The process handle no longer exists. Front summary and trace plus the aggregate contain only NUL bytes at resumption. The cause of the file corruption is unmeasured. Three completed cases remain verifiable. End time records administrative closeout.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-humanoid-tool-walk-001 (dir, sha256 86d729d8a6c7b2ce19709e8ee43e0182c90a759142467f586149308623c99ec1)
