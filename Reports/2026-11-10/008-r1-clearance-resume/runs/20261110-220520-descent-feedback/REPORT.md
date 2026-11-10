# Run report: Replan short descent steps after each observed pose

- Run ID: 2026-11-10-008/20261110-220520-descent-feedback
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:05:20+11:00
- Ended: 2026-11-10T22:06:12+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_out.json (6726d0e381ba652a17b281f6538e558b18dd1a4b3adb57dbb9f8f4ce54260535)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with local descent feedback
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-adaptive-descent-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether re-observed short descent steps can reach the larger shifted box.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Three executed corrections reduced the larger box alignment error to about 5 mm and enabled closure. The probe then lost grip before the required rise.

## Regressions observed

The former descent refusal cleared on the larger box, but the full lift gate still failed.

## Interruptions and failures

Grip loss during the probe.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-adaptive-descent-001 (dir, sha256 6d361b167bc36b05710bda0db254b71ab87bb5b58674bbc63f5f3e6370bbc4f0)
