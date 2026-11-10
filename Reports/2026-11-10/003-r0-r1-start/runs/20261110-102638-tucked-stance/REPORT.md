# Run report: Closer stance with arms tucked at reset

- Run ID: 2026-11-10-003/20261110-102638-tucked-stance
- Parent task: 2026-11-10-003
- Started: 2026-11-10T10:26:38+11:00
- Ended: 2026-11-10T10:27:17+11:00
- Git commit: efd6872f4c375f948244e04c5beb71eade626925 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_tucked.json (e1e74b7cc4aa69f0a8afaf771fcc1aa6f2a8a335163fb6bcfea7cd1d734b0e9a)
- Seeds: 0
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None; instrumented physics trace
- Robot embodiment: Free-standing Unitree G1 with Dex3
- Robot controller: NVIDIA WBC with monitored posture targets
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_tucked.json --out C:/Users/24adi/DREAM-r1-evidence/tucked-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether starting in the existing arms-down pose clears the box during settling and exposes reach feasibility. Preserve the stance, box, policy, seed and safety limits.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The G1 settled, obtained a valid plan and began executing the reach under WBC. The supervisor stopped it at 1.60 s for requested torque at the right wrist pitch (1.865 times the model limit). The evaluator logged hand contact with the right hip pitch link at the stop. No closure or lift occurred.

## Regressions observed

Not measured. The tuck is a diagnostic fixture condition, not a changed default. It cleared initial box contact but introduced hand-hip contact.

## Interruptions and failures

The torque stop ended the reach. Hand-hip contact at the stop is observed evidence, but a controlled test has not yet isolated its contribution to the torque request.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/tucked-001 (dir, sha256 97f968399c1e63cb33c6d38def2bed0d614d9a430d7fe6c63f75e5441b0e6022)
