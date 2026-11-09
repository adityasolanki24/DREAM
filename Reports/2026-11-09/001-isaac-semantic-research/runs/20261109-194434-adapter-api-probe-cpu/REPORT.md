# Run report: Adapter API probe on CPU: OWLv2 and Grounding DINO with random weights, transformers 4.57.1, torch 2.8.0

- Run ID: 2026-11-09-001/20261109-194434-adapter-api-probe-cpu
- Parent task: 2026-11-09-001
- Started: 2026-11-09T19:44:34+11:00
- Ended: 2026-11-09T19:49:40+11:00
- Git commit: b1ed707c370a60c42faed81b03c143e35882e84d
- Branch: worktree-agent-ab8c50d8ef4f5ed08
- Config files and SHA-256: None (command-line arguments only; see Command)
- Seeds: torch.manual_seed(0)
- Model and checkpoint: random-initialised OWLv2 base and Grounding DINO tiny architectures (no pretrained weights)
- Simulator and version: none
- Renderer: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot embodiment: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot controller: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Hardware: local CPU, probe venv outside the repository
- Command: `python probe.py (scratch; builds models from pinned repos' config and tokenizer files, random weights, one captured frame)`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 3
- Cases completed: 3

## Purpose

API probe of the OWLv2 and Grounding DINO adapters on CPU with random weights and fake outputs, at the cloud lock versions (transformers 4.57.1, torch 2.8.0), to confirm box coordinate conventions before the GPU comparison.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
OWLv2 boxes are in pixels of the padded max(h,w) square: a full-square box becomes (0,0,320,320) and validation clips it to (0,0,320,240). Grounding DINO boxes are in input-image pixels; label mapping on real outputs is not verified (random weights emit whole-caption labels, reported as unknown, not guessed). OWLv2's post_process_object_detection is deprecated (removed in transformers v5; the pinned 4.57.1 is unaffected).

## Regressions observed

Not applicable (probe).

## Interruptions and failures

None. The probe ran from a scratch script outside the repository (since deleted), so its command is not reproducible from the repository.

## Evidence

No output retained: the probe ran from a scratch script that was deleted. The adapters it exercised are dream/perception/box_detectors.py (commit 5943bdd) and tests/test_box_detectors.py.
