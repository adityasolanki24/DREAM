# Run report: Nested-object replay, l_shape development frames, span-inflated surrogate boxes (mechanism test)

- Run ID: 2026-11-09-001/20261109-192657-nested-replay-span
- Parent task: 2026-11-09-001
- Started: 2026-11-09T19:26:57+11:00
- Ended: 2026-11-09T19:27:33+11:00
- Git commit: 87d746ec84057d8675b060ea6b718907dc44cef1 (dirty)
- Branch: worktree-agent-ab8c50d8ef4f5ed08
- Config files and SHA-256: None (command-line arguments only; see Command)
- Seeds: none (deterministic replay of 39 captured frames)
- Model and checkpoint: none (truth-derived surrogate boxes, countertop box encloses the sink)
- Simulator and version: none (offline replay of Isaac Sim 6.2 capture)
- Renderer: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot embodiment: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot controller: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Hardware: local CPU
- Command: `python experiments/nested_replay.py --raw <capture item> --source span --out results/nested_association_dev/span.json`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 10
- Cases completed: 10

## Purpose

Mechanism test with a surrogate for the failure seen in Isaac: the countertop box is widened to enclose the sink (as Qwen's spanning boxes did), labels otherwise from truth; same five variants and poses as the control run.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
True pose. Option off: sink 0.70 (verified), countertop 0.50, cabinet 0.50, 15 tracks. As shipped: sink 0.88, countertop 0.78, cabinet 0.40, 19 tracks. The surrogate does not reproduce the 0.39 failure (the sink already verifies with the option off), so this shows the exclusion mechanism working, not a repair of the observed failure.

## Regressions observed

Cabinet purity 0.50 -> 0.40; tracks 15 -> 19.

## Interruptions and failures

None.

## Evidence

- results/nested_association_dev/span.json (file, sha256 6c1aca1258d05ced1a7456d122e0d3967580a31777a6c3162615d5a206e80152)
