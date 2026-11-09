# Run report: Nested-object replay, l_shape development frames, Qwen answers recorded in the capture (2 frames)

- Run ID: 2026-11-09-001/20261109-192733-nested-replay-qwen-recorded
- Parent task: 2026-11-09-001
- Started: 2026-11-09T19:27:33+11:00
- Ended: 2026-11-09T19:27:42+11:00
- Git commit: 87d746ec84057d8675b060ea6b718907dc44cef1 (dirty)
- Branch: worktree-agent-ab8c50d8ef4f5ed08
- Config files and SHA-256: None (command-line arguments only; see Command)
- Seeds: none (deterministic replay)
- Model and checkpoint: Qwen3-VL-4B-Instruct ebb281ec (answers recorded in the capture job, not re-run)
- Simulator and version: none (offline replay of Isaac Sim 6.2 capture)
- Renderer: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot embodiment: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot controller: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Hardware: local CPU
- Command: `python experiments/nested_replay.py --raw <capture item> --source qwen_recorded --out results/nested_association_dev/qwen_recorded.json`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 10
- Cases completed: 10

## Purpose

Replay of the Qwen3-VL-4B answers stored in the job 4 capture through the same variants.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Uninformative: only 2 of 39 frames carry a Qwen answer (frame 22: a cabinet box spanning the image bottom and a sink box; frame 38: empty). Every variant yields 2 single-view tracks; nothing can be concluded.

## Regressions observed

Not measured (no informative data).

## Interruptions and failures

None in execution; the input data were insufficient.

## Evidence

- results/nested_association_dev/qwen_recorded.json (file, sha256 dcbe239848a391f9d6cd74f6b9cca82d09f29f3080f49653ee84ad979d920c3d)
