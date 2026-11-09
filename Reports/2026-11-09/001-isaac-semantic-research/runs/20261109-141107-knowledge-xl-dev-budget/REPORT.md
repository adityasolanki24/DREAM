# Run report: Semantic knowledge xl: development budget check

- Run ID: 2026-11-09-001/20261109-141107-knowledge-xl-dev-budget
- Parent task: 2026-11-09-001
- Started: 2026-11-09T14:11:07+11:00
- Ended: 2026-11-09T14:40:43+11:00
- Git commit: 6825395d493290adfdce46e2d6b5bebedf731d74
- Branch: worktree-agent-a6670edb145684437
- Config files and SHA-256: experiments/configs/semantic_knowledge_xl.json (0bbd5efb8f68ef835a40f0264b9651c4963a23f8c85c3e092c242845eec73d64)
- Seeds: 100-103 (development)
- Model and checkpoint: None (no learned model runs in these episodes; the oracle-assisted detector is a rendered-segmentation noise model).
- Simulator and version: MuJoCo 3.x
- Renderer: MuJoCo software renderer (segmentation and depth only); the base is the kinematic surrogate.
- Robot embodiment: kinematic surrogate base, oracle-assisted detector
- Robot controller: kinematic surrogate base
- Hardware: owner laptop, CPU, 3 workers
- Command: `python experiments/semantic_search_eval.py experiments/configs/semantic_knowledge_xl.json --split dev_budget --workers 3 --out results/semantic_knowledge_xl/dev_budget`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 64
- Cases completed: 64

## Purpose

Apply the pre-declared time-budget rule on development data only: keep 600 s unless A_frontier finds fewer than 80% of present targets on dev xl homes within 600 s, in which case set 900 s. Arm: A_frontier only (gain frontier, current verifier). 64 target-present xl homes: scenarios B and D, 8 targets, runner seeds 100-103.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
A_frontier found 58 of 64 target-present dev xl homes (90.6%), above the 80% threshold, so the budget stays at 600 s. Of the 6 misses, 5 were honest stops after the explorer considered the reachable space covered (3 'target_not_yet_found' with 11-25 m^2 of unseeable area left, 2 'possible_target_unconfirmed'), and only 1 (sofa, seed 103) ran into the 600 s budget. Median sim time to verified find 109.75 s; mean wall time 78.5 s per episode. No crashes, no scene failures, no false claims. Evidence: results/semantic_knowledge_xl/dev_budget.

## Regressions observed

None. This is a first run on the new xl generator; there is no earlier xl result to regress against.

## Interruptions and failures

None. Run completed without interruption. Note: the dev data are development evidence, are not part of K1, and nothing but the budget rule was decided from them.

## Evidence

- results/semantic_knowledge_xl/dev_budget/summary.json (file, sha256 4d637b763241e758a0ae53c9c81756f5a1153709d9ee05ce6b579565edc28539)
