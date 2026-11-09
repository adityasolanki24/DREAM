# Run report: Semantic knowledge xl: development smoke, all six arms

- Run ID: 2026-11-09-001/20261109-144043-knowledge-xl-dev-smoke
- Parent task: 2026-11-09-001
- Started: 2026-11-09T14:40:43+11:00
- Ended: 2026-11-09T15:45:13+11:00
- Git commit: 6825395d493290adfdce46e2d6b5bebedf731d74 (dirty)
- Branch: worktree-agent-a6670edb145684437
- Config files and SHA-256: experiments/configs/semantic_knowledge_xl.json (0bbd5efb8f68ef835a40f0264b9651c4963a23f8c85c3e092c242845eec73d64)
- Seeds: 101 (development)
- Model and checkpoint: None (no learned model runs in these episodes). The qwen and combined arms read the saved Qwen3-VL-4B text-only room priors (results/qwen_room_priors, pinned model revision); no model is run.
- Simulator and version: MuJoCo 3.x
- Renderer: MuJoCo software renderer (segmentation and depth only); the base is the kinematic surrogate.
- Robot embodiment: kinematic surrogate base, oracle-assisted detector
- Robot controller: kinematic surrogate base
- Hardware: owner laptop, CPU, 3 workers
- Command: `python experiments/semantic_search_eval.py experiments/configs/semantic_knowledge_xl.json --split dev_smoke --workers 3 --out results/semantic_knowledge_xl/dev_smoke`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 72
- Cases completed: 72

## Purpose

Development smoke of the full six-arm K1 design on xl homes: confirm every arm runs without crashes, measure time per episode (to estimate the K1 wall time) and room-inference accuracy per knowledge source. Runner seed 101 only, scenarios B, D, E, targets refrigerator, stove, washing_machine, sofa (12 homes x 6 arms = 72 episodes). Nothing was tuned.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
72 of 72 episodes completed; 0 crashes, 0 scene failures, 0 false claims in any arm. All six arms found 8 of 8 present targets and correctly refused 4 of 4 absent-target homes. Median time to find was 102.9 s (sem_qwen, sem_combined), 106.5 s (sem_conceptnet, sem_none), 116.1 s (A_frontier), 111.4 s (A_frontier_point); mean path 40 m for the frontier arms and 51-56 m for the utility arms. Top-1 room-type accuracy (regions with evidence): sem_conceptnet 36/64, sem_qwen 35/64, sem_combined 36/62, sem_none 8/62 (meaningless: uniform knowledge). Wall time per episode: about 96 s for the frontier arms, 171-184 s for the utility arms (scenario E episodes run to the 600 s budget and take 6-8 min), so K1 (576 episodes) is estimated at about 8 hours with 3 workers. The gates script ran on this data as a dry run (no gate is meaningful at n = 8 pairs). Evidence: results/semantic_knowledge_xl/dev_smoke.

## Regressions observed

None observed. The default (conceptnet) knowledge path was separately replayed before and after the change on six dev-seed episodes (outside this run) and matched exactly apart from wall time and random memory ids.

## Interruptions and failures

None. Run completed without interruption. The sample is far too small to say anything about the knowledge sources; it is a pipeline check.

## Evidence

- results/semantic_knowledge_xl/dev_smoke/summary.json (file, sha256 99f66f2f4c79d88784bdfe22a62bd73360a239049cbb0f78f311585e55c69a67)
- results/semantic_knowledge_xl/dev_smoke/summary.json (file, sha256 99f66f2f4c79d88784bdfe22a62bd73360a239049cbb0f78f311585e55c69a67)
