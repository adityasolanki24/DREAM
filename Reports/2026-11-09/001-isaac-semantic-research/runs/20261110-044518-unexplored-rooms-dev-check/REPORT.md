# Run report: Dev check of unexplored-room evidence: with and without it, all knowledge sources (seeds 100-103)

- Run ID: 2026-11-09-001/20261110-044518-unexplored-rooms-dev-check
- Parent task: 2026-11-09-001
- Started: 2026-11-10T04:45:18+11:00
- Ended: 2026-11-10T09:44:13+11:00
- Git commit: 27d653514c7bddaa321034981c0efe76f4da3038 (dirty)
- Branch: worktree-agent-a88fa05480d69c248
- Config files and SHA-256: experiments/configs/semantic_exploration_dev.json (4a4913d3a1fd449a8e61ceedde4c336a249baa561bcfe33e5bf4950f6328c158)
- Seeds: 100-103 (development)
- Model and checkpoint: None (Qwen3-VL-4B text-only room priors read from results/qwen_room_priors)
- Simulator and version: MuJoCo 3.x
- Renderer: Not applicable (kinematic base, segmentation oracle detector)
- Robot embodiment: kinematic surrogate base, oracle-assisted NOMINAL detector
- Robot controller: kinematic surrogate base
- Hardware: owner laptop, CPU, 3 workers
- Command: `python experiments/semantic_search_eval.py experiments/configs/semantic_exploration_dev.json --split dev_check --diagnose --workers 3 --out results/semantic_exploration_dev/dev_check (detached worktree at 27d6535)`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 320
- Cases completed: 320

## Purpose

Development check, seeds 100-103 only, of BeliefConfig.unexplored_rooms (room evidence from objects seen across doorways for unexplored space) against the same arms without it. Arms without it for sem_none, sem_conceptnet and sem_qwen come from run 20261109-235922-exploration-diagnosis-dev on the same homes; sem_none with the option is identical by construction and was not run.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Found of 64: A_frontier 57, sem_none 59, sem_conceptnet 60, sem_qwen 60, sem_combined 55; with the option: conceptnet 55, qwen 55, combined 60. Median paired time difference 0.0 s for every comparison with and without the option; mean +26.9 (conceptnet), +32.5 (qwen), -14.2 (combined) s, one-sided Wilcoxon p 0.97, 0.90, 0.08. The option fired in about 70% of decisions (unplaced objects existed) but changed the chosen action versus uniform factors in only 6.4% (conceptnet) to 11.9% (combined), the same rate as without it. Outcome differences are mostly misses landing at the 600 s budget edge. Not helpful on dev. Also (dev only): all utility arms including sem_none were faster than A_frontier on these 64 homes (median 0, one-sided p 0.004-0.095), unlike K1. Details: results/semantic_exploration_dev/dev_compare.json.

## Regressions observed

Conceptnet and qwen found 5 fewer of 64 homes with the option (all misses ended near the 600 s budget); combined found 5 more. No false success claims were examined beyond the standard scoring (result values 'false_success' absent in these arms).

## Interruptions and failures

None (320/320). The agent session hit a usage limit while it ran; the detached background process was unaffected.

## Evidence

- results/semantic_exploration_dev/dev_compare.json (file, sha256 846916c9db6a6fec7d4093c3fd8074f7f53d395df0449fe8f0a661b7f03bad88)
