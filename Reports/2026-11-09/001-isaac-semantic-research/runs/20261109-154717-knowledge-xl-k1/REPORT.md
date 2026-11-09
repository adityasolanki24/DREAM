# Run report: Semantic knowledge xl: K1 held-out comparison (6 arms, 576 episodes)

- Run ID: 2026-11-09-001/20261109-154717-knowledge-xl-k1
- Parent task: 2026-11-09-001
- Started: 2026-11-09T15:47:17+11:00
- Ended: 2026-11-09T23:48:02+11:00
- Git commit: 5a8ffedd7159fd33a78e7009d4c65f8da92f3f83
- Branch: worktree-agent-a6670edb145684437
- Config files and SHA-256: experiments/configs/semantic_knowledge_xl.json (5b1903c4014f2f66e03b77f2c160964c64b283db7fb57f2b21b52c0d625ba4aa)
- Seeds: 9300-9303 (held-out, fresh)
- Model and checkpoint: None (no model runs in these episodes). The qwen and combined arms read the saved Qwen3-VL-4B text-only room priors (results/qwen_room_priors, produced earlier with a prompt that holds only the object name and the knowledge extract's place vocabulary).
- Simulator and version: MuJoCo 3.x
- Renderer: MuJoCo software renderer (segmentation and depth only); the base is the kinematic surrogate.
- Robot embodiment: kinematic surrogate base, oracle-assisted NOMINAL detector
- Robot controller: kinematic surrogate base
- Hardware: owner laptop, CPU, 3 workers
- Command: `python experiments/semantic_search_eval.py experiments/configs/semantic_knowledge_xl.json --split K1 --workers 3 --out results/semantic_knowledge_xl/K1  (from detached worktree at 5a8ffed)`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 576
- Cases completed: 576

## Purpose

Pre-registered held-out comparison K1: does better semantic knowledge make DREAM's object search faster than frontier exploration in larger (xl, 12-room) homes without costing correctness? Six arms (A_frontier_point, A_frontier, sem_none, sem_conceptnet, sem_qwen, sem_combined), xl generator, kinematic base, NOMINAL detector, scenarios B, D, E, 8 targets, fresh seeds 9300-9303, 576 episodes, 600 s budget. Config frozen at 1f06db2 (field recording that commit added in 5a8ffed); code ran from a detached worktree pinned at 5a8ffed. Caveat carried: the generator's furnishing tables and the evaluation share an author, so agreement with the tables can inflate a knowledge source; scenarios D and E are the check.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
576 of 576 episodes completed, no crashes, no scene failures, no duplicates, run in one pass (7.8 h wall) with no resume. K-G1 PASS: 0 false success claims in each of the five current-verifier arms (0/96 each, one-sided 95% upper bound 3.07%); the old-verifier arm A_frontier_point also made 0. K-G3 PASS: target-present successes (B and D, 64 pairs) were 61 (conceptnet), 61 (qwen), 62 (combined) against 59 for A_frontier, threshold 56. All six arms correctly refused 32 of 32 absent-target homes. K-G4 (descriptive): sem_qwen vs sem_conceptnet 61/61 found, McNemar p = 1.0, Wilcoxon p = 0.61; sem_combined vs sem_qwen 62/61, p = 1.0 and 0.85. Room-inference top-1 on regions with evidence: conceptnet 326/599 (laundry 0/57, storage 1/56), qwen 365/586 (laundry 29/53, storage 4/56), combined 360/592. Details in results/semantic_knowledge_xl/gates.json. K-G2 (the primary, confirmatory gate) FAILED for all three comparisons: against A_frontier on 64 B+D pairs, one-sided paired Wilcoxon (time censored at 600 s) gave p = 0.59 (conceptnet), 0.95 (qwen), 0.80 (combined); Holm-adjusted p = 1.0 for all. Median paired differences were 0.0 s and mean paired differences were positive (the semantic arms were slower on average by 8.3, 10.0 and 8.9 s). Better knowledge therefore did not make search faster than frontier exploration in xl homes. sem_none (information gain only, no knowledge) was equally fast and found as many (62/64), and each knowledge source against sem_none was indistinguishable (Wilcoxon p 0.26-0.55). The semantic arms found 2-3 more targets than A_frontier (only 2 to 4 homes where they found and A_frontier did not, 0 to 1 the reverse, McNemar not significant); the extra finds come from the utility planner's inspection points, not from the room priors, since sem_none gets them too. The Qwen priors did not help room-inference-driven search despite better laundry recognition (29/53 vs 0/57).

## Regressions observed

Against the pre-existing H1-H4 picture: no regression in correctness (still 0 false claims), but the hint of benefit in H2 does not reappear as speed. The utility arms walk further than the frontier arms (mean path 58 m vs 45 m). Caveats: oracle-assisted detector and odometry, kinematic base, one generator whose tables share an author with the evaluation (which would favour Qwen if anything, and it still did not help), 4 seeds per condition; scenario D shows no knowledge penalty (29/32 for conceptnet and qwen, 30/32 combined).

## Interruptions and failures

None in the run: it completed in one pass from the pinned worktree, with no resume. The author session of this analysis was interrupted twice by usage limits while the run kept going; this did not affect the run.

## Evidence

- results/semantic_knowledge_xl/gates.json (file, sha256 b7cde34f626848808f98a3a019eef0e3207cefb56efd949d03a06be9f632f69d)
