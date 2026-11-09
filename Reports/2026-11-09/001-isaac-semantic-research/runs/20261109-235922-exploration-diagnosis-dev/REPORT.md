# Run report: Instrumented diagnosis of room knowledge in xl dev homes (sem_none, sem_conceptnet, sem_qwen; B and D; seeds 100-103)

- Run ID: 2026-11-09-001/20261109-235922-exploration-diagnosis-dev
- Parent task: 2026-11-09-001
- Started: 2026-11-09T23:59:22+11:00
- Ended: 2026-11-10T04:43:37+11:00
- Git commit: 6acf99745f05196fc6873971aa8dfbd8fe106a27 (dirty)
- Branch: worktree-agent-a88fa05480d69c248
- Config files and SHA-256: experiments/configs/semantic_exploration_dev.json (d0f745325bcba24bf7e487d48a00ab710f470da4735c7e6f9bd180123b68eb1a)
- Seeds: 100-103 (development)
- Model and checkpoint: None (no language model in these episodes; Qwen3-VL-4B text-only room priors are read from results/qwen_room_priors)
- Simulator and version: MuJoCo 3.x
- Renderer: Not applicable (kinematic base, segmentation oracle detector, no RGB rendering)
- Robot embodiment: kinematic surrogate base, oracle-assisted NOMINAL detector
- Robot controller: kinematic surrogate base
- Hardware: owner laptop, CPU, 3 workers
- Command: `python experiments/semantic_search_eval.py experiments/configs/semantic_exploration_dev.json --split dev_diag --diagnose --workers 3 --out results/semantic_exploration_dev/dev_diag (from detached worktree at 6acf997)`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 192
- Cases completed: 192

## Purpose

Explain, on development seeds only, why room knowledge did not speed up the utility planner in held-out K1: log each decision's terms and counterfactual choices, and ground-truth timelines of room labelling and target sighting.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
192/192 episodes (found: sem_conceptnet 60/64, sem_qwen 60/64, sem_none 59/64). Knowledge changed 6.7% (ConceptNet, 571 decisions) and 12.1% (Qwen, 561) of decisions versus the same decision with all room factors at 1; in 22/64 and 28/64 episodes at least once. Median time to first view of the target 85-86 s for all three arms; typical room first labelled at about 55-59 s but mapped at all in only 36-37 of 64 homes; the target was first seen before or when the typical room entered the map in 42 of 60 (Qwen). Target seen in the initial look in 8 homes. Exploration 91-93% of robot time, verification 7-9%. Median largest semantic shift 1.4 (CN) / 3.6 (Qwen) m^2 against a median gap of 2.1-2.3 m^2 between best and second candidate. Details: results/semantic_exploration_dev/dev_diag/diagnosis.json.

## Regressions observed

Not applicable: instrumentation only; a smoke episode with and without --diagnose gave identical simulated time (117.64 s).

## Interruptions and failures

None in the run (192/192, no crashes). The agent session that launched it hit a usage limit during the run; the detached background process was unaffected and did not need resuming.

## Evidence

- results/semantic_exploration_dev/dev_diag/diagnosis.json (file, sha256 f33a1d2de86c2070192456022177b897e28d69833f1cc1bd650880ae6b81e1e2)
