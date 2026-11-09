# Run report: Nested-object replay, l_shape development frames, truth-derived tight boxes (control)

- Run ID: 2026-11-09-001/20261109-192629-nested-replay-tight
- Parent task: 2026-11-09-001
- Started: 2026-11-09T19:26:29+11:00
- Ended: 2026-11-09T19:27:04+11:00
- Git commit: 87d746ec84057d8675b060ea6b718907dc44cef1
- Branch: worktree-agent-ab8c50d8ef4f5ed08
- Config files and SHA-256: experiments/configs/detector_compare.json (7d68d4d912abbaf8a2297d5daca467c3ff14ee5ac08a09fc052bdef6dec3774e)
- Seeds: none (deterministic replay of 39 captured frames)
- Model and checkpoint: none (truth-derived boxes)
- Simulator and version: none (offline replay of Isaac Sim 6.2 capture)
- Renderer: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot embodiment: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot controller: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Hardware: local CPU
- Command: `python experiments/nested_replay.py --raw <capture item> --source tight --out results/nested_association_dev/tight.json`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 10
- Cases completed: 10

## Purpose

Control for the nested-object association option. The l_shape development frames (39, job 4 capture) are replayed through SemanticMap with perfect class labels and tight boxes taken from the segmentation truth (sink rim, basin and sink merged), so any mixing left in the sink track comes from back-projection and association, not from the detector. Five variants: off, as shipped (containment exclusion plus size/height gate), exclusion only, gate only, smaller-detection-owns-shared-pixels; true poses and odometry-frame poses.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Verifier-view label purity, true pose. Option off: sink 0.24, countertop 0.31, cabinet 0.45, 16 tracks. As shipped: sink 0.26, countertop 0.75 (verified), cabinet 0.41, 24 tracks. In every variant and in the odometry frame the sink track stays mixed (0.21-0.28): the option does NOT fix the sink failure on these frames. Cause (from the replay): the sink's own tight box is largely countertop surface, and adjacent countertop and cabinet instances lie within the tracker's 0.35 m association distance. Development evidence only; the gate rule was changed from footprint area to longest side after the first replay split the refrigerator into 8 tracks.

## Regressions observed

Fragmentation with the option on: tracks 16 -> 24, cabinet support 22 -> 17, refrigerator support 7 -> 4 (still verified).

## Interruptions and failures

None in execution. Record flaws: the first run-finish recorded wrong case counts and a stale note metric; a second call corrected the counts, but the stale note and a duplicated evidence line remain in metadata.json. experiments/configs/detector_compare.json is listed as a config although it is not an input to this replay.

## Evidence

- results/nested_association_dev/tight.json (file, sha256 9af323d2b702973bc3c67c1a030cb7122cb6fdca75842201dcabfeeae052ae6d)
- results/nested_association_dev/tight.json (file, sha256 9af323d2b702973bc3c67c1a030cb7122cb6fdca75842201dcabfeeae052ae6d)
