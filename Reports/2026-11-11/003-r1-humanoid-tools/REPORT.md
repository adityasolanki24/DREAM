# DREAM Research / Development Report: Expose the R1 grasp through the live humanoid toolbox

## Task identification
- Task title: Expose the R1 grasp through the live humanoid toolbox
- Task ID: 2026-11-11-003
- Date and time: 2026-11-11T21:43:17+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53
- Final commit: fc3fa22
- Task status: partially completed

## A. Objective

Expose the existing continuous walk, grasp, and sensed hold on one live humanoid capability interface and typed toolbox.

## B. Starting state

At cf2851d, walking cube grasps existed as a standalone diagnostic. The adopted R1 protocol and toolbox were missing. A cylinder still slipped after walking.

## C. Work performed

### Implementation
Added the Humanoid protocol, G1Humanoid over one live environment and WBC supervisor, and HumanoidToolbox. The R1 diagnostic dispatches grasp_tabletop through the tool and reports sensed hold separately from independent physics scoring. Added head camera capture and bounded cancellation in live and rehearsal execution. Prepared a stratified development runner.

### Evaluation
Two intact matched cube results pass sensed and independently scored holds. The intact cylinder fails both on grip loss. The front-case and aggregate outputs are corrupted. A separate camera capture experiment passed.

### Runs
Run records: see runs/ and the daily index
The walking-tool-replay run is partial because one case cannot be verified. The head-camera run completed. Both run records contain source revision, evidence hashes, and limitations.

## D. What was achieved

A real humanoid tool executes continuous walking and grasping. Two intact walking cube passes are retained. Head RGB-D capture is verified on the same physics owner before and after walking.

## E. What was not achieved

The 50-episode development and complete held-out R1 gates remain unmeasured. Broad grasp reliability is unresolved. The original front-case replay is unverifiable.

## F. Passed tests and experiments

Focused tool outcome, isolation, cancellation, arm and hand, and cohort generation checks passed. Camera capture produced 240 by 320 RGB and depth, finite depth fraction 0.6845182292, and camera translation 0.1156231429 m over 1.2 s.

## G. Failed tests and experiments

The cylinder failed its hold and stopped on grip loss. Front summary and trace and the campaign aggregate contain only NUL bytes. Those files are preserved and the run is partial.

## H. Improvements

The real grasp has a typed tool entry point and shared safety state. Sensed tool success is independent of evaluator contact and clearance labels.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
The two intact previous cube passes are retained and the cylinder still fails. The front replay is unverifiable. Complete fast and full suites on the final tree were not measured within this task.

## J. Comparison against baseline

Offset and rotated cubes pass before and after integration. The cylinder fails before and after integration. A fresh front replay is needed.

## K. Research integrity and assumptions

Classified as general algorithmic engineering integration. Object and table geometry and noisy odometry remain oracle-assisted. Camera capture is measured, while grasp decisions use the labelled tracker. World geometry, material ranges, and the 10 cm and 3 s acceptance gate were preserved. The current diagnostic supports one target per episode.

## L. Cloud resource usage

No cloud resources or paid compute. The local camera run used native OpenGL. Other work used local CPU.

## M. Remaining work

Recover the front replay, execute the 50-case development campaign, diagnose physical failures, pass 35 of 50 before freezing, and run the complete R1 held-out gate.

## N. Final conclusion

The live humanoid tool path is implemented with partial physical evidence. Overall R1 remains incomplete.
