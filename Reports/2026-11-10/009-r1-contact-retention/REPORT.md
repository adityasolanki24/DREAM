# DREAM Research / Development Report: Improve R1 grasp retention and transfer

## Task identification
- Task title: Improve R1 grasp retention and transfer
- Task ID: 2026-11-10-009
- Date and time: 2026-11-10T22:16:21+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572
- Final commit: eecbebea13ca001f457d8e37039a051176627bac
- Task status: partially completed

## A. Objective

Advance the full R1 goal without fixed motions. Isolate the box hold failure, test geometry-based grasp selection, and introduce shape variation through a shared observation and control path.

## B. Starting state

The fixed-stance baseline at d5cc887 verified three of six cube cases. One stopped before closing and two slipped during hold. The wider R1 distribution and stance selection were still pending.

## C. Work performed

### Implementation

Added a geometric pinch selector that evaluates the articulated hand sweep against the observed object and tabletop. It scores opposing contact directions, moment about the pinch axis, number of supporting links and closure reserve. The selected pose feeds the same approach, force limits, measured descent, and lift controller. The old pose selection remains available as the comparison and default for the existing runner. The fixture accepts rectangular boxes, cylinders and spheres using configured observed extents. Shape-specific code builds collision geometry only.

### Evaluation

Instrumented evaluator-only contact forces, normals, pad links and relative rotation on one passing and two failing holds. Ran a planning-only sweep, matched live-physics comparisons, a position-uncertainty screen and three additional shapes. Archived the unsuccessful uncertainty variant outside Git.

### Runs

See runs/ for every execution, configurations and evidence hashes. The selected final outcomes are consolidated under results/.

## D. What was achieved

The contact ranking recovered the farther cube hold. The smaller cube progressed from closure refusal to a lift above the height threshold but slipped before the time threshold. The original passing fixtures were retained. A rectangular box, cylinder and sphere reached the probe with the shared control path.

## E. What was not achieved

Curved and rectangular fixtures did not pass the full hold gate. The full R1 development distribution, stance selection and held-out evaluations are still pending. The experimental selector remains opt-in.

## F. Passed tests and experiments

Ruff lint and formatting passed. All 88 targeted G1 grasp, hand, posture, supervisor and isolation tests passed, including the former far-box slip case.

## G. Failed tests and experiments

Both failed baseline holds reproduced. The top-ranked pose still lost the larger cube. The pose-uncertainty screen did not improve the matched hold outcomes. The rectangle, cylinder and sphere all failed the independent hold gate.

## H. Improvements

The same geometric and tactile algorithm selected poses for new shape geometry. On the six matched cube cases it recovered a formerly slipping case without losing a previous pass.

## I. Regressions and dis-improvements

No loss of a previous cube pass in the matched contact-ranking comparison. The uncertainty screen lost the recovered far-box pass relative to nominal geometric ranking. The full repository suite was not run for this development checkpoint.

## J. Comparison against baseline

The passing baseline maintained a third pad contact on the proximal middle link. The failing grasps had mainly two distal contacts, and their thumb normal turned downward shortly before release despite substantial opposing force. Static ranking helped one drop, but its highest score did not predict stability on every object.

## K. Research integrity and assumptions

The selector uses predicted contacts in a separate kinematic state, never evaluator-only live contact identity. Object pose and fixture geometry are oracle-assisted. Score weights and angle bounds are declared heuristics. There are no case or seed conditions in control selection. These are development observations and do not close R1.

## L. Cloud resource usage

None. All runs used local CPU.

## M. Remaining work

Evaluate candidate retention under gravity in a planning copy using the observed geometry, then test whether a stability check improves over static ranking on fresh widths and shapes. Continue to stance choice and the full development and claim gates after retention improves.

## N. Final conclusion

Contact-based selection expands the verified cube envelope and exposes real lift failures on broader geometry. The original R1 goal remains active.
