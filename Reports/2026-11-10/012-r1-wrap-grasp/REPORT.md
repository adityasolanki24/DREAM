# DREAM Research / Development Report: R1 enveloping grasp contact search

## Task identification
- Task title: R1 enveloping grasp contact search
- Task ID: 2026-11-10-012
- Date and time: 2026-11-10T23:00:45+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: acc45456ad2c216390d156e2dc43ae01b72debab
- Final commit: b7ac5dcff658b00bc677f383a589ea13f5822e58
- Task status: partially completed

## A. Objective

Identify whether a more enveloping closure can retain a round object and expand the physical R1 fixtures to the specified bottle and mug families. Verify sensing and the actual clearance gate independently.

## B. Starting state

The previous selector held 8 of 9 matched fixtures and both additional geometry checks. The sphere lost opposition during its first probe. The test scene represented every object with one collision primitive.

## C. Work performed

### Implementation
Add rigid compound bottle and open mug collision fixtures, including a narrower bottle neck and a hollow mug cavity and handle. Use every object collider during the same grasp search and clearance checks. Use lower body-relative grasp targets after the normal rehearsal candidates fail. Explicit object contact friction takes effect in live execution, while the prediction retains its own fixed material prior. Score the lowest collision point above the table and separately report the robot's sensed hold from encoder travel, tactile opposition and oracle-assisted observation.

### Evaluation
Screen three articulation profiles under gravity on the sphere and cylinder. Pilot four new bottle and mug fixtures, identify friction masking with a contact regression, then rerun the old matched cases, earlier geometry shifts and new compound cases under the stricter gate.

### Runs
Run records: see runs/ and the daily index
See runs/ for stage-only, compound pilot and corrected physical replays. Source snapshots, configurations and full traces are preserved in the referenced evidence directories.

## D. What was achieved

All eight previously matched holds and both previous geometry shifts passed the independent lowest-surface gate. The sensed outcome agreed with evaluation on all 15 replay episodes. Two open mugs held under the configured contact friction. The compound fixture generator retains mass, one rigid object and the neck or cavity and handle surfaces.

## E. What was not achieved

The sphere and both compound bottle cases failed. A reusable power-grasp selector, walking into a selected stance, the complete 50-case development distribution and the 60-case held-out gate remain pending.

## F. Passed tests and experiments

All 107 focused grasp, hand, posture, supervisor, fixture and observation-isolation tests passed. Repository-wide Ruff lint and formatting checks passed. The contact-friction regression and tilted-handle clearance regression passed. Complete slow-suite acceptance remains pending for the R1 milestone.

## G. Failed tests and experiments

The initial material regression reproduced configured friction 0.7 turning into contact friction 1.0. Assigning the object's explicit material higher priority corrected that defect. The wrapping stage produced no sphere holds. Both bottle replays failed despite reaching or approaching the lift height.

## H. Improvements

R1 can now evaluate compound rigid objects through the same geometry-based planner. The independent evaluator accounts for an object's lowest surface and includes tilt and handles. The robot's sensed success is recorded separately from physical scoring.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
All ten previously verified live outcomes were retained in the stricter replay. All 15 sensed claims agreed with independent evaluation. The larger scene and additional search options increase compute. Full-suite regressions and broad material transfer were not measured.

## J. Comparison against baseline

The old nine matched cases remain at 8/9, and both earlier shifts remain at 2/2. The additional compound families pass 2/4, with both mugs successful and both bottles unsuccessful. The new gate measures clearance of the entire object.

## K. Research integrity and assumptions

Classification includes a general algorithmic correction to explicit material combination and independent clearance scoring, plus task-family heuristics for compound fixture proportions and lower grasp samples. The bottle and mug shapes are procedural approximations. Planning sees oracle-assisted geometry and retains the common material prior. Live contact parameters are not copied into planning. No wrap articulation was promoted after its sphere failures.

## L. Cloud resource usage

None. Local CPU only.

## M. Remaining work

Attribute the bottle's failed predicted hold to slip, tilt, reach or verification. Develop a physically supported grasp for the sphere. Integrate reachability-based walking and the humanoid interface, then freeze and run the full R1 gates.

## N. Final conclusion

The validation now covers complete object clearance and the robot's sensed claim on compound fixtures. Two mug holds expand the verified development scope. R1 remains active with bottle, sphere and mobility work open.
