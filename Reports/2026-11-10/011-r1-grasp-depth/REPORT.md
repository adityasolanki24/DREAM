# DREAM Research / Development Report: R1 depth-aware candidate search

## Task identification
- Task title: R1 depth-aware candidate search
- Task ID: 2026-11-10-011
- Date and time: 2026-11-10T22:49:08+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: c04ffd81e7a2adffeb84f65a47e64afee342467e
- Final commit: d574b37c0fb0169a35ca1dc55ccc5889856683e7
- Task status: partially completed

## A. Objective

Recover the remaining fixed-stance grip failures by testing how far the fingers wrap around the object and how the thumb closes, using the same observable selection criteria for all shapes.

## B. Starting state

The whole-body preview passed 6/9 matched live trials. The inward cube, cylinder and sphere all formed opposing contacts. The inward cube cleared 10 cm and slipped before hold. The cylinder slipped around 5 cm while the wrist pad force rose. The sphere lost opposition in the first probe.

## C. Work performed

### Implementation
Scale front-to-back proposals by observed object extent. Rehearse a bounded number of poses per depth band and verify up to three distinct proposals in a full-body simulation. If the default closure cannot support the load in rehearsal, geometrically screen symmetric thumb abduction offsets and use the same choice in clearance, closure and lift. Fall back to the original whole-body selector when the expanded predictions all fail.

### Evaluation
Compare depth-only and depth-plus-thumb selection across the same nine fixed-stance cases. Isolate the hand in the rehearsal stage to test all feasible base poses under three thumb angles. Compare the final selector with the preceding baseline on two additional geometry, mass and table-height changes. These are development checks.

### Runs
Run records: see runs/ and the daily index
See runs/ for the separate stage-only, matched development and geometry-shift runs. The evidence folders retain the complete source snapshots and physical traces.

## D. What was achieved

Verified live 10 cm lifts and 3 s holds on the cylinder and inward cube. In the outward cube case the second depth proposal passed after the first whole-body prediction slipped. The shifted cylinder also held with the expanded selector.

## E. What was not achieved

The sphere continues to lose opposing contact in the initial lift probe. The supported choices are pinch variants. A power-grasp family, walking into a reachability-selected stance, the unified humanoid toolbox and the complete R1 distribution and held-out gates are pending.

## F. Passed tests and experiments

All 94 focused grasp, hands, posture, supervisor and observation-isolation tests passed. Repository-wide Ruff lint and format checks passed. The combined selector verified 8 of 9 matched live holds and both geometry-shift checks. Full-suite acceptance remains part of the R1 milestone.

## G. Failed tests and experiments

The depth-only selector did not recover the inward cube or sphere. A broader sphere-only rehearsal tested 119 feasible wrist poses at each of three thumb offsets and produced no stable hold. The breadth of the pose sweep did not resolve the sphere's contact loss.

## H. Improvements

The selection varies the contact depth with observed geometry and lets measured retention choose the thumb articulation. It held two more of the original nine fixtures and one more of two geometry-shift checks. No special-case control condition keys off object identity or seed.

## I. Regressions and dis-improvements

<!-- If a metric was not measured, write "Not measured", not "No regression". -->
No prior verified holds were lost across the matched and two additional development cases. Additional geometric evaluation and optional rollout attempts increase computational cost. End-to-end benchmark throughput and hardware transfer were not measured.

## J. Comparison against baseline

The prior whole-body preview passed 6/9. Depth search passed 7/9. Depth with thumb adaptation passed 8/9. On the added fixed-stance shifts, the prior selector passed 1/2 and the expanded selector 2/2. See results/g1_r1_depth_dev_20261110/summary.json for exact configurations and episode evidence.

## K. Research integrity and assumptions

Class: task-family heuristic. The size-scaled depth and symmetric thumb candidates are bounded hand-search options. The same option set and physical criteria are used across cases. Planning uses the existing nominal material prior and independent noise, while object geometry and odometry remain oracle-assisted. Only independent live lift and contact samples count toward verification. The changed cube and cylinder are two development transfer checks rather than the frozen R1 claim gate.

## L. Cloud resource usage

None. Local CPU only, with at most three concurrent simulation or test processes.

## M. Remaining work

Test a reusable enveloping grasp with multiple support contacts for round objects, keeping the current failed sphere as a challenge. Add the other required geometries and material variation, use the existing stance and whole-body interfaces to join walking with grasp execution, and run the adopted development and held-out gates.

## N. Final conclusion

Depth and thumb selection improved the live fixed-stance comparison without dropping previous passes. Sphere retention, mobility integration and R1 acceptance remain open.
