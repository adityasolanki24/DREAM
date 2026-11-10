# DREAM Research / Development Report: Continue the sequential G1 roadmap through the R0 gate

## Task identification
- Task title: Continue the sequential G1 roadmap through the R0 gate
- Task ID: 2026-11-10-005
- Date and time: 2026-11-10T10:57:57+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: codex/g1-integration
- Starting commit: efd6872f4c375f948244e04c5beb71eade626925
- Final commit: dbedd2ce820e644f6a190ccb5eee1f59ca5a7f12
- Task status: partially completed

## A. Objective

Complete the R0 integration checks before continuing R1, then use reproducible development controls to isolate the earliest grasp blocker. Keep R2 pending.

## B. Starting state

The integration worktree held the port and monitored settling fix at efd6872. The original reach regression had passed. A shifted-target run had started before the previous turn was interrupted.

## C. Work performed

### Implementation

Completed the earlier reporting records and closed the deferred semantic-research task on this branch. Marked R0 green only after the fast suite, full suite, Ruff and isolation had passed. Committed and pushed the integration as 33e23c0. After that gate, extended the R1 diagnostic runner with evaluator traces for requested and applied right wrist torque, both hands' contact partners, and fixed starting shoulder roll controls.

### Evaluation

Recovered both passed reach results from the prior run. Ran the complete fast and full suites on the integration source tree. The gate logs are fast.txt and full.txt under C:/Users/24adi/DREAM-r1-evidence/r0-suite-001. Then reran the prior tucked configuration and two controlled clearance variants.

### Runs

Three R1 development runs are indexed under runs/. The original tucked config remains reproducible. The new hip-clearance and both-arms-clear configs isolate the changed initial shoulder angles.

## D. What was achieved

R0 integration passed and was published on origin/codex/g1-integration. R1 reproduced the right wrist saturation and hip contact. Clearing only the right hip removed the right wrist stop but revealed a left wrist stop. Clearing both arms at reset enabled a complete reach and opposed touch. Independent contact evidence identified the tabletop as the object touched.

## E. What was not achieved

No box lift, verified grasp, R1 capability claim or held-out evaluation. The fixture changes do not implement collision-aware approach planning. No carry work was started.

## F. Passed tests and experiments

R0: 990 passed and 2 skipped in the fast suite; 1061 passed and 2 skipped in the full suite. Original and shifted reach checks passed. Repository-wide Ruff lint, Ruff format and isolation passed. The post-R0 diagnostics passed Ruff and the focused hand and isolation suite (50 passed).

## G. Failed tests and experiments

The original R1 trial stopped at the right wrist. The right-only clearance control stopped at the left wrist. The bilateral clearance control stopped at the hand contact-force limit while trying to lift after touching the table. R1 remains open.

## H. Improvements

The diagnostic controls isolated initial hip clearance from the separate tabletop clearance failure. In the original trace the wrist requested 9.32 Nm and applied torque saturated at 5.00 Nm. In the both-arms-clear fixture the G1 completed its reach and attempted physical closure.

## I. Regressions and dis-improvements

The integration suite had no failures. The first clearance control exposed a separate idle-arm stop. The bilateral control removed both wrist stops, but tactile feedback alone still confused tabletop contact with potential purchase. Other R1 variants were not measured.

## J. Comparison against baseline

The original tucked trial and both controls retained the same table, box, base stance, seed and limits. Only the initial shoulder rolls changed between controls. This supports the hip-interference diagnosis without establishing robust transfer.

## K. Research integrity and assumptions

The extra shoulder roll is a development fixture change. The evaluator contact identity is excluded from the grasp decision. Oracle-assisted setup and headless CPU physics are documented. No force or torque limit was relaxed, and no result was scored as a grasp without independent lift evidence.

## L. Cloud resource usage

None.

## M. Remaining work

Add a regression for tabletop contact during closure. Check the open hand and closing sweep against the observed support surface, and require measured object response before attempting a full lift. Complete the fixed-stance grasp gate before moving to object variation and carry.

## N. Final conclusion

R0 is green and checkpointed. R1 advanced from a wrist stop to physical closure, then exposed a repeatable tabletop clearance and contact attribution failure.
