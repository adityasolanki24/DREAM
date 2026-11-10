# DREAM Research / Development Report: Strategic audit against the end goal and a plan for future agents

## Task identification
- Task title: Strategic audit against the end goal and a plan for future agents
- Task ID: 2026-11-10-001
- Date and time: 2026-11-10T10:00:47+11:00
- Original prompt: [PROMPT.md](PROMPT.md)
- Git branch: claude/isaac-semantic-research-20261109
- Starting commit: efd6872f4c375f948244e04c5beb71eade626925
- Final commit: efd6872f4c375f948244e04c5beb71eade626925
- Task status: completed

## A. Objective

Find why DREAM has stopped making visible progress toward its end goal (general-purpose embodied intelligence and whole-body autonomy on a humanoid), by auditing the implementation, the architecture, the process and every branch, and draft a plan that future agents can follow. No rung of the plan is advanced by this task; it defines the rungs.

## B. Starting state

Branch `claude/isaac-semantic-research-20261109` at `efd6872`, 90 commits ahead of `main` (`2484361`, the 7 November closeout). Task 2026-11-09-001 was open with 23 run records and an unfilled `REPORT.md`. `claude/wholebody-autonomy` (`8f47475`) was unmerged after W6 failed two development gates. The README status table had 31 rows, 10 of them PASS.

## C. Work performed

### Implementation
None. No code, config, `AGENTS.md` or `README.md` change. The only files written are this report, [PLAN.md](PLAN.md) and the regenerated indexes.

### Evaluation
None. This was a read-only audit: `AGENTS.md`, `README.md`, `dream/runtime/agent.py`, `dream/tools/goals.py`, `dream/tools/dexterous.py`, `dream/types.py`, module headers in `dream/tools`, `dream/runtime`, `dream/memory` and `dream/robot`, and read-only git commands. Three read-only subagents audited (1) `Reports/2026-11-09/001-isaac-semantic-research/`, (2) the whole-body branch, the G1 manipulation state and the learned-policy attempts, (3) the commit timeline, effort split and branch inventory. One subagent unpacked `archive/documentation-wholebody-20261029.zip` into a temporary directory outside the repository to read the W1 to W6 definitions.

### Runs
Run records: see runs/ and the daily index
No experimental runs. Nothing was executed against a simulator, a model or the cloud.

## D. What was achieved

- A diagnosis with evidence, in [PLAN.md](PLAN.md) sections 1 and 2. The central finding: the G1 has never picked up an object with its own hands under physics, and no work has advanced that since 6 November.
- Seven strategic decisions (PLAN section 3) and an eight-rung capability ladder R0 to R7 with deliverables, development targets, claim gates and kill criteria (section 4).
- A list of work to stop (section 5), the contract edits the plan needs (section 6), the risks and the decisions that will return to the owner (section 7), and the next three sessions (section 8).

## E. What was not achieved

- The plan is not adopted. `AGENTS.md` section 8 still lists the old priorities; changing the contract is the owner's decision.
- No consolidation was done: branches, worktrees and the open task 2026-11-09-001 are as they were.
- `results/research_closeout_20261106/inventory.json` and `Reports/HISTORY.md` were not read in detail.
- The whole-body branch's code was not reviewed line by line; its state is taken from its results, its archived plan and its handoff notes, which are worker-written.

## F. Passed tests and experiments

Not applicable. No tests or experiments were run.

## G. Failed tests and experiments

Not applicable. No tests or experiments were run.

## H. Improvements

Not measured. This task changes no behaviour.

## I. Regressions and dis-improvements

Not measured. No code or configuration was changed, and no suite was run.

## J. Comparison against baseline

Not applicable.

## K. Research integrity and assumptions

- Timeline dates come from commit-message keyword searches and are approximate. Commit counts per path and line counts are exact outputs of git commands on 10 November.
- The commit-message buckets (evaluation 28%, infrastructure 24%, apps 16%, documentation 13%) are keyword counts that overlap; they indicate proportion and are not a partition.
- W5 and W6 numbers come from `results/g1_w6_closeout_20261106/summary.json` and the archived handoff on the whole-body branch, as reported by a subagent; they were not recomputed.
- The claim that no G1 grasp under physics exists rests on the audit of `dream/sim`, `dream/robot`, `dream/tools` and `results/`; a result outside those paths would have been missed.
- The gate thresholds in the plan are proposals, to be fixed once by the agent that freezes each rung's config.
- The plan's judgement that manipulation is the binding constraint is an argument from the evidence above. It is not an experimental result.

## L. Cloud resource usage

None. No cloud resource was created or used.

## M. Remaining work

1. Owner decision: adopt, amend or reject the plan, and approve the `AGENTS.md` edits in PLAN section 6.
2. R0, part 1: close task 2026-11-09-001 and build the integration trunk (PLAN section 8).
3. Commit this report and the plan; they are uncommitted at the end of this task.

## N. Final conclusion

DREAM is stalled because its effort went into search, benchmarks, infrastructure and apps while the capability its end goal depends on, a humanoid that manipulates with its own body, was parked after one failed gate and never resumed. The contract's own priority list has no step that builds it. The plan puts one robot on one trunk in one simulator and climbs from a first finger grasp (R1) through carry and fetch (R2, R3) to unseen task compositions (R4) and discovery on the humanoid (R5). It does not promise general-purpose intelligence; it reaches the first points at which that can be measured.
