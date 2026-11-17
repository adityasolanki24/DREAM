# Prompt for task 2026-11-17-001

- Task ID: 2026-11-17-001
- Submitted: 2026-11-17T13:44:42+11:00
- Created: 2026-11-17T13:44:42+11:00
- Branch: claude/architecture-cleanup-20261117
- Starting commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53
- Redactions: 0

## Original prompt
DREAM: Architectural Cleanup, Simplification and Stabilisation
Repository: https://github.com/adityasolanki24/DREAM
I want you to perform a serious, evidence-driven architectural cleanup of DREAM.
The repository has grown considerably through AI-assisted development. It contains many partially integrated components, overlapping abstractions, experimental pipelines and features that are difficult to understand, maintain or operate together.
My goal is to turn DREAM into a smaller, coherent, understandable and reproducible research codebase without destroying valuable work.
This is an implementation task, not merely a request for recommendations. Inspect the repository thoroughly, establish a baseline, make justified improvements and verify your changes.
1. Understand the existing project
Before changing anything:

* Read `AGENTS.md` completely. It is the authoritative engineering contract.
* Read `README.md`, especially the 7 November research closeout.
* Inspect the repository structure, dependencies, Git branches, recent commits, experimental configurations and tests.
* Read `results/research_closeout_20261106/inventory.json`.
* Check the working tree for uncommitted changes. Preserve all existing user work.
* Understand how the main components connect.

Specifically inspect:

* `dream/agent/`
* `dream/runtime/`
* `dream/robot/`
* `dream/sim/`
* `dream/perception/`
* `dream/navigation/`
* `dream/skills/`
* `dream/tools/`
* `dream/memory/`
* `dream/eval/`
* `experiments/`
* `tests/`
* Cloud execution and external model integrations.

Determine which components are genuinely operational, which are experimental, which duplicate functionality and which are unused.
Do not assume that a passing unit test establishes an end-to-end robotics capability.
2. Establish the actual state
Create a concise technical inventory covering:

1. Working and independently verified functionality.
2. Functionality operating only under narrow simulation assumptions.
3. Partially implemented features.
4. Failed experimental integrations.
5. Duplicate implementations.
6. Unnecessary abstraction layers.
7. Unused or unreachable code.
8. Broken or unclear interfaces.
9. Dependencies that are no longer needed.
10. Components that could be isolated without affecting the core system.

For each significant component, identify its entry point, dependencies, tests and actual research purpose.
Use repository evidence rather than documentation claims alone.
Pay particular attention to the previously reported failures:

* G1 whole-body W6 failing acceptance criteria.
* Qwen request-to-goal substitution and false-success claims.
* Inconsistent room-search generalisation.
* Camera-based manipulation underperforming oracle-assisted manipulation.
* MolmoAct2 failing complete manipulation tasks.
* Unvalidated integrations and excessive complexity.

Do not attempt to solve all these research problems during the cleanup.
3. Define a coherent architecture
DREAM's long-term ambition is general-purpose embodied intelligence, but its present implementation must have a manageable foundation.
Prioritise a clear core execution pipeline:

1. Receive and preserve the user's original goal.
2. Obtain sensor observations.
3. Build or update an estimated environment state.
4. Select a feasible action through an explicit planning interface.
5. Execute through a robot-specific controller.
6. Observe the actual outcome.
7. Independently verify progress against the original goal.
8. Continue, recover, refuse or stop.

Keep high-level reasoning separate from deterministic control and safety.
The primary embodiment to organise around is the Unitree G1 in MuJoCo, since the immediate research direction concerns language-guided navigation and physically reliable interaction.
Preserve the Panda manipulation and active-discovery work as separately testable research capabilities.
Do not pretend that the G1 can already perform reliable whole-body interaction. Preserve that as an experimental research direction.
Prefer straightforward functions, explicit interfaces and small modules.
Avoid speculative frameworks, unnecessary factories, plugin registries, excessive inheritance and layers that merely forward calls.
The architecture should reflect what the code actually does today, not every capability DREAM might eventually have.
4. Perform the cleanup incrementally
Create a dedicated refactoring branch from the appropriate validated commit.
Use small, logical commits.
Identify opportunities to:

* Consolidate genuinely duplicated code.
* Remove unnecessary wrappers and indirection.
* Simplify confusing interfaces.
* Clarify module ownership and responsibilities.
* Reduce circular or excessive dependencies.
* Separate validated functionality from experimental integrations.
* Make execution entry points obvious.
* Remove obsolete configuration paths when proven unused.
* Simplify setup and execution instructions.
* Eliminate redundant state representations where safe.
* Improve naming where it materially improves understanding.

Do not restructure working code merely for aesthetic consistency.
Do not introduce new models, frameworks, major dependencies or capabilities.
If an apparent simplification would change physical behaviour, evaluation results or an important interface, stop and assess it as a functional change rather than silently including it in the refactor.
Preserve the repository's existing safety and simulator-state isolation invariants.
5. Preserve research integrity
This is critical.
Do not delete or rewrite:

* Historical evaluation results.
* Failed experiment records.
* Reproducibility metadata.
* Frozen benchmark configurations.
* Validated simulation models.
* External licences and attribution.
* Original research evidence.
* Existing user changes or unmerged work.

Do not remove a component merely because it currently fails an experiment.
Classify it appropriately and isolate it if necessary.
Do not alter environments, seeds, scoring thresholds or success criteria to make results appear better.
Never substitute simulator ground truth for robot perception.
Never replace real simulated execution with hardcoded state changes.
Keep the original user goal immutable through planning and verification. A robot must not silently substitute an easier goal and report success.
Respect the existing `AGENTS.md` rule that the repository contains only `README.md` and `AGENTS.md` as Markdown documentation files. Improve those files rather than introducing numerous new documentation files.
6. Make DREAM understandable to its developer
The project owner is a mechatronics engineering student who wants to understand and continue developing the system independently.
Optimise for technical clarity.
After the cleanup, a developer should be able to answer:

* Where does execution begin?
* What happens when a goal is submitted?
* Which component interprets the goal?
* Where does perception happen?
* What information is estimated versus provided by an oracle?
* Which module plans robot motion?
* How does the Unitree G1 controller receive commands?
* How is success independently verified?
* Where are failures handled?
* Which components are experimental?
* How can one reproduce a published result?

Improve `README.md` with a concise architecture explanation, a trustworthy capability-status table and the shortest working commands for the principal validated workflows.
Do not turn the README into another enormous collection of speculative design documents.
7. Verification requirements
Before modification, record the baseline:

* Git commit and branch.
* Working-tree status.
* Relevant Python tests.
* Import and isolation checks.
* Ruff results.
* Existing reproducible simulation checks.

After each logical refactor, run focused tests.
At appropriate checkpoints, run the relevant integration tests.
Before declaring the cleanup complete, run all feasible repository checks required by `AGENTS.md`, including:

* Python fast tests.
* Python slow tests where the required environment is available.
* Import and architecture isolation tests.
* Ruff checks and formatting validation.
* Relevant simulation regression tests.
* Web or Android checks only where affected or required for a complete milestone.

Clearly distinguish tests that passed from tests that could not run.
Do not launch expensive cloud jobs, download huge model weights, create paid infrastructure or start long-running GPU evaluations without asking me first.
Do not silently change benchmark results.
8. Scope and safety
You may edit, reorganise and refactor code in a dedicated branch.
However:

* Do not force-push.
* Do not rewrite existing Git history.
* Do not delete historical branches or tags.
* Do not discard uncommitted changes.
* Do not delete or overwrite irreplaceable research evidence.
* Do not modify cloud infrastructure or credentials.
* Do not push to GitHub without my approval.
* Do not perform irreversible bulk deletions without showing me the exact proposed changes.

If removal of a component is justified but could destroy useful work, first propose a reversible archival or isolation strategy.
Do not stop after writing a plan. Proceed with safe, incremental improvements and tests.
Only request my input when a meaningful architectural choice, destructive operation or permission boundary requires it.
9. Final deliverables
At the end, provide:
A. Before-and-after architecture
Explain the original organisation and what changed.
Show the core execution pipeline and the responsibility of each major module.
B. Cleanup summary
List:

* Files and components simplified.
* Duplicate implementations consolidated.
* Obsolete paths isolated or removed.
* Interfaces clarified.
* Dependencies reduced.
* Important components deliberately preserved.

C. Verified functionality
Provide a table showing:

* Component.
* Previous verified status.
* Current verified status.
* Tests or experiments run.
* Remaining limitations.

Do not claim improvements without evidence.
D. Reproduction instructions
Provide the simplest verified commands to run the main supported workflows.
E. Remaining technical debt
Identify the most consequential remaining problems, ranked by their impact on research progress.
F. Research readiness
Explain whether the cleaned repository is suitable as a foundation for the proposed research:
Language-Guided Navigation and Whole-Body Interaction for Humanoid Robots.
Identify the minimum additional technical work needed to establish a reproducible G1 navigation-to-contact baseline.
Do not implement the entire research proposal as part of this cleanup.
Guiding principle
Reduce complexity without reducing scientific honesty.
I want a research system I can understand, reproduce, debug and extend.
I do not want another impressive-looking architecture full of partially connected AI components.
A smaller, genuinely working research foundation is preferable to a large system that appears capable but fails when tested.
Begin by inspecting the repository and establishing the baseline. Then perform the cleanup methodically, verifying each meaningful change.
