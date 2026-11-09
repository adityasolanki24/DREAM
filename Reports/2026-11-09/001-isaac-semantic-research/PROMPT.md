# Prompt for task 2026-11-09-001

- Task ID: 2026-11-09-001
- Submitted: 2026-11-09T13:48:18+11:00
- Created: 2026-11-09T13:58:11+11:00
- Branch: claude/isaac-semantic-research-20261109
- Starting commit: df00ae9e30cb6c32ade4d3155e4c83cac489b6a2
- Redactions: 0

## Original prompt
Continue DREAM development from the latest local working tree, including the semantic-goal-search research branch.

Repository: https://github.com/adityasolanki24/DREAM

You are acting as DREAM's lead embodied-AI research engineer, experiment engineer and technical documentation owner.

I want you to continue the existing research using NVIDIA Isaac Sim, cloud GPUs, genuine visual perception and semantic reasoning.

I also want you to implement a PERMANENT, HIERARCHICAL RESEARCH REPORTING SYSTEM.

This reporting system is mandatory for this assignment and every subsequent DREAM development prompt.

Do not start the project again, discard existing progress or rewrite historical results.

# PART 1: PERMANENT REPORTING SYSTEM

## 1. Create a top-level Reports directory

Create a directory named exactly:

`Reports/`

This directory should become DREAM's organised development and research history.

Organise it hierarchically:

```text
DREAM/
├── Reports/
│   ├── INDEX.md
│   ├── 2026-11-09/
│   │   ├── INDEX.md
│   │   ├── 001-semantic-exploration/
│   │   │   ├── PROMPT.md
│   │   │   ├── REPORT.md
│   │   │   ├── metadata.json
│   │   │   ├── runs/
│   │   │   │   ├── run-001/
│   │   │   │   │   ├── REPORT.md
│   │   │   │   │   └── metadata.json
│   │   │   │   ├── run-002/
│   │   │   │   │   ├── REPORT.md
│   │   │   │   │   └── metadata.json
│   │   │   └── comparisons/
│   │   │       └── BASELINE_COMPARISON.md
│   │   ├── 002-isaac-perception/
│   │   │   ├── PROMPT.md
│   │   │   ├── REPORT.md
│   │   │   ├── metadata.json
│   │   │   └── runs/
│   │   └── 003-cloud-streaming/
│   │       ├── PROMPT.md
│   │       ├── REPORT.md
│   │       └── runs/
│   ├── 2026-11-10/
│   │   ├── INDEX.md
│   │   └── 001-object-search/
│   │       ├── PROMPT.md
│   │       ├── REPORT.md
│   │       ├── metadata.json
│   │       └── runs/
│   └── ...
```

This is an illustrative structure. Implement a practical version following these principles.

Use the actual local date and time, preferably Australia/Sydney, rather than hardcoding November 9.

Each independently submitted development prompt must receive its own unique task directory.

Each individual experimental execution must receive its own run directory or record.

Repeated attempts must never overwrite earlier reports.

A task that runs across midnight should retain a stable task ID and provide date-linked progress records so it is discoverable from both dates.

Use readable names and unique IDs to prevent collisions.

## 2. Record every prompt

For every future DREAM development request, create a new reporting entry before beginning substantial work.

Store the original instructions in `PROMPT.md`, with the submission timestamp and task ID.

Preserve the complete research requirements, evaluation instructions and acceptance criteria.

Do not store secrets, access tokens, private credentials or sensitive information in reports. Redact such data while preserving the technical substance.

This requirement applies to:
- Research assignments.
- Feature development.
- Bug fixes.
- Experiments.
- Cloud evaluations.
- Model comparisons.
- Simulation work.
- Android development.
- Frontend development.
- Integration.
- Performance optimisation.
- Refactoring.
- Technical investigations.

Even if a task fails, is blocked or is abandoned, it must have a report explaining what happened.

For purely conversational questions that do not initiate repository work, a development report is unnecessary.

## 3. Every prompt must generate a complete REPORT.md

Use the following mandatory structure:

### DREAM Research / Development Report

**Task identification**
- Task title.
- Unique task ID.
- Date and time.
- Original prompt reference.
- Git branch.
- Starting commit.
- Final commit.
- Task status: completed, partially completed, failed, blocked or cancelled.

**A. Objective**

Explain what the user requested and what problem the work was intended to solve.

**B. Starting state**

Describe what existed before the work began.

Include relevant baseline metrics, existing failures and previous experimental results.

**C. Work performed**

Describe the actual work completed.

Identify:
- Algorithms introduced.
- Existing algorithms modified.
- Models integrated.
- Files and components changed.
- Experiments designed.
- Bugs investigated.
- Bugs fixed.
- Infrastructure changes.

Clearly separate implementation from evaluation.

**D. What was achieved**

State exactly what worked.

Include quantitative evidence where available.

Avoid vague claims such as "significantly improved" unless the measurements support them.

**E. What was not achieved**

Explicitly identify:
- Objectives not completed.
- Unsupported functionality.
- Unvalidated claims.
- Experiments not run.
- Incomplete integration.
- Features that still require development.

Do not omit unsuccessful objectives.

**F. Passed tests and experiments**

Record:
- Test names.
- Test counts.
- Experiment IDs.
- Evaluation conditions.
- Success criteria.
- Actual measured results.
- Evidence locations.

**G. Failed tests and experiments**

Record:
- Exact failure.
- Expected behaviour.
- Actual behaviour.
- Reproduction conditions.
- Error logs.
- Root cause, if known.
- Whether the failure was fixed.
- Whether the fix was retested.

Distinguish a known root cause from a hypothesis.

**H. Improvements**

Compare against the previous relevant implementation.

Report measurable improvements in:
- Task completion.
- Accuracy.
- Safety.
- Execution time.
- Navigation distance.
- Model inference.
- Memory use.
- GPU utilisation.
- Camera delivery.
- System reliability.

Only include metrics relevant to the actual task.

**I. Regressions and dis-improvements**

This section is mandatory, even when no regressions are detected.

Explicitly record anything that became worse:
- Lower task success.
- Increased false claims.
- Longer execution time.
- Greater travel distance.
- More collisions.
- More GPU usage.
- Increased cost.
- Higher latency.
- New bugs.
- Reduced compatibility.
- More complicated dependencies.

If a metric was not measured, state "Not measured", not "No regression".

**J. Comparison against baseline**

Provide a comparison table when comparable evidence exists.

Example:

| Metric | Previous | New | Difference | Interpretation |
|---|---|---|---|---|
| Task success | 70% | 82% | +12 percentage points | Improved |
| False success | 3/30 | 0/30 | -3 cases | Encouraging, sample limited |
| Mean task time | 42 s | 48 s | +6 s | Regressed |
| Navigation distance | 12 m | 15 m | +3 m | Regressed |
| Safety stops | 0 | 0 | Unchanged | No observed difference |

These numbers are examples only. Never use fabricated metrics in actual reports.

Use paired tests and statistical uncertainty where appropriate.

Do not compare runs with different physics, renderers, observation models or task distributions without stating the differences.

**K. Research integrity and assumptions**

Document:
- Simulation versus physical hardware.
- Oracle information usage.
- Learned versus scripted behaviour.
- Hardcoded assumptions.
- Heuristics.
- Model limitations.
- Development versus held-out experiments.
- Known biases.
- Generalisation limitations.

**L. Cloud resource usage**

When applicable, report:
- Cloud provider/project.
- Region.
- Machine type.
- GPU type.
- GPU count.
- Runtime.
- Estimated cost.
- Actual billed cost, if available.
- Job IDs.
- Storage and output locations.
- Automatic expiry configuration.
- Cleanup verification.
- Remaining resources.

Never present estimates as actual charges.

**M. Remaining work**

Specify the exact unresolved problems.

Do not introduce unrelated future projects.

**N. Final conclusion**

End with a concise assessment:

- What genuinely succeeded.
- What genuinely failed.
- Whether the original objective was achieved.
- Whether the result represents a verified improvement.
- Whether the work is ready to merge.
- What remains uncertain.

## 4. Every experimental run gets its own report

A prompt-level report is not enough.

For each experiment, model evaluation, simulation campaign or cloud execution, create a separate run record.

Every run should identify:
- Unique run ID.
- Parent prompt/task ID.
- Run date and timestamps.
- Exact Git commit.
- Config file and its hash.
- Random seeds.
- Model name and checkpoint.
- Simulator and version.
- Renderer.
- Robot embodiment.
- Robot controller.
- Hardware.
- Command executed.
- Start and end time.
- Exit status.
- Whether the run completed.
- Number of cases attempted and completed.
- Passed and failed cases.
- Quantitative metrics.
- Observed regressions.
- Logs and raw evidence paths.
- Cloud costs and cleanup status.

If a run crashes halfway through, keep the partial results and report the interruption.

Never silently delete failed runs or replace them with successful reruns.

If one prompt requires 10 different experiments, I expect 10 distinct run records and one overall report consolidating them.

## 5. Maintain searchable indexes

`Reports/INDEX.md` should contain a chronological overview of all reported tasks.

Suggested columns:

| Date | Task ID | Prompt | Status | Main finding | Report |
|---|---|---|---|---|---|

Each date's `INDEX.md` should contain the same type of summary for that day.

Keep indexes updated automatically.

Provide links to individual reports and experiments.

Make it possible to understand DREAM's development history without searching thousands of Git commits.

Do not replace the existing README, which remains the project-level description and scientific status.

The Reports directory should complement the README and historical `results/` records.

## 6. Automate reporting

Implement lightweight reporting helpers or scripts to:

- Create a report entry when a new task begins.
- Generate a unique ID.
- Capture starting Git metadata.
- Record experimental commands and configuration hashes.
- Register runs.
- Append run results.
- Update daily and root indexes.
- Mark a task complete, failed or blocked.
- Validate report completeness.

Use a simple, maintainable implementation.

Do not build an unnecessarily large reporting platform.

Update AGENTS.md and relevant developer instructions so this reporting workflow remains mandatory in future Claude Code sessions.

If a required report field cannot be populated automatically, leave it explicitly marked for completion. Do not invent values.

Every subsequent implementation prompt should conclude by updating its report.

## 7. Preserve reproducibility and avoid report duplication

Raw experimental data should continue to live under the existing `results/` structure or another appropriate immutable evidence location.

Reports should link to those files using repository-relative paths.

Do not duplicate multi-gigabyte recordings, model checkpoints or raw RGB-D datasets inside Reports.

Where possible, include SHA-256 checksums or immutable evidence references.

Preserve historical reports and old results.

Do not retroactively create invented execution logs for earlier work.

You may create a clearly labelled historical index referencing verified previous experiments.

If results are revised, append an amendment or create a new version rather than silently changing the original record.

# PART 2: CONTINUE DREAM'S RESEARCH IN ISAAC SIM

## 8. Research objective

The principal research question is:

Can semantic world knowledge improve how an embodied robot searches unfamiliar environments when it must rely on genuine visual observations and physical navigation?

The latest DREAM results show:
- Original-goal consistency has improved.
- The stronger verifier passed its fresh-seed gates.
- Semantic search did not significantly outperform frontier exploration in small homes.
- ConceptNet room knowledge was weak for laundry and storage rooms.
- Qwen-derived room associations looked more promising in an offline comparison.
- Most MuJoCo object-search experiments relied on simplified perception.
- The latest Isaac evaluation was incomplete because of missing assets and skipped cases.

Continue from this evidence.

Do not pretend these problems have already been solved.

## 9. Make Isaac Sim the primary realistic simulation environment

Use NVIDIA Isaac Sim on Google Cloud GPUs where appropriate.

Use:
- Unitree G1.
- Existing validated pretrained locomotion controller.
- Real rendered camera observations.
- Depth sensing.
- Genuine learned object detection or VLM perception.
- Existing navigation system.
- Existing semantic memory.
- Typed action tools.
- Independent verification.

Do not rebuild the robotics architecture unnecessarily.

MuJoCo can remain the local regression and ablation environment.

Isaac should be used for the next primary model-guided evaluation.

Do not describe Isaac simulation as equivalent to physical robot testing.

## 10. Audit hardcoding and oracle information

I want a direct answer to whether DREAM's success depends on hardcoded assumptions.

Inspect for:
- Fixed object locations.
- Known room layouts.
- Environment-specific routes.
- Hardcoded target identities.
- Hardcoded success conditions tied to scenes.
- Model prompts containing evaluation answers.
- Simulator ground-truth leaking into perception.
- Scripted actions presented as model-generated reasoning.
- Hidden use of procedural generator metadata.
- Evaluation-seed-specific logic.
- Privileged object segmentation.
- Privileged odometry.
- Artificially simplified camera observations.

Classify every significant simplification.

Not all deterministic engineering is inappropriate.

Safety limits, valid tool schemas, collision checking, coordinate transformations and general geometric constraints are legitimate.

What I do not want is a robot that knows the answers because its developer encoded the benchmark.

Write the audit findings into the relevant task report.

Never hide an identified simplification.

## 11. Improve genuine visual perception

Replace the simplified segmentation-based object-search sensor with actual learned perception in the primary Isaac evaluation.

Investigate:
- Qwen3-VL.
- Existing vision detector interfaces.
- Object detection.
- Room classification.
- Instance tracking.
- RGB-D back-projection.
- Multiple-view consistency.
- Occlusion handling.
- Look-alike objects.
- Counter-mounted objects such as sinks.

Validate model box coordinates before transforming them into 3D.

Do not weaken the verifier just to improve the success rate.

Track confidence, evidence and failure modes.

The robot must not have access to hidden scene truth during action selection.

## 12. Improve semantic knowledge

Evaluate:
- ConceptNet.
- Qwen-derived semantic room priors.
- A combination of both.
- No semantic knowledge.

The model should understand that refrigerators are typically found in kitchens, while remaining capable of finding them elsewhere.

Knowledge should guide exploration, not fabricate observations.

A kitchen hypothesis cannot automatically establish the existence of a refrigerator.

Use room-level evidence, object observations and uncertainty.

Do not treat uncalibrated scores as calibrated probabilities.

Run new held-out evaluations after changing knowledge sources.

## 13. Improve active exploration

Use context to choose which regions or viewpoints to inspect next.

Consider:
- Room relevance.
- Search uncertainty.
- Expected information gain.
- Object visibility.
- Occlusion.
- Navigation distance.
- Reachability.
- Safety.
- Time budget.
- Existing memory.

Reuse Dijkstra, RPP, occupancy mapping, the pretrained G1 controller and the current agent architecture.

Do not automatically go to the kitchen for every refrigerator request.

The robot must still handle atypical placements, absent targets and conflicting clues.

Preserve the immutable original-goal specification.

Intermediate tasks must never substitute for the user's requested outcome.

## 14. Evaluate in larger, more varied environments

The old environments were too small to demonstrate a convincing benefit from semantic search.

Create or identify appropriate, genuinely diverse multi-room Isaac environments.

Include:
- Different room types.
- Larger layouts.
- Long travel distances.
- Unexpected object locations.
- Missing targets.
- Partial occlusions.
- Look-alike objects.
- Conflicting semantic clues.
- Multiple plausible target locations.
- Rooms with incomplete furnishings.

Verify that all required assets load successfully before freezing benchmarks.

Do not repeat the missing refrigerator/oven asset failures.

If an asset is missing, record that failure in the run report.

Do not turn a target-present experiment into an absent-target experiment without explicitly updating its design before evaluation.

## 15. Compare against proper baselines

Compare:
- Classical frontier exploration.
- Frontier exploration with stronger verification.
- ConceptNet semantic exploration.
- Qwen semantic exploration.
- Combined semantic exploration and information gain.

Include appropriate ablations.

Keep robot configuration, physics, camera and experiment conditions matched.

Use identical paired scenarios wherever possible.

Freeze success criteria and evaluation seeds before held-out runs.

Do not tune on held-out results.

Measure:
- Original-goal success.
- False success.
- Correct refusal.
- Time to target.
- Distance travelled.
- Number of model calls.
- Object-detection accuracy.
- Room-classification accuracy.
- Safety stops and contacts.
- Recovery success.
- Inference overhead.
- GPU costs.
- Generalisation across environments.

Every experiment must have its own report under Reports.

# PART 3: CLOUD AUTONOMY AND RELIABLE SELF-DELETION

## 16. Cloud authorisation

I authorise extensive Google Cloud GPU use for this assignment.

You may provision and execute the cloud workloads needed for the approved research without requesting confirmation for every ordinary run.

Use the project's existing resource safeguards, access controls, cost ledger and per-job limits.

Do not interpret this as permission for unbounded expenditure, unlimited parallel GPU machines or uncontrolled resource creation.

Use appropriate operational cost guardrails and record all spending estimates.

If additional spending would exceed those safeguards, request confirmation before increasing them.

Do not launch unrelated workloads.

## 17. Self-deletion must be reliable

All temporary cloud resources must automatically terminate and be deleted after completion.

Do not leave paid GPU instances idle.

The self-deletion lifetime must be long enough for the experiment to complete.

Determine the deadline using:

Expected setup time
+ model loading time
+ experiment runtime
+ runtime contingency
+ evidence upload allowance
+ cleanup allowance.

Avoid setting expiration times that interrupt experiments unnecessarily.

Keep the experiment timeout shorter than the infrastructure lifetime so that results can be uploaded before the VM expires.

Checkpoint long-running experiments.

If the deadline approaches:
- Stop admitting new experimental cases.
- Save completed results.
- Upload evidence.
- Record the incomplete state.
- Exit cleanly.
- Resume from saved progress when justified.

Do not continuously relaunch an experiment that fails because of an insufficient lifetime.

Do not disable automatic deletion.

Verify cleanup independently after each session.

Keep only intentionally retained images and assets.

Do not modify unrelated NERVA or Quartus cloud resources.

## 18. Cloud reporting

Every cloud execution must produce its own run report.

Record:
- VM and GPU type.
- Region.
- Session ID.
- Job ID.
- Estimated runtime.
- Actual runtime.
- Estimated and actual cost, separately.
- Resource maximum lifetime.
- Checkpoint status.
- Raw evidence location.
- Result upload status.
- Cleanup status.
- Whether any resources remain.

After the experiment, do not mark the run complete until the recorded result and cleanup checks are finished.

Use clear labels for experiments that ended early, crashed, timed out or were interrupted.

# PART 4: EXECUTION AND FINAL DELIVERY

## 19. Execute in stages

Proceed in this order:

1. Inspect the latest repository and research status.
2. Implement the Reports system.
3. Add permanent reporting instructions to AGENTS.md.
4. Create the report for this prompt.
5. Audit hardcoding and simulator shortcuts.
6. Validate Isaac environments and assets.
7. Integrate genuine learned visual perception.
8. Compare semantic knowledge sources.
9. Improve active exploration.
10. Run focused local development tests.
11. Freeze evaluation configurations.
12. Run cloud GPU experiments.
13. Perform baseline and ablation comparisons.
14. Analyse improvements and regressions.
15. Complete all run-level reports.
16. Complete the prompt-level report.
17. Update indexes and README.
18. Run final regression tests.
19. Commit verified work on an isolated development branch.

Do not stop after creating the Reports directory.

Continue the actual research.

## 20. Final deliverables

I expect:

- A functioning permanent Reports directory.
- Daily report indexes.
- One report per development prompt.
- One report per experimental run.
- Clear pass/fail reporting.
- Clear achievement and nonachievement reporting.
- Improvements and regressions documented quantitatively.
- Reproducible evidence references.
- Hardcoding and oracle-dependency audit.
- Updated Isaac semantic exploration implementation.
- Genuine visual perception integration.
- Evaluated semantic knowledge alternatives.
- Larger multi-room experiments.
- Baseline comparisons.
- Cloud GPU execution records.
- Verified automatic cleanup.
- Updated research summary.
- Verified Git commits.

The final message to me should provide:

1. What you completed.
2. What passed.
3. What failed.
4. What improved.
5. What became worse.
6. What remains incomplete.
7. Cloud costs and cleanup status.
8. Exact Git branch and commit.
9. The path to the main report for this prompt.

## 21. Permanent rule for all future prompts

From now on, whenever I give a DREAM development or research instruction, you must:

- Create a dated task report.
- Save the task's original requirements.
- Record the starting state.
- Document each experimental run.
- Record both successes and failures.
- Compare relevant results against previous baselines.
- Identify measurable improvements and regressions.
- Preserve raw evidence.
- Update the task report after each milestone.
- Complete the report before declaring the task finished.
- Update Reports/INDEX.md and the relevant daily index.
- Link the final report in your response.

This is a permanent requirement of the DREAM project.

The reporting process must not be skipped because an experiment failed, a task was interrupted, or the implementation took longer than expected.

Do not claim success simply because the code builds.

Every result must distinguish implementation, testing and scientific validation.

Do not let reporting become a substitute for doing the research. The reports should document the work, not consume most of the development time.

The objective is to develop a reproducible embodied-AI research system while building a clear, chronological, evidence-backed record of every development task and experiment.

Begin implementing the reporting system and continue the Isaac semantic exploration work now.

For the report bit, add it in the agents.md as well, so you remember it every time.
