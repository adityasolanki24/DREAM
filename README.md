# DREAM

Research toward general-purpose embodied intelligence: receive an unfamiliar physical goal,
observe, investigate, compose available skills, verify the outcome, and recognise limits.
**General-purpose task completion is the objective, not an established capability.**
`AGENTS.md` is the authoritative engineering and behavioural contract; `CLAUDE.md` aliases it.

## Current status — 29 October 2026

**This checkout: `codex/g1-runtime-builder-20261028`.** Runtime-image build, genuine Isaac/Qwen app run, idle-balance correction and local inspection-failure audit. Inspection/planning corrections are not implemented yet.

| Component | Evidence-backed status | Limits / next work |
|---|---|---|
| Active physical discovery | Force-limited probing, Bayesian mechanism belief and information-gain experiment selection; held-out results preserved under `results/` | Abstract gantry and narrowed Panda tasks; no general physical reasoning claim |
| Panda manipulation and discovery | Contact-based skills and independently verified structured tasks; historical camera pick-and-place 39/40; compartment drawers/lids 27/30 each | Camera mechanism discovery remains unreliable; validated compartment work uses oracle-assisted tracking and disclosed scene simplifications |
| G1 mobility in MuJoCo | NVIDIA Decoupled WBC contract audited; historical locomotion and search gates passed | Velocity-commanded locomotion; arms held; this is a pretrained motor controller |
| Classical room search | Held-out nominal 20/20, shifted 8/10, no-cabinet 6/6 correct refusals | Oracle-assisted odometry; colour/depth perception; furniture contact occurred in 18/66 historical episodes |
| Navigation, memory and interruption | RPP navigation, SQLite home/experience memory, scene graph and interruptible phone API implemented on the later kitchen/interface branches | Use the frozen results and their caveats; none establish unrestricted autonomy or validated SLAM |
| App and model selection | Desktop/mobile mission control, camera, chat, provider/model selection, task events and pause/cancel; genuine app-to-cloud execution recorded | Full camera frame is contained, not cropped. Voice UI exists; reliable end-to-end spoken task completion has not been established |
| Natural-language goals | Configured VLM can interpret nonempty language into typed supported goals | Current mobile capabilities are cabinet inspection and navigation to remembered places; other requests must be declined |
| Cached cloud runtime | Immutable image `4812387390768042548`; locked CUDA 12.8, L4, MuJoCo 3.16 and actual inference/physics preflight verified | Image-build receipt records recovery from a post-creation JSON parsing error; its manipulation attempt failed release verification |
| Genuine Isaac/Qwen run | Isaac Sim 6.2 / PhysX / Unitree G1, Qwen3-VL-4B-Instruct; pinned checkpoint revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Development demonstration only; cabinet task completion failed |
| GPT/VoLoAgent work | Official ChatGPT sign-in backend and a small frozen-decision interface pilot; VoLoAgent non-actuating contract work | No completed matched physical GPT/Qwen comparison; no validated VoLoAgent actuation |
| Whole-body W6 | Development agent/baseline/reference runs and review preserved on `claude/wholebody-autonomy` | Failed gates and rollback/reference/attribution questions remain; no frozen held-out result or readiness for integration |

Historical measurements above are scoped to their original code, configurations and seeds.
A documentation refresh does not rerun or revalidate them. Detailed earlier reports,
commands, licence notes and simplification audits remain in the documentation archive and Git.
Original results and hard failure cases have not been removed or rescored.

## Latest Isaac failure and exact local audit

The first live task passed ten seconds of controlled standing, then detected a 62-degree
pelvis tilt at 10.02 s. Kit updates in the idle server advanced physics without WBC ticks.
The correction removes uncontrolled idle updates, captures a frame before API readiness,
and rejects further tasks after a physical safety stop. Three focused CPU regressions passed.

A real retry at `65ce8b3` completed `look_around` and `inspect(candidate=2)` without a fall,
reaching 26.52 s of simulation time including standing. It **did not complete the goal**:
inspection reported tool success while `at_inspection_pose=false`, the verifier reported
"not on the handle's side of the object", and Qwen explicitly called `finish(success=false)`.
Raw decisions and both failed attempts remain in `results/isaac_live_idle_failure` on the
runtime branch and in the external cloud results.

The exact saved refusal has now been replayed locally. An exhaustive audit of the saved
estimated map found:

- Candidate 2: 41 reachable cells satisfy the existing inspection-pose predicate, but the
  old sampler omitted them. Its only offered side-5 viewpoint was 2.12 m from the handle
  and was not a valid inspection pose. Its detected handle was assigned side 1.
- Candidate 9: 364 reachable cells satisfy that predicate. Its offered side-4 and side-5
  viewpoints have paths on the saved map and satisfy the pose predicate; side 0 does not.
- These are **estimated-map reachability results**, not proof of physical reachability or
  correct handle association in Isaac. Original raw RGB-D was not preserved for this run.

Audit source, exact input and original reproduction are under
`experiments/audit_mobile_inspection.py` and `results/mobile_inspection_semantics` on the
runtime branch. No inspection-semantics correction, recovery regression set, deterministic
physical loop validation or fresh Qwen evaluation has run since this diagnosis.

The next engineering gate is explicit action/goal semantics and bounded model-led recovery:
wrong-side observation must not count as completed inspection; consider other admissible
viewpoints/candidates; refuse when admissible alternatives or the task budget are exhausted.
Retain original robot dynamics, safety limits, idle handling and the unchanged kitchen scenario.
Run the exact saved regressions and a complete deterministic perception/navigation/verification
loop locally **before another paid evaluation**. The requested fresh Qwen evaluation is pending.

## Branches and integration

| Branch | Role |
|---|---|
| `main` | Stable integration line; only changes that pass the applicable checks are integrated |
| `codex/phase1-agentic-interface` | Active interface, natural-language routing and model-comparison checkpoints |
| `codex/g1-runtime-builder-20261028` | Active runtime image, Isaac app integration and unfinished inspection audit |
| `claude/wholebody-autonomy` | Separate active whole-body research, with unresolved W6 gates |

Retired branch heads are retained by annotated `archive/2026-10-29/...` tags. Do not merge
unfinished research solely to remove a branch. Runtime/kitchen integration is currently
blocked by the existing `dream/skills/g1_reach.py` import of `mujoco`, which violates the
simulator-isolation test. The invariant must not be weakened. W6 is separately blocked by
its development review. Work-in-progress commits preserve work; they do not certify it.

## Setup and focused checks

Core CPU environment (Python 3.10+; use the project environment, not an unrelated system Python):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,demo,locomotion]"
.venv/Scripts/python.exe -m pytest tests/test_isolation.py -q
```

MuJoCo is pinned to 3.16.0. Keep Isaac and VLM dependencies in their separately locked
runtimes. Do not resolve them into the core CPU environment. Inspect `pyproject.toml` for
optional scene-graph, SLAM, model and dataset extras. Preserve upstream licences/provenance.

On the interface/runtime branches:

```powershell
.venv/Scripts/python.exe -m experiments.phase1_interface --help
cd web
npm ci
npm run dev
```

Use the backend's protected local token; do not commit tokens, credentials or checkpoints.
The structured provider is explicitly labelled as having no language model. Select a configured
VLM for language interpretation. Model selection never grants raw joint control.

Runtime audit reproduction (CPU; not a new model call or physics evaluation):

```powershell
.venv/Scripts/python.exe -m experiments.audit_mobile_inspection --saved results/mobile_inspection_semantics/saved_failure.json --out <new-audit-output.json>
```

The historical fast suite exceeds the intended 60-second target. Run focused tests during
changes; complete the required integration/physical checks at a milestone. Never use a mock
inference result as evidence of a genuine VLM demonstration.

## Cloud lifecycle and results

DREAM cloud jobs use the scoped `cloud/` tooling in project `nerva-adityapersonal` with
estimated job/cumulative cost guards, immutable source snapshots and machine-enforced VM
expiry. Actual billed cost has not been established; estimates must not be labelled bills.
The latest Isaac session's scoped compute/network/identity cleanup returned no errors.
The initial stop rejected a missing root result manifest; recovery independently verified all
66 saved output objects against GCS MD5/size metadata and retained local SHA-256 hashes before
deleting the bucket (absence confirmed by a 404). The reconciled runtime-based estimate is
US$5.30, against the original US$7.65 maximum-lifetime estimate; actual billing is unavailable.
`results/cloud_isaac_20261029_cleanup` records the recovery and its limits. The post-audit also
lists a separate older DREAM router/network; this is not a claim that all historical project
resources have been removed. No fresh paid evaluation was launched during this consolidation.

Weights, caches, credentials and large raw recordings remain outside Git/OneDrive, normally
under `%LOCALAPPDATA%/DREAM`. Compact evidence, source and result metadata are committed.
The supplied small portfolio recording is maintained in the separate Portfolio repository.

## Documentation and preserved history

This README is the maintained status/setup entry point. First-party handoff prompts,
transcripts and duplicate status documents were consolidated and preserved verbatim in
`archive/documentation-runtime-20261029.zip`, including this branch's previous full README.
Licence/provenance files, `AGENTS.md`, its `CLAUDE.md` alias, and original experiment records
are retained. The whole-body scratch copy is archived separately rather than left as another
working source tree. Archives are historical evidence, not current instructions.

A pre-consolidation backup contains Git bundles for every branch, staged/unstaged binary
patches, untracked-file ZIPs and a SHA-256 inventory. Its location is recorded outside Git at
`%LOCALAPPDATA%/DREAM/archives/latest-consolidation.txt`. Credential-pattern scanning of the
pending material found no matches; private runtime credentials remain outside these commits.
