# DREAM: strategic audit and plan, 10 November 2026

Task 2026-11-10-001. Written for the owner and for every agent that works on DREAM next.

**Status of this document.** It is a proposal until the owner adopts it. Once adopted, section 4
(the ladder) replaces the "Current" list in `AGENTS.md` section 8, and section 6 lists the other
contract edits it needs. Nothing in `AGENTS.md` or `README.md` was changed by this task.

**How an agent uses it.** Read sections 1 and 3, find the lowest rung in section 4 that is not
yet claimed, and work only on that rung. Section 5 lists work that is stopped. If a rung's kill
criterion is reached, stop and put the decision to the owner; do not open a new line of work.

Evidence for every number is in `README.md`, `Reports/2026-11-09/001-isaac-semantic-research/`,
`results/`, the `claude/wholebody-autonomy` branch, or read-only git commands run on 10 November.
Timeline dates come from commit-message searches and are approximate; commit counts are exact.

---

## 1. The finding in one page

DREAM's end goal is a humanoid that takes an unfamiliar physical goal and accomplishes it with
its own body. After 20 days and 464 commits, **the G1 has never picked up an object with its own
hands under physics.** Every G1 "grasp" so far is a scripted test on a fixed pelvis, a MuJoCo
`connect` constraint, or an Isaac `FixedJoint`. The G1 walks and looks; the Panda manipulates; a
third fixed-pelvis G1 moves its fingers. These are three robots behind three interfaces that do
not compose.

Progress stalled because the work moved sideways instead of up:

1. **The bottleneck is whole-body manipulation, and nothing has advanced it since 6 November.**
   The reach work (W6) failed two development gates and was parked unmerged. The grasp stage
   (W7) was never started. The last four days went to object search, which was already the
   strongest part of the system.
2. **The last week's research question had no room to succeed.** Semantic search was measured
   against DREAM's own frontier explorer in homes small enough that a sweep finds almost
   everything. Result: no speed-up in two separate evaluations.
3. **The task layer cannot be general as built.** Goals are a closed set of seven hand-written
   types, each with its own verifier and planner code. A new kind of task needs new code.
4. **Process is larger than capability.** About two thirds of commits are evaluation
   bookkeeping, infrastructure, apps or documentation. Gates were written for capabilities that
   did not exist yet, some on samples too small to decide.
5. **Work is fragmented.** `main` is three days stale. Unmerged work sits on about ten branches
   and twelve worktrees, including 5,363 lines of whole-body code that the next step needs.

The plan: stop the sideways work, put one robot on one trunk, and climb a ladder whose first
rung is the G1 lifting an object with its own fingers. Each rung is a physical capability a
person can watch, measured once properly, and each one is needed by the rung above it.

---

## 2. Diagnosis, with evidence

### 2.1 What the G1 can and cannot do

| Capability | State | Evidence |
|---|---|---|
| Walk, stand, resist pushes | Works. Pretrained NVIDIA Decoupled WBC, legs only | `results/g1_locomotion`: 0 falls, 30/30 go-to |
| Explore a room and find an object by looking | Works, oracle-assisted odometry and detector | H4: 14/15, 0 falls |
| Move an arm while balancing | Code exists (right-arm IK over the WBC's PD arm targets); no validated result | `dream/robot/g1_assisted_demo.py` |
| Reach a handle at 0.35 / 0.75 / 1.10 m | Touch only. Dev gates failed, unmerged | W6: 31/38 feasible; off-band 2/3; refusal 19/22 |
| Close fingers on an object, free-standing | **Never attempted** | W7 not started |
| Grasp with fingers, fixed pelvis | Scripted test only, no perception | `results/g1_dex3_checkpoint` |
| Carry while walking | **Never attempted** | Arms are PD-held in every walking result |
| Open a drawer or door | Only through a `connect` constraint | `dream/robot/g1_assisted_demo.py` |
| Discovery (probing a mechanism) | Panda and gantry only | `dream/tools/discovery.py` |

Of the 10 PASS rows in the README status table, 1 is the G1 doing something physical (walking
and looking), 2 are the Panda, 3 are apps and infrastructure, and 4 are evaluation-integrity
mechanisms. Panda and gantry work was last touched around 26 October.

### 2.2 The contract's own priority list skips the missing capability

`AGENTS.md` section 8 orders the current milestone as A mobility, B exploration, C investigation
of an object, D composition ("navigate, discover, manipulate, carry back"), E memory. No step
builds manipulation on the G1. D assumes it. So agents following the list finished A, deepened B
for a week, and had no instruction that led to the thing D depends on.

### 2.3 Three embodiments, three interfaces

`dream/types.py` defines `Embodiment` (Panda: TCP, gripper, compliance), `DexterousEmbodiment`
(fixed-pelvis G1: palms, fingers) and `MobileBase` (walking G1: velocity, odometry, camera).
Each has its own toolbox: `Toolbox`, `DexterousToolbox`, `MobileToolbox`. Discovery, grasping
and placing run only through `Embodiment`. Navigation and search run only through `MobileBase`.
No object in the repository can both walk to a thing and act on it.

### 2.4 The task layer is closed

- `dream/tools/goals.py`: `Goal = InRegion | OnTop | Inspect | GoToPlace | GoToObserved |
  FindObject | UnsupportedGoal`. `Inspect` accepts one target, "cabinet". String goals are
  parsed by regular expressions.
- `dream/runtime/agent.py`: `RuleBasedPlanner` is a hand-written decision procedure for the
  Panda's two placement goals.
- `dream/runtime/mobile_model.py`: the language model chooses among a handful of tools bound to
  observed candidates. It selects; it does not compose.

The goal-consistency work (immutable request record, binding check, completion gate) is sound
and worth keeping. But it protects a vocabulary that is too small to test generality on. The
"bounded task-program composition" in the architecture table is still only planned.

### 2.5 The last week, measured

Semantic object search, 8 to 10 November:

- Held-out primary comparison against the existing explorer: median paired difference 0 s,
  p = 0.73 (gate G3 failed).
- K1, 576 episodes: knowledge arms found 61, 61 and 62 targets against 59 for the frontier
  baseline; speed gate K-G2 failed with Holm p = 1.0.
- Knowledge changed 6.7% (ConceptNet) and 12.1% (Qwen) of decisions; median time to first
  target view was 85 to 86 s in every arm.
- The "unexplored room" addition did not help on development homes.
- The Isaac half: 13 cloud run records over four or five sessions, roughly US$25 to 30
  estimated, almost all spent on asset loading, PhysX crashes and the robot falling at spawn.
  No object search has run in a composed Isaac home, and no learned detector has run in any
  search loop.

These are honest negative results and should stay recorded. They also show the question was
badly placed: all of this used an oracle-assisted detector on a kinematic base, and its best
possible outcome was a robot that looks at the right thing sooner. It could not have produced a
robot that does more.

The one component that did matter is the multi-view verifier (single-view acceptance made 17
false claims in 50). It is done.

### 2.6 Process outweighs capability

- Lines: `experiments/` 54,186; all of `dream/` about 21,500; `tests/` 14,091; `results/` holds
  3,964 files.
- Commit messages (keyword buckets, overlapping, indicative): evaluation and gates 28%,
  infrastructure 24%, apps and streaming 16%, documentation and reports 13%.
- The Reports system took 38 commits in its first two days. The open task 2026-11-09-001 has 23
  run records and a `REPORT.md` that is still an unfilled template.
- Gates on samples that cannot decide: W6's off-band gate was 2 of 3 against a 0.70 threshold
  (Wilson 95% interval 0.21 to 0.94). Its refusal gate was 19 of 22 against 0.90; one episode
  decides it. The branch was then left unmerged with no cause established.
- Held-out seeds are consumed quickly. Search seeds H1 to H4 and 9200 to 9205, room seeds, and
  the W5 held-out set are all now development data.

Rigour applied before a capability exists produces well-documented failures. The rules
in `AGENTS.md` section 4 are right about claims. They were being applied to bring-up.

### 2.7 Fragmentation

- `main` is at the 7 November closeout. The current branch is 90 commits ahead of it.
- `claude/wholebody-autonomy` (35 commits, last touched 6 November) holds the posture planner,
  the supervisor, the unified 29+14 DoF G1 with Dex3 fingers, the reach tools and the SONIC
  port: 5,363 added lines in `dream/` that exist nowhere else. It was not merged because its
  gates failed, so code and verdict were parked together.
- Five agent worktrees, two superseded semantic branches, two history-rewrite backups, three
  diagnostic branches. The drafted K2 config exists only inside worktrees.
- Two simulators. Isaac costs a cloud session per attempt and has mostly returned asset
  problems. MuJoCo runs locally and already hosts every manipulation result.

### 2.8 Smaller issues

- Learned motor policies were tried only zero-shot on mismatched contracts (MolmoAct2 on the
  Panda: 0/1; GR00T: open-loop only, no verified G1 hand contract). Nothing has ever been
  trained or fine-tuned in this project.
- Odometry and tracking are oracle-assisted almost everywhere. This is acceptable for now and
  is labelled; it is not the binding constraint.
- `AGENTS.md` section 2 is stale against the README (it still describes milestone 11 as in
  progress and no Isaac embodiment as validated).
- The web app (24 commits), the Android app (5) and camera streaming are not on the path to
  the end goal. Apps and streaming appear in about 16% of commit messages.

---

## 3. Strategic decisions

**D1. One robot.** All new capability is built on the free-standing G1 with Dex3 hands, in one
model, behind one interface and one toolbox. The Panda and the gantry stay as validated
testbeds and regression checks. Nothing new is built on them.

**D2. One simulator for development.** MuJoCo, locally, CPU. Isaac is used for one
cross-simulator replication per claimed rung from R3 upward, and not before. No Isaac home,
asset or scene work until then.

**D3. One trunk.** A single integration branch that becomes `main`. Code merges when tests,
ruff and the isolation guard pass. A failed gate is recorded in the README; it does not keep
code off the trunk.

**D4. Classical motor layer first, logging demonstrations from the start.** R1 to R3 use what the
repository already has: the pretrained WBC for the legs, inverse kinematics and the posture
planner for the arms, contact-based finger closure. Reasons: the parts exist, the laptop has no
GPU, and the zero-shot learned policies failed on embodiment and camera contracts. From R1
onward every episode is saved as synchronised observations and actions in one fixed schema.
When the classical grasp ceiling has been measured, that data is the basis for training or
fine-tuning a policy on DREAM's own contract (R7). A learned policy is adopted only if it beats
the classical one on held-out objects.

**D5. Nominal conditions before extremes.** W5 and W6 started with reaches at 0.35 m and 1.10 m,
where torque limits and crouching dominate (W5 attributed 14 of its feasible losses to
supervisor stops, mostly torque, and 14 more to planner refusals). R1 starts at table height in the arm's comfortable workspace. The height
envelope returns in R7.

**D6. Generality is measured by unseen task compositions, on the real robot stack.** The claim
DREAM is aiming at is "a task nobody coded for was completed". R4 measures exactly that. No
other result may be described as progress toward general-purpose behaviour.

**D7. Two stages of evidence.** Bring-up: development seeds, short notes, no gates, no held-out
data. Claim: one frozen config, one held-out run, sized so that it can decide. A rung moves to
the claim stage only when its development success rate is already near the target.

---

## 4. The ladder

Each rung has a deliverable, a development target that must be met before freezing, a claim
gate, and a kill criterion. The gate thresholds below are proposals; the agent that freezes a
rung's config may adjust them once, with a reason, before any held-out run. In every rung:
0 falls, 0 false success claims, and the evaluator asserts that no equality, weld or attachment
constraint is active.

"Session" below means one working session of an agent, roughly half a day of work.

### R0. One trunk, one robot model (consolidation)

No new capability. Without it, R1 has nothing to build on.

1. Close task 2026-11-09-001: fill its `REPORT.md`, status `partially_completed`, main finding
   as in 2.5. Record the detector comparison (session 5) as a run or state that it has no
   record. Copy the K2 and Isaac homes configs out of the worktrees into the branch, marked
   "drafted, not run".
2. Create `integration/trunk` from the current branch. Merge `claude/wholebody-autonomy` into
   it as code, with the README stating "W6 development gates failed; not validated". If the
   merge is unworkable, port W1 (height and torso channels), W2 (unified Dex3 model and
   actuator bridge) and W3 (posture planner, supervisor) and say what was left behind.
3. Fast suite, slow suite, isolation guard and ruff green on the merged tree.
4. Remove merged and superseded worktrees and branches after confirming each is contained in
   the trunk. Keep the diagnostic branches listed as "never merge" in the README, and the
   pinned worktree used for frozen held-out replays.
5. Put the ladder scorecard (section 4.9) at the top of the README.

Done when: one branch contains everything, the suites pass, and `git worktree list` shows the
main checkout plus any deliberately pinned evaluation worktree.
Budget: 2 sessions. Owner approval is needed before `main` is moved and before branches are
deleted.

### R1. Stand and grasp

**The G1, free-standing under the WBC, picks an object off a table with its own fingers and
holds it.**

- Scene: a table with its surface between 0.70 and 0.90 m, one object within reach after base
  placement. The robot starts 1 to 2 m away and chooses where to stand.
- Objects, development set: box, cylinder, sphere, bottle, mug-like; graspable width 4 to 8 cm;
  mass 0.05 to 0.30 kg; a stated friction range.
- Build: one `Humanoid` embodiment interface (base velocity, both arms, both hands, head
  camera, proprioception, fingertip forces) and one `HumanoidToolbox`; a stance chooser from
  arm reachability; a grasp chooser from observed object geometry (power grasp or pinch);
  finger closure that stops on measured contact; the supervisor from W3 for torque and tilt.
- Perception: the oracle-assisted tracker is allowed and labelled in this rung.
- The agent's success claim comes from its own sensing: fingertip forces, finger closure
  short of fully closed, and the object seen above the table. The evaluator checks ground
  truth: object at least 0.10 m above the surface for 3 s, touched only by hand geometry.
- Development target before freezing: at least 70% over 50 development episodes.
- Claim gate: 60 held-out episodes (new seeds; object sizes and masses shifted; two shapes not
  in the development set). At least 75% on in-distribution shapes, reported separately for the
  unseen shapes with no threshold.
- Kill criterion: if development success is under 40% after 5 sessions, stop and bring the
  owner the failure attribution and three options: a simpler hand (labelled an environment
  simplification), a single-shape scope, or moving the learned-policy work (R7) forward.

### R2. Pick, carry, place

**The G1 picks an object, walks at least 3 m holding it, places it on a second surface and lets
go.**

- Adds: holding the grasp while the WBC walks, a slip check during the carry (fingertip force
  and camera), placing on a surface from measured geometry, release, and a verified result.
- Recovery from the loop, not from special cases: a dropped object is seen on re-observation
  and picked up again, within an attempt budget.
- Perception for the claim: object pose from the head RGB-D camera (`dream/perception/`), not
  the tracker. Odometry stays oracle-assisted and labelled.
- Development target: at least 60% over 50 episodes. Claim gate: 60 held-out episodes, at least
  65%, and every drop either recovered or correctly reported.
- Kill criterion: under 30% after 5 sessions. Bring the owner the attribution between grasp
  loss while walking, placement, and camera pose error.

### R3. Fetch from language in an unknown room

**"Bring the red bottle to the table by the door": the G1 searches a room it has not seen,
picks, carries, places, and reports the truth about what happened.**

- Composes existing parts: the request record and binding check (`dream/tools/intent.py`),
  search and verification (`dream/runtime/semantic_search.py`, `dream/semantics/verify.py`),
  navigation, and R1 and R2. Objects sit on open surfaces, not inside containers.
- Includes absent targets, unreachable targets and unsupported requests; refusals are scored.
- Comparison arms: the same stack with oracle perception and a constraint-assisted grasp as a
  labelled upper reference, and the same stack with re-planning after failures disabled.
- Development target: at least 50% end to end. Claim gate: 60 held-out episodes over new room
  layouts; at least 50% completion on feasible requests; at least 90% correct refusal on at
  least 20 infeasible ones.
- After the MuJoCo claim: one Isaac replication on a single simple room. A different result
  there is reported, not tuned away.
- This rung is the first demonstration of whole-body autonomy. It gets a recorded video.

### R4. Unseen task compositions

**Tasks nobody wrote code for, completed without changing code.** This is the generality
measurement.

- Replace the closed `Goal` union for the humanoid with goals that are conjunctions of
  predicates over observed entities: `on(a, b)`, `in(a, region)`, `at(robot, place)`,
  `holding(a)`, `clear(a)`, `open(c)`, `visible(a)`. Each predicate has one verifier that reads
  observations only.
- Each skill declares its preconditions and effects in the same predicates. A deterministic
  planner searches over skills to reach the goal and replans after every action.
- The language model translates the request into a predicate goal grounded in observed
  entities. It does not choose motions or sequence skills. The request record and binding
  check apply unchanged. A deterministic stub replaces it in tests.
- Evaluation: task templates are generated from a grammar over the predicates. Development
  uses some combinations; the sealed test set uses combinations and lengths that never occur
  in development (for example two-object goals, ordering constraints, clearing an obstacle
  first, moving between rooms, and impossible requests). The owner or a separate session
  writes the sealed set, and development agents are told not to read it.
- Report: completion on unseen compositions against completion on seen ones, correct
  refusals, false claims. The gap between seen and unseen is the result, whatever it is.
- Claim gate: at least 12 unseen templates, 5 episodes each. No completion threshold is fixed
  in advance for unseen compositions; 0 false claims is required.

### R5. Discovery on the humanoid

**"Get the bottle out of the closed drawer": the G1 finds out how a container opens by probing
it with its own hand, opens it, and fetches from inside.**

- Give the `Humanoid` interface the compliant-probe operations that `dream/agent/` and
  `dream/skills/articulation.py` need, so the belief, the predictive model and the
  information-gain probe selection run unchanged on the G1.
- The handle is held by fingers. The `connect` constraint in `g1_assisted_demo.py` is retired
  to a labelled baseline.
- `open(c)` becomes a predicate the R4 planner can require, so discovery is invoked because a
  precondition fails, not because a script says so.
- Mechanisms: drawer, hinged door, fixed panel, and the out-of-class case, which today fails
  silently (0 of 16 flagged) and must be reported as such.
- This rung unifies DREAM's original scientific contribution with the whole-body robot. It
  also restores the scientific question in `AGENTS.md` section 1 as an experiment on the G1:
  the same fetch tasks with and without active discovery.

### R6. Experience that helps

**A second visit is measurably better than the first, and knowledge from one room helps in
another.**

- Uses `dream/memory/`. Two measurements against the same system with memory disabled:
  repeated tasks in the same home (time and exploration saved), and a mechanism learned on one
  container reused on a similar one elsewhere (probes saved, with withdrawal when it is wrong).
- The earlier home-memory result was worse when an object had moved; that case stays in the
  evaluation.

### R7. Remove the supports and widen the envelope

Each item starts when the rung analysis above shows it is the limiting factor, or after R5.

- A learned open-vocabulary detector in the loop in place of the oracle-assisted one (the
  detector comparison from task 2026-11-09-001 resumes here).
- Odometry from the robot's own sensors.
- Reaching at low and high targets (the W6 work, with gates sized to decide).
- A learned grasp or manipulation policy trained on the demonstrations logged since R1,
  adopted only if it beats the classical one on held-out objects.
- Larger and more varied homes, including Isaac, once fetch works in simple ones.

### 4.9 Scorecard for the top of the README

| Rung | Capability | Stage | Result |
|---|---|---|---|
| R0 | One trunk, one robot model | not started | |
| R1 | Stand and grasp with own fingers | not started | |
| R2 | Pick, carry, place | not started | |
| R3 | Fetch from language in an unknown room | not started | |
| R4 | Unseen task compositions | not started | |
| R5 | Discovery on the humanoid | not started | |
| R6 | Experience that helps | not started | |
| R7 | Supports removed, envelope widened | not started | |

Stage is one of: not started, bring-up, development target met, claimed (with the held-out
numbers), failed (with the numbers and the attribution).

---

## 5. Work that stops

Stopped until the owner asks for it by name, or until the rung named here is reached:

| Work | Why | Resumes |
|---|---|---|
| Semantic search efficiency, K2, knowledge-source comparisons, kappa calibration | Two evaluations show no effect; the homes cannot show one | R7, in homes large enough to matter |
| Composed Isaac homes, asset validation, `l_island`, office scene inventory | Infrastructure for a capability that does not exist yet | R3 replication, R7 |
| Learned detector in the search loop | Not the limiting factor for R1 to R3 | R7 |
| Panda and gantry features, camera compartment opening (11/20) | Not the target embodiment | Superseded by R5 |
| W6 gate repair at extreme heights | Sequencing: nominal grasp first | R7 |
| MolmoAct2, GR00T, VoLoAgent, SONIC port | Contract mismatches; no path to R1 | R7 decision |
| Web app, Android app, camera streaming, phone API | Not on the path | Owner request |
| Renderer comparisons, navigation follower comparisons | No decision depends on them | Never, unless a rung fails on them |

Existing results and their tests stay. Nothing validated is deleted.

---

## 6. Process changes that need the owner's approval

These change `AGENTS.md`, which is the owner's contract. They are listed, not applied.

1. **Section 8:** replace the "Current" list A to E with the ladder R0 to R7 and the stop list.
2. **Section 2:** bring "What exists today" in line with the README, including the plain
   statement that the G1 has not manipulated anything.
3. **Bring-up and claim stages (D7).** In bring-up: no pre-registered gates, no held-out seeds,
   and progress recorded as dated notes on the task. In the claim stage: sections 4, 11 and 12
   apply in full.
4. **Gate sizing.** No threshold gate on fewer than 20 episodes in a cell. Every gate states
   the interval it can resolve.
5. **Merge rule (D3).** Passing tests, ruff and isolation is the bar for the trunk. Gate results
   go in the README.
6. **Reports during bring-up.** One task per prompt stays. Proposed: a bring-up iteration loop
   is one run record with a table of iterations, and a separate run record is required for
   every run whose numbers are cited in the README. This relaxes "ten experiments mean ten run
   records" for bring-up only; the owner may prefer to keep the rule as it is.
7. **Session rule.** Every task's report states, in section A, which rung it advances. Work on
   no rung needs an explicit owner request.
8. **Episode logging (D4).** One fixed schema for observations and actions, written by the
   humanoid evaluation runner from R1 onward, stored outside Git and OneDrive.

---

## 7. Risks and the decisions that will come back to the owner

| Risk | Sign | Decision it forces |
|---|---|---|
| Dex3 finger grasps are unreliable in MuJoCo contact while the robot balances | R1 development under 40% after 5 sessions | Simpler hand, narrower object set, or learned policy first |
| Arm torque limits stop reaches even at table height | Supervisor stops dominate R1 losses | Lighter objects and closer stance, or a whole-body controller that uses the waist and legs (SONIC) |
| Carrying disturbs the walking policy | Falls or drops in R2 | Limit carry mass and arm pose, or revisit the controller |
| Merging the whole-body branch is costly | R0 exceeds 2 sessions | Port W1 to W3 only |
| The predicate planner solves seen compositions and fails unseen ones | Large seen-to-unseen gap in R4 | That is the research result; decide whether to add learned components |
| Agents drift back to benchmark work | A task with no rung in section A | Enforce the session rule |
| Oracle odometry hides a navigation failure | Fetch failures attributed to base placement | Pull sensor odometry forward from R7 |

What this plan does not promise: general-purpose intelligence. It reaches the first point at
which generality can be measured on a humanoid that acts (R4), and the first point at which
DREAM's discovery idea is tested on that humanoid (R5). Everything is simulation.

---

## 8. The next three sessions

1. **Close and consolidate (R0, part 1).** Finish and close task 2026-11-09-001. Create
   `integration/trunk`. Attempt the whole-body merge; report conflicts and the suite results.
   No experiments.
2. **Finish R0 and start R1.** Clean up worktrees and branches with the owner's approval. Add
   the scorecard. Define the `Humanoid` interface in `dream/types.py` from the existing three
   protocols. Stand the unified Dex3 G1 at a table under the WBC and close the fingers on a
   box: first physical grasp attempt, development seed, recorded video.
3. **R1 bring-up.** Stance chooser, grasp chooser, contact-stopped closure, the success sensing
   and the evaluator. Run the 50-episode development set, attribute every failure using the
   categories in `AGENTS.md` section 3.5, and iterate on the largest category only.

Cloud is not needed for any of these. Run MuJoCo locally with at most three workers, redirect
background output to log files, and check free disk space (27 GB on 10 November) before each
batch.
