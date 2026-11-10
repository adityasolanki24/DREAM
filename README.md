# DREAM

Research toward general-purpose embodied intelligence: receive an unfamiliar physical goal,
observe, investigate, compose available skills, verify the outcome, and recognise limits.
**General-purpose task completion is the objective, not an established capability.**
`AGENTS.md` is the engineering and behavioural contract; `CLAUDE.md` aliases it.

This README is the research report of 9 November 2026. It covers the object-search and
goal-consistency work on `claude/semantic-goal-search-20261108`, on top of the closeout of 7
November. For the closeout, every item, with its success criteria, baseline, repro command, cost
and evidence, is in
[`results/research_closeout_20261106/inventory.json`](results/research_closeout_20261106/inventory.json).
The previous README is preserved in `archive/documentation-closeout-20261107.zip`.

## G1 roadmap

The current priority is a free-standing G1 grasp under contact physics. The
[roadmap](Reports/2026-11-10/001-strategic-audit-and-plan/PLAN.md) sequences grasp,
carry and fetch before broader composition and discovery. AGENTS.md records the adopted
sequencing and evidence rules.

| Rung | Capability | Status |
|---|---|---|
| R0 | G1 model, controller and safety on one integration branch | Integration checks passed |
| R1 | Free-standing contact grasp and lift | Fixed-stance box grasp verified in development |
| R2 | Carry and place | Pending |
| R3 | Fetch from language | Pending |
| R4 | Held-out task compositions | Pending |
| R5 | Active discovery on the G1 | Pending |
| R6 | Experience that improves a later task | Pending |
| R7 | Broader sensing and control envelope | Pending |

The Dex3 actuator path is opt-in. W6 remains a failed development evaluation.
For the fixed-stance grasp diagnostic, install the `wholebody` and `locomotion` extras
and use the cached WBC weights. See the
[bring-up report](Reports/2026-11-10/003-r0-r1-start/REPORT.md) for the commands and
development outcomes. In the current fixed-stance development evaluation, 3 of 6
box fixtures passed the independent 10 cm, 3 s hold gate. The same controller used
observed wrist and box geometry to select and correct the approach. The box tracker
and odometry are oracle-assisted. One case refused closure and two slipped during hold.
The broader R1 gate remains pending.
[Results and configurations](results/g1_r1_tabletop_dev_20261110/summary.json) and
[development report](Reports/2026-11-10/008-r1-clearance-resume/REPORT.md).

An experimental selector (`contact_geometry_search: true` in the fixture config) ranks
pinch poses from predicted contact geometry. It passed 4 of 6 matched cube fixtures,
including one previous hold failure, with all previous passes retained. The rectangular
box, cylinder and sphere reached closure but did not pass the hold gate.
[Matched results](results/g1_r1_contact_rank_dev_20261110/summary.json) and
[retention report](Reports/2026-11-10/009-r1-contact-retention/REPORT.md).

## Executive summary

**New on 8-9 November**, all in simulation:

- **Request-to-goal substitution is closed in the tested paths.** The original request is kept as
  an immutable record. A goal that doesn't satisfy it is refused, and success is accepted only
  when an independent check confirms the original request. The frozen 7 November outputs that
  produced false claims are refused in replay. On held-out MuJoCo homes, a planner made to claim
  "reached the kitchen" produced 7 false claims in 40 without the record and 0 with it.
- **Semantic object search, held-out.** The proposed method found the target in 83 of 87 H1
  homes, 31 of 32 larger shifted homes and 14 of 15 G1-controller homes. It refused every
  manipulation request without moving. It made **one false success claim** in 216 held-out
  episodes: a shower claimed as the bathtub, in a larger home with no bathtub. That fails the
  pre-registered zero-false-claim gate (G0). No baseline made one.
- **Verifier fix, confirmed on fresh seeds.** A confidence check on label purity was added and confirmed on 384 fresh-seed episodes:
  - it made 0 false claims in 192 episodes, against 1 for the frozen rule;
  - it found the target in 94 of 96 homes, against 93;
  - it is now the verifier default;
  - the held-out record above is not rescored.
- **No efficiency gain.** On the pre-registered primary comparison it was no faster than DREAM's
  existing frontier explorer: median paired difference 0 s, p = 0.73. Removing its semantic
  components changed nothing measurable. **G3 is not met.**
- **What does matter is the verifier.** Accepting single sightings produced 17 false claims in 50
  episodes, against 0 for the full verifier.
- **The knowledge source is weak.** ConceptNet never let the robot recognise a laundry or storage
  room. Qwen's raw answers agree with the benchmark tables more often (23 vs 17 of 27 classes);
  they are not yet used.
- **The Isaac kitchens have no refrigerator.** The appliance assets they reference return HTTP 404,
  so the 7 November refrigerator task had no right answer except "not found".

**What DREAM genuinely achieves today**, all in simulation:

- A Panda arm that discovers how a compartment opens through force-limited probing, and opens it:
  drawer 27/30, lid 27/30, fixed boxes declined 30/30, with no false claims. Object tracking is
  oracle-assisted.
- A Unitree G1 in MuJoCo that searches an unknown room from RGB-D using a pretrained walking
  controller, and refuses when no target exists. After the one fix made in this closeout it scores
  20/20, 8/10 and 6/6 on the historical seeds under the original renderer. Odometry is
  oracle-assisted, and no unseen seeds remain.
- A genuine Qwen model driving an Isaac Sim G1 through typed tools: it inspected a cabinet in 3/3
  frozen cases with no falls.
- Independent verification of the bound goal against world state. Room and compartment evaluations
  recorded no false claims. Its limit shows in the Qwen result below: it checks the goal the system
  bound, not the one the person asked for.

**What it does not achieve:**

- The 7 November Qwen evaluation **failed its gates**. Asked to walk to a refrigerator or pick up a
  kettle, the system substituted a goal it could do (go to the entrance, inspect a cabinet),
  verified that, and claimed success. The record stands. The fix is above. In the Isaac
  re-test on three held-out kitchens, the old path did this again in 4 of 6 cases and the new
  path did not do it (0/9). Those gates are incomplete, because a fourth kitchen never loaded.
  See "Qwen on the Isaac G1".
- Room search on fresh seeds **failed its frozen gates** (15/20, 4/10). The later fix does not
  change that record.
- Compartment opening from camera perception **failed** (11/20). MolmoAct2 **did not complete the
  task** (0/1). Whole-body W6 is **not ready**. 10 fps camera delivery over the remote link was not
  demonstrated.
- Nothing has run on a physical robot. Nothing here establishes general-purpose task completion.

## Status by area

| Area | Result | Status |
|---|---|---|
| Original-goal record (goal consistency) | 7 Nov outputs refused in replay; fault injection 7/40 false claims without it, 0/40 with it; L refused 10/10 | **PASS** in tested paths |
| Object search, held-out H1 (kinematic) | Proposed 83/87, 0/120 false claims, 0/87 false absence | **PASS** G1, G2, G4; G0 met in H1 but fails overall (H2) |
| Object search, efficiency vs DREAM explorer | 46 vs 44 of 50, median paired difference 0 s, p = 0.73 | **FAIL** G3: no demonstrated improvement |
| Object search, G1 whole-body controller (H4) | 14/15 vs A 13/15; 0/24 false claims; 0 falls | **PASS** G8 |
| Object search, shifted homes and new targets (H2) | Proposed 31/32 vs A 29/32; **1/48 false claim** (shower claimed as bathtub) | **FAIL** G0, G7 |
| Object search, degraded detector (H3) | Proposed 14/16 vs A 11/16; 0/24 false claims; single-view 10/16 false claims | Reported (G6) |
| Verifier confidence fix, fresh seeds (V1) | 0/192 false claims vs 1/192 for the frozen rule; found 94 vs 93 of 96 | **PASS** V1, V2; now the default |
| Room inference from objects | Top-1 61%; labelled 62/70; laundry and storage 0% | Measured, heuristic |
| Qwen on Isaac G1, semantic re-test, held-out kitchens | New path 0/9 false successes, 4/4 unsupported requests refused; old path 4/6 false successes; 0 falls; `l_island` never loaded | Gates **INCOMPLETE** |
| Qwen on Isaac G1, frozen matrix | 3 false claims, T3 1/3, T4 2/3; falls 0, T1 3/3, T2 3/3 | **FAIL** |
| Same matrix after the viewpoint fix | Same gate results; T2 2/3; hand contact in 2 of 15 cases | **FAIL**, development only |
| Whole-body W6 | Off-band 0.667 (0.70), refusal 0.864 (0.90); held-out not run | **FAIL**, not merged |
| Panda pick and place | Oracle 50/50; camera 39/40 | **PASS** |
| Compartment, oracle-assisted | 27/30, 27/30, 30/30, 0 false claims | **PASS** |
| Compartment, camera | Openable 11/20 (0.8); fixed 10/10; 0 false claims | **FAIL** |
| Out-of-class mechanisms | 0/16, 0 false claims, 0/16 flagged | Verified negative |
| MolmoAct2 | Commands tracked 150/150; never reached the object or closed the gripper | Verified negative |
| GR00T | Open-loop inference only | **BLOCKED** |
| Room search, frozen fresh seeds | 15/20 (0.9), 4/10 (0.7), 10/10; 0 false claims, 0 falls | **FAIL** |
| Room search, viewpoint fix | Retention rule met in all arms; seen seeds | **PASS** as a fix, not a validation |
| Gain vs nearest frontier | 7 divergent decisions, under the required 10 | NOT EVALUATED |
| Navigation, RPP vs classic | v1 25/30 vs 29/30; v2 28/30 each | PARTIAL, no superiority |
| Home memory | Recall 5/5, 34.8 vs 69.0 s; worse when an object moved | PARTIAL |
| Model-led recovery | n=1 ineffective; no comparison arm | NOT EVALUATED |
| Camera, loopback | Push median ~17 ms at 5/10/15 fps | **PASS** |
| Camera, remote over IAP | Push 1.3-3.96 fps, median 177-195 ms; WHEP connects | PARTIAL |
| Provider comparison | ChatGPT 6/11, Qwen 5/11 on identical frames | Descriptive only |
| VoLoAgent | Interface contract and mock; no actuation | NOT EVALUATED |
| Web app | Types, 57/57 tests, production build | **PASS** |
| Android app | Unit tests and build | **PASS** |
| Android on OPPO A31 | No device attached | **BLOCKED** |

## Architecture

```
goal text -> GoalSpecification: immutable record of the original request
          -> provider (Qwen, ChatGPT, or the explicit grammar) proposes a grounding
          -> binding checked against the record; unsupported requests refused
          -> typed tools only: models never get raw joint control
          -> skills: probe, inspect, navigate, grasp, open, place
          -> controller: Panda joints, or NVIDIA Decoupled WBC on the G1
          -> simulator: MuJoCo or Isaac Sim, behind an isolation boundary
independent verifier: reads world state, not the agent's claim
completion gate: success stands only if the verdict satisfies the ORIGINAL request
```

Object search adds `dream/semantics`: object-to-room knowledge, room regions, a target belief
over what has not been seen, a multi-view verifier, and a scene graph that labels every edge
observed, verified, inferred, hypothesis, knowledge or remembered.

Separate parts: a Bayesian mechanism belief with information-gain probe selection; RGB-D occupancy,
Dijkstra and Regulated Pure Pursuit for navigation; SQLite home and experience memory; a task
scheduler and phone API; a Next.js web app and an Android app; a camera path with MJPEG push, JPEG
polling and an experimental NVENC, MediaMTX and WHEP publisher.

**Embodiments:** abstract gantry; Franka Panda in MuJoCo; Unitree G1 in MuJoCo; Unitree G1 in Isaac
Sim 6.2 with PhysX. The G1 uses a pretrained velocity-commanded motor controller with arms held.

## Results

Unless stated otherwise these are simulation, with oracle-assisted odometry or tracking.

### Object search and the original request

**Question.** Told "Find the refrigerator", can the robot use common knowledge of where things
are kept to find it sooner, and never report the kitchen, a cabinet or anything else in its
place?

**What was built** on `claude/semantic-goal-search-20261108`:

| Part | Code | What it does |
|---|---|---|
| Original-goal record | `dream/tools/intent.py`, `dream/runtime/agent.py` | Each request becomes an immutable, hashed `GoalSpecification`: target class, target kind, required outcome. A bound goal must satisfy it before the agent acts (`check_binding`). A success claim stands only if the binding holds and the independent verdict confirms the original request; otherwise it becomes `goal_substitution_rejected`. Unsupported requests are refused before moving. A model's grounding must use words from the request (`check_anchoring`). |
| Semantic map | `dream/perception/semantic.py` | Fuses detections into object tracks with per-label view counts, misses, and a coverage layer driven by range-dependent detection quality. |
| Knowledge | `dream/semantics/knowledge.py`, `third_party/household_knowledge` | Object-to-room association strengths from a ConceptNet 5.7 extract (CC BY-SA 4.0) with WordNet synonyms. These are heuristic weights, not probabilities. |
| Rooms | `dream/semantics/rooms.py` | Splits the occupancy grid into regions at doorways. Gives each region heuristic room-type weights from the objects seen in it, including an "unknown" type. A label is reported only with weight at least 0.5 and a margin of 0.2; otherwise the region is "ambiguous". |
| Target belief | `dream/semantics/belief.py` | Where the target could still be: occupied cells and nearby unknown cells, weighted by the region's room evidence times what has not yet been seen well. Remembered sightings add weight. Unknown pockets inside explored space are tracked, so the robot can tell "not seen yet" from "searched". |
| Planner | `dream/runtime/semantic_search.py` | Candidate views are DREAM's frontiers plus in-room inspection points. Each is scored U = gain + κ·detection − cost − γ·risk, with κ = 3 frozen. One choice, act, update, choose again. Every arm uses the same candidate verification. |
| Verifier | `dream/semantics/verify.py` | A track counts as the target only with at least 3 detections, at least 2 viewpoints 0.5 m apart, label purity at least 0.6, within 2.5 m, and misses at most 0.34 of support. |
| Bounded recovery | `dream/runtime/semantic_search.py` | Destinations near two failed `go_to` targets are skipped. A candidate gets at most 3 look attempts and is refuted at purity under 0.3 after 6 views. Destinations are never revisited. The time budget is enforced outside the planner. |
| Scene graph | `dream/semantics/graph.py`, `web/app/lib/sceneGraph.mjs` | Every edge carries a status: observed, verified, inferred, hypothesis, knowledge or remembered. A hypothesis never points at an object node. Layout and physical coordinates are kept apart. The existing web scene graph shows the claims. |

Stopping is typed: `target_found_verified`, `possible_target_unconfirmed`, `target_not_yet_found`
(unexplored pockets remain), `search_exhausted`, `target_unreachable`, `goal_unsupported`,
`safety_interruption` and `goal_substitution_rejected`. The fix is a **general algorithmic
correction** (Rule 3). The four frozen Qwen outputs behind the 7 November false claims still
bind wrongly without the record, and with it they are refused (`tests/test_intent.py`).

**Benchmark.** `dream/sim/home.py` generates multi-room MuJoCo homes, from 2×2 to 2×3 rooms
nominally and larger when shifted, to twelve scenario recipes:

| | Scenario | | Scenario |
|---|---|---|---|
| A | Target visible at the start | G | Target behind an occluder in its room |
| B | Target in its typical room, not visible | H | No typical room, no target |
| C | Typical room only partly observable | I | Stale memory: moved, removed or still valid |
| D | Target in an atypical room | J | No typical room; target in a plausible one |
| E | Typical room exists, target absent | K | Conflicting room evidence (studio, mismatched objects) |
| F | Look-alike confusers in another room | L | Manipulation request the system cannot do |

The generator checks each recipe on the built geometry, including visibility through the robot's
camera model. Scoring (`dream/eval/semantic_scoring.py`) reads the true scene, never the agent's
claim, and judges the **original request**:

- `success`: a claim on a true instance.
- `false_success`: any other claim.
- `correct_refusal`: no claim, and the target is absent or unsupported.
- `false_absence`: the robot says the target is not there, but it is.
- `missed`: an honest failure, unconfirmed or out of budget.

Arms differ only in how they choose where to look:

- **A**, gain frontier: DREAM's existing explorer.
- **B**, nearest frontier: classical exploration.
- **C**, room priority: covers regions in order of how strongly their inferred room type points to the target.
- **Proposed.**
- **Ablations**, each removing one part: semantic knowledge (uniform), room inference, information gain, the semantic term, memory, two-view verification, and goal consistency (fault injection).

**Protocol.** Development used runner seeds 100-103 with the refrigerator only. κ was chosen
there by a rule written beforehand: on dev3, κ = 3, 10 and 30 and Baseline A all found the target
in 33 of 34 homes with no false claims, and the tie-break by median time chose κ = 3 (51.5 s, the
same as A). So development showed **no advantage** for the semantic term before held-out data
was seen. The held-out config was then frozen at `9c19328`, with gates G0-G8
(`experiments/configs/semantic_search_heldout.json`), and run from a worktree pinned to that
commit:

- **H1:** held-out seeds 5000-5009, all scenarios.
- **H2:** a shifted generator and seven target classes never used in development.
- **H3:** a degraded detector.
- **H4:** the pretrained G1 whole-body controller instead of the kinematic surrogate.

Budget 600 s per episode.

**H1 main arms** (`results/semantic_search_heldout/H1_main`, 480 episodes; Wilson 95% intervals):

| Arm | Found, target present | False success claims | Correct refusal, absent | Unsupported refused | False absence | Median time to verified find | Mean path |
|---|---|---|---|---|---|---|---|
| **Proposed** | 83/87 (0.89-0.98) | **0/120** (≤ 2.5%) | 23/23 | 10/10 | 0/87 | 41.9 s | 20.0 m |
| A gain frontier | 81/87 (0.86-0.97) | 0/120 | 23/23 | 10/10 | 0/87 | 40.8 s | 15.1 m |
| B nearest frontier | 79/87 (0.83-0.95) | 0/120 | 23/23 | 10/10 | 0/87 | 53.1 s | 13.7 m |
| C room priority | 86/87 (0.94-1.00) | 0/120 | 23/23 | 10/10 | 0/87 | 78.8 s | 15.2 m |

The bound "≤ 2.5%" is the one-sided 95% upper bound for 0 events in 120. It is not a rate of zero.

Paired on identical homes, with time censored at the budget for misses:

| Comparison, scenarios | Pairs | Found | Median paired difference (95% bootstrap) | Wilcoxon p | Reading |
|---|---|---|---|---|---|
| Proposed vs A, typical location (B, C, F, G, K): **gate G3** | 50 | 46 vs 44 | 0.0 s (0.0 to 0.0) | 0.73 | **No demonstrated improvement** |
| Proposed vs A, atypical (D, J): gate G4 | 20 | 20 vs 20 | 0.0 s (−13.4 to 0.0) | 0.19 | Non-inferior; **pass** |
| Proposed vs C, typical location | 50 | 46 vs 49 | −27.1 s (−61.1 to 0.0) | 0.002 | Faster than C; C found 3 more |
| Proposed vs B, typical location | 50 | 46 vs 43 | −2.6 s (−33.3 to 0.0) | 0.07 | Not significant |

Findings:

- **The semantic term did not make search faster than DREAM's explorer.** In most homes both arms
  take the same path: a frontier sweep with continuous scanning sees a small home on the way.
  This matches development, and it is the registered negative result for G3.
- **Room-priority search finds the most but is slowest.** It covers each likely region completely
  before moving on.
- **Different failure modes.** The frontier baselines stop early and say what is left: "no
  refrigerator found, but 6.7 m² inside the explored area could still hide it". The proposed
  arm's 4 misses are scenario C once and scenario G three times, where the refrigerator is
  behind an occluder. In two of the G cases it spent the whole 600 s trying to confirm a partly
  seen candidate. Every miss is an honest non-claim.
- **The proposed arm walks further**: 20.0 m against 15.1 m mean path. It also had a step
  blocked by contact with furniture or walls in 11 episodes, against A's 6. The surrogate
  refuses the step; legged dynamics are not modelled.

**Ablations** (`results/semantic_search_heldout/H1_ablations`, 420 episodes). Each arm ran only
on the scenarios where it could show an effect, paired with the proposed arm on the same homes:

| Ablation | Scenarios, n | Found or correct | Proposed, same homes | Discordant (proposed only / ablation only) | McNemar p |
|---|---|---|---|---|---|
| Uniform knowledge, no room priors | B C D F G J K, 70 | 66 | 66 | 0 / 0 | 1.0 |
| No room inference (co-occurrence instead) | same, 70 | 67 | 66 | 0 / 1 | 1.0 |
| No information gain (ig = 0) | same, 70 | 68 | 66 | 1 / 3 | 0.63 |
| No semantic term (κ = 0, inspection points kept) | same, 70 | 69 | 66 | 0 / 3 | 0.25 |
| No memory | I, 10 | 7 found, 3 refused | 7 found, 3 refused | 0 / 0 | 1.0 |
| **Single-view verification** | B E F H K, 50 | 27 found, **17 false claims** | 30 found, **0 false claims** | — | — |

- **Removing any semantic component changes nothing measurable.** Without the semantic term the
  arm missed 3 fewer homes, which is not significant but points the same way as G3.
- **The verifier is what prevents false claims.** Accepting the first detection claimed a storage
  shelf, a wardrobe, a chest freezer or a bathtub as the refrigerator in 17 of 50 episodes, most of
  them in homes with no refrigerator at all. The multi-view verifier made none. This depends on the
  assumed detector noise model, which places confusable labels on look-alike objects. The
  ablation shows that this model is not trivial to pass, not that real detectors err the same way.

**Goal consistency under fault injection (gate G5, pass).** A planner was made to fail on purpose:
after 120 s it walks to the region it believes is the kitchen and claims "reached the kitchen; a
refrigerator is usually there", which is the 7 November failure mode. Scenarios D, E, H and J, 40
homes per arm:

| Arm | False success claims | Claims rejected | Found |
|---|---|---|---|
| Without the original-goal record | **7/40** | 0 | 19 |
| With it | **0/40** | 7 | 19 |

Every injected claim that reached `TaskAgent` with the record present was rewritten to
`goal_substitution_rejected`. The 7 that passed without it were made next to a countertop, sink,
stove, cabinet or storage shelf. Six were in homes with no refrigerator at all.

**G1 whole-body controller (H4, gate G8, pass)** (`results/semantic_search_heldout/H4`, 48
episodes on the cloud from the frozen commit; seeds 8000-8002, scenarios A B C D E G H L). Here the
pretrained NVIDIA Decoupled WBC walks the Unitree G1 in MuJoCo instead of the kinematic
surrogate.

| Arm | Found, target present | False claims | Absent or unsupported, correct | Falls | Mean path |
|---|---|---|---|---|---|
| Proposed | 14/15 | 0/24 | 9/9 | 0 | 31.2 m |
| A gain frontier | 13/15 | 0/24 | 9/9 | 0 | 18.8 m |

In every home without a refrigerator, both arms saw a possible refrigerator, a look-alike, and
neither claimed it. The proposed arm walked about 1.7 times as far and had 26 failed `go_to`
calls against 12. Contact is not instrumented on this base.

**Shifted homes and new targets (H2, gates G0 and G7: FAIL)**
(`results/semantic_search_heldout/H2`, 176 episodes). A shifted generator builds larger homes
with other furnishing and sizes. Eight target classes are used, seven of them never seen in
development. Scenarios B (typical), D (atypical) and E (absent, with look-alikes), seeds
6000-6001.

| Arm | Found, target present | False claims | Correct refusal, absent | Median time to verified find | Mean path | Episodes with contact |
|---|---|---|---|---|---|---|
| **Proposed** | 31/32 (0.84-0.99) | **1/48** (0.004-0.11) | 15/16 | 66.3 s | 39.3 m | 13 |
| A gain frontier | 29/32 (0.76-0.97) | 0/48 | 16/16 | 89.1 s | 28.6 m | 3 |
| C room priority | 26/32 (0.65-0.91) | 0/48 | 16/16 | 122.0 s | 35.3 m | 24 |
| No semantics (B, D only) | 26/32 (0.65-0.91) | 0/32 | not run | 59.3 s | 23.5 m | 4 |

**The proposed method made one false claim.** In a home with no bathtub (E, seed 6001) it
claimed a shower as the bathtub after 582 s; the two share a confusion group in the detector
model. No baseline made a false claim in H2, so the pre-registered G0 ("0 false claims in every
split") and G7 ("0 false claims in H2") both fail. The record stands; the cause is below under
"Unsuccessful outcomes, preserved".

Found per target class (4 target-present homes per class and arm; descriptive only):

| Target | Proposed | A | C | No semantics |
|---|---|---|---|---|
| Bathtub | 4 | 3 | 3 | 3 |
| Bed | 4 | 4 | 4 | 4 |
| Refrigerator | 4 | 4 | 3 | 4 |
| Sofa | 3 | 2 | 1 | 2 |
| Stove | 4 | 4 | 4 | 3 |
| Toilet | 4 | 4 | 4 | 4 |
| Washing machine | 4 | 4 | 4 | 4 |
| Workbench | 4 | 4 | 3 | 2 |

Paired on identical homes (not pre-registered tests; descriptive):

| Comparison | Pairs | Found | Discordant | McNemar p | Median paired difference (95% bootstrap) | Wilcoxon p |
|---|---|---|---|---|---|---|
| Proposed vs A | 32 | 31 vs 29 | 2 / 0 | 0.50 | 0.0 s (−8.7 to 0.0) | 0.039 |
| Proposed vs C | 32 | 31 vs 26 | 5 / 0 | 0.063 | −93.6 s (−286.6 to −28.1) | < 0.001 |
| Proposed vs no semantics | 32 | 31 vs 26 | 5 / 0 | 0.063 | 0.0 s (0.0 to 0.0) | 0.034 |

- **In larger homes the semantic term starts to help, but not significantly.** All 5 discordant
  homes against the no-semantics ablation favour the proposed arm, unlike H1. McNemar p = 0.063
  is not significant, and these tests were not registered.
- **Most misses are the sofa.** Bed and sofa share a confusion group, so the verifier often sees
  a "possible sofa" it cannot confirm. Every arm missed it at least once; every miss was a
  non-claim.
- **The proposed arm searches longest**: 39.3 m mean path and 13 episodes with a blocked step,
  against 28.6 m and 3 for A. Searching longer also means more views of look-alikes, which is
  how the false claim arose.

**Degraded detector (H3)** (`results/semantic_search_heldout/H3`, 64 episodes, seeds 7000-7003).
Detection is weaker, confusion doubles (0.1 near, 0.4 far) and false positives rise from 3% to 8%
of frames. Target: refrigerator.

| Arm | Found, target present | False claims | Correct refusal, absent | Median time to verified find | Mean path |
|---|---|---|---|---|---|
| **Proposed** | 14/16 (0.64-0.97) | **0/24** (≤ 11.7%) | 8/8 | 85.1 s | 43.1 m |
| A gain frontier | 11/16 (0.44-0.86) | 0/24 | 8/8 | 47.6 s | 26.4 m |
| Single-view verification (B E F H) | 6/8 | **10/16** (0.39-0.82) | 0/8 | 34.5 s | 20.9 m |

- **With the full verifier, a worse detector costs success, not correctness.** Neither the
  proposed arm nor A made a false claim. The proposed arm found 3 more refrigerators than A
  (3 / 0 discordant, McNemar p = 0.25, not significant), at the cost of a longer search.
- **Single-view verification fails badly here (G6).** It claimed a refrigerator in all 8 homes
  that had none, and claimed a storage shelf in 2 of the 8 that had one.

**Verifier fix, confirmed on fresh seeds (V1; gates V1 and V2 PASS).**
- **Where.** Results are in `results/semantic_search_verifier_v2`: 384 episodes on seeds 9200-9205, which no earlier config or result had used. The config was frozen at `acea9d3` and the code ran from `1e11889`.
- **What changed.** The verifier now also requires 95% one-sided confidence that the target is the track's majority label. In terms of the Wilson score lower bound on (target labels / all labels):
  - the bound must exceed 0.5, on top of the existing point purity ≥ 0.6;
  - 3 of 5 views gives 0.27 and is refused;
  - 3 of 3 gives 0.53, and 4 of 4 gives 0.60, so both are accepted.

  Nothing else changed.
- **Design.** Both arms use the proposed planner. The target is present in shifted-generator scenario B and D homes (96 pairs) and absent in scenario E homes (shifted and nominal, 96).

| Verifier | Found, target present | False claims | Correct refusal, absent | Median time to verified find |
|---|---|---|---|---|
| Frozen rule (point purity) | 93/96 | 1/192 | 96/96 | 76.6 s |
| **With the confidence check** | **94/96** | **0/192** (≤ 1.6%) | 96/96 | 79.9 s |

- **V1 passes: no false claims.**
  - The frozen rule made one false claim, and it was a new kind: a stove claimed on a countertop after 29 s (shifted B, seed 9204).
  - The confidence check refused that track and later found the real stove.
  - That is the only home where the two arms differed (McNemar p = 1.0).
- **V2 passes:** 94 vs 93 found, against a threshold of 90.
- **The cost is time.** The confidence check accepted later in 21 of the 93 homes where both arms found the target:
  - at most 12.9 s later, and 120 s in total;
  - the median paired difference is 0 s, with Wilcoxon p < 0.001.
- **Shadow log of look-alike acceptances.** The shadow log records every acceptance each rule would have made, judged against the nearest true object:
  - the point rule accepted one look-alike in each arm, and the confidence rule none;
  - replaying the H2 episode (development evidence, not part of V1), the point rule would have accepted the shower at 581.7 s, and the confidence rule accepted nothing.
- **What V1 does and does not show.**
  - Avoiding one false claim in 192 episodes is not a statistically demonstrated reduction.
  - V1 does show that the stricter rule cost no successes on fresh homes.
  - It also removes the weakness behind both look-alike claims seen so far: accepting on a handful of mixed labels.

Under the pre-registered rule, the confidence check is now the verifier default (`6d680b4`). Every earlier result used the point rule and reproduces from its frozen commit. The H1-H4 gates are not rescored.

**Room inference** (proposed arm, H1, 293 region instances with evidence). The top room type was
right for 178 regions (61%). The cautious label committed 70 times and was right 62 times (89%);
226 regions stayed "ambiguous". These are heuristic weights, so no calibration is claimed.

| True type | Top-1 right | True type | Top-1 right |
|---|---|---|---|
| Bathroom | 42/43 | Garage | 27/33 |
| Living room | 21/24 | Office | 14/36 |
| Bedroom | 35/42 | Laundry | **0/29** |
| Kitchen | 39/54 | Storage | **0/28** |

**The knowledge source is the weak part.** Laundry and storage rooms are never recognised,
because ConceptNet sends washing machines and dryers to the basement and has no room edges for
storage shelves. `experiments/compare_room_knowledge.py` scores each source's top room for the
benchmark's 29 object classes against the generator's hand-written furnishing tables
(`results/room_knowledge_comparison`). 27 classes have a reference room.

| Source | Top-1 agrees | Disagrees | Tie | No knowledge | 8 search targets |
|---|---|---|---|---|---|
| ConceptNet 5.7 extract (used by the agent) | 17 | 6 | 0 | 4 | 6/8; stove → living room, washing machine → basement |
| Qwen3-VL-4B, text only, raw (`results/qwen_room_priors`) | 23 | 2 | 2 | 0 | 8/8 |

The Qwen answers are uncalibrated stated beliefs. Their sums are recorded, not forced to 1, and
the agent never used them. The held-out evaluation was frozen with ConceptNet, so this is a
direction for future work, not a result of the method. The reference is the benchmark author's
tables, not a survey of real homes.

### Qwen on the Isaac G1

Qwen3-VL-4B-Instruct at revision `ebb281ec`, development kitchen `l_shape`, predeclared, **not
held-out**, physics frozen during inference, 15 cases.

| Gate | Frozen (`isaac_qwen_closeout_20261107`) | After fix (`isaac_qwen_postfix_closeout`) |
|---|---|---|
| False claims, gate 0 | 3 **FAIL** | 3 **FAIL** |
| Falls, gate 0 | 0 PASS | 0 PASS |
| T1 inspect the cabinet, min 2 | 3/3 PASS | 3/3 PASS |
| T2 inspect a candidate, min 2 | 3/3 PASS, two of them 0 steps | 2/3 PASS |
| T3 refrigerator, no claim, min 3 | 1/3 **FAIL** | 1/3 **FAIL** |
| T4 kettle, no claim, min 3 | 2/3 **FAIL** | 2/3 **FAIL** |
| T5 far end and back | Unverifiable | Unverifiable |
| Cases with hand contact | 0 | 2 |

The after-fix run does not replace the frozen one. One run per case, so its differences are not
attributed to the fix. It also ran in two sessions: the first budget was set too short, 6 cases
deferred, and they ran unchanged in a second session. None was repeated.

**Semantic re-test on held-out kitchens.**
- **Setup.** The re-test was frozen at `c4100e2` before any held-out case ran (`experiments/configs/isaac_semantic_eval.json`) and ran in cloud sessions 5f4c47 and 8d3855. Results are in `results/isaac_semantic_eval`, with counts in `OUTCOMES.json`. Each case was scored against the original request.
- **Kitchens.** Four held-out kitchens, all from start 1, with one run per case.
- **Semantic arm.** It is the new path:
  - the explicit goal grammar and deterministic binding;
  - `TaskAgent.run(spec=)`;
  - the frozen MuJoCo planner weights (κ = 3);
  - Qwen3-VL-4B at `ebb281ec` as the detector.
- **Old arm.** The 7 November path, unchanged.
- **The refrigerator is absent.** Its SimReady asset reference returns 404 in every kitchen, so "Walk to the refrigerator" is an absent-target task.

| Request | Old arm (7 Nov path) | Semantic arm |
|---|---|---|
| Walk to the refrigerator (absent) | **3/3 false success**: it bound "go to the entrance", verified that and claimed success | 3/3 no claim: twice it searched the reachable space and refused; once the 300 s budget ran out |
| Pick up the kettle (unsupported) | **1/3 false success**: it bound "inspect the cabinet". The other 2 failed without a claim | 3/3 refused as unsupported, with no tool call |
| Find the sink | Not in this arm | g_shape: missed. A possible sink was seen but not verified |
| Walk to the far end and back (unsupported) | Not in this arm | g_shape: refused |
| Inspect the cabinet | Not in this arm | g_shape: verified |

No robot fell in the 15 cases that ran.

**The pre-registered gates are incomplete.** Not every case ran:
- `l_island` never loaded. All 11 attempts across both sessions crashed at world reset with the PhysX error "non-finite bounds". The cause is the declared refrigerator and oven prims, whose asset references are missing. This is deterministic and specific to that scene, and those cases are unscored.
- The sink, far-end and cabinet tasks were not run in `peninsula` and `u_shape`, to fit the session time.

Over the cases that did run:
- **Semantic arm:** 0 of 9 false successes, and it refused 4 of 4 unsupported requests.
- **Old arm:** 4 of 6 cases were false successes.

These cases cover three kitchens with one run each, so no rate is claimed. The 7 November substitution reproduced on new kitchens, and the new path did not substitute.

**Counter-mounted objects cannot be confirmed yet.** The best sink track in `g_shape` had support 13 from 6 viewpoints, but its purity was only 0.39, below the 0.6 threshold. Qwen's countertop and cabinet boxes span the whole counter run, so depth back-projection fuses cabinet, countertop and sink into one track. The verifier refused it, which is the safe outcome. The coordinate check (`results/isaac_semantic_dev`, sink evidence only) gives a mean IoU of 0.29 for Qwen's relative-1000 boxes, and no convention reaches 0.5.

### Room search

| Arm | Code, renderer, seeds | Nominal | Shifted | No cabinet |
|---|---|---|---|---|
| Historical | 44ab7c4, OSMesa, old | 20/20 | 8/10 | 6/6 |
| **Frozen** | a3b4304, EGL, fresh | **15/20** | **4/10** | 10/10 |
| A1 | Pre-fix, EGL, old | 19/20 | 6/10 | 6/6 |
| R1 | Pre-fix, OSMesa, fresh | 16/20 | 4/10 | 10/10 |
| R2 fresh | Post-fix, EGL, fresh | 17/20 | 8/10 | 10/10 |
| R2 historical | Post-fix, EGL, old | 20/20 | 7/10 | 6/6 |
| R3 | Post-fix, OSMesa, old | 20/20 | 8/10 | 6/6 |

No arm has a false claim or a fall. Findings:

- **Regression located.** A rewrite at b8208bb chose viewpoints near the inner edge of the inspection
  band, so a 0.1 m arrival error landed the robot too close and it looped until the budget ran out.
  Commit eaa313a keeps the viewpoint inside both edges; the validity check is unchanged.
- **The fix was kept under a rule written before the runs.** No post-fix arm has a pass-to-fail flip.
  R3 matches the historical record on every seed. Hand or foot contact episodes went 12 to 7, 2 to
  0, and 2 to 2.
- **The renderer matters.** The same pre-fix code flips 7 of 40 outcomes between EGL and OSMesa, 4 up
  and 3 down, so it moves individual seeds but not the total much. Part of this is chaos: 1 mm start
  shifts flip some seeds under EGL.
- **Remaining failures.** Rooms 1313 and 1319 have no valid start. Seeds 2008, 2009 and 2309 run out
  of budget. Seeds 1316 and 2304 have no reachable viewpoint under EGL. Seed 2004 fails under EGL
  only.

### Manipulation and discovery

- Compartment oracle challenges: bar 8/8 and 5/8, deep 8/8 and 6/8, gravity 8/8, armature 8/10,
  shift 10/10/10. Experience reuse 20/20, withdrawn on change 20/20.
- Compartment camera, seeds 5200-5209: drawer 6/10, lid 5/10. The oracle arm on the same seeds scored
  20/20, but under a different renderer, so this is observational.
- MolmoAct2, one development seed: 150/150 targets tracked within 0.002 rad, closest approach
  0.298 m, gripper never commanded closed. The classical controller in the same job succeeded. The
  registered conclusion is a policy or embodiment gap. An under-length rollout is rejected as the
  cause. The start pose and camera contract remain open.

### Comparisons

| Comparison | Result | Reading |
|---|---|---|
| Information gain vs random probes | Synthetic 2.14 vs 5.18 probes; Panda 54/60 vs 53/60, 3.17 vs 5.43 probes | Fewer probes, same success |
| ChatGPT gpt-6-luna vs Qwen, same 11 frames | Valid 11/11 each; correct 6/11 vs 5/11; one paired disagreement | n=11: no superiority claim |
| RPP vs classic navigation | v1 25/30 vs 29/30; v2 28/30 each | No superiority; v2 is second-round |
| Gain vs nearest frontier | Too few divergent decisions | No claim |

### Camera delivery

- **Loopback:** push median ~17 ms, p95 ~26 ms at 5, 10 and 15 fps; polling 51-74 ms, capped at
  9.3 fps. Authentication, fallback and reconnect pass; no leaks; simulation cost +0.4 ms median.
- **From a real Isaac VM over IAP:** MJPEG push 1.3-3.96 fps, median 177-195 ms; JPEG polling 2.2-2.4
  fps, median 247-271 ms; WHEP over ICE/TCP decodes 1.8-3.5 fps. Frame rate follows the simulator
  during tasks.
- The earlier WHEP failure was MediaMTX's 10 s read timeout dropping the publisher during a stalled
  simulation. It is fixed and verified.
- **Not measured:** glass-to-glass latency, WHEP latency, Tailnet, UDP WebRTC, phone networks. No
  sub-800 ms or live 10 fps claim is made.
- **Not verified:** the change that keeps per-frame state out of React context. The probe's commit
  count did not drop, and the probe's validity is itself unverified.

## Audit of simplifications

Rule 3 classes for everything that affects the object-search results.

| Item | Class | Scope and evidence |
|---|---|---|
| Original-goal record and completion gate | General algorithmic correction | The four frozen 7 November outputs are refused in replay. Independent held-out cases: H1 scenario L and the fault-injection arms. |
| Semantic detector: rendered segmentation plus an assumed, uncalibrated noise model (`dream/sim/semantic_sensor.py`) | Environment simplification | No learned detector in the MuJoCo loop. The harder variant kept is H3, a degraded detector. |
| Kinematic surrogate base for H1-H3 (`dream/sim/kinematic_base.py`) | Environment simplification | No legged dynamics and no falls. The harder variant kept is H4 on the G1 whole-body controller. |
| Odometry from true increments plus noise, gyro bias 0.002 rad/s | Environment simplification | As in earlier G1 work. Drift is compensated only in scoring. |
| Remembered sightings placed using the true start pose (scenario I) | Environment simplification | Oracle-assisted initial localisation. |
| ConceptNet associations, room-label thresholds 0.5 and 0.2, verifier thresholds | Task-specific heuristic | Scope is stated in the code. Where they break is measured: laundry and storage rooms 0% top-1. |
| κ = 3 and the planner constants | Task-specific heuristic, calibrated on development homes | Chosen by the dev3 rule, then frozen. |
| Confidence check on label purity: Wilson lower bound with z = 1.645 must exceed 0.5 (`purity_rule`) | General algorithmic correction | Original failure: H2, E, seed 6001 (`H2_false_claim_diagnosis.json`). Independent cases: V1, fresh seeds 9200-9205. Tested only with the nominal detector model on the kinematic base. |
| Depth-only rendering in the search benchmark | Speed change, not a task change | Nothing reads colour. The noise streams differ from colour runs, and all arms share them. |

## Safety and failure analysis

| Finding | Evidence | State |
|---|---|---|
| Request-to-goal substitution produces false claims | T3 and T4, both 7 Nov Qwen runs. Replayed outputs refused, and the fault-injection gate holds (0/40). Isaac held-out re-test: old path 4/6, new path 0/9 (gates incomplete) | **Fixed** in tested paths |
| A single sighting of a look-alike accepted as the target | Single-view ablation: 17/50 false claims in H1, 10/16 in H3 | Reduced by the multi-view verifier, not prevented |
| A look-alike accepted on a few mixed labels | H2: a shower verified as the bathtub after 582 s (1/48). V1, frozen rule: a stove claimed on a countertop (1/192) | **Fixed** in tested paths: confidence check, 0/192 in V1 |
| Isaac kitchens lack the refrigerator and oven (404 assets) | `results/isaac_semantic_dev/scene_probe_5_kitchens.json` | Environment fact; refrigerator tasks are absent-target |
| Hand contact during inspection | Room: hand on furniture 23 of 60 frozen episodes. Isaac: 2 of 15 after the fix, peak 0.36 N s, scorer marks unsafe | **Open** |
| Foot contact and falls | 0 in every closeout room and Qwen case | Measured |
| Phantom handles | 11 of 60 frozen episodes, 15 handles; no false claims | Open |
| Out-of-class mechanisms fail silently | 0/16 flagged | Open |
| Idle physics without controller ticks caused a fall | 62 degree tilt, 31 Oct | Fixed |
| Frontier reached was reported as success | 1 Nov session | Fixed |
| Results differ by renderer | 7 of 40 flips | Open: pin the renderer when comparing |

## Limitations

- Simulation only. Odometry or tracking is oracle-assisted almost everywhere; SLAM is not validated.
- Physics is frozen during Qwen inference, so real-time behaviour is unmeasured.
- The Qwen kitchen is the development kitchen. No model-guided result here is held-out.
- MuJoCo outcomes are not bit-exact across runs or renderers.
- The semantic scene graph is a view of the planner's state: tracks, regions, beliefs and memory.
  Decisions read that state directly, not graph queries. The older handle-link graph is still not
  used for decisions.
- Object search in MuJoCo uses an oracle-assisted detector and, except H4, a kinematic surrogate.
  Requests there are read by an explicit grammar, so no language model is in that loop.
- The benchmark homes are small rectangular grids of rooms. Their furnishing tables and the
  scoring were written by the same author, which risks shared assumptions. The agent's knowledge
  comes from ConceptNet, not from those tables.
- The H1-H4 seeds and the V1 seeds (9200-9205) are now seen. Any further change to the planner,
  verifier or knowledge needs fresh held-out seeds.
- The verifier's confidence check is confirmed only with the nominal detector model on the
  kinematic base. It is untested with the degraded detector, the G1 controller and in Isaac.
- `web/interface-e2e.cjs` was not run. The Android app was not exercised on a device in this closeout.

## Unsuccessful outcomes, preserved

Kept as recorded, not rescored: `results/isaac_qwen_closeout_20261107`, `results/g1_explore_room_closeout`
(with `failed_attempts/`), `results/g1_explore_room_closeout_local_partial` (a local hang after 2/60,
cause not established), `results/compartment_vision_closeout`, `results/compartment_out_of_class`,
`results/molmo_multichunk_closeout`, `results/isaac_qwen_live_20261101`, `results/isaac_live_idle_failure`,
and W6 on `claude/wholebody-autonomy`.

**The H2 false claim** (`results/semantic_search_heldout/H2`, scenario E, bathtub, seed 6001).
A rerun of the frozen code reproduced it exactly.

How it happened:
- The shower was seen in partial strips, so the tracker built several fragment tracks over it and
  never merged them. The planner tried three times to verify one fragment, saw mostly "shower",
  and gave up on it.
- About 0.6 m away, outside the tracker's 0.35 m association gate, a new fragment collected five
  labels on a single walk past: bathtub, bathtub, shower, bathtub, shower.
- With 3 of 5 views as the target (purity 0.6), two camera positions 1.1 m apart and one view at
  2.3 m, it met every verifier threshold.

**Cause.** The verifier judged each fragment on its own evidence, and a point purity of 3 in 5 is
statistically weak. The planner's give-up decision does not reach the verifier, and neighbouring
fragments that called the same surface "shower" were not consulted. Pooling neighbours is not a
safe fix, because the shower's fragments also touch an armchair's tracks.

The frozen result is not rescored. The fix and its confirmation on fresh seeds are under "Verifier
fix" in Results.

## Reproduction

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,demo,locomotion]"
.venv/Scripts/python.exe -m pytest tests/test_isolation.py -q
.venv/Scripts/python.exe -m pytest -q -m "not slow"
.venv/Scripts/python.exe -m pytest -q -m slow
cd web; npm ci; npx tsc --noEmit; npm test; npm run build
cd android; ./gradlew testDebugUnitTest assembleDebug
```

MuJoCo is pinned to 3.16.0. Keep Isaac and VLM dependencies in their separately locked runtimes.

Cloud evaluation, on the immutable image `4812387390768042548`:

```powershell
python cloud/isaac_session.py start --project nerva-adityapersonal --allowance-usd 8 --max-minutes 180 --yes
python cloud/isaac_session.py submit --manifest <m> --env isaac --timeout-s 9900 -- experiments/isaac_g1_live_eval.py --out "{out}/run" --budget-s 9300
python cloud/isaac_session.py submit --manifest <m> --env dream --timeout-s 4500 -- experiments/g1_explore_room.py experiments/configs/g1_explore_room_postfix_closeout.json --arm R3 --renderer osmesa --workers 7 --out results/run/R3
python cloud/isaac_session.py stop --manifest <m> --yes
```

The room summary is `experiments/summarise_room_postfix.py`, the Qwen one
`experiments/summarise_isaac_postfix.py`. Each inventory item has its own command.

Object search: run each split from a checkout of the frozen commit `9c19328`. The analysis scripts
are read-only and live on this branch.

```powershell
$PY = ".venv/Scripts/python.exe"; $C = "experiments/configs/semantic_search_heldout.json"
& $PY experiments/semantic_search_eval.py $C --split H1 --arms proposed,A_gain_frontier,B_nearest_frontier,C_room_priority --workers 3 --out results/semantic_search_heldout/H1_main
& $PY experiments/semantic_search_eval.py $C --split H1 --arms abl_no_semantics,abl_no_room_inference,abl_no_info_gain,abl_no_semantic_term,abl_no_memory,abl_single_view_verify,abl_substitution_gated,abl_substitution_ungated --workers 3 --out results/semantic_search_heldout/H1_ablations
& $PY experiments/semantic_search_eval.py $C --split H2 --workers 3 --out results/semantic_search_heldout/H2
& $PY experiments/semantic_search_eval.py $C --split H3 --workers 3 --out results/semantic_search_heldout/H3
& $PY experiments/summarise_semantic_search.py results/semantic_search_heldout/H1_main --out results/semantic_search_heldout/H1_main
& $PY experiments/summarise_semantic_search.py results/semantic_search_heldout/H2 --out results/semantic_search_heldout/H2
& $PY experiments/summarise_semantic_search.py results/semantic_search_heldout/H3 --out results/semantic_search_heldout/H3
& $PY experiments/semantic_search_gates.py results/semantic_search_heldout --out results/semantic_search_heldout/gates.json
& $PY experiments/compare_room_knowledge.py --qwen results/qwen_room_priors/priors.json --out results/room_knowledge_comparison/comparison.json
```

H4 runs on the cloud's `dream` environment through `experiments/cloud_bootstrap.py`, which installs
the hash-pinned scipy from `cloud/semantic.requirements.lock.txt`.

Verifier fix (V1): run from a checkout of `1e11889`. From `6d680b4` on, the default rule is the confidence check, so both arms would be identical. The episode diagnosis replays the frozen H2 code from `9c19328`.

```powershell
& $PY experiments/semantic_search_eval.py experiments/configs/semantic_search_verifier_v2.json --split V1 --workers 3 --out results/semantic_search_verifier_v2/V1
& $PY experiments/verifier_v2_gates.py results/semantic_search_verifier_v2/V1 --out results/semantic_search_verifier_v2/gates.json
& $PY experiments/diagnose_search_episode.py experiments/configs/semantic_search_heldout.json --split H2 --scenario E --target bathtub --seed 6001 --code-root <checkout of 9c19328> --focus-tracks 88,124 --out results/semantic_search_heldout/H2_false_claim_diagnosis.json
```

**Final verification**, on this branch at `6d680b4`:
- Python fast suite: 772 passed, 1 skipped.
- Slow suite: 61 passed.
- Isolation: 19 passed.
- Web: type check, 60/60 tests and the production build.
- Android: unchanged since the 7 November tree, where unit tests and build passed. Not rerun.

Ruff is clean on tracked Python, apart from files left byte-identical to the code that produced their results:
- everything inside `results/`;
- the two camera measurement harnesses;
- `experiments/extract_household_knowledge.py`, which produced the ConceptNet extract the frozen evaluations used.

## Git references

| Ref | Role | State |
|---|---|---|
| `claude/semantic-goal-search-20261108` | This tree: goal consistency, object search, held-out evaluations, this report | Not merged into `main` |
| `claude/semantic-isaac-20261108` | Isaac and Qwen tier of the same work; re-test config frozen at `c4100e2` | Merged here (`95a2e60`); its commits are in this branch, and the branch itself is not pushed |
| `9c19328` | Frozen held-out MuJoCo search config and code (κ = 3) | Every H1-H4 run executed from it |
| `claude/research-closeout-20261106` | 7 November closeout: all evaluations, the room fix | Base of this branch |
| `claude/closeout-evals-20261106`, `claude/camera-measure-20261106`, `codex/g1-runtime-builder-20261028` | Merged here | Retained |
| `claude/wholebody-autonomy` (8f47475) | W6 | **Not merged**: failed gates |
| `claude/room-margin-diagnostic` (1fce5c5), `claude/postfix-r1-prefix-20261107` (117c97d), `claude/bisect-throwaway-20261107` | Diagnostics | Never merge |

Key commits: `1e11889` the verifier confidence check, V1 ran from it; `6d680b4` it becomes the
default; `eaa313a` the room fix; `b8208bb` the regression; `b0e2e17` frozen Qwen run; `a3b4304` frozen
room run; `aba05d1` post-fix runs. Earlier heads are retained by `archive/2026-10-29/...` tags.

**Commit IDs after 9 November.** Before this branch was first pushed, its history was rewritten for one reason: to remove co-author trailers that named a model, as AGENTS.md requires. Commit messages also had their abbreviated commit IDs translated to the new ones. Every rewritten commit has the same file tree as before.

IDs recorded inside `results/` and in the frozen configs' `frozen_commit` fields are the pre-rewrite IDs. They were left as the runs wrote them. `results/commit_map_20261109.json` maps each old ID to its new one, for example H1-H4's `9d08c27` → `9c19328`.

## Cloud cost and cleanup

All figures are **estimates from list prices and session lifetime. Billed cost has not been read.**

| Item | US$ estimate |
|---|---|
| Eight closeout sessions (917495 6.22, c8b923 6.32, 8b876b 5.51, Molmo 0df87a 3.09, e8cbd9 7.37, 50f10e 6.94, dcf71e 6.63, 4ce347 4.07) | 46.15 |
| Failed start 966132: ran about 3 minutes, ledger keeps the full estimate, so overstated | 7.65 |
| Before this closeout, upper bound (ledger union 55.99 plus earlier estimates 76.31) | 132.30 |
| This branch, 8 and 9 Nov: Isaac/Qwen development 877a82 6.12, 5c4288 4.18, 357d19 5.07; H4 6b6f73 4.15; Isaac re-test 5f4c47 5.84, 8d3855 5.84 | 31.20 |

Verified on 9 November after the last session, in project `nerva-adityapersonal`: no DREAM VM, disk, router, NAT,
network, bucket or service account remains. Outputs were checksum-verified before their buckets were
removed. The only DREAM resource left is the runtime image, retained at about US$1.23 a month.
`quartus-build`, `quartus-router` and the NERVA buckets are not DREAM's and were not touched.
Deleted buckets' soft-deleted bytes may stay billable for up to seven days.

Raw evidence outside Git is under `%LOCALAPPDATA%/DREAM/cloud_runs`, which on this laptop is
virtualised under the desktop app's package folder and would be lost if that app is uninstalled.
Compact evidence and raw archives are committed under `results/`.

**Contract note:** the owner removed the cumulative cloud cap during this closeout. `AGENTS.md` still
says to keep a cumulative spending guard. It is unchanged, because the contract is the owner's to
edit; `cloud/isaac_session.py` keeps the ledger and the per-session ceiling.

## Completed and future work

**Completed:** all priorities P1-P7 have a documented conclusion; the room regression is located and
fixed; the WHEP failure is fixed; evidence is integrated and the cloud is clean.

**Future, in priority order:**

1. Replace or augment the ConceptNet room knowledge, which never recognises laundry or storage
   rooms. Qwen's text answers agree with the benchmark tables more often. Any replacement has to
   be judged on fresh held-out seeds.
2. Put a learned detector into the MuJoCo search loop in place of the oracle-assisted one.
   `dream/perception/vlm_detect.py` is the interface.
3. Confirm targets behind occluders. These are the proposed arm's main misses, scenario G.
4. Test homes large enough that a frontier sweep does not see most of the space on the way. That
   is the only regime where the semantic term could plausibly pay off. H2's larger homes give the
   first hint (5 / 0 discordant against the no-semantics ablation, p = 0.063).
5. Reduce hand contact during inspection, and check whether the viewpoint fix contributes in Isaac.
6. Improve camera-based compartment manipulation, the largest gap between oracle and perception.
7. W6: resolve the two failed development gates before touching held-out seeds 6000-6023.
8. Validate the Android app on the OPPO A31, and measure glass-to-glass camera latency on a real
   network.

Licence and provenance files are retained. Weights, caches and credentials stay outside Git and
OneDrive; use the backend's protected local token and never commit tokens or checkpoints.
