# DREAM

Research toward general-purpose embodied intelligence: receive an unfamiliar physical goal,
observe, investigate, compose available skills, verify the outcome, and recognise limits.
**General-purpose task completion is the objective, not an established capability.**
`AGENTS.md` is the engineering and behavioural contract; `CLAUDE.md` aliases it.

This README is the research closeout report of 7 November 2026. Every item, with its success
criteria, baseline, repro command, cost and evidence, is in
[`results/research_closeout_20261106/inventory.json`](results/research_closeout_20261106/inventory.json).
The previous README is preserved in `archive/documentation-closeout-20261107.zip`.

## Executive summary

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

- The Qwen evaluation **fails its gates**. Asked to walk to a refrigerator or pick up a kettle, the
  system can substitute a goal it can do (go to the entrance, inspect a cabinet), verify that, and
  claim success. This is the most important open defect, and it is not fixed.
- Room search on fresh seeds **failed its frozen gates** (15/20, 4/10). The later fix does not
  change that record.
- Compartment opening from camera perception **failed** (11/20). MolmoAct2 **did not complete the
  task** (0/1). Whole-body W6 is **not ready**. 10 fps camera delivery over the remote link was not
  demonstrated.
- Nothing has run on a physical robot. Nothing here establishes general-purpose task completion.

## Status by area

| Area | Result | Status |
|---|---|---|
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
goal text -> provider (Qwen, ChatGPT, or structured with no language model)
          -> typed tools only: models never get raw joint control
          -> skills: probe, inspect, navigate, grasp, open, place
          -> controller: Panda joints, or NVIDIA Decoupled WBC on the G1
          -> simulator: MuJoCo or Isaac Sim, behind an isolation boundary
independent verifier: reads world state, not the agent's claim
```

Separate parts: a Bayesian mechanism belief with information-gain probe selection; RGB-D occupancy,
Dijkstra and Regulated Pure Pursuit for navigation; SQLite home and experience memory; a task
scheduler and phone API; a Next.js web app and an Android app; a camera path with MJPEG push, JPEG
polling and an experimental NVENC, MediaMTX and WHEP publisher.

**Embodiments:** abstract gantry; Franka Panda in MuJoCo; Unitree G1 in MuJoCo; Unitree G1 in Isaac
Sim 6.2 with PhysX. The G1 uses a pretrained velocity-commanded motor controller with arms held.

## Results

Unless stated otherwise these are simulation, with oracle-assisted odometry or tracking.

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

## Safety and failure analysis

| Finding | Evidence | State |
|---|---|---|
| Request-to-goal substitution produces false claims | T3 and T4, both Qwen runs | **Open** |
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
- Scene graph links are stored but nothing uses them for decisions.
- `web/interface-e2e.cjs` was not run. The Android app was not exercised on a device in this closeout.

## Unsuccessful outcomes, preserved

Kept as recorded, not rescored: `results/isaac_qwen_closeout_20261107`, `results/g1_explore_room_closeout`
(with `failed_attempts/`), `results/g1_explore_room_closeout_local_partial` (a local hang after 2/60,
cause not established), `results/compartment_vision_closeout`, `results/compartment_out_of_class`,
`results/molmo_multichunk_closeout`, `results/isaac_qwen_live_20261101`, `results/isaac_live_idle_failure`,
and W6 on `claude/wholebody-autonomy`.

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

**Final verification on this tree:** Python fast 595 passed and 1 skipped; slow 30 passed; isolation
17 passed; web types, 57/57 tests and production build; Android unit tests and build. Ruff is clean on
tracked Python except inside `results/` and two camera measurement harnesses, left byte-identical to
the code that produced their results.

## Git references

| Ref | Role | State |
|---|---|---|
| `claude/research-closeout-20261106` | This tree: all evaluations, the fix, this report | Passes the checks above |
| `claude/closeout-evals-20261106`, `claude/camera-measure-20261106`, `codex/g1-runtime-builder-20261028` | Merged here | Retained |
| `claude/wholebody-autonomy` (8f47475) | W6 | **Not merged**: failed gates |
| `claude/room-margin-diagnostic` (1fce5c5), `claude/postfix-r1-prefix-20261107` (117c97d), `claude/bisect-throwaway-20261107` | Diagnostics | Never merge |

Key commits: `eaa313a` the fix; `b8208bb` the regression; `b0e2e17` frozen Qwen run; `a3b4304` frozen
room run; `aba05d1` post-fix runs. Earlier heads are retained by `archive/2026-10-29/...` tags.

## Cloud cost and cleanup

All figures are **estimates from list prices and session lifetime. Billed cost has not been read.**

| Item | US$ estimate |
|---|---|
| Eight closeout sessions (917495 6.22, c8b923 6.32, 8b876b 5.51, Molmo 0df87a 3.09, e8cbd9 7.37, 50f10e 6.94, dcf71e 6.63, 4ce347 4.07) | 46.15 |
| Failed start 966132: ran about 3 minutes, ledger keeps the full estimate, so overstated | 7.65 |
| Before this closeout, upper bound (ledger union 55.99 plus earlier estimates 76.31) | 132.30 |

Verified after the last session, in project `nerva-adityapersonal`: no DREAM VM, disk, router, NAT,
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

1. Fix request-to-goal substitution: an unsupported or unobserved target must never bind to a
   different goal. It is the cause of all three false claims.
2. Evaluate on a kitchen and room seeds that have never been seen.
3. Reduce hand contact during inspection, and check whether the viewpoint fix contributes in Isaac.
4. Improve camera-based compartment manipulation, the largest gap between oracle and perception.
5. W6: resolve the two failed development gates before touching held-out seeds 6000-6023.
6. Validate the Android app on the OPPO A31, and measure glass-to-glass camera latency on a real
   network.

Licence and provenance files are retained. Weights, caches and credentials stay outside Git and
OneDrive; use the backend's protected local token and never commit tokens or checkpoints.
