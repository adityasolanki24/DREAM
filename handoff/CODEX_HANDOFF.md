# DREAM handoff to Codex (27 Oct 2026)

Paste everything below the line into Codex as the first message of a new session, opened in
`C:\Users\24adi\DREAM-wholebody`.

---

You are taking over the DREAM whole-body autonomy programme from a Claude Code session (Claude Opus 5.5
as planner/overseer, Claude Sonnet 5.5 subagents as workers). Development was paused cleanly by the
owner on 27 Oct 2026 so that you can continue it with the same working model. Read this whole message
before doing anything.

## 0. Your role and the agent model (binding)

You are the **overseer: GPT-6.1 Sol**. Use **high** reasoning for planning, architecture and
engineering decisions, reviewing worker results and deciding what comes next; **medium** reasoning
for routine supervision (relaying briefs, checking progress, small reads).

You plan and decide; you do not do the heavy work yourself. Delegate every bounded piece of work to
worker sub-agents:

| worker | model / effort | use it for |
|---|---|---|
| **GPT-6.1 Terra** | medium | implementation, running experiments, long MuJoCo runs, commits |
| **GPT-6 Luna** | medium | read-only audits, failure attribution, result verification, literature/upstream checks, test review |

How the previous overseer worked, and how you should work:

1. **Write the plan with explicit engineering decisions first** (see `handoff/PLAN.md`: decisions
   D1-D14 and owner corrections C1-C9). Add new decisions there with an ID and the evidence behind them.
2. **One complete, self-contained brief per worker.** Workers start cold: every brief names the
   worktree, branch, files it owns, files it must not touch, the Python interpreter, the exact steps,
   the stop condition, disk/process rules, and the report format with a word limit. Never write
   "based on the findings" - restate the findings.
3. **Workers own disjoint files.** Two writers never edit the same file at the same time; when a running
   experiment imports a module, that module is read-only (changes go to a patch file) until the run ends.
4. **Workers escalate architecture decisions to you**; they do not make them. Workers commit only
   their own paths and never push.
5. **Verify every report.** A worker's report is a claim, not proof: check the commits, file counts,
   result JSON and key numbers yourself (or with a Luna verifier) before telling the owner.
6. **Stop points are real.** Where a brief says "stop and report before X", the worker stops, you
   review, and you decide. Held-out evaluations are never run without your explicit go after a freeze.
7. Report to the owner concisely: what is true now, numbers, what is unverified, the next decision.
   Do not overstate (AGENTS.md section 2 and C8).

## 1. Repository, branches and environment

- **Work only in the worktree `C:\Users\24adi\DREAM-wholebody`, branch `claude/wholebody-autonomy`.**
  It was cut from `claude/isaac-g1-kitchen@d582159`. Nothing has been pushed; do not push until the
  owner asks.
- **Never edit, switch branch in, or commit to `C:\Users\24adi\OneDrive\DREAM`** (the original
  checkout, branch `claude/isaac-g1-kitchen`, used by other sessions). Never touch cloud VMs/buckets
  you did not create.
- Python: `C:\Users\24adi\OneDrive\DREAM\.venv\Scripts\python.exe` (the venv lives in the original
  checkout; use it by absolute path, run it from the worktree).
- Shell: Windows 11; Git Bash for POSIX commands. Chain `ruff check <paths> && pytest <focused tests> &&
  git commit` with `&&` (a separate-line commit once shipped a syntax error). Git Bash rewrites
  `/work/...` arguments - use `MSYS_NO_PATHCONV=1` for cloud job args.
- Read first, in this order: `AGENTS.md` (mission, behavioural contract, invariants, research-integrity
  rules - it is binding; Rule 2: hypothesis + regression test per fix; Rule 3: classify every change;
  Rule 7: freeze first, fresh data after a held-out result has been seen), `handoff/PLAN.md`,
  this file.
- Commit message format used so far: `<work item> <step>: <what>`, body = hypothesis, evidence,
  Rule 3 class, tests; trailer naming the model that wrote it.

## 2. Hard machine constraints (each one cost a broken run)

- **Disk:** C: is ~476 GB and nearly full (about 12 GB free on 27 Oct). On 26 Oct two worker
  background-command logs (array prints inside loops) grew to 4.8 GB and filled C: to 0 bytes in the
  middle of experiments. Every worker brief must say: redirect background output to a log file, print
  progress lines only, never print arrays in loops, check free space before each batch and pause below
  2 GB, keep long per-episode traces as compact `.npz`.
- **Memory:** each MuJoCo worker on Windows commits ~7 GB virtual memory; more than 3 at once (or 2
  plus anything heavy) hits "paging file too small". Default: at most 2 MuJoCo processes at a time.
  The reach runner already uses one episode per worker process and reruns crashed checkpoints.
- No Docker on the laptop or the GCP image. Long MuJoCo runs can go to the GCP session VM
  (`cloud/isaac_session.py submit --env dream`) within the cloud cap: US$100 config cap, US$6 per
  session ceiling, one DREAM session at a time; propose anything larger to the owner first.
- Isaac Sim 6.2 quirks are in AGENTS.md / cloud tooling (offline uv lock `cloud/isaac62.lock.txt`,
  `file://` USD references, slow SimulationApp teardown).

## 3. What has been done (verify against git log; newest last)

| item | result | commits |
|---|---|---|
| W1 WBC command channels + envelope v2 | height/torso-rpy commands promoted per upstream contract; frozen 415-trial envelope, 5 seeds, Wilson intervals; `results/g1_wbc_envelope_v2/envelope.json` | ddcac4e, 41e99f4 |
| W2 unified 29+14 DoF G1 with Dex3 fingers | name-mapped actuator bridge (C2) | earlier on branch |
| W3 Mink posture planner + supervisor | `dream/robot/g1_posture.py`, `g1_supervisor.py`, `g1_rollout.py`, `g1_correction.py`, `wholebody.py` (C7 interface) | earlier on branch |
| W4 SONIC feasibility -> unofficial CPU port of GEAR-SONIC `gear_sonic_deploy` as second backend | `dream/robot/sonic.py` (NOT verified against the C++ binary; labelled) | b4ccef1 |
| W5 unfamiliar-height reach slice, v1 protocol | dev run 1 + fixes + held-out run once on frozen code 60138f7 | 126f87c, 10e1928, 60138f7, b550bba, 3c2a2e9 |
| S2 SONIC frozen envelope | 205/205 trials, seeds 20-24 | fb58b62, 2293425 |
| W6 general corrections + protocol v2 (in progress, see 4) | steps 1-4 committed, step 5 dev rerun was running at pause | a3ada5b, 62a5bdb, 8ee77d1, 1ae1c2a, d5040b7 |

Naming note: PLAN.md section 3 lists "W6 contact grasp" and "W7 retirement". In practice W6 became
"general corrections + protocol v2" after the W5 failure attribution. Renumber: contact grasp (D12) is
now **W7**, retirement of `isaac_task.py` (D9) is **W8**. Record this in PLAN.md.

### W5 v1 held-out result (frozen, scored under v1, do not rescore)
24 episodes, 17 reference-feasible, 7 infeasible. 1 certified success (s5011); 2 physical touches the
agent could not certify (s5000, s5022; claim threshold 23 mm sat under the 13-25 mm contact floor);
6/7 correct refusals; 14 feasible non-successes. Dev run 1: 5/24 feasible successes.

### W5 failure attribution (41 feasible, held-out + dev run 1; 6 successes, 35 lost)
1. Planner/rollout screens refused feasible targets: 14 (6 held-out zero-candidate plans: s5001, 5008,
   5009, 5010, 5015, 5020, although the reference stance lies on the agent's own grid; the 4 cm
   open-loop hand_error screen is a cliff - 21 of 35 rejections fall in 0.04-0.06 m).
2. Supervisor stops, mostly torque: 14 (peaks 1.0-1.32 x `jnt_actfrcrange`, static planner estimate
   0.27-0.45, i.e. 2-3x underprediction; mostly during correction).
3. Perception + navigation: 7 (reported target sigma ~4 mm vs real 35-70 mm; perceived fixture-box top
   0.3 m low).
4. Certification definition: 2.
Also: "reference feasible" used a denser, truth-input search, so "missed opportunity" mixed search
effort with capability. Scripts: `handoff/w5_analysis/`.

### S2 SONIC envelope (frozen; `results/sonic_envelope/`)
Only pelvis 0.55 and 0.62 m pass 5/5 (Wilson lower bound 0.57). 0.74-0.40 m fail height tracking only
(SONIC pelvis sits 4-7 cm above command; idle 0.787 m). Kneel cells (<=0.35 m) saturate torque, lose
foot support, slip ~0.9 m. All 24 push cells fail only via the height offset. Joint-reference mode
passes only the 1.05 m reference (hand FK error 5.5-17.7 cm elsewhere), so `kinds = ["posture"]`,
`height_m = [0.55, 0.62]`. Body has hands, WBC v2 did not: not mass-identical. Incidents (one 0-byte
trial quarantined and rerun) in `incidents.json`. Treat `experiments/sonic_envelope.py` and
`results/sonic_envelope*` as a frozen record.

## 4. W6 state at pause

W6 brief (what it was asked to do): (1) diagnose H3 - is zero-candidate planning caused by planner
inputs (covariance, approach_axis, perceived boxes) rather than the stance grid - by running `plan()`
offline with inputs swapped one at a time; record which joint triggers each torque/joint-limit stop;
(2) calibrate the target covariance on DEV data only, fix the fixture-box estimate if general;
(3) replace the open-loop 4 cm screen with a closed-loop rollout gate (same execute-then-correct
procedure and supervisor limits on the planning copy, gate on final predicted hand error and measured
peak torque/support, rank by measured torque, torque-aware corrections); (4) write protocol v2
`experiments/configs/g1_reach_heights_v2.json` before any v2 run (contact-based success: hand/handle
contact >= 0.3 s, no collapse, no unresolved stop, force limit; agent claim from Dex3 touch-sensor sites
+ FK; two reference labels; fresh held-out seeds 6000-6023; dev = v1 dev + v1 held-out seeds; gates
unchanged; backend "wbc"); (5) rerun the full dev set (36 dev + 24 v1-held-out scenes), report per-cause
changes against the attribution, then STOP before freezing.

Status:
- Steps 1-4 committed (a3ada5b H3 diagnosis script + joint-named stops; 62a5bdb planner: canonical
  obstacle boxes, run-time bounding volumes, shoulder-torso limit, axis fallback; 8ee77d1 covariance
  fitted on dev, unseen box tops treated as unknown; 1ae1c2a closed-loop rollout gate, measured-torque
  ranking, torque-aware corrections, tactile claim; d5040b7 protocol v2 draft, NOT frozen).
- Step 5 (dev rerun into `results/g1_reach_heights_v2_dev/`) was in progress at pause; partial
  episodes there. Untracked: `experiments/g1_reach_attribute.py`, `results/g1_reach_heights_v2_dev/`,
  `scratch/` (H3 diagnosis outputs `scratch/h3_*.json/.log`).
- Known pre-existing test failures (fail at the parent too, left alone): the slow v1-era
  `test_rollout_gate_then_monitored_real_execution...`, and `tests/test_isolation.py::skills`
  (`skills/g1_reach.py` imports mujoco).
- **Check this first:** the `config` block embedded in `results/g1_reach_heights_v2_dev/meta.json`
  shows the v1 `success_definition` (distance to handle centre <= 0.03 m). Verify whether the v2 runner
  actually scores with the contact-based definition (judge(..., success_mode)) or whether this is a
  stale config field; fix before trusting any v2 dev numbers.
- W6 pause report (the worker's own summary) is in section 7 below if present.

## 5. Next steps (in order)

1. **Luna, verify:** the W6 commits (tests pass, ruff clean), the meta.json success-definition
   question above, the H3 findings, and the state of the partial dev rerun. Report gaps.
2. **Terra, finish W6 step 5:** rerun (or resume) the full v2 dev set under the current code -
   36 dev + 24 v1-held-out scenes, agent + scripted baseline - with both reference labels. Report
   per-cause changes against the W5 attribution table (planner/rollout refusals, torque stops,
   perception/navigation, certification), joint-level stop causes, and the calibrated covariance.
   Stop.
3. **You decide** from the dev evidence whether more general corrections are needed (each with a
   hypothesis, a dev-only test and a regression test; never tune on held-out scenes). Then freeze:
   record code SHA and config hash, mark v2 config FROZEN.
4. **Terra, run v2 held-out once** (seeds 6000-6023, backend "wbc"), with your explicit go. Luna
   attributes failures afterwards exactly as in W5. Report feasible-reach success and correct refusals
   separately, with Wilson intervals (C6).
5. Then, per PLAN.md: SONIC as a planner backend only inside its measured envelope (and decide whether
   its height offset is a calibration issue worth an explicit, pre-registered re-measurement); **W7**
   contact-only Dex3 grasp (D12); **W8** retire `experiments/isaac_task.py` to a labelled scripted
   baseline (D9) and update README status (owner approval for README edits); Milestones D-F (VLA,
   open-vocabulary perception, discovery) under D10/D11/D13.

## 6. Integrity rules you must keep enforcing

- Scene truth only in `dream/eval`; decision modules see perceived inputs only (isolation tests).
- No assisted grasps in the new path; any assisted result is flagged and labelled.
- Gates and configs are written before runs; held-out seeds run exactly once on frozen code; seen
  held-out data becomes development evidence (v1 held-out seeds 5000-5023 already are).
- Every fix: hypothesis, evidence, regression test, Rule 3 class in the commit message.
- Unofficial ports and unmeasured regions are labelled as such; nothing outside a measured envelope is
  executed autonomously (C1).
- W5/W6 are geometric and control milestones with a structured target, not language-conditioned
  autonomy (C8).

## 7. W6 pause report (worker's own summary, 27 Oct; not yet independently verified)

- Steps 1-4 done and committed (SHAs in section 4). Step 5: 12 of 60 dev episodes done, all the
  h=0.35 scenes, all refused, all labelled infeasible by reference (i), so all are correct refusals.
  **Open question:** why reference (i) finds 0.35 m infeasible - check it before trusting the label.
- H3 result (offline `plan()`, saved perceived inputs, one input swapped at a time):
  - Covariance inflation and true target: no effect anywhere.
  - True fixture box flipped s5008/5009/5010/5020 from 0 to 13-17 candidates although the perceived
    box was within 4 mm. Cause: `mj_geomDistance` gave different mesh-box clearances for the same box
    described width-first vs depth-first. Fixed by canonical boxes + correct run-time bounding volumes.
  - Removing approach_axis flipped s5001 (0 to 14) and s5015 (0 to 6). Fixed by per-candidate axis
    relaxation.
  - After fixes all six zero-candidate episodes plan (14, 16, 16, 16, 6, 22 candidates).
- Joint-level stops: the planner could adduct the arm into the torso; the controller then pressed
  shoulder_yaw into the torso at ~70 N and shoulder-roll torque hit 1.1-1.5 x limit (static estimate
  0.25). A shoulder-to-torso limit removed it (peak 0.33).
- Two harness bugs fixed: the rollout never settled (38 spurious "foot support lost" stops at
  0.1-0.2 s), and the planning copy had zero odometry (first correction commanded 2.6 rad).
- Calibration (33 dev episodes): face-frame error rms 6.1 mm normal, 15.3 mm along the face, 1.1 mm
  vertical. Old isotropic model covered 64% of errors within 2 sigma; new model 85%.
- Smoke: dev s100 mid-height, rollout predicted ~8 mm final error; agent claimed 0.5 s tactile
  contact; evaluator confirmed 1.9 s contact at 9.8 N.
- Uncommitted: `experiments/g1_reach_attribute.py` (untested, 2 lint errors; copy in
  `handoff/w6_wip.patch`, note `handoff/w6_wip.NOTE.txt`); `results/g1_reach_heights_v2_dev/`
  (12/60, untracked); `scratch/` (worker scripts/logs; `ovl.py`, overlay and `scratch/headcopy` - a
  checkout of the parent commit used to confirm pre-existing test failures - are obsolete; delete
  `scratch/headcopy` to save disk once verified).
- Uninvestigated: in the real execution of dev s101 (h=1.10), a support-lost stop at 24.6 s although
  the rollout passed.
- Resume command (skips finished episodes):
  `python experiments/g1_reach_heights.py experiments/configs/g1_reach_heights_v2.json --set dev --system agent --workers 2`
  (log `scratch/dev_v2_agent.log`). Then the scripted baseline, then `agent_truth_inputs` on the
  non-successes with `--where-agent-failed`, then tidy and run
  `experiments/g1_reach_attribute.py --v2 results/g1_reach_heights_v2_dev`, write the per-cause
  report against the W5 attribution, and STOP before freezing v2.
- Two miniconda multiprocessing Python processes not started by DREAM workers were running at pause;
  leave them alone unless the owner says otherwise.
