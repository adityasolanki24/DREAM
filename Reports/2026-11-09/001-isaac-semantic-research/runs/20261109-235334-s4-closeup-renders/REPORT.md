# Run report: Session 4 job 4: close-up marker renders of bed, workbench, desk, shelf, sink

- Run ID: 2026-11-09-001/20261109-235334-s4-closeup-renders
- Parent task: 2026-11-09-001
- Started: 2026-11-09T23:53:34+11:00
- Ended: 2026-11-10T04:44:58+11:00
- Git commit: 3f9963b73cd3405c47cdfcce8f961804153bfe21 (dirty)
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_asset_candidates.json (ccc98bac1988b3a483b6e20e5b0a48bdfe4123081d44bc1a99d097e87e098804)
- Seeds: n/a
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: none
- Robot controller: none
- Hardware: GCP g2-standard-8 L4
- Command: `isaac_asset_validation.py --indexes bed:0,workbench:0,desk:0,bookshelf:1,sink:1 --dist-factor 0.9`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 5
- Cases completed: 5

## Purpose

Close-up orientation-marker renders (camera distance factor 0.9) of sink, bed, desk, bookshelf and workbench to read their fronts.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Completed in 35 seconds of job time, all five assets validated again. Fronts read: sink -y (door and handle; fixtures on the +y back), workbench +y (drawers and shelves; pegboard at the back), bed has its long axis along x with ends at +-x (foot end -x chosen, low confidence; the earlier -y was wrong), desk and shelf are symmetric (+x chosen). experiments/configs/isaac_asset_fronts.json updated.

## Regressions observed

None.

## Interruptions and failures

None in the job. Driver notes: the four jobs were submitted back to back by a script whose submit processes for jobs 3 and 4 gave up waiting (their wait deadline counts from submission, the jobs queued behind jobs 1 and 2), so jobs 3 and 4 have no per-job output folder; the jobs themselves ran (exit 0) and their outputs were retrieved by `stop` into the session's out/jobs/3 and out/jobs/4. An earlier version of the driver issued `stop` by mistake 5 minutes after the start (a VM-presence check returned empty); I killed it and removed the STOP marker before the guest started any job, so no job was lost.

## Evidence

results/isaac_env_validation/closeup/*.json; experiments/configs/isaac_asset_fronts.json.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/4/val (dir, sha256 1b996ce057d36694264962f8a7888b7038005ce1593485a613fb5668cc82957f)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (zone a)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-235410-1d7369
- job_ids: 4
- max_lifetime_min: 140
- estimated_runtime_min: 10
- actual_runtime_min: 1
- estimated_cost_usd: 4.79
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-item JSON and per-frame files written as produced
- evidence_location: %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/<n> (outside Git and OneDrive); compact copies in results/isaac_home_dev, results/isaac_env_validation/closeup, manifests in results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified by `stop` before bucket removal
- cleanup_status: verified by the coordinator: no DREAM VM, disk, bucket, network, router or service account; cleanup_errors empty
- resources_remaining: runtime image only
