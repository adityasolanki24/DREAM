# Run report: Isaac asset front-face renders of the 27 chosen assets and ready-made scene inventory (cloud session 2, job 1)

- Run ID: 2026-11-09-001/20261109-155656-asset-front-renders-and-scene-inventory
- Parent task: 2026-11-09-001
- Started: 2026-11-09T15:56:56+11:00
- Ended: 2026-11-09T18:58:55+11:00
- Git commit: cfc6ce97fde639a2efab382c2548b978818a8362
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_asset_candidates.json (714054c432ed93c87ef3ccc517f53314a101e64105dd7d3e023928c86991a353)
- Seeds: n/a
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: none
- Robot controller: none
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_asset_validation.py --indexes <27 chosen candidates> --environments --budget-s 900`
- Exit status: 0
- Completed (yes/no/partial): partial (partial)
- Cases attempted: 27
- Cases completed: 16

## Purpose

Render four views of the 27 chosen assets so that each one's front face can be read, and inventory ready-made scenes.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Partial. 16 of 27 chosen assets rendered inside the 900 s budget (refrigerator through coffee_table), all with failures=[] again; the budget stopped the job before armchair onward and before the scene inventory. The renders showed that the snapshot loop lags the camera: the shelf's native extents (0.51 x 1.51 m) match the images only if each image shows the side one step earlier in the px, py, nx, ny order, and the first image of every asset is framed from the previous asset's pose. So front faces read from these renders are unverified.

## Regressions observed

None.

## Interruptions and failures

Budget stop by design (exit 0). The stale-frame defect of job 2 of session 1 was only partly fixed: the extra updates did not remove a one-view lag.

## Evidence

Job 1 output in the cloud run folder named in the Cloud section; snapshots are not committed.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/1/out/val (dir, sha256 26489d333931bc0e248e88236b8cb18009a9ebd68fda915b6b545ab08a231d6d)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone b)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-160004-5f4423
- job_ids: 1
- max_lifetime_min: 170
- estimated_runtime_min: 15
- actual_runtime_min: 15
- estimated_cost_usd: 7.4
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-candidate / per-item JSON and frames written as produced
- evidence_location: raw job outputs under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-160004-5f4423/jobs/<n>/out (outside Git and OneDrive); compact copies in results/isaac_kitchen_repair, results/isaac_env_validation, results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified before bucket removal
- cleanup_status: verified independently after stop: no DREAM VM, disk, address, router, network, bucket or service account remains; runtime image retained by design
- resources_remaining: runtime image only
