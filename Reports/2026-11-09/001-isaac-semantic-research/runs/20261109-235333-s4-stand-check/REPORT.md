# Run report: Session 4 job 1: G1 standing check in composed homes 100 B and 101 D (ground and scene fix)

- Run ID: 2026-11-09-001/20261109-235333-s4-stand-check
- Parent task: 2026-11-09-001
- Started: 2026-11-09T23:53:33+11:00
- Ended: 2026-11-10T04:44:57+11:00
- Git commit: 3f9963b73cd3405c47cdfcce8f961804153bfe21
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_home_dev.json (1ffe0bc74057d081935aee1ad39788ead3c9fec427972cf4c859f033f8f792f9)
- Seeds: home generator seed 100 (B, refrigerator) and 101 (D, sofa)
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: Unitree G1
- Robot controller: NVIDIA Decoupled WBC
- Hardware: GCP g2-standard-8 L4
- Command: `isaac_home_runs.py --homes 100,101 --stand-only --budget-s 900`
- Exit status: 0
- Completed (yes/no/partial): no (failed)
- Cases attempted: 1
- Cases completed: 0

## Purpose

Standing-only check of the G1 in composed homes after the ground and scene fix (one ground collider at z=0, scene at /physicsScene, 0.25 s height trace, spawn diagnostics).

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Home 100 B refrigerator: the G1 still did not stand, but no longer launched: torso 0.46 m at 0.25 s, pelvis tilt 118 degrees. The spawn volume was free of colliders. The live stage listed FOUR physics scenes: ours at /physicsScene and one inside each of three placed assets (/World/Home/o007..o009/Physics/PhysicsScene). Only home 100 was run in this job (the budget left no time for the second item: the job ran 12 minutes, mostly loading).

## Regressions observed

None.

## Interruptions and failures

Standing failed. Per the coordinator, cloud debugging of the launch stops after this job; the fix was not tried in this job.

## Evidence

results/isaac_home_dev/s4_job1_*.json.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/1/runs (dir, sha256 09fb05bc096920ba0045bed1a5952e3444bb5fe1547aa31c5e810564a4f0fe54)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (zone a)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-235410-1d7369
- job_ids: 1
- max_lifetime_min: 140
- estimated_runtime_min: 10
- actual_runtime_min: 12
- estimated_cost_usd: 4.79
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-item JSON and per-frame files written as produced
- evidence_location: %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/<n> (outside Git and OneDrive); compact copies in results/isaac_home_dev, results/isaac_env_validation/closeup, manifests in results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified by `stop` before bucket removal
- cleanup_status: verified by the coordinator: no DREAM VM, disk, bucket, network, router or service account; cleanup_errors empty
- resources_remaining: runtime image only
