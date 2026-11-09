# Run report: Session 4 job 3: l_shape capture, several trajectories, both resolutions

- Run ID: 2026-11-09-001/20261109-235334-s4-lshape-capture
- Parent task: 2026-11-09-001
- Started: 2026-11-09T23:53:34+11:00
- Ended: 2026-11-10T04:44:57+11:00
- Git commit: 3f9963b73cd3405c47cdfcce8f961804153bfe21 (dirty)
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_home_dev.json (1ffe0bc74057d081935aee1ad39788ead3c9fec427972cf4c859f033f8f792f9)
- Seeds: l_shape start seeds 1, 2, 3, 4 (development kitchen)
- Model and checkpoint: Qwen/Qwen3-VL-4B-Instruct ebb281ec (recognisability only)
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: Unitree G1
- Robot controller: NVIDIA Decoupled WBC
- Hardware: GCP g2-standard-8 L4
- Command: `isaac_home_runs.py --no-homes --kitchens l_shape --budget-s 1500`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

l_shape perception capture in several trajectories at both resolutions, target at least 150 frames.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Completed: 174 frames in 4 trajectories (35, 50, 49, 40 frames), 20.1 m of path, no fall, 174 frames at 320x240 and 174 at 640x480 from identical poses (190.4 MB), with odometry, true pose and camera, and segmentation ground truth (8 labelled instances in the first frame) at both resolutions. Qwen3-VL was not run during capture. Recognisability afterwards (development evidence): kitchen_cabinet 2 of 3 views with a detection (none at IoU 0.5), sink 0 of 2. It ran in 21 minutes of the 28 allowed.

## Regressions observed

None.

## Interruptions and failures

None in the job. Driver notes: the four jobs were submitted back to back by a script whose submit processes for jobs 3 and 4 gave up waiting (their wait deadline counts from submission, the jobs queued behind jobs 1 and 2), so jobs 3 and 4 have no per-job output folder; the jobs themselves ran (exit 0) and their outputs were retrieved by `stop` into the session's out/jobs/3 and out/jobs/4. An earlier version of the driver issued `stop` by mistake 5 minutes after the start (a VM-presence check returned empty); I killed it and removed the STOP marker before the guest started any job, so no job was lost.

## Evidence

results/isaac_home_dev/s4_job3_*.json; capture manifest results/isaac_perception_capture/manifest_s4_l_shape.json (includes the format documentation).

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/3/runs (dir, sha256 f91a7120c8ed29aab1943c0fde1cf1583dfc5822f84601fca16e196e0cffce50)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (zone a)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-235410-1d7369
- job_ids: 3
- max_lifetime_min: 140
- estimated_runtime_min: 28
- actual_runtime_min: 21.5
- estimated_cost_usd: 4.79
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-item JSON and per-frame files written as produced
- evidence_location: %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-235410-1d7369/out/jobs/<n> (outside Git and OneDrive); compact copies in results/isaac_home_dev, results/isaac_env_validation/closeup, manifests in results/isaac_perception_capture
- upload_status: outputs retrieved and SHA-256 verified by `stop` before bucket removal
- cleanup_status: verified by the coordinator: no DREAM VM, disk, bucket, network, router or service account; cleanup_errors empty
- resources_remaining: runtime image only
