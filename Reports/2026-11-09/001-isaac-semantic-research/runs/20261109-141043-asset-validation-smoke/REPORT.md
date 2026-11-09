# Run report: Isaac asset validation smoke (3 classes) in cloud session 1

- Run ID: 2026-11-09-001/20261109-141043-asset-validation-smoke
- Parent task: 2026-11-09-001
- Started: 2026-11-09T14:10:43+11:00
- Ended: 2026-11-09T15:30:34+11:00
- Git commit: a1173733c72ecf25acae14065e29652a07e0d9f9
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_asset_candidates.json (714054c432ed93c87ef3ccc517f53314a101e64105dd7d3e023928c86991a353)
- Seeds: n/a (asset validation)
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: none
- Robot controller: none
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_asset_validation.py --classes refrigerator,toilet,bathtub --max-per-class 2`
- Exit status: 0
- Completed (yes/no/partial): partial (partial)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Smoke test of the asset validator (experiments/isaac_asset_validation.py) on three classes (refrigerator, toilet, bathtub, two candidates each) before the full run, to see that the Isaac-side code path works at all.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
1 candidate attempted and completed (Residential Refrigerator_A01): it resolves, composes, has 20 meshes and a finite normalised bound of about 0.67 x 1.23 x 1.52 m. The record nevertheless shows three defects of the validator itself, not of the asset: (1) the physics play with the raw asset took 664 s (PhysX cooking of 20 SDF colliders), (2) the asset's dynamic bodies fell 2.5 m (bound after play z -2.46 to -0.94) yet the record listed no failure, (3) the snapshot list was empty. The job stopped admitting candidates at its 600 s budget, so toilet and bathtub were not attempted. Status partial: 1 of 6 planned candidates.

## Regressions observed

None (new tool).

## Interruptions and failures

Not interrupted. Budget reached by design after the first candidate. The validator defects above were fixed before the next run (commit 6cb869d): static treatment with a bounding-box collider, replicator-step rendering, a movement check.

## Evidence

Job 1 output copied to %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/1/out/val (checksummed in this run's metadata). Its only candidate record is superseded by the later run and kept under results/isaac_env_validation/assets/refrigerator only where not superseded.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/1/out/val (dir, sha256 012513446dda3ad51be6aadd9fd9bb70fc8fef5602ff1af6a55f967ad6796989)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone b after a-zone stockout)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-141110-8b500f
- job_ids: 1
- max_lifetime_min: 75
- estimated_runtime_min: 10
- actual_runtime_min: 12
- estimated_cost_usd: 4.97
- billed_cost_usd: not read: no billing data was consulted; the figure above is an estimate, not a charge
- checkpoint_status: per-candidate JSON written before the next candidate started
- evidence_location: raw job outputs under %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-141110-8b500f/jobs/<n>/out (outside Git and OneDrive); compact per-candidate records in results/isaac_env_validation/assets
- upload_status: outputs retrieved and SHA-256 verified before bucket removal
- cleanup_status: verified: no DREAM VM, disk, address, router, NAT, network, bucket or service account remains; runtime image retained by design
- resources_remaining: runtime image only
