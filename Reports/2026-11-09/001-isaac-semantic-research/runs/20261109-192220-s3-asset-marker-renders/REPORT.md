# Run report: Session 3 job 1: orientation-marker renders and re-validation of the chosen assets (Island counter, vanity sink)

- Run ID: 2026-11-09-001/20261109-192220-s3-asset-marker-renders
- Parent task: 2026-11-09-001
- Started: 2026-11-09T19:22:20+11:00
- Ended: 2026-11-09T23:48:27+11:00
- Git commit: 9d8a9e3c4959dfd2623dc9f22180387f60837f43
- Branch: worktree-agent-a2e255eadffeafc65
- Config files and SHA-256: experiments/configs/isaac_asset_candidates.json (ccc98bac1988b3a483b6e20e5b0a48bdfe4123081d44bc1a99d097e87e098804)
- Seeds: n/a
- Model and checkpoint: none
- Simulator and version: Isaac Sim 6.2
- Renderer: RaytracedLighting, headless
- Robot embodiment: none
- Robot controller: none
- Hardware: GCP g2-standard-8 L4
- Command: `cloud/isaac_session.py submit experiments/isaac_asset_validation.py --indexes <23 chosen candidates> --budget-s 1500`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 22
- Cases completed: 22

## Purpose

Render four views of 22 chosen assets with coloured orientation markers (+x red, -x yellow, +y green, -y blue on the ground just outside each asset) so every image names its own camera side, fix the one-view render lag of session 2, and re-validate the chosen assets including the new Island_A01 counter and the vanity sink.

Render four views of 22 chosen assets with coloured orientation markers (+x red, -x yellow, +y green, -y blue on the ground just outside each asset) so every image names its own camera side, fix the one-view render lag of session 2, and re-validate the chosen assets including the new Island_A01 counter and the vanity sink.

## Results

Completed in 16.5 minutes. All 22 assets validated with no failure, including countertop = Island_A01 (0.63 x 1.24 x 0.90 m, matching the generator counter) and sink = Vanity_Sink_Faucet. Render lag fixed: in all 88 images the marker nearest the camera is the camera's own side, and no image was re-rendered more than twice. The automatic marker-order check returned True where both lateral markers were visible and never False; it had no verdict in about half the views, so fronts were read by eye from the markers. Fronts: 13 classes clear (refrigerator, stove, microwave, kitchen_cabinet, wardrobe, nightstand, chair, office_chair, washing_machine, filing_cabinet, sofa, armchair, toilet), 10 symmetric or ambiguous with an arbitrary marker-referenced choice (countertop, sink, dining_table, bed, desk, bookshelf, storage_shelf, coffee_table, washbasin, workbench). One unexplained inconsistency: the shelf's recorded bound (0.51 x 1.51 m) does not match how wide it looks in the renders.

<!-- Passed and failed cases; quantitative metrics. -->
Completed in 16.5 minutes. All 22 assets validated with no failure, including countertop = Island_A01 (0.63 x 1.24 x 0.90 m, matching the generator counter) and sink = Vanity_Sink_Faucet. Render lag fixed: in all 88 images the marker nearest the camera is the camera's own side, and no image was re-rendered more than twice. The automatic marker-order check returned True where both lateral markers were visible and never False; it had no verdict in about half the views, so fronts were read by eye from the markers. Fronts: 13 classes clear (refrigerator, stove, microwave, kitchen_cabinet, wardrobe, nightstand, chair, office_chair, washing_machine, filing_cabinet, sofa, armchair, toilet), 10 symmetric or ambiguous with an arbitrary marker-referenced choice (countertop, sink, dining_table, bed, desk, bookshelf, storage_shelf, coffee_table, washbasin, workbench). One unexplained inconsistency: the shelf's recorded bound (0.51 x 1.51 m) does not match how wide it looks in the renders.

## Regressions observed

None.

None.

## Interruptions and failures

None; exit 0.

None; exit 0.

## Evidence

results/isaac_env_validation/assets (22 records, with per-view marker verdicts), experiments/configs/isaac_asset_fronts.json, experiments/configs/isaac_assets.json.

- C:/Users/24adi/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Local/DREAM/cloud_runs/dream-qwen-isaac-20261109-192221-327e9a/jobs/1/out/val (dir, sha256 132cb0912cfee6af88254d483860908538aead96e830cf34b7c2bca930f34019)

## Cloud

- provider: GCP
- project: nerva-adityapersonal
- region: asia-southeast1 (VM in zone c)
- machine_type: g2-standard-8
- gpu_type: NVIDIA L4
- gpu_count: 1
- session_id: dream-qwen-isaac-20261109-192221-327e9a
- job_ids: 1
- max_lifetime_min: 175
- estimated_runtime_min: 25
- actual_runtime_min: 16.5
- estimated_cost_usd: 7.53
- billed_cost_usd: not read: an estimate above, not a charge
- checkpoint_status: per-asset / per-item JSON written as produced
- evidence_location: %LOCALAPPDATA%/DREAM/cloud_runs/dream-qwen-isaac-20261109-192221-327e9a/jobs/<n>/out (outside Git and OneDrive); compact copies in results/isaac_env_validation/assets and results/isaac_home_dev
- upload_status: outputs retrieved and SHA-256 verified by `stop` (run by the coordinator) before bucket removal
- cleanup_status: verified independently by the coordinator: no DREAM VM, disk, address, router, network, bucket or service account; runtime image kept
- resources_remaining: runtime image only
