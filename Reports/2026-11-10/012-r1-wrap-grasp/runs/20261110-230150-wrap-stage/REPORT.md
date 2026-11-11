# Run report: Compare pinch and wrap articulation with wrist roll around round objects

- Run ID: 2026-11-10-012/20261110-230150-wrap-stage
- Parent task: 2026-11-10-012
- Started: 2026-11-10T23:01:50+11:00
- Ended: 2026-11-10T23:02:36+11:00
- Git commit: acc45456ad2c216390d156e2dc43ae01b72debab (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_sphere.json (7765331ea21723f3cf865a23638312077868948214294e28a2c429ecd078ac14); experiments/configs/g1_grasp_bringup_cylinder.json (ecb6de06dd040e4a769537eaa933a0462ccfe70d962cf308c713a20fb345dcd8)
- Seeds: 43,41
- Model and checkpoint: Dex3 articulation
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Dex3 rehearsal stage
- Robot controller: Force-limited vertical stage
- Hardware: local CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-wrap-stage-001/screen.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 6
- Cases completed: 6

## Purpose

Try proximal-first wrapping and distal hooking with larger wrist rolls, while retaining the real finger and table collision geometry.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The sphere had zero stage holds across 195 screened poses and closure profiles. The cylinder retained 21 pinch, 19 proximal-wrap and 42 distal-hook candidates in rehearsal.

## Regressions observed

Live performance was not measured in this run. The cylinder served as a positive control for the stage.

## Interruptions and failures

The sphere continued to slip despite stable opposing closure in 194 of 195 candidates. No alternate articulation was adopted on that evidence.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-wrap-stage-001 (dir, sha256 a0140476cff5440040b03d970a424c02e09ee9530967ba9566fb2b25114540c0)
