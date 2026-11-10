# Reports for 2026-11-10

[Root index](../INDEX.md)

## Tasks created

| Date | Task ID | Prompt | Status | Main finding | Runs | Report |
|---|---|---|---|---|---|---|
| 2026-11-10 | 2026-11-10-006 | [Resolve the R1 tabletop grasp failure](006-r1-tabletop-grasp/PROMPT.md) | partially_completed | The guarded R1 approach prevents the reproduced tabletop-directed lift. The right index trajectory still loses clearance during the raise, and no box lift was verified. | 13 | [REPORT](006-r1-tabletop-grasp/REPORT.md) |
| 2026-11-10 | 2026-11-10-005 | [Continue the sequential G1 roadmap through the R0 gate](005-r0-integration-gate/PROMPT.md) | partially_completed | R0 passed the full suite and was published. R1 reached and closed its fingers, but independent evidence showed tabletop contact and no box lift. | 3 | [REPORT](005-r0-integration-gate/REPORT.md) |
| 2026-11-10 | 2026-11-10-004 | [Implement the G1 roadmap one validated rung at a time](004-sequential-roadmap-gates/PROMPT.md) | partially_completed | Ruff and the reproduced reach regression passed. Full R0 verification continued in a new task after interruption. | 5 | [REPORT](004-sequential-roadmap-gates/REPORT.md) |
| 2026-11-10 | 2026-11-10-003 | [Start the adopted G1 roadmap](003-r0-r1-start/PROMPT.md) | partially_completed | Started R0 and R1 on the integration branch. The fixed-stance G1 reached monitored arm execution before a right wrist pitch torque stop. Grasp and lift remain unverified. | 4 | [REPORT](003-r0-r1-start/REPORT.md) |
| 2026-11-10 | 2026-11-10-001 | [Strategic audit against the end goal and a plan for future agents](001-strategic-audit-and-plan/PROMPT.md) | completed | The G1 has never grasped an object with its own hands under physics and nothing has advanced that since 6 November; plan R0-R7 drafted, awaiting owner adoption | 0 | [REPORT](001-strategic-audit-and-plan/REPORT.md) |

## Continued on this day

- [2026-11-09-001](../2026-11-09/001-isaac-semantic-research/REPORT.md): Permanent Reports system; Isaac semantic exploration with learned perception, knowledge sources and larger homes

## Runs started

| Run | Task | Status | Started | Report |
|---|---|---|---|---|
| 20261110-115046-pose-baseline | 2026-11-10-006 | failed | 2026-11-10T11:50:46+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115046-pose-baseline/REPORT.md) |
| 20261110-115238-grasp-frame | 2026-11-10-006 | completed | 2026-11-10T11:52:38+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115238-grasp-frame/REPORT.md) |
| 20261110-115328-pinch-stance | 2026-11-10-006 | completed | 2026-11-10T11:53:28+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115328-pinch-stance/REPORT.md) |
| 20261110-115415-hand-proxy-control | 2026-11-10-006 | completed | 2026-11-10T11:54:15+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115415-hand-proxy-control/REPORT.md) |
| 20261110-115504-articulated-guard | 2026-11-10-006 | completed | 2026-11-10T11:55:04+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115504-articulated-guard/REPORT.md) |
| 20261110-115544-raise-waypoint | 2026-11-10-006 | completed | 2026-11-10T11:55:44+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115544-raise-waypoint/REPORT.md) |
| 20261110-115639-ready-fixture | 2026-11-10-006 | completed | 2026-11-10T11:56:39+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115639-ready-fixture/REPORT.md) |
| 20261110-115717-ready-native-legs | 2026-11-10-006 | completed | 2026-11-10T11:57:17+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115717-ready-native-legs/REPORT.md) |
| 20261110-115821-cartesian-feedback | 2026-11-10-006 | completed | 2026-11-10T11:58:21+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-115821-cartesian-feedback/REPORT.md) |
| 20261110-120001-nominal-posture | 2026-11-10-006 | completed | 2026-11-10T12:00:01+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-120001-nominal-posture/REPORT.md) |
| 20261110-120055-narrow-posture | 2026-11-10-006 | completed | 2026-11-10T12:00:55+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-120055-narrow-posture/REPORT.md) |
| 20261110-120137-clearance-link | 2026-11-10-006 | completed | 2026-11-10T12:01:37+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-120137-clearance-link/REPORT.md) |
| 20261110-120235-compact-raise | 2026-11-10-006 | completed | 2026-11-10T12:02:35+11:00 | [REPORT](006-r1-tabletop-grasp/runs/20261110-120235-compact-raise/REPORT.md) |
| 20261110-114509-wrist-torque | 2026-11-10-005 | completed | 2026-11-10T11:45:09+11:00 | [REPORT](005-r0-integration-gate/runs/20261110-114509-wrist-torque/REPORT.md) |
| 20261110-114557-hip-clearance | 2026-11-10-005 | completed | 2026-11-10T11:45:57+11:00 | [REPORT](005-r0-integration-gate/runs/20261110-114557-hip-clearance/REPORT.md) |
| 20261110-114645-both-arms-clear | 2026-11-10-005 | completed | 2026-11-10T11:46:45+11:00 | [REPORT](005-r0-integration-gate/runs/20261110-114645-both-arms-clear/REPORT.md) |
| 20261110-103624-rollback-diagnosis | 2026-11-10-004 | failed | 2026-11-10T10:36:24+11:00 | [REPORT](004-sequential-roadmap-gates/runs/20261110-103624-rollback-diagnosis/REPORT.md) |
| 20261110-103738-rollback-arm-only | 2026-11-10-004 | failed | 2026-11-10T10:37:38+11:00 | [REPORT](004-sequential-roadmap-gates/runs/20261110-103738-rollback-arm-only/REPORT.md) |
| 20261110-103816-rollback-hold | 2026-11-10-004 | completed | 2026-11-10T10:38:16+11:00 | [REPORT](004-sequential-roadmap-gates/runs/20261110-103816-rollback-hold/REPORT.md) |
| 20261110-103937-rollback-stability | 2026-11-10-004 | completed | 2026-11-10T10:39:37+11:00 | [REPORT](004-sequential-roadmap-gates/runs/20261110-103937-rollback-stability/REPORT.md) |
| 20261110-104028-rollback-independent | 2026-11-10-004 | completed | 2026-11-10T10:40:28+11:00 | [REPORT](004-sequential-roadmap-gates/runs/20261110-104028-rollback-independent/REPORT.md) |
| 20261110-102246-nominal-grasp | 2026-11-10-003 | completed | 2026-11-10T10:22:46+11:00 | [REPORT](003-r0-r1-start/runs/20261110-102246-nominal-grasp/REPORT.md) |
| 20261110-102409-closer-stance | 2026-11-10-003 | completed | 2026-11-10T10:24:09+11:00 | [REPORT](003-r0-r1-start/runs/20261110-102409-closer-stance/REPORT.md) |
| 20261110-102544-contact-attribution | 2026-11-10-003 | completed | 2026-11-10T10:25:44+11:00 | [REPORT](003-r0-r1-start/runs/20261110-102544-contact-attribution/REPORT.md) |
| 20261110-102638-tucked-stance | 2026-11-10-003 | completed | 2026-11-10T10:26:38+11:00 | [REPORT](003-r0-r1-start/runs/20261110-102638-tucked-stance/REPORT.md) |
| 20261110-044518-unexplored-rooms-dev-check | 2026-11-09-001 | completed | 2026-11-10T04:45:18+11:00 | [REPORT](../2026-11-09/001-isaac-semantic-research/runs/20261110-044518-unexplored-rooms-dev-check/REPORT.md) |
