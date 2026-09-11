# Complete tokendrop native development pilot

All 90 native cases from 30 source episodes completed. Snapshot: 2026-09-11T07:13:02.335766+00:00. Physical coverage: 40/90; completed bounded RGB first cases: 2/3.

| Task | Cases | Action recovery | Offline persistence |
| --- | ---: | ---: | ---: |
| PatternLock | 30 | +0.00089773 | +0.00025440 |
| RouteStick | 30 | +0.00102874 | -0.00008546 |
| VideoPlaceButton | 30 | +0.00037540 | +0.00029689 |

Means include ten clean k=0 cases per task. Small positive action recovery is not evidence of superiority over controls. The complete raw records retain random, negative and interpolation controls. Persistence remains mixed.

| Task | Condition | Status | Steps | Subgoals advanced |
| --- | --- | --- | ---: | ---: |
| PatternLock | interfered | fail | 29 | 0 |
| PatternLock | repaired | fail | 29 | 0 |
| PatternLock | teacher | success | 51 | 2 |
| PatternLock | interpolation | fail | 29 | 0 |
| PatternLock | negative | fail | 29 | 0 |
| PatternLock | random | fail | 29 | 0 |
| RouteStick | interfered | success | 121 | 2 |
| RouteStick | repaired | success | 121 | 2 |
| RouteStick | teacher | success | 99 | 2 |
| RouteStick | interpolation | success | 121 | 2 |
| RouteStick | negative | success | 121 | 2 |
| RouteStick | random | success | 121 | 2 |

Closed-loop conditions share the same initial physical and RGB/state fingerprints. The maximum is ten queries; a branch ends earlier on terminal success/failure. These are first-case development checks with one fresh diffusion seed, not a full closed-loop benchmark.

All uncompressed source SHA-256 values and repository-relative paths are in `summary.json`. Existing identical reports are reused. JSON gzip exports preserve original bytes. The native first-case reports used for closed loop are earlier independent runs, explicitly linked by source hash; they must not be substituted with same-named later pilot reports.

Remaining physical, frame-sampling, and both-mechanism ablation jobs continue without fixed outcome gates. The queue snapshot retains an earlier preflight error whose successful v2 correction was already published.
