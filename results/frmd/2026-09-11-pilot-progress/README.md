# Tokendrop pilot progress snapshot

Snapshot: 2026-09-11T06:47:21.169901+00:00. 82/90 native cases and 36/90 physical cases completed. Other cases remain running or queued.

Raw reports are preserved byte-for-byte in deterministic gzip. `snapshot.json` references existing published files when their uncompressed SHA-256 is identical. Paths are relative to the repository root. Physical sources and evaluation seeds were validated against native reports.

This is a partial development snapshot; task blocks have unequal coverage. No fixed outcome threshold stops subsequent experiments. All controls and neutral/negative outcomes are retained. Offline persistence and one-chunk physics do not establish closed-loop task success.

| Task | Native cases | Mean recovery | Mean offline persistence |
| --- | ---: | ---: | ---: |
| PatternLock | 30 | +0.00089773 | +0.00025440 |
| RouteStick | 30 | +0.00102874 | -0.00008546 |
| VideoPlaceButton | 22 | +0.00000536 | +0.00032663 |

Physical conditions (noise replicas averaged within each case):

| Condition | Success fraction | Subgoal advancement | Ongoing fraction |
| --- | ---: | ---: | ---: |
| interfered | 0.038194 | 0.104167 | 0.937500 |
| repaired | 0.038194 | 0.104167 | 0.937500 |
| teacher | 0.083333 | 0.187500 | 0.916667 |
| interpolation | 0.038194 | 0.104167 | 0.937500 |
| negative | 0.038194 | 0.104167 | 0.937500 |
| random | 0.038194 | 0.104167 | 0.937500 |

FRMD and the interfered baseline have equal case-averaged success and subgoal advancement in this snapshot. Ongoing chunks are reported separately from terminal failures. See `analysis.json` for case-level physical comparisons and `case-metrics.csv` for all action controls.
